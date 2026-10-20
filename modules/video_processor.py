"""
Video Processor Module.

Handles local video ingestion from live webcams and recorded video files using OpenCV.
Extracts video metadata (dimensions, frame rate, frame count), manages frame retrieval
with configurable rate pacing, tracks anonymous individuals, computes group motion dynamics,
and guarantees clean resource release without saving raw media to disk.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from config.settings import settings
from modules.person_tracker import CentroidTracker, Track

logger = logging.getLogger(__name__)


class VideoStatus(str, Enum):
    """Operational status of the video processor."""
    UNINITIALIZED = "UNINITIALIZED"
    OPENED = "OPENED"
    PROCESSING = "PROCESSING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass
class VideoMetadata:
    """Metadata and hardware specifications for a video source."""
    width: int = 0
    height: int = 0
    source_fps: float = 0.0
    total_frames: int = 0  # 0 for live streams
    duration_seconds: float = 0.0
    is_live: bool = False
    source_description: str = ""

    @property
    def resolution_str(self) -> str:
        """Formatted resolution string (e.g. '1280x720')."""
        if self.width > 0 and self.height > 0:
            return f"{self.width}x{self.height}"
        return "Unknown"


class VideoProcessor:
    """
    Manages OpenCV VideoCapture streams for webcams and video files.
    Decouples frame decoding, person detection, tracking, and rate pacing.
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        target_fps: Optional[int] = None,
        tracker: Optional[CentroidTracker] = None,
        allow_save_frames: bool = False,
        custom_detector: Optional[Callable[[np.ndarray], List[Tuple[int, int, int, int]]]] = None,
    ) -> None:
        """
        Initialize VideoProcessor.

        Args:
            source: Camera device index (int) or path to video file (str/Path).
            target_fps: Target processing frame rate (defaults to settings.video.target_fps).
            tracker: CentroidTracker instance for anonymous trajectory tracking.
            allow_save_frames: Guardrail flag. If False, saving frames to disk is prohibited.
            custom_detector: Optional callable returning List of (x, y, w, h) bounding boxes.
        """
        self.source = self._normalize_source(source)
        raw_fps = target_fps if target_fps is not None else settings.video.target_fps
        self.target_fps = max(1, int(raw_fps))
        self.tracker = tracker or CentroidTracker()
        self.allow_save_frames = allow_save_frames
        self.custom_detector = custom_detector

        self.cap: Optional[cv2.VideoCapture] = None
        self.metadata = VideoMetadata()
        self.status = VideoStatus.UNINITIALIZED
        self.error_message: Optional[str] = None
        
        self.frame_index: int = 0
        self.last_frame_time: float = 0.0
        self.prev_gray_frame: Optional[np.ndarray] = None

        # Safe detector initialization: check if Haar cascade XMLs exist on disk
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        self.upper_body_cascade: Optional[cv2.CascadeClassifier] = None

        face_xml = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        if Path(face_xml).exists():
            fc = cv2.CascadeClassifier(face_xml)
            if not fc.empty():
                self.face_cascade = fc

        body_xml = getattr(cv2.data, "haarcascades", "") + "haarcascade_upperbody.xml"
        if Path(body_xml).exists():
            bc = cv2.CascadeClassifier(body_xml)
            if not bc.empty():
                self.upper_body_cascade = bc

        # Built-in HOG People detector (guaranteed zero external file dependency)
        self.hog = cv2.HOGDescriptor()
        detector = (
            cv2.HOGDescriptor.getDefaultPeopleDetector()
            if hasattr(cv2.HOGDescriptor, "getDefaultPeopleDetector")
            else cv2.HOGDescriptor_getDefaultPeopleDetector()
        )
        self.hog.setSVMDetector(detector)

    @staticmethod
    def _normalize_source(source: Union[int, str, Path]) -> Union[int, str]:
        """Convert input source into an integer device index or resolved path string."""
        if isinstance(source, Path):
            return str(source.resolve())
        if isinstance(source, str) and source.strip().isdigit():
            return int(source.strip())
        return source

    def open_source(self, source: Optional[Union[int, str, Path]] = None) -> bool:
        """
        Open the video source (webcam device or video file) and inspect metadata.

        Args:
            source: Optional override for the video source.

        Returns:
            True if source opened successfully, False otherwise.
        """
        # Release any previously opened capture
        self.release()

        if source is not None:
            self.source = self._normalize_source(source)

        try:
            # Validate video file existence if a path string is provided
            if isinstance(self.source, str):
                file_path = Path(self.source)
                if not file_path.exists():
                    self.status = VideoStatus.ERROR
                    self.error_message = f"Video file not found at: {self.source}"
                    logger.error(self.error_message)
                    return False
                if not file_path.is_file():
                    self.status = VideoStatus.ERROR
                    self.error_message = f"Path is not a regular file: {self.source}"
                    logger.error(self.error_message)
                    return False

            # Initialize OpenCV VideoCapture
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap or not self.cap.isOpened():
                self.status = VideoStatus.ERROR
                if isinstance(self.source, int):
                    self.error_message = (
                        f"Cannot open webcam device index {self.source}. "
                        "Check camera permissions or ensure device is connected."
                    )
                else:
                    self.error_message = (
                        f"OpenCV failed to decode video file: {self.source}. "
                        "The file may be corrupted or use an unsupported codec."
                    )
                logger.error(self.error_message)
                self.release()
                return False

            # Extract hardware/file metadata
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Webcams typically report <= 0 or NaN for total frames
            is_live = isinstance(self.source, int) or total_frames <= 0
            if is_live:
                total_frames = 0
                duration_sec = 0.0
                desc = f"Live Webcam (Device {self.source})"
            else:
                duration_sec = (total_frames / source_fps) if source_fps > 0 else 0.0
                desc = f"Video File: {Path(str(self.source)).name}"

            # Fallback for FPS if source reports 0 (common on some virtual webcams)
            effective_source_fps = source_fps if source_fps > 0 else 30.0

            self.metadata = VideoMetadata(
                width=width,
                height=height,
                source_fps=round(effective_source_fps, 2),
                total_frames=total_frames,
                duration_seconds=round(duration_sec, 2),
                is_live=is_live,
                source_description=desc,
            )

            self.status = VideoStatus.OPENED
            self.error_message = None
            self.frame_index = 0
            self.last_frame_time = 0.0
            self.prev_gray_frame = None

            logger.info(
                "Video source opened successfully: %s [%s, %s FPS, %s frames]",
                desc,
                self.metadata.resolution_str,
                self.metadata.source_fps,
                total_frames if not is_live else "Live",
            )
            return True

        except Exception as e:
            self.status = VideoStatus.ERROR
            self.error_message = f"Unexpected error opening video source: {str(e)}"
            logger.exception(self.error_message)
            self.release()
            return False

    def start_capture(self) -> bool:
        """Alias for open_source() to maintain backward compatibility."""
        return self.open_source()

    def get_next_frame(self, apply_pacing: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieve the next video frame from the capture stream.

        Args:
            apply_pacing: If True, sleeps briefly if needed to respect target_fps.

        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray]).
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        if apply_pacing and self.target_fps > 0 and self.last_frame_time > 0:
            target_period = 1.0 / self.target_fps
            elapsed = time.time() - self.last_frame_time
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

        ret, frame = self.cap.read()
        self.last_frame_time = time.time()

        if not ret or frame is None:
            # End of stream or hardware disconnect
            if self.metadata.is_live:
                self.status = VideoStatus.ERROR
                self.error_message = "Live camera stream interrupted or disconnected."
            else:
                self.status = VideoStatus.COMPLETED
                self.error_message = None
            return False, None

        self.frame_index += 1
        self.status = VideoStatus.PROCESSING
        return True, frame

    def read_next_frame(self) -> Optional[np.ndarray]:
        """Convenience method returning frame or None."""
        success, frame = self.get_next_frame(apply_pacing=False)
        return frame if success else None

    def release(self) -> None:
        """Release OpenCV VideoCapture hardware handles and reset internal state."""
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception as e:
                logger.warning("Error releasing VideoCapture handle: %s", e)
            finally:
                self.cap = None

        if self.status not in (VideoStatus.COMPLETED, VideoStatus.ERROR):
            self.status = VideoStatus.STOPPED

        self.prev_gray_frame = None
        logger.info("Video capture resources released cleanly.")

    def stop_capture(self) -> None:
        """Alias for release()."""
        self.release()

    def is_opened(self) -> bool:
        """Check whether the video stream is currently open and valid."""
        return bool(self.cap and self.cap.isOpened())

    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect human subjects in frame using Haar cascades (if available) or built-in HOG.

        Args:
            frame: BGR frame as NumPy ndarray.

        Returns:
            List of bounding boxes as (x, y, w, h).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) < 2:
            return []

        if self.custom_detector is not None:
            return self.custom_detector(frame)

        detections: List[Tuple[int, int, int, int]] = []
        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 1. Detect faces using Haar cascade if loaded
        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                )
                for (x, y, w, h) in faces:
                    person_w = int(w * 1.4)
                    person_h = int(h * 2.2)
                    person_x = max(0, int(x - (person_w - w) / 2))
                    person_y = max(0, y)
                    detections.append((person_x, person_y, person_w, person_h))
            except Exception as e:
                logger.debug("Face cascade detection error: %s", e)

        # 2. Fallback to upper body detector if loaded and no faces found
        if not detections and self.upper_body_cascade is not None and not self.upper_body_cascade.empty():
            try:
                bodies = self.upper_body_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(50, 50),
                )
                for (x, y, w, h) in bodies:
                    detections.append((int(x), int(y), int(w), int(h)))
            except Exception as e:
                logger.debug("Upper body cascade detection error: %s", e)

        # 3. Fallback to built-in HOG People Detector (zero external model dependency)
        if not detections and self.hog is not None:
            try:
                boxes, _ = self.hog.detectMultiScale(
                    frame,
                    winStride=(8, 8),
                    padding=(4, 4),
                    scale=1.05,
                )
                for (x, y, w, h) in boxes:
                    detections.append((int(x), int(y), int(w), int(h)))
            except Exception as e:
                logger.debug("HOG people detection error: %s", e)

        # 4. Fallback: silhouette/contour detection for high-contrast subjects & synthetic demo videos
        if not detections:
            try:
                _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame_h, frame_w = frame.shape[:2]
                min_area = (frame_w * frame_h) * 0.008
                max_area = (frame_w * frame_h) * 0.40
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if min_area < area < max_area:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        aspect = bh / float(bw) if bw > 0 else 0.0
                        if 0.6 <= aspect <= 4.5:
                            detections.append((int(bx), int(by), int(bw), int(bh)))
            except Exception as e:
                logger.debug("Silhouette contour detection error: %s", e)

        return detections

    def compute_motion_index(self, frame: np.ndarray) -> float:
        """
        Calculate normalized group movement index [0.0 to 100.0] via frame differencing.

        Args:
            frame: Current BGR frame.

        Returns:
            Normalized motion intensity percentage.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray_frame is None:
            self.prev_gray_frame = gray
            return 0.0

        # Absolute frame difference
        frame_diff = cv2.absdiff(self.prev_gray_frame, gray)
        self.prev_gray_frame = gray

        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_pixel_ratio = np.sum(thresh > 0) / float(thresh.size)

        # Scale non-linearly to 0-100 range
        normalized_score = min(100.0, (motion_pixel_ratio / 0.15) * 100.0)
        return round(float(normalized_score), 2)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[int, Track], float]:
        """
        Process a single frame: detect persons, update tracker, calculate motion,
        and render anonymized visual overlays.

        Args:
            frame: Raw BGR frame.
            timestamp: Frame capture timestamp.

        Returns:
            Tuple of (annotated_frame, active_tracks, motion_index).
        """
        current_time = timestamp if timestamp is not None else time.time()
        
        # 1. Detect person bounding boxes
        detections = self.detect_persons(frame)

        # 2. Update multi-object tracker
        active_tracks = self.tracker.update(detections, timestamp=current_time)

        # 3. Compute motion index
        motion_index = self.compute_motion_index(frame)

        # 4. Draw privacy-preserving visual overlays
        annotated = frame.copy()
        for track_id, track in active_tracks.items():
            x, y, w, h = track.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 115), 2)
            
            label = f"Person {track_id}"
            cv2.putText(
                annotated,
                label,
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 115),
                2,
                cv2.LINE_AA,
            )

            cx, cy = track.centroid
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # Summary watermark showing dimensions, FPS, and detected count
        fps_display = self.target_fps if self.target_fps > 0 else self.metadata.source_fps
        watermark = (
            f"Src: {self.metadata.resolution_str} @ {fps_display:.0f}FPS | "
            f"Active: {len(active_tracks)} | Motion: {motion_index:.1f}%"
        )
        cv2.putText(
            annotated,
            watermark,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated, active_tracks, motion_index

    def save_frame_diagnostic(self, frame: np.ndarray, target_path: Union[str, Path]) -> Path:
        """
        Diagnostic-only frame saver. Protected by strict privacy policy guardrail.

        Raises:
            PermissionError: If allow_save_frames is False.
        """
        if not self.allow_save_frames:
            raise PermissionError(
                "Privacy Policy Violation: Saving raw frames to disk is disabled by default. "
                "Enable 'allow_save_frames=True' explicitly only during controlled diagnostic tests."
            )
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        return path

    def __enter__(self) -> "VideoProcessor":
        self.open_source()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
