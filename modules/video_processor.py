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
from modules.person_tracker import CentroidTracker, Track, apply_non_maximum_suppression

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

        # Safe detector initialization: check bundled project cascades, models, and cv2.data
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        self.face_alt_cascade: Optional[cv2.CascadeClassifier] = None
        self.profile_cascade: Optional[cv2.CascadeClassifier] = None
        self.upper_body_cascade: Optional[cv2.CascadeClassifier] = None
        self.yunet_detector = None
        self.yunet_model_path: Optional[str] = None
        self._yunet_input_size: Optional[Tuple[int, int]] = None

        cascade_search_dirs = [
            Path(__file__).resolve().parent.parent / "data" / "cascades",
            Path(__file__).resolve().parent / "cascades",
            Path(getattr(cv2.data, "haarcascades", "")),
        ]

        def _find_cascade(filename: str) -> Optional[cv2.CascadeClassifier]:
            for sdir in cascade_search_dirs:
                cpath = sdir / filename
                if cpath.exists():
                    cc = cv2.CascadeClassifier(str(cpath.resolve()))
                    if not cc.empty():
                        return cc
            return None

        self.face_alt_cascade = _find_cascade("haarcascade_frontalface_alt2.xml")
        self.face_cascade = _find_cascade("haarcascade_frontalface_default.xml")
        self.profile_cascade = _find_cascade("haarcascade_profileface.xml")
        self.upper_body_cascade = _find_cascade("haarcascade_upperbody.xml")

        # Check for YuNet ONNX face detection model
        model_dirs = [
            Path(__file__).resolve().parent.parent / "data" / "models",
            Path(__file__).resolve().parent / "models",
        ]
        for mdir in model_dirs:
            mpath = mdir / "face_detection_yunet_2023mar.onnx"
            if mpath.exists() and hasattr(cv2, "FaceDetectorYN"):
                self.yunet_model_path = str(mpath.resolve())
                break

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
        Detect human subjects in frame using modern YuNet DNN (if available),
        multi-cascade Haar detectors (alt2, default, profile), upper-body cascade,
        built-in HOG people detection, and synthetic video contour fallback.

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
        frame_h, frame_w = frame.shape[:2]

        # Target processing width for detection speed
        det_scale = 640.0 / float(frame_w) if frame_w > 640 else 1.0
        if det_scale < 1.0:
            det_w = 640
            det_h = max(1, int(frame_h * det_scale))
            det_frame = cv2.resize(frame, (det_w, det_h))
        else:
            det_w = frame_w
            det_h = frame_h
            det_frame = frame

        # Convert to grayscale for Haar / contour pipelines
        if len(det_frame.shape) == 2:
            det_gray = det_frame
        else:
            det_gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------------------------------------------------
        # Tier 1: OpenCV YuNet Deep Neural Network Face Detector
        # ---------------------------------------------------------------------
        if self.yunet_model_path is not None and hasattr(cv2, "FaceDetectorYN"):
            try:
                score_thresh = max(0.65, getattr(settings.video, "person_detection_confidence", 0.65))
                if self.yunet_detector is None or self._yunet_input_size != (det_w, det_h):
                    self.yunet_detector = cv2.FaceDetectorYN.create(
                        self.yunet_model_path, "", (det_w, det_h), score_threshold=score_thresh, nms_threshold=0.30
                    )
                    self._yunet_input_size = (det_w, det_h)
                else:
                    self.yunet_detector.setInputSize((det_w, det_h))
                    self.yunet_detector.setScoreThreshold(score_thresh)

                _, faces = self.yunet_detector.detect(det_frame)
                if faces is not None and len(faces) > 0:
                    min_full_dim = max(55, int(0.045 * min(frame_w, frame_h)))
                    raw_faces: List[Tuple[int, int, int, int]] = []
                    for face in faces:
                        fx, fy, fw, fh = face[:4]
                        score = float(face[14]) if len(face) > 14 else 1.0
                        if score < score_thresh:
                            continue
                        # Reject sub-scale specks and background artifacts in detection frame
                        if fw < 26 or fh < 26:
                            continue
                        # Reject non-face aspect ratios
                        face_aspect = float(fh) / float(fw) if fw > 0 else 0.0
                        if face_aspect < 0.65 or face_aspect > 1.85:
                            continue

                        rx = int(round(fx / det_scale))
                        ry = int(round(fy / det_scale))
                        rw = int(round(fw / det_scale))
                        rh = int(round(fh / det_scale))
                        if rw < min_full_dim or rh < min_full_dim:
                            continue
                        raw_faces.append((rx, ry, rw, rh))

                    clean_faces = apply_non_maximum_suppression(raw_faces, iou_threshold=0.30, iom_threshold=0.40)
                    for (rx, ry, rw, rh) in clean_faces:
                        person_w = int(rw * 1.40)
                        person_h = int(rh * 2.20)
                        person_x = max(0, int(rx - (person_w - rw) / 2))
                        person_y = max(0, int(ry - rh * 0.15))
                        person_w = min(frame_w - person_x, person_w)
                        person_h = min(frame_h - person_y, person_h)
                        detections.append((person_x, person_y, person_w, person_h))
            except Exception as e:
                logger.debug("YuNet DNN face detection error: %s", e)

        # ---------------------------------------------------------------------
        # Tier 2: Multi-Cascade Haar Face Detection (alt2, default, profile, CLAHE)
        # ---------------------------------------------------------------------
        if not detections and (self.face_alt_cascade is not None or self.face_cascade is not None or self.profile_cascade is not None):
            try:
                min_face = max(32, int(45 * det_scale))
                candidate_faces: List[Tuple[int, int, int, int]] = []

                # Frontal Face Alt2 (highest accuracy, lowest false-positive rate)
                if self.face_alt_cascade is not None and not self.face_alt_cascade.empty():
                    alt_faces = self.face_alt_cascade.detectMultiScale(
                        det_gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_face, min_face)
                    )
                    for (x, y, w, h) in alt_faces:
                        candidate_faces.append((x, y, w, h))

                # Frontal Face Default
                if not candidate_faces and self.face_cascade is not None and not self.face_cascade.empty():
                    def_faces = self.face_cascade.detectMultiScale(
                        det_gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
                    )
                    for (x, y, w, h) in def_faces:
                        candidate_faces.append((x, y, w, h))

                # CLAHE enhancement for challenging lighting conditions
                if not candidate_faces and (self.face_alt_cascade or self.face_cascade):
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    clahe_gray = clahe.apply(det_gray)
                    casc = self.face_alt_cascade or self.face_cascade
                    cl_faces = casc.detectMultiScale(
                        clahe_gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
                    )
                    for (x, y, w, h) in cl_faces:
                        candidate_faces.append((x, y, w, h))

                # Profile Face Cascade (turned heads)
                if not candidate_faces and self.profile_cascade is not None and not self.profile_cascade.empty():
                    prof_faces = self.profile_cascade.detectMultiScale(
                        det_gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
                    )
                    for (x, y, w, h) in prof_faces:
                        candidate_faces.append((x, y, w, h))

                if candidate_faces:
                    raw_faces = []
                    for (x, y, w, h) in candidate_faces:
                        rx = int(round(x / det_scale))
                        ry = int(round(y / det_scale))
                        rw = int(round(w / det_scale))
                        rh = int(round(h / det_scale))
                        raw_faces.append((rx, ry, rw, rh))

                    clean_faces = apply_non_maximum_suppression(raw_faces, iou_threshold=0.30, iom_threshold=0.40)
                    for (rx, ry, rw, rh) in clean_faces:
                        person_w = int(rw * 1.40)
                        person_h = int(rh * 2.20)
                        person_x = max(0, int(rx - (person_w - rw) / 2))
                        person_y = max(0, int(ry - rh * 0.15))
                        person_w = min(frame_w - person_x, person_w)
                        person_h = min(frame_h - person_y, person_h)
                        detections.append((person_x, person_y, person_w, person_h))
            except Exception as e:
                logger.debug("Haar cascade face detection error: %s", e)

        # ---------------------------------------------------------------------
        # Tier 3: Upper Body Cascade
        # ---------------------------------------------------------------------
        if not detections and self.upper_body_cascade is not None and not self.upper_body_cascade.empty():
            try:
                min_body = max(45, int(65 * det_scale))
                bodies = self.upper_body_cascade.detectMultiScale(
                    det_gray,
                    scaleFactor=1.10,
                    minNeighbors=5,
                    minSize=(min_body, min_body),
                )
                for (x, y, w, h) in bodies:
                    aspect = float(h) / float(w) if w > 0 else 0.0
                    if 0.65 <= aspect <= 2.40:
                        rx = int(round(x / det_scale))
                        ry = int(round(y / det_scale))
                        rw = int(round(w / det_scale))
                        rh = int(round(h / det_scale))
                        detections.append((rx, ry, rw, rh))
            except Exception as e:
                logger.debug("Upper body cascade detection error: %s", e)

        # ---------------------------------------------------------------------
        # Tier 4: Built-in HOG People Detector (Supported on both Webcam & Files)
        # ---------------------------------------------------------------------
        if not detections and self.hog is not None:
            try:
                boxes, _ = self.hog.detectMultiScale(
                    det_frame,
                    winStride=(8, 8),
                    padding=(4, 4),
                    scale=1.05,
                )
                for (x, y, w, h) in boxes:
                    aspect = float(h) / float(w) if w > 0 else 0.0
                    if 0.70 <= aspect <= 2.80:
                        rx = int(round(x / det_scale))
                        ry = int(round(y / det_scale))
                        rw = int(round(w / det_scale))
                        rh = int(round(h / det_scale))
                        detections.append((rx, ry, rw, rh))
            except Exception as e:
                logger.debug("HOG people detection error: %s", e)

        # ---------------------------------------------------------------------
        # Tier 5: Fallback: Silhouette/contour detection for synthetic demo videos
        # ---------------------------------------------------------------------
        if not detections and not self.metadata.is_live:
            try:
                raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) > 2 else frame
                _, thresh = cv2.threshold(raw_gray, 40, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        # Final pass: deduplicate any overlapping person bounding boxes with IoM containment check
        return apply_non_maximum_suppression(detections, iou_threshold=0.30, iom_threshold=0.40)

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
        if timestamp is not None:
            current_time = timestamp
        elif not self.metadata.is_live and self.metadata.source_fps > 0:
            fps = self.target_fps if self.target_fps > 0 else self.metadata.source_fps
            current_time = float(self.frame_index) / float(fps)
        else:
            current_time = time.time()
        
        # 1. Detect person bounding boxes
        detections = self.detect_persons(frame)

        # 2. Update multi-object tracker
        active_tracks = self.tracker.update(detections, timestamp=current_time)

        # 3. Compute motion index
        motion_index = self.compute_motion_index(frame)

        # 4. Draw privacy-preserving visual overlays
        annotated = frame.copy()
        visible_count = 0
        for track_id, track in active_tracks.items():
            # Keep bounding box visible through brief detector dropouts (<= 3 frames)
            if not track.is_visible and track.consecutive_missing_frames > 3:
                continue
            visible_count += 1
            x, y, w, h = track.bbox
            color = (0, 220, 115)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            label = f"Person {track_id}"
            cv2.putText(
                annotated,
                label,
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            cx, cy = track.centroid
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # Summary watermark showing dimensions, FPS, and confirmed active count
        fps_display = self.target_fps if self.target_fps > 0 else self.metadata.source_fps
        watermark = (
            f"Src: {self.metadata.resolution_str} @ {fps_display:.0f}FPS | "
            f"Active: {visible_count} | Motion: {motion_index:.1f}%"
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
