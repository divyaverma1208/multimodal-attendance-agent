"""
Person Tracker Module.

Implements anonymous spatial-temporal multi-object tracking (MOT) for classroom presence.
Assigns ephemeral tracking IDs (e.g., Track 1, Track 2) using Euclidean centroid distance matching.
Handles entry/exit dynamics, brief detection failures, and duplicate suppression (NMS).
Supports optional MediaPipe head orientation proxy estimation where supported.

ETHICAL CONSTRAINTS:
- Strictly tracks anonymous spatial bounding boxes; never performs facial identification or biometric matching.
- Does not infer psychological states, attentiveness, or emotions from facial expressions.
- Keeps all processing 100% local.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class HeadOrientation:
    """
    Approximate observable head pose proxy (in degrees).
    Used strictly as a physical activity indicator, NOT an attentiveness ground truth.
    """
    yaw: float = 0.0      # Negative = oriented left, Positive = oriented right
    pitch: float = 0.0    # Negative = oriented downwards, Positive = oriented upwards
    orientation_label: str = "Frontal"  # "Frontal", "Left", "Right", "Down", "Unknown"


@dataclass
class Track:
    """Represents an anonymous individual trajectory over time."""
    track_id: int
    bbox: Tuple[int, int, int, int]    # (x, y, w, h)
    centroid: Tuple[int, int]           # (cx, cy)
    first_seen_timestamp: float
    last_seen_timestamp: float
    total_frames_detected: int = 1
    consecutive_missing_frames: int = 0
    is_visible: bool = True              # False if temporarily missed in recent frame
    movement_delta_px: float = 0.0       # Displacement since previous frame
    head_orientation: Optional[HeadOrientation] = None
    trajectory: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def total_active_seconds(self) -> float:
        """Total time span between first and last detection."""
        return max(0.0, self.last_seen_timestamp - self.first_seen_timestamp)


@dataclass
class TrackingFrameResult:
    """Structured detection and tracking result for a processed frame."""
    frame_index: int
    timestamp: float
    active_tracks: List[Track]
    new_tracks_count: int
    lost_tracks_count: int
    total_active_count: int
    raw_detections_count: int


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes (x, y, w, h).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - inter_area)

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def apply_non_maximum_suppression(
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.40,
) -> List[Tuple[int, int, int, int]]:
    """
    Suppresses redundant overlapping bounding boxes (e.g. multiple detector scales).
    Prioritizes larger bounding boxes.
    """
    if not boxes:
        return []

    # Sort boxes by area descending
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept_boxes: List[Tuple[int, int, int, int]] = []

    for box in sorted_boxes:
        overlap = False
        for kept in kept_boxes:
            if compute_iou(box, kept) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept_boxes.append(box)

    return kept_boxes


class CentroidTracker:
    """
    Centroid-based multi-object tracker for anonymous human presence monitoring.
    Associates detections with existing tracks using Euclidean distance minimization.
    Handles temporary occlusions and missed detections.
    """

    def __init__(
        self,
        max_lost_frames: Optional[int] = None,
        max_distance_px: Optional[float] = None,
    ) -> None:
        """
        Initialize the tracker.

        Args:
            max_lost_frames: Frame tolerance before deregistering a lost track.
            max_distance_px: Max centroid distance to match a detection to an existing track.
        """
        self.max_lost_frames = max_lost_frames or settings.video.max_lost_frames
        self.max_distance_px = max_distance_px or settings.video.max_centroid_distance_px
        
        self.next_track_id: int = 1
        self.active_tracks: Dict[int, Track] = {}
        self.deregistered_tracks: Dict[int, Track] = {}

    def _compute_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Compute the center point (cx, cy) of a bounding box."""
        x, y, w, h = bbox
        return (int(x + w / 2.0), int(y + h / 2.0))

    def register(self, bbox: Tuple[int, int, int, int], timestamp: float) -> Track:
        """Register a new detected subject with a fresh anonymous ID."""
        centroid = self._compute_centroid(bbox)
        track = Track(
            track_id=self.next_track_id,
            bbox=bbox,
            centroid=centroid,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            total_frames_detected=1,
            consecutive_missing_frames=0,
            is_visible=True,
            movement_delta_px=0.0,
            trajectory=[centroid],
        )
        self.active_tracks[self.next_track_id] = track
        self.next_track_id += 1
        return track

    def deregister(self, track_id: int) -> None:
        """Move a vanished track from active list to historical archive."""
        if track_id in self.active_tracks:
            self.deregistered_tracks[track_id] = self.active_tracks.pop(track_id)

    def update(
        self,
        detections: List[Tuple[int, int, int, int]],
        timestamp: Optional[float] = None,
    ) -> Dict[int, Track]:
        """
        Update tracker with new bounding box detections for the current frame.

        Args:
            detections: List of bounding boxes as (x, y, w, h).
            timestamp: Frame timestamp. Defaults to current epoch time.

        Returns:
            Dictionary of currently active Track instances.
        """
        current_time = timestamp if timestamp is not None else time.time()

        # Step 1: Suppress duplicate overlapping detections
        clean_detections = apply_non_maximum_suppression(detections, iou_threshold=0.40)

        # Case 1: No detections in this frame (handle brief detection failure / occlusion)
        if len(clean_detections) == 0:
            for track_id in list(self.active_tracks.keys()):
                track = self.active_tracks[track_id]
                track.consecutive_missing_frames += 1
                track.is_visible = False
                track.movement_delta_px = 0.0
                if track.consecutive_missing_frames > self.max_lost_frames:
                    self.deregister(track_id)
            return self.active_tracks

        input_centroids = [self._compute_centroid(b) for b in clean_detections]

        # Case 2: No existing tracks; register all clean detections
        if len(self.active_tracks) == 0:
            for bbox in clean_detections:
                self.register(bbox, current_time)
            return self.active_tracks

        # Case 3: Match existing tracks with incoming detections via Euclidean distance
        track_ids = list(self.active_tracks.keys())
        existing_centroids = [self.active_tracks[tid].centroid for tid in track_ids]

        # Compute distance matrix: (num_existing, num_detections)
        D = np.zeros((len(existing_centroids), len(input_centroids)), dtype=float)
        for i, (ex, ey) in enumerate(existing_centroids):
            for j, (cx, cy) in enumerate(input_centroids):
                D[i, j] = math.hypot(ex - cx, ey - cy)

        # Pairwise greedy distance minimization
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # If distance exceeds threshold, do not associate (person is too far)
            if D[row, col] > self.max_distance_px:
                continue

            track_id = track_ids[row]
            track = self.active_tracks[track_id]
            prev_centroid = track.centroid
            new_centroid = input_centroids[col]
            new_bbox = clean_detections[col]

            # Update track state
            track.movement_delta_px = math.hypot(
                new_centroid[0] - prev_centroid[0],
                new_centroid[1] - prev_centroid[1],
            )
            track.centroid = new_centroid
            track.bbox = new_bbox
            track.last_seen_timestamp = current_time
            track.total_frames_detected += 1
            track.consecutive_missing_frames = 0
            track.is_visible = True
            track.trajectory.append(new_centroid)
            if len(track.trajectory) > 50:
                track.trajectory.pop(0)

            used_rows.add(row)
            used_cols.add(col)

        # Handle existing tracks that were not matched (brief detection failure)
        unused_rows = set(range(len(existing_centroids))) - used_rows
        for row in unused_rows:
            track_id = track_ids[row]
            track = self.active_tracks[track_id]
            track.consecutive_missing_frames += 1
            track.is_visible = False
            track.movement_delta_px = 0.0
            if track.consecutive_missing_frames > self.max_lost_frames:
                self.deregister(track_id)

        # Register new detections not matched with any existing track (entering persons)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.register(clean_detections[col], current_time)

        return self.active_tracks

    def get_all_session_tracks(self) -> List[Track]:
        """Return combined list of active and deregistered session tracks."""
        return list(self.active_tracks.values()) + list(self.deregistered_tracks.values())


class HeadOrientationEstimator:
    """
    Estimates approximate head orientation proxy using MediaPipe or OpenCV geometry.
    ETHICAL BOUNDARY: Measures observable directional movement vectors only.
    Never infers emotion, sentiment, or internal cognitive focus.
    """

    def __init__(self, use_mediapipe: bool = True) -> None:
        """Initialize orientation estimator."""
        self.use_mediapipe = use_mediapipe
        self._mediapipe_landmarker = None

    def estimate_orientation(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> HeadOrientation:
        """
        Estimate coarse head orientation from bounding box crop.

        Args:
            frame: Full BGR frame.
            bbox: (x, y, w, h) of the detected person/face.

        Returns:
            HeadOrientation instance.
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        # Ensure valid crop boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return HeadOrientation(0.0, 0.0, "Unknown")

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Fast geometric proxy: Split crop horizontally into left and right halves
        mid_x = (x2 - x1) // 2
        left_half = gray[:, :mid_x]
        right_half = gray[:, mid_x:]

        mean_left = float(np.mean(left_half))
        mean_right = float(np.mean(right_half))

        # Asymmetry ratio as rough yaw proxy
        diff = mean_right - mean_left
        yaw = max(-45.0, min(45.0, diff * 1.2))

        # Vertical balance as rough pitch proxy (top vs bottom third)
        third_h = (y2 - y1) // 3
        top_third = gray[:third_h, :]
        bottom_third = gray[2 * third_h:, :]
        v_diff = float(np.mean(bottom_third)) - float(np.mean(top_third))
        pitch = max(-30.0, min(30.0, v_diff * 0.8))

        if abs(yaw) < 12.0 and abs(pitch) < 12.0:
            label = "Frontal"
        elif yaw <= -12.0:
            label = "Oriented Left"
        elif yaw >= 12.0:
            label = "Oriented Right"
        elif pitch <= -12.0:
            label = "Oriented Down"
        else:
            label = "Frontal"

        return HeadOrientation(yaw=round(yaw, 1), pitch=round(pitch, 1), orientation_label=label)


class PersonTracker:
    """
    High-level Person Tracker Engine.
    Coordinates person detection, Non-Maximum Suppression, temporal centroid tracking,
    head orientation proxy estimation, and configurable frame-interval pacing.
    """

    def __init__(
        self,
        processing_interval_frames: int = 1,
        max_lost_frames: Optional[int] = None,
        max_distance_px: Optional[float] = None,
        estimate_orientation: bool = True,
    ) -> None:
        """
        Initialize PersonTracker.

        Args:
            processing_interval_frames: Execute full detection every N frames (1 = every frame).
            max_lost_frames: Frames to retain lost track before deregistering.
            max_distance_px: Max centroid displacement to associate detections.
            estimate_orientation: Whether to compute coarse head orientation proxy.
        """
        self.interval = max(1, processing_interval_frames)
        self.tracker = CentroidTracker(
            max_lost_frames=max_lost_frames,
            max_distance_px=max_distance_px,
        )
        self.orientation_estimator = (
            HeadOrientationEstimator() if estimate_orientation else None
        )
        self.frame_counter: int = 0
        self._last_result: Optional[TrackingFrameResult] = None

    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[Tuple[int, int, int, int]],
        timestamp: Optional[float] = None,
    ) -> TrackingFrameResult:
        """
        Process incoming frame detections and return structured tracking results.

        Args:
            frame: Full BGR frame as NumPy ndarray.
            detections: Bounding boxes from detector as (x, y, w, h).
            timestamp: Frame timestamp.

        Returns:
            Structured TrackingFrameResult instance.
        """
        now = timestamp if timestamp is not None else time.time()
        self.frame_counter += 1

        prev_active_ids = set(self.tracker.active_tracks.keys())

        # Check processing interval: if skipping, retain existing tracks without new matches
        if (self.frame_counter % self.interval) != 0 and self._last_result is not None:
            return self._last_result

        # Update tracker
        active_dict = self.tracker.update(detections, timestamp=now)
        current_active_ids = set(active_dict.keys())

        new_tracks = len(current_active_ids - prev_active_ids)
        lost_tracks = len(prev_active_ids - current_active_ids)

        # Estimate orientation proxy for visible active tracks
        if self.orientation_estimator is not None:
            for track in active_dict.values():
                if track.is_visible:
                    track.head_orientation = self.orientation_estimator.estimate_orientation(
                        frame=frame,
                        bbox=track.bbox,
                    )

        result = TrackingFrameResult(
            frame_index=self.frame_counter,
            timestamp=now,
            active_tracks=list(active_dict.values()),
            new_tracks_count=new_tracks,
            lost_tracks_count=lost_tracks,
            total_active_count=len(active_dict),
            raw_detections_count=len(detections),
        )
        self._last_result = result
        return result
