"""
Unit Tests for Person Tracker Module.

Tests anonymous spatial-temporal multi-object tracking, duplicate suppression (NMS),
brief detection failure resilience, entry/exit deregistration, and structured frame results.
"""

import time
import numpy as np
import pytest

from modules.person_tracker import (
    CentroidTracker,
    HeadOrientationEstimator,
    PersonTracker,
    Track,
    TrackingFrameResult,
    apply_non_maximum_suppression,
    compute_iou,
)


def test_iou_and_non_maximum_suppression() -> None:
    """Test that overlapping detections are merged via NMS to eliminate duplicate tracks."""
    box1 = (100, 100, 50, 50)
    box2 = (105, 105, 50, 50)  # Heavy overlap
    box3 = (300, 300, 50, 50)  # Separate person

    iou = compute_iou(box1, box2)
    assert iou > 0.50

    cleaned = apply_non_maximum_suppression([box1, box2, box3], iou_threshold=0.40)
    assert len(cleaned) == 2  # box1/box2 merged into one, box3 preserved


def test_track_registration_and_trajectory() -> None:
    """Test initial track registration, centroid calculation, and trajectory history."""
    tracker = CentroidTracker(max_lost_frames=5, max_distance_px=50.0)
    now = time.time()

    # Frame 1: Person at (100, 100, 40, 60) -> Centroid = (120, 130)
    tracks = tracker.update([(100, 100, 40, 60)], timestamp=now)
    assert len(tracks) == 1
    assert 1 in tracks
    assert tracks[1].centroid == (120, 130)
    assert tracks[1].is_visible is True
    assert tracks[1].total_frames_detected == 1


def test_track_continuation_and_movement() -> None:
    """Test that small displacements maintain the same track ID and update movement delta."""
    tracker = CentroidTracker(max_lost_frames=5, max_distance_px=50.0)
    now = time.time()

    # Frame 1
    tracker.update([(100, 100, 40, 40)], timestamp=now)
    
    # Frame 2: Person shifts 5px right and 5px down
    tracks = tracker.update([(105, 105, 40, 40)], timestamp=now + 0.1)
    assert len(tracks) == 1
    assert 1 in tracks
    assert tracks[1].total_frames_detected == 2
    assert tracks[1].movement_delta_px > 0.0
    assert tracks[1].consecutive_missing_frames == 0


def test_brief_detection_failure_handling() -> None:
    """
    Test resilience to brief detection failures:
    Track should be retained in memory when missed for a frame, and re-associated
    when re-detected, preserving the same track ID.
    """
    tracker = CentroidTracker(max_lost_frames=3, max_distance_px=50.0)
    now = time.time()

    # Frame 1: Detected
    tracker.update([(100, 100, 50, 50)], timestamp=now)
    assert tracker.active_tracks[1].is_visible is True

    # Frame 2: Brief detection failure (e.g. subject turned away, 0 detections)
    tracker.update([], timestamp=now + 0.1)
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[1].consecutive_missing_frames == 1
    assert tracker.active_tracks[1].is_visible is False  # Marked invisible but retained

    # Frame 3: Subject re-detected nearby
    tracker.update([(102, 102, 50, 50)], timestamp=now + 0.2)
    assert len(tracker.active_tracks) == 1
    assert 1 in tracker.active_tracks
    assert tracker.active_tracks[1].consecutive_missing_frames == 0
    assert tracker.active_tracks[1].is_visible is True
    assert tracker.active_tracks[1].total_frames_detected == 2


def test_person_leaving_camera_view() -> None:
    """Test that subjects leaving camera view are deregistered after max_lost_frames."""
    tracker = CentroidTracker(max_lost_frames=2, max_distance_px=50.0)
    now = time.time()

    # Frame 1: Person present
    tracker.update([(50, 50, 40, 40)], timestamp=now)
    assert 1 in tracker.active_tracks

    # Frames 2, 3, 4: Person walked away (0 detections)
    tracker.update([], timestamp=now + 0.1)  # missed 1
    tracker.update([], timestamp=now + 0.2)  # missed 2
    tracker.update([], timestamp=now + 0.3)  # missed 3 > max_lost_frames

    # Track 1 should now be deregistered
    assert len(tracker.active_tracks) == 0
    assert 1 in tracker.deregistered_tracks
    assert len(tracker.get_all_session_tracks()) == 1


def test_multiple_people_entry_and_tracking() -> None:
    """Test multiple people entering camera view receiving distinct IDs."""
    tracker = CentroidTracker(max_lost_frames=5, max_distance_px=60.0)
    now = time.time()

    # Frame 1: Person A and Person B enter
    detections = [(50, 50, 40, 40), (250, 250, 40, 40)]
    tracks = tracker.update(detections, timestamp=now)
    assert len(tracks) == 2
    assert set(tracks.keys()) == {1, 2}

    # Frame 2: Person C enters, A and B move slightly
    detections_f2 = [(52, 52, 40, 40), (252, 252, 40, 40), (450, 100, 40, 40)]
    tracks_f2 = tracker.update(detections_f2, timestamp=now + 0.1)
    assert len(tracks_f2) == 3
    assert set(tracks_f2.keys()) == {1, 2, 3}


def test_person_tracker_high_level_orchestration() -> None:
    """Test PersonTracker class returning structured TrackingFrameResult."""
    pt = PersonTracker(processing_interval_frames=1, estimate_orientation=True)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1
    result = pt.process_frame(
        frame=dummy_frame,
        detections=[(100, 100, 60, 80), (300, 150, 60, 80)],
        timestamp=time.time(),
    )

    assert isinstance(result, TrackingFrameResult)
    assert result.frame_index == 1
    assert result.total_active_count == 2
    assert result.new_tracks_count == 2
    assert len(result.active_tracks) == 2
    assert result.active_tracks[0].head_orientation is not None


def test_head_orientation_estimator_proxy() -> None:
    """Test observable head orientation proxy output without emotional inference."""
    estimator = HeadOrientationEstimator()
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    
    orientation = estimator.estimate_orientation(dummy_frame, (50, 50, 100, 100))
    assert orientation is not None
    assert isinstance(orientation.yaw, float)
    assert isinstance(orientation.pitch, float)
    assert orientation.orientation_label in ["Frontal", "Oriented Left", "Oriented Right", "Oriented Down", "Unknown"]
