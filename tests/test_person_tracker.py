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
    compute_overlap_metrics,
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


def test_tracker_reset_lifecycle() -> None:
    """Test resetting CentroidTracker clears active/deregistered collections and restarts IDs."""
    tracker = CentroidTracker(max_lost_frames=1)
    now = time.time()

    # Register two tracks
    tracker.update([(10, 10, 40, 40), (100, 100, 40, 40)], timestamp=now)
    assert len(tracker.active_tracks) == 2
    assert tracker.next_track_id == 3

    # Deregister one track
    tracker.update([(10, 10, 40, 40)], timestamp=now + 0.1)
    tracker.update([(10, 10, 40, 40)], timestamp=now + 0.2)
    assert len(tracker.deregistered_tracks) == 1

    # Reset
    tracker.reset()
    assert len(tracker.active_tracks) == 0
    assert len(tracker.deregistered_tracks) == 0
    assert tracker.next_track_id == 1

    # Next registration starts from ID 1 again
    new_tracks = tracker.update([(50, 50, 30, 30)], timestamp=now + 1.0)
    assert 1 in new_tracks
    assert len(new_tracks) == 1


def test_tracker_get_track_helper() -> None:
    """Test retrieving tracks by ID across active and deregistered collections."""
    tracker = CentroidTracker(max_lost_frames=1)
    now = time.time()

    tracker.update([(20, 20, 40, 40)], timestamp=now)
    track_1 = tracker.get_track(1)
    assert track_1 is not None
    assert track_1.track_id == 1
    assert tracker.get_track(999) is None

    # Let track 1 expire
    tracker.update([], timestamp=now + 0.1)
    tracker.update([], timestamp=now + 0.2)
    assert 1 in tracker.deregistered_tracks

    # get_track should still find it in deregistered records
    track_archived = tracker.get_track(1)
    assert track_archived is not None
    assert track_archived.track_id == 1


def test_degenerate_and_float_bounding_boxes() -> None:
    """Test NMS and CentroidTracker resilience to float coords and degenerate boxes."""
    # Degenerate boxes with zero or negative width/height
    bad_boxes = [(10, 10, 0, 50), (20, 20, -10, 40), (100.4, 100.6, 50.2, 60.8)]
    cleaned = apply_non_maximum_suppression(bad_boxes)
    assert len(cleaned) == 1
    assert cleaned[0] == (100, 100, 50, 60)

    # Centroid tracker registration with float coordinates
    tracker = CentroidTracker()
    tracks = tracker.update([(150.8, 120.2, 40.0, 60.0)])
    assert 1 in tracks
    assert tracks[1].centroid == (170, 150)
    assert tracks[1].bbox == (150, 120, 40, 60)


def test_person_tracker_frame_interval_pacing_and_reset() -> None:
    """Test PersonTracker frame skipping propagates active tracks and reset clears state."""
    pt = PersonTracker(processing_interval_frames=2, estimate_orientation=False)
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    now = time.time()

    # Frame 1: full processing
    res1 = pt.process_frame(dummy_frame, [(20, 20, 30, 30)], timestamp=now)
    assert res1.frame_index == 1
    assert res1.total_active_count == 1
    assert res1.new_tracks_count == 1

    # Frame 2: skipped pacing interval (interval = 2)
    res2 = pt.process_frame(dummy_frame, [(25, 25, 30, 30)], timestamp=now + 0.05)
    assert res2.frame_index == 2
    assert res2.timestamp == now + 0.05
    assert res2.total_active_count == 1
    assert res2.new_tracks_count == 0
    assert res2.active_tracks[0].movement_delta_px == 0.0

    # Frame 3: full processing again
    res3 = pt.process_frame(dummy_frame, [(25, 25, 30, 30)], timestamp=now + 0.1)
    assert res3.frame_index == 3
    assert res3.total_active_count == 1

    # Reset
    pt.reset()
    assert pt.frame_counter == 0
    assert len(pt.tracker.active_tracks) == 0


def test_head_orientation_empty_or_degenerate_frame() -> None:
    """Test orientation estimator returns Unknown on empty, None, or tiny crops without crashing."""
    estimator = HeadOrientationEstimator()

    # None frame
    res_none = estimator.estimate_orientation(None, (10, 10, 50, 50))  # type: ignore
    assert res_none.orientation_label == "Unknown"

    # Empty frame (0x0)
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = estimator.estimate_orientation(empty_frame, (10, 10, 50, 50))
    assert res_empty.orientation_label == "Unknown"

    # Tiny crop (< 20px)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res_tiny = estimator.estimate_orientation(dummy_frame, (10, 10, 5, 5))
    assert res_tiny.orientation_label == "Unknown"


def test_containment_iom_suppresses_nested_sub_boxes() -> None:
    """
    Test that a smaller sub-box largely contained within a person box
    (which has low standard IoU but high IoM containment) is suppressed.
    """
    # Main person upper body box: (200, 100, 250, 350) -> area = 87500
    person_box = (200, 100, 250, 350)
    # Spurious nested box on hair/ear: (220, 120, 90, 100) -> area = 9000
    # Box 2 is completely inside Box 1: IoU = 9000 / 87500 = 0.103, but IoM = 1.0!
    sub_box = (220, 120, 90, 100)

    iou, iom = compute_overlap_metrics(person_box, sub_box)
    assert iou < 0.20  # Standard IoU is small
    assert iom > 0.90  # Sub-box is almost 100% inside person box

    # Standard NMS with containment suppression must keep only 1 box!
    cleaned = apply_non_maximum_suppression([person_box, sub_box])
    assert len(cleaned) == 1
    assert cleaned[0] == person_box


def test_single_person_tracker_does_not_double_count() -> None:
    """Test CentroidTracker registers exactly 1 track when detector outputs multi-scale nested boxes."""
    tracker = CentroidTracker()
    now = time.time()

    # Detections for a single person: one main body box and one nested head/hair artifact
    detections = [(400, 200, 280, 400), (420, 210, 120, 140)]
    active = tracker.update(detections, timestamp=now)

    assert len(active) == 1
    assert 1 in active
    assert active[1].track_id == 1


def test_track_stability_and_active_duration_growth() -> None:
    """
    Acceptance test for Phase 2:
    - Track ID remains stable across continuous observations
    - total_frames_detected increases steadily
    - total_active_seconds becomes positive
    - Brief missed detection does not create a duplicate track
    - Stale track eventually expires after max_lost_frames
    """
    tracker = CentroidTracker(max_lost_frames=5, max_distance_px=100.0)
    t0 = 1000.0

    # 1. First observation
    active = tracker.update([(100, 100, 150, 200)], timestamp=t0)
    assert len(active) == 1
    assert 1 in active
    tr = active[1]
    assert tr.track_id == 1
    assert tr.total_frames_detected == 1
    assert tr.total_active_seconds == 0.0

    # 2. Continuous observations across 9 more frames (total 10 frames, 1.0s elapsed)
    for i in range(1, 10):
        t_cur = t0 + (i * 0.1)
        # Bounding box shifts slightly (5px per frame) with small size jitter
        bbox = (100 + i * 2, 100 + i, 150, 200)
        active = tracker.update([bbox], timestamp=t_cur)
        assert len(active) == 1
        assert 1 in active
        assert active[1].track_id == 1  # ID remains perfectly stable

    tr = active[1]
    assert tr.total_frames_detected == 10
    assert abs(tr.total_active_seconds - 0.9) < 1e-4

    # 3. Brief occlusion/missed detection for 2 frames
    active = tracker.update([], timestamp=t0 + 1.0)
    assert len(active) == 1
    assert active[1].is_visible is False
    assert active[1].consecutive_missing_frames == 1

    active = tracker.update([], timestamp=t0 + 1.1)
    assert len(active) == 1
    assert active[1].consecutive_missing_frames == 2

    # 4. Reappearance at frame 13: matches same track, does NOT create duplicate
    active = tracker.update([(120, 110, 150, 200)], timestamp=t0 + 1.2)
    assert len(active) == 1
    assert 1 in active
    assert active[1].track_id == 1
    assert active[1].total_frames_detected == 11
    assert active[1].consecutive_missing_frames == 0
    assert active[1].is_visible is True
    assert abs(active[1].total_active_seconds - 1.2) < 1e-4

    # 5. Stale expiration after max_lost_frames (5 frames)
    for j in range(6):
        active = tracker.update([], timestamp=t0 + 1.3 + (j * 0.1))

    # Track 1 should now be expired and archived in deregistered_tracks
    assert len(active) == 0
    assert 1 not in active
    assert 1 in tracker.deregistered_tracks
    assert tracker.deregistered_tracks[1].total_frames_detected == 11


def test_ema_bounding_box_smoothing_eliminates_jitter() -> None:
    """Verify that EMA smoothing prevents sudden jumps in bounding box coordinates."""
    tracker = CentroidTracker(max_lost_frames=5)
    now = time.time()

    # Frame 1: Person at (100, 100, 50, 50)
    tracker.update([(100, 100, 50, 50)], timestamp=now)
    assert tracker.active_tracks[1].bbox == (100, 100, 50, 50)

    # Frame 2: Sudden 20px jump in raw detection (120, 100, 50, 50)
    tracker.update([(120, 100, 50, 50)], timestamp=now + 0.1)
    smoothed_x = tracker.active_tracks[1].bbox[0]
    # alpha=0.65 -> 0.65 * 120 + 0.35 * 100 = 78 + 35 = 113
    assert smoothed_x == 113
    assert smoothed_x < 120
    assert smoothed_x > 100


def test_spurious_single_frame_track_fast_purge() -> None:
    """Verify that a single-frame spurious detection is purged after 2 missing frames instead of max_lost_frames."""
    tracker = CentroidTracker(max_lost_frames=10)
    now = time.time()

    # Frame 1: Spurious detection appears once
    tracker.update([(200, 200, 50, 50)], timestamp=now)
    assert 1 in tracker.active_tracks
    assert tracker.active_tracks[1].total_frames_detected == 1

    # Frame 2: Missing for 1 frame
    tracker.update([], timestamp=now + 0.1)
    assert 1 in tracker.active_tracks
    assert tracker.active_tracks[1].consecutive_missing_frames == 1

    # Frame 3: Missing for 2 frames -> fast purged
    tracker.update([], timestamp=now + 0.2)
    assert 1 not in tracker.active_tracks
    assert 1 in tracker.deregistered_tracks




