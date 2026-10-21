"""
Comprehensive End-to-End Tests for ClassroomPipeline.

Validates the full 11-stage classroom intelligence lifecycle:
1. Create Session
2. Open Video / Optional Audio
3. Extract Frames and Audio Windows
4. Detect and Track People
5. Calculate Attendance Observations
6. Extract Visual and Audio Features
7. Aggregate Features by Time Window
8. Calculate Activity Indicators
9. Generate Teaching Suggestions
10. Save Results in SQLite
11. Update Dashboard and Reports
"""

from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pytest

from modules.database import DatabaseManager
from modules.pipeline import ClassroomPipeline, PipelineStepResult
from modules.session_manager import SessionState


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> str:
    """Create a temporary synthetic 30-frame video file with moving subjects."""
    file_path = tmp_path / "synthetic_classroom.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(file_path), fourcc, 10.0, (320, 240))

    # Draw moving rectangles representing tracked people
    for i in range(30):
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        # Subject 1 moving horizontally
        x1 = 40 + i * 4
        cv2.rectangle(frame, (x1, 50), (x1 + 40, 150), (220, 220, 220), -1)
        # Subject 2 stationary
        cv2.rectangle(frame, (200, 60), (240, 160), (200, 200, 200), -1)
        out.write(frame)

    out.release()
    return str(file_path)


def test_end_to_end_pipeline_lifecycle(synthetic_video_path: str) -> None:
    """
    Test the complete 11-stage pipeline from creation through live stepping to conclusion.
    """
    db = DatabaseManager(":memory:")

    pipeline = ClassroomPipeline(
        db=db,
        course_code="CS101",
        session_title="End-to-End Pipeline Verification",
        video_source=synthetic_video_path,
        target_fps=10,
        camera_enabled=True,
        audio_enabled=False,  # Headless video verification
        sustained_threshold_pct=50.0,
        window_duration_sec=1,  # Short window to trigger fusion during test
        custom_detector=lambda frame: [(50, 50, 40, 80)],
    )

    # Stage 1: Create Session
    sess = pipeline.create_session()
    assert sess.session_id is not None
    assert sess.status == SessionState.READY
    db_sess = db.get_session(sess.session_id)
    assert db_sess is not None
    assert db_sess["status"] == "READY"

    # Stage 2: Open Video / Audio Sources & Stage 3: Start Session
    pipeline.start()
    assert pipeline.session_metadata.status == SessionState.ACTIVE

    # Stages 4 through 10: Step through frames
    step_count = 0
    while True:
        res = pipeline.step(apply_pacing=False)
        if not res.success:
            break

        step_count += 1
        assert res.session_status == SessionState.ACTIVE
        assert res.frame is not None
        assert res.annotated_frame is not None

        # Verify tracking was performed
        assert isinstance(res.active_tracks, dict)

    assert step_count >= 10, f"Expected at least 10 frames processed, got {step_count}"

    # Stage 11: Conclude Session, Save Results, & Export Reports
    summary = pipeline.conclude()
    assert summary["session_id"] == sess.session_id
    assert summary["total_active_seconds"] >= 0.0
    assert summary["total_observed_tracks"] >= 1

    # Verify SQLite persistence of session status
    db_concluded = db.get_session(sess.session_id)
    assert db_concluded["status"] == "CONCLUDED"

    # Verify SQLite persistence of attendance records
    records = db.get_session_attendance(sess.session_id)
    assert len(records) >= 1

    # Verify attendance events were logged in SQLite
    events = db.get_attendance_events(sess.session_id)
    assert len(events) >= 1


def test_pipeline_audio_only_mode() -> None:
    """Verify that the pipeline operates gracefully when camera is disabled."""
    db = DatabaseManager(":memory:")

    pipeline = ClassroomPipeline(
        db=db,
        course_code="SEMINAR500",
        session_title="Audio-Only Seminar Discussion",
        camera_enabled=False,
        audio_enabled=False,  # Offline test mode
        window_duration_sec=1,
    )

    sess = pipeline.create_session()
    pipeline.start()

    res = pipeline.step(apply_pacing=False)
    assert res.success is True
    assert res.headcount == 0
    assert res.frame is None

    summary = pipeline.conclude()
    assert summary["total_observed_tracks"] == 0
    assert db.get_session(sess.session_id)["status"] == "CONCLUDED"


def test_pipeline_pause_and_resume(synthetic_video_path: str) -> None:
    """Verify state transitions and time slicing across pause and resume."""
    db = DatabaseManager(":memory:")

    pipeline = ClassroomPipeline(
        db=db,
        course_code="MATH201",
        session_title="Calculus with Breaks",
        video_source=synthetic_video_path,
        target_fps=10,
        camera_enabled=True,
        audio_enabled=False,
    )

    pipeline.start()
    pipeline.step()

    # Pause
    pipeline.pause()
    assert pipeline.session_metadata.status == SessionState.PAUSED
    res_paused = pipeline.step()
    assert "idle" in res_paused.message.lower()

    # Resume
    pipeline.resume()
    assert pipeline.session_metadata.status == SessionState.ACTIVE
    res_resumed = pipeline.step()
    assert res_resumed.success is True

    summary = pipeline.conclude()
    assert summary["session_id"] is not None


def test_pipeline_run_batch_mode(synthetic_video_path: str) -> None:
    """Verify automated headless batch processing of recorded video files."""
    db = DatabaseManager(":memory:")

    pipeline = ClassroomPipeline(
        db=db,
        course_code="PHYS101",
        session_title="Batch Physics Lecture",
        video_source=synthetic_video_path,
        target_fps=10,
        camera_enabled=True,
        audio_enabled=False,
        window_duration_sec=1,
        custom_detector=lambda frame: [(50, 50, 40, 80)],
    )

    summary = pipeline.run_batch(max_frames=15)
    assert summary["total_observed_tracks"] >= 1
    assert db.get_session(summary["session_id"])["status"] == "CONCLUDED"


def test_pipeline_headcount_accuracy_and_inactive_track_filtering(tmp_path: Path) -> None:
    """Verify headcount counts only active, visible persons and excludes lost/spurious tracks."""
    db_file = tmp_path / "headcount_test.db"
    db = DatabaseManager(db_path=db_file)

    detector_boxes = [(100, 100, 80, 120)]

    pipeline = ClassroomPipeline(
        db=db,
        course_code="CS101",
        session_title="Headcount Test",
        video_source=0,  # Mocked via custom_detector
        camera_enabled=True,
        audio_enabled=False,
        custom_detector=lambda frame: detector_boxes,
    )
    pipeline.start()

    # Step 1: Exactly 1 person detected
    res = pipeline.step(apply_pacing=False)
    assert res.success is True
    assert res.headcount == 1

    # Step 2: Person disappears from camera view
    detector_boxes = []
    # Frame 1 missing (<= 3 frames tolerance)
    res = pipeline.step(apply_pacing=False)
    assert res.headcount == 1

    # Frame 2 missing
    pipeline.step(apply_pacing=False)
    # Frame 3 missing
    pipeline.step(apply_pacing=False)
    # Frame 4 missing (> 3 frames): Headcount must drop to 0 rather than staying 1 for 30 frames
    res4 = pipeline.step(apply_pacing=False)
    assert res4.headcount == 0

    pipeline.conclude()
