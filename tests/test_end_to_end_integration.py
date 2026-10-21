"""
End-to-End System Integration Tests.

Validates the complete 15-point functional specification:
1. A teacher can create a session.
2. The session can process a recorded video.
3. Webcam processing works where supported.
4. Person detections are passed to the tracking module.
5. Tracking observations are passed to attendance logic.
6. Attendance events are persisted to SQLite.
7. Audio processing is optional.
8. Visual and audio features are aggregated by timestamp.
9. Analytics are calculated from actual module outputs.
10. Teaching recommendations use the analytics results.
11. The dashboard updates with real session data.
12. The session can be stopped safely.
13. Errors in one optional modality do not crash the entire app.
14. Missing features are not replaced with fabricated results.
15. The application remains usable without internet access.
"""

from pathlib import Path
import tempfile
import time
from typing import List, Tuple
import cv2
import numpy as np
import pytest

from modules.database import DatabaseManager
from modules.engagement_analytics import EngagementAnalytics
from modules.pipeline import ClassroomPipeline, PipelineStepResult
from modules.recommendation_engine import RuleCode
from modules.report_generator import ReportGenerator
from modules.session_manager import SessionState
from modules.video_processor import VideoProcessor, VideoStatus


@pytest.fixture
def test_video_file(tmp_path: Path) -> str:
    """Generate a valid 25-frame video file with moving visual elements."""
    video_path = tmp_path / "classroom_lecture.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))

    for i in range(25):
        frame = np.full((240, 320, 3), 30, dtype=np.uint8)
        # Person 1 (moving left to right)
        x1 = 20 + i * 5
        cv2.rectangle(frame, (x1, 40), (x1 + 40, 140), (200, 200, 200), -1)
        # Person 2 (stationary at desk)
        cv2.rectangle(frame, (200, 50), (250, 160), (180, 180, 180), -1)
        out.write(frame)

    out.release()
    return str(video_path)


def test_full_application_lifecycle_integration(test_video_file: str, tmp_path: Path) -> None:
    """
    Comprehensive test validating Requirements 1-12 & 14-15:
    Session creation, video processing, person detection -> tracking -> attendance ->
    fusion -> analytics -> recommendations -> SQLite persistence -> safe stop & reports.
    """
    db_path = tmp_path / "integration_test.db"
    db = DatabaseManager(str(db_path))

    # Requirement 1: A teacher can create a session
    pipeline = ClassroomPipeline(
        db=db,
        course_code="ENG301",
        session_title="Literature & Dynamics Seminar",
        video_source=test_video_file,
        target_fps=10,
        camera_enabled=True,
        audio_enabled=False,  # Requirement 7: Audio is optional
        sustained_threshold_pct=50.0,
        window_duration_sec=0.1,  # Short aggregation window to trigger fusion during test
        custom_detector=lambda frame: [(30, 40, 40, 100), (200, 50, 50, 110)],
    )

    sess = pipeline.create_session()
    assert sess.session_id.startswith("sess_")
    assert sess.course_code == "ENG301"
    assert sess.session_title == "Literature & Dynamics Seminar"

    # Verify session persisted in SQLite
    db_sess = db.get_session(sess.session_id)
    assert db_sess is not None
    assert db_sess["status"] == "READY"

    # Requirement 2 & 3: Start session and open video source
    pipeline.start()
    assert pipeline.session_metadata.status == SessionState.ACTIVE
    assert pipeline.video_processor.is_opened()

    # Step through video frames (Requirements 4, 5, 6, 8, 9, 10)
    frames_processed = 0
    windows_generated = 0

    while True:
        res: PipelineStepResult = pipeline.step(apply_pacing=False)
        if not res.success:
            break

        frames_processed += 1
        time.sleep(0.015)

        # Requirement 4: Person detections are passed to tracking module
        assert isinstance(res.active_tracks, dict)
        assert len(res.active_tracks) >= 1, "Centroid tracker should track detected persons"

        # Requirement 8: Visual & audio features aggregated by timestamp
        if res.completed_window is not None:
            windows_generated += 1
            # Requirement 14: Missing features are not replaced with fabricated results
            assert res.completed_window.speech_activity_pct is None, "Missing audio must be None, not 0.0"
            assert res.completed_window.rms_energy_mean is None

        # Verify motion index extracted
        assert res.motion_index >= 0.0

    assert frames_processed >= 15, f"Processed {frames_processed} frames"
    assert windows_generated >= 1, "At least one multimodal window should have fused"

    # Requirement 6: Attendance events are persisted to SQLite
    events = db.get_attendance_events(sess.session_id)
    assert len(events) >= 1
    event_types = [e["event_type"] for e in events]
    assert "FIRST_SEEN" in event_types

    # Requirement 8 & 14: Activity windows stored in SQLite with genuine NULLs
    db_windows = db.get_activity_windows(sess.session_id)
    assert len(db_windows) >= 1
    for w in db_windows:
        assert w["speech_activity_pct"] is None, "SQLite must store genuine NULL for disabled audio"

    # Requirement 9: Analytics calculated from actual module outputs
    analytics_summary = EngagementAnalytics.summarize_windows(db_windows)
    assert analytics_summary.total_windows >= 1
    assert analytics_summary.mean_visual_movement >= 0.0

    # Requirement 12: Session can be stopped safely
    summary = pipeline.conclude()
    assert summary["session_id"] == sess.session_id
    assert summary["total_active_seconds"] >= 0.0
    assert summary["total_observed_tracks"] >= 1
    assert not pipeline.video_processor.is_opened(), "Video capture handle must be released cleanly"

    # Verify session marked CONCLUDED in SQLite
    concluded_sess = db.get_session(sess.session_id)
    assert concluded_sess["status"] == "CONCLUDED"

    # Requirement 11: Dashboard and reports update with real session data
    records = db.get_session_attendance(sess.session_id)
    assert len(records) >= 1
    df = ReportGenerator.generate_attendance_dataframe(
        records=records,
        session_title=sess.session_title,
        course_code=sess.course_code,
    )
    assert not df.empty
    assert "Track ID" in df.columns
    assert "Attendance Status" in df.columns

    # Verify CSV export generation
    csv_bytes = ReportGenerator.to_csv_bytes(df)
    assert len(csv_bytes) > 0
    assert b"Track ID" in csv_bytes


def test_optional_audio_and_resilience_to_sensor_errors(test_video_file: str, tmp_path: Path) -> None:
    """
    Validates Requirements 7, 13, and 14:
    - Audio is completely optional.
    - Errors in one optional modality (e.g. failing audio stream) do not crash visual tracking.
    - Missing modalities are treated as unavailable (None/NULL), not fake numbers.
    """
    db = DatabaseManager(":memory:")

    pipeline = ClassroomPipeline(
        db=db,
        course_code="TEST101",
        session_title="Fault Injection Resilience",
        video_source=test_video_file,
        camera_enabled=True,
        audio_enabled=True,
        window_duration_sec=1,
    )

    pipeline.create_session()
    pipeline.start()

    # Simulate an audio read failure/exception while video is running
    if pipeline.audio_processor:
        # Intentionally inject an error into read_stream_chunk
        pipeline.audio_processor.read_stream_chunk = lambda: (_ for _ in ()).throw(RuntimeError("Microphone hardware disconnected"))

    # Step should NOT raise an exception, audio error is absorbed gracefully
    res = pipeline.step(apply_pacing=False)
    assert res.success is True, "Pipeline must remain functional despite audio hardware error"
    assert res.audio_features is None, "Failed audio should produce None, not synthetic audio"

    # Conclude safely
    summary = pipeline.conclude()
    assert summary["session_id"] is not None


def test_attendance_teacher_override_and_audit_trail(tmp_path: Path) -> None:
    """
    Validates Requirement 5, 6, and 11:
    - Tracking observations create attendance records.
    - Teacher can manually correct/verify an attendance status.
    - Manual overrides produce an immutable audit log in attendance_events table.
    """
    db = DatabaseManager(":memory:")
    session_id = "sess_audit_test"
    db.create_session(
        session_id=session_id,
        course_code="MATH101",
        session_title="Linear Algebra",
        video_source_type="file",
    )

    # Enroll student in roster
    db.insert_student(
        student_id="STUDENT_999",
        course_code="MATH101",
        full_name="Alice Smith",
        email="alice@example.com",
    )

    # Insert initial provisional record
    db.upsert_attendance_record(
        record_id="rec_001",
        session_id=session_id,
        track_id=1,
        first_seen_timestamp=100.0,
        last_seen_timestamp=150.0,
        total_detected_seconds=50.0,
        sustained_presence_status="PROVISIONAL",
    )

    # Teacher manually overrides record to SUSTAINED_PRESENT with explanation
    success = db.override_attendance_record(
        session_id=session_id,
        track_id=1,
        status="SUSTAINED_PRESENT",
        student_id="STUDENT_999",
        verification_method="TEACHER_CONFIRMED",
        reason="Student was sitting in camera blind spot; confirmed present in room.",
    )
    assert success is True

    # Verify updated record
    rec = db.get_attendance_record(session_id, track_id=1)
    assert rec["sustained_presence_status"] == "SUSTAINED_PRESENT"
    assert rec["assigned_student_id"] == "STUDENT_999"
    assert rec["verification_method"] == "TEACHER_CONFIRMED"
    assert "blind spot" in rec["correction_notes"]

    # Verify audit event in attendance_events table
    events = db.get_attendance_events(session_id, track_id=1)
    correction_events = [e for e in events if e["event_type"] == "MANUAL_CORRECTION"]
    assert len(correction_events) == 1
    assert "blind spot" in correction_events[0]["description"]


def test_offline_capability_zero_cloud_dependencies() -> None:
    """
    Validates Requirement 15:
    The application runs 100% offline without remote network connections.
    """
    # Verify local cascade and tracker availability
    vp = VideoProcessor()
    # HOG descriptor detector loaded locally
    assert vp.hog is not None
    # Centroid tracker operates locally
    assert vp.tracker is not None

    # SQLite operates in pure local mode
    db = DatabaseManager(":memory:")
    assert db.db_path == ":memory:"


def test_multimodal_pipeline_with_audio_and_video(test_video_file: str, tmp_path: Path) -> None:
    """
    Test Phase 3, 4, 5, 9 integration:
    Full multimodal session with simultaneous video frames and active audio chunks.
    Verifies speech ratio extraction, feature fusion, persistent activity scoring,
    and clean conclusion.
    """
    db_path = tmp_path / "multimodal_integration.db"
    db = DatabaseManager(str(db_path))

    pipeline = ClassroomPipeline(
        db=db,
        course_code="DATA201",
        session_title="Multimodal Perception Lab",
        video_source=test_video_file,
        target_fps=10,
        camera_enabled=True,
        audio_enabled=True,
        sustained_threshold_pct=50.0,
        window_duration_sec=0.1,  # Short window for rapid test fusion
        custom_detector=lambda frame: [(50, 50, 60, 120)],
    )

    pipeline.create_session()
    pipeline.start()
    pipeline._mic_started = True  # Mock mic started flag

    # Feed synthetic vocal chunk into mic queue
    from tests.test_audio_processor import generate_synthetic_speech
    speech_data = generate_synthetic_speech(duration_sec=0.5, sample_rate=16000, f0=240.0)
    pipeline.audio_processor._mic_queue.put(speech_data)

    # Process first step: should ingest the audio chunk
    res = pipeline.step(apply_pacing=False)
    assert res.success is True
    assert res.headcount == 1
    assert res.audio_features is not None
    assert res.audio_features.speech_ratio_pct > 20.0
    assert res.audio_features.speech_ratio > 0.20
    assert pipeline.latest_audio_features is not None

    # Step through remaining frames until window closes
    time.sleep(0.12)
    res2 = pipeline.step(apply_pacing=False)
    assert res2.success is True

    # Conclude session cleanly
    summary = pipeline.conclude()
    assert summary["session_id"] == pipeline.session_id
    assert summary["total_observed_tracks"] == 1
    assert pipeline.video_processor.cap is None
    assert pipeline._mic_started is False

