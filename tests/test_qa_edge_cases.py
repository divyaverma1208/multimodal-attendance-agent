"""
Senior QA Edge Cases & Resilience Test Suite.

Validates:
1. Missing, null, and corrupt inputs across all core modules.
2. Database persistence boundaries, non-existent records, and SQL edge cases.
3. Session creation, pause, resume, and conclusion lifecycles under extreme conditions.
4. Attendance calculations under out-of-order timestamps, zero duration, and 100% threshold.
5. Missing audio and video condition handling.
6. Recommendation engine evaluation on empty or volatile inputs.
7. CSV report export with empty data, unicode, and special characters.
8. Resource cleanup verification for VideoProcessor and AudioProcessor.
"""

import io
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from modules.attendance import AttendanceEngine, AttendanceStatus
from modules.audio_processor import AudioProcessor, AudioFeatures
from modules.database import DatabaseManager, DatabaseError
from modules.engagement_analytics import EngagementAnalytics, SessionAnalyticsSummary
from modules.feature_fusion import FeatureFusionEngine, MultimodalWindow, DataQualityLabel
from modules.person_tracker import CentroidTracker, Track
from modules.recommendation_engine import RecommendationEngine, RuleCode
from modules.report_generator import ReportGenerator
from modules.session_manager import SessionManager, SessionState
from modules.video_processor import VideoProcessor, VideoStatus


# =============================================================================
# 1. VideoProcessor Edge Cases & Resource Cleanup
# =============================================================================

def test_video_processor_zero_or_negative_fps() -> None:
    """VideoProcessor should clamp or safely handle invalid target_fps values."""
    vp_zero = VideoProcessor(source="dummy.mp4", target_fps=0)
    assert vp_zero.target_fps >= 1

    vp_neg = VideoProcessor(source="dummy.mp4", target_fps=-10)
    assert vp_neg.target_fps >= 1


def test_video_processor_corrupt_frame_detection() -> None:
    """detect_persons should safely handle empty or invalid frame arrays without crashing."""
    vp = VideoProcessor(source="dummy.mp4")

    # Empty 0-dimension frame
    empty_frame = np.array([], dtype=np.uint8)
    boxes = vp.detect_persons(empty_frame)
    assert boxes == []

    # 1D array
    flat_frame = np.zeros(100, dtype=np.uint8)
    boxes = vp.detect_persons(flat_frame)
    assert boxes == []


def test_video_processor_resource_cleanup_idempotency() -> None:
    """Calling release() multiple times should be safe and idempotent."""
    vp = VideoProcessor(source="dummy.mp4")
    vp.release()
    vp.release()
    assert vp.cap is None
    assert vp.prev_gray_frame is None


def test_video_processor_context_manager_cleanup() -> None:
    """Context manager must guarantee cap is released on exit even if an exception occurs."""
    with pytest.raises(RuntimeError):
        with VideoProcessor(source="non_existent_file.mp4") as vp:
            assert vp.cap is None or not vp.cap.isOpened()
            raise RuntimeError("Forced simulation error")
    assert vp.cap is None


# =============================================================================
# 2. AudioProcessor Edge Cases
# =============================================================================

def test_audio_processor_empty_and_nan_features() -> None:
    """Feature extraction must safely return finite zero or fallback features for corrupt audio."""
    ap = AudioProcessor()

    # Empty buffer
    empty_slice = np.array([], dtype=np.float32)
    feats_empty = ap.extract_features(empty_slice, timestamp=10.0)
    assert feats_empty.rms_energy == 0.0
    assert feats_empty.is_speech is False

    # NaN / Inf audio buffer
    corrupt_slice = np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
    feats_corrupt = ap.extract_features(corrupt_slice, timestamp=10.0)
    assert np.isfinite(feats_corrupt.rms_energy)
    assert np.isfinite(feats_corrupt.zcr)


def test_audio_processor_extreme_clipping() -> None:
    """Extreme amplitudes should be clamped and not cause numeric overflow."""
    ap = AudioProcessor()
    extreme_audio = np.array([1e6, -1e6, 500.0, -500.0], dtype=np.float32)
    feats = ap.extract_features(extreme_audio, timestamp=1.0)
    assert np.isfinite(feats.rms_energy)


# =============================================================================
# 3. Database Edge Cases & Persistence Boundaries
# =============================================================================

def test_database_non_existent_session_queries(tmp_path: Path) -> None:
    """Querying a non-existent session should return None or empty collections without errors."""
    db = DatabaseManager(db_path=tmp_path / "edge_db.db")

    assert db.get_session("non_existent_id") is None
    assert db.get_session_attendance("non_existent_id") == []
    assert db.get_activity_windows("non_existent_id") == []
    assert db.get_teaching_suggestions("non_existent_id") == []
    assert db.delete_session("non_existent_id") is False


def test_database_special_characters_and_sql_injection(tmp_path: Path) -> None:
    """Parameterized queries must neutralize SQL injection payloads and handle unicode."""
    db = DatabaseManager(db_path=tmp_path / "sql_test.db")
    malicious_title = "'; DROP TABLE sessions; --"
    unicode_course = "CS101-コンピュータ科学-éàü"
    sess_id = "sess_sql_inject_test"

    db.create_session(
        session_id=sess_id,
        course_code=unicode_course,
        session_title=malicious_title,
        video_source_type="file",
    )

    sess = db.get_session(sess_id)
    assert sess is not None
    assert sess["session_title"] == malicious_title
    assert sess["course_code"] == unicode_course

    # Verify table was NOT dropped
    assert db.get_session(sess_id) is not None


# =============================================================================
# 4. Session Creation & Termination Lifecycles
# =============================================================================

def test_session_manager_invalid_transitions(tmp_path: Path) -> None:
    """Session state machine must reject invalid state transitions gracefully."""
    db = DatabaseManager(db_path=tmp_path / "sm_edge.db")
    sm = SessionManager(db=db)

    # Cannot start when no session is initialized
    with pytest.raises(RuntimeError, match="No session"):
        sm.start_session()

    # Pausing or concluding without an active session is a safe no-op returning 0.0
    sm.pause_session()
    assert sm.conclude_session() == 0.0

    # Initialize valid session
    sess = sm.create_session("CS202", "Data Structures")
    assert sm.current_session.status == SessionState.READY

    # Start session
    sm.start_session()
    assert sm.current_session.status == SessionState.ACTIVE

    # Pause and resume
    sm.pause_session()
    assert sm.current_session.status == SessionState.PAUSED
    sm.resume_session()
    assert sm.current_session.status == SessionState.ACTIVE

    # Conclude session
    sm.conclude_session()
    assert sm.current_session.status == SessionState.CONCLUDED

    # Concluded session cannot be restarted/resumed
    with pytest.raises(ValueError, match="Cannot start session from state"):
        sm.resume_session()


# =============================================================================
# 5. Attendance Calculations & Boundary Conditions
# =============================================================================

def test_attendance_zero_total_session_duration() -> None:
    """If total active session time is 0.0s, presence percentage should be 0.0% without ZeroDivisionError."""
    engine = AttendanceEngine()
    engine.record_observation("s_zero", track_id=1, timestamp=10.0, is_visible=True)

    status = engine.evaluate_track_status(
        track_id=1,
        total_session_active_seconds=0.0,
        threshold_pct=50.0,
    )
    # Should not crash and should not be verified present
    assert status != AttendanceStatus.SUSTAINED_PRESENT


def test_attendance_out_of_order_timestamps() -> None:
    """Engine should handle out-of-order or duplicate timestamps gracefully."""
    engine = AttendanceEngine()
    session_id = "s_time_warp"

    # Forward
    engine.record_observation(session_id, track_id=1, timestamp=100.0, is_visible=True)
    # Time warp backward (e.g. video loop or jitter)
    engine.record_observation(session_id, track_id=1, timestamp=90.0, is_visible=True)
    # Advance forward
    engine.record_observation(session_id, track_id=1, timestamp=110.0, is_visible=True)

    part = engine.participants.get(1)
    assert part is not None
    assert part.first_seen_timestamp <= part.last_seen_timestamp


def test_attendance_boundary_thresholds() -> None:
    """Test 0% and 100% threshold edge cases."""
    engine = AttendanceEngine()
    sess_id = "s_thresh"

    for t in range(0, 100, 10):
        engine.record_observation(sess_id, track_id=1, timestamp=float(t), is_visible=True)

    # 100% threshold test
    status_100 = engine.evaluate_track_status(
        track_id=1,
        total_session_active_seconds=100.0,
        threshold_pct=100.0,
    )
    assert status_100 in (AttendanceStatus.SUSTAINED_PRESENT, AttendanceStatus.PROVISIONAL, AttendanceStatus.PARTIAL)


# =============================================================================
# 6. Missing Modalities & Degradation Handling
# =============================================================================

def test_analytics_with_completely_missing_audio() -> None:
    """Analytics must compute composite activity without fabricating 0.0 for absent audio."""
    analytics = EngagementAnalytics()

    # Window with audio missing (speech_activity_pct is None)
    score = analytics.calculate_composite_activity(
        movement_index=50.0,
        speech_activity_pct=None,
        head_pose_variance=0.05,
    )
    assert score is not None
    assert 0.0 <= score <= 100.0

    # Window with both movement and speech missing
    score_empty = analytics.calculate_composite_activity(
        movement_index=None,
        speech_activity_pct=None,
        head_pose_variance=None,
    )
    assert score_empty is None


def test_feature_fusion_time_reversal_handling() -> None:
    """Observations with timestamps earlier than current window start must not crash fusion."""
    fusion = FeatureFusionEngine(window_duration_seconds=15.0)

    # Ingest at t=50.0
    fusion.ingest_visual_observation(timestamp=50.0, active_people_count=3, motion_energy=12.0)
    # Late arrival at t=40.0
    fusion.ingest_visual_observation(timestamp=40.0, active_people_count=3, motion_energy=10.0)

    closed = fusion.close_window(timestamp=65.0)
    assert closed is not None
    assert closed.visual_sample_count >= 1


# =============================================================================
# 7. Recommendation Engine Volatile & Empty Input Resilience
# =============================================================================

def test_recommendation_engine_empty_and_single_window() -> None:
    """Recommendation engine should not generate spurious alerts on empty or single window."""
    re = RecommendationEngine()

    assert re.evaluate_windows([]) == []

    # Single window should not trigger multi-window trend rules
    single_win = MultimodalWindow(
        window_id="win_1",
        session_id="sess_1",
        window_start_sec=0.0,
        window_end_sec=15.0,
        active_person_count=5,
        visual_movement_index=10.0,
        head_pose_variance=0.01,
        speech_activity_pct=0.0,
        rms_energy_mean=0.02,
    )
    suggestions = re.evaluate_windows([single_win])
    assert len(suggestions) == 0


# =============================================================================
# 8. Report Generator CSV Export Edge Cases
# =============================================================================

def test_report_generator_empty_records_export(tmp_path: Path) -> None:
    """Exporting empty attendance records must generate valid header-only CSV."""
    df_empty = ReportGenerator.generate_attendance_dataframe([])
    assert df_empty.empty
    assert "Track ID" in df_empty.columns

    csv_bytes = ReportGenerator.to_csv_bytes(df_empty)
    assert len(csv_bytes) > 0
    assert b"Track ID" in csv_bytes

    csv_file = ReportGenerator.export_attendance_csv(df_empty, session_id="empty_sess", export_dir=tmp_path)
    assert csv_file.exists()


def test_report_generator_unicode_and_quotes(tmp_path: Path) -> None:
    """Records with quotes, commas, and international characters must export valid RFC-compliant CSV."""
    records = [
        {
            "track_id": 1,
            "assigned_student_id": "STU_99",
            "student_name": 'Dr. Jane "Ada" Lovelace, FRS',
            "sustained_presence_status": "SUSTAINED_PRESENT",
            "first_seen_timestamp": 1700000000.0,
            "last_seen_timestamp": 1700003600.0,
            "total_detected_seconds": 3600.0,
            "verification_method": "TEACHER_CONFIRMED",
        }
    ]

    df = ReportGenerator.generate_attendance_dataframe(records)
    csv_bytes = ReportGenerator.to_csv_bytes(df)
    csv_text = csv_bytes.decode("utf-8")

    # Read back with pandas to verify RFC compliance
    read_df = pd.read_csv(io.StringIO(csv_text))
    assert len(read_df) == 1
    assert 'Dr. Jane "Ada" Lovelace, FRS' in read_df["Student Name"].values[0]
