"""
Unit Tests for Attendance Engine.

Validates sustained presence verification, provisional attendance, brief occlusion bridging,
intermittent detection intervals, duplicate event suppression, session boundaries,
anonymous identity decoupling, double-counting prevention, and manual teacher corrections.
"""

from pathlib import Path
import tempfile
import time
import pytest

from modules.attendance import (
    AttendanceEngine,
    AttendanceEventType,
    AttendanceStatus,
)
from modules.database import DatabaseManager
from modules.person_tracker import Track


def test_single_detection_guardrail() -> None:
    """Test constraint: A single solitary observation must NEVER confirm attendance."""
    engine = AttendanceEngine(min_detections_for_presence=2, transient_filter_sec=30.0)
    now = time.time()

    # Single observation at t = now
    engine.record_observation(session_id="s1", track_id=101, timestamp=now, is_visible=True)

    status = engine.evaluate_track_status(track_id=101, total_session_active_seconds=600.0)
    assert status == AttendanceStatus.TRANSIENT


def test_sustained_presence_achieved() -> None:
    """Test that continuous presence >= threshold_pct earns SUSTAINED_PRESENT."""
    engine = AttendanceEngine(
        default_threshold_pct=60.0,
        transient_filter_sec=30.0,
        gap_tolerance_sec=10.0,
    )
    now = 1000.0
    session_id = "s_sustained"

    # Ingest continuous observations from t = 1000 to t = 1450 (450 seconds out of 600s = 75%)
    for t in range(1000, 1451, 5):
        engine.record_observation(session_id=session_id, track_id=1, timestamp=float(t), is_visible=True)

    status = engine.evaluate_track_status(track_id=1, total_session_active_seconds=600.0)
    assert status == AttendanceStatus.SUSTAINED_PRESENT

    evals = engine.evaluate_session_presence(total_session_active_seconds=600.0)
    assert len(evals) == 1
    assert evals[0].presence_percentage >= 70.0
    assert evals[0].status == AttendanceStatus.SUSTAINED_PRESENT


def test_provisional_attendance() -> None:
    """Test that presence >= provisional_threshold (20%) but < sustained (60%) is PROVISIONAL."""
    engine = AttendanceEngine(
        default_threshold_pct=60.0,
        provisional_threshold_pct=20.0,
        transient_filter_sec=30.0,
    )
    now = 100.0
    session_id = "s_prov"

    # Present for 150 seconds out of 500s session (30% presence)
    for t in range(100, 251, 5):
        engine.record_observation(session_id=session_id, track_id=2, timestamp=float(t), is_visible=True)

    status = engine.evaluate_track_status(track_id=2, total_session_active_seconds=500.0)
    assert status == AttendanceStatus.PROVISIONAL


def test_brief_occlusion_gap_bridging() -> None:
    """Test that brief detection gaps (<= gap_tolerance_sec) are bridged as continuous presence."""
    engine = AttendanceEngine(gap_tolerance_sec=10.0)
    session_id = "s_gap"

    engine.record_observation(session_id=session_id, track_id=3, timestamp=100.0, is_visible=True)
    engine.record_observation(session_id=session_id, track_id=3, timestamp=105.0, is_visible=True)

    # 4-second gap (subject briefly turned head or detector missed frame)
    events = engine.record_observation(session_id=session_id, track_id=3, timestamp=109.0, is_visible=True)

    state = engine.participants[3]
    # Gap bridged: presence interval spans 100.0 to 109.0 continuously
    assert len(state.presence_intervals) == 1
    assert state.presence_intervals[0] == (100.0, 109.0)
    assert len(state.absence_intervals) == 0

    event_types = [e.event_type for e in events]
    assert AttendanceEventType.BRIEF_OCCLUSION_BRIDGED in event_types


def test_intermittent_detection_and_leaving_view() -> None:
    """Test subject leaving camera view (> gap_tolerance_sec) and re-entering."""
    engine = AttendanceEngine(gap_tolerance_sec=10.0)
    session_id = "s_intermittent"

    # Period 1: Present from 100 to 130
    for t in range(100, 131, 5):
        engine.record_observation(session_id=session_id, track_id=4, timestamp=float(t), is_visible=True)

    # Absence from 130 to 200 (70s absence > gap tolerance)
    events = engine.record_observation(session_id=session_id, track_id=4, timestamp=200.0, is_visible=True)

    # Period 2: Present from 200 to 240
    for t in range(205, 241, 5):
        engine.record_observation(session_id=session_id, track_id=4, timestamp=float(t), is_visible=True)

    state = engine.participants[4]
    # Must record 2 presence intervals and 1 absence interval
    assert len(state.presence_intervals) == 2
    assert len(state.absence_intervals) == 1
    assert state.absence_intervals[0] == (130.0, 200.0)
    assert any(e.event_type == AttendanceEventType.RE_ENTERED for e in events)


def test_duplicate_events_suppression() -> None:
    """Test that identical event triggers for the same track and timestamp are suppressed."""
    engine = AttendanceEngine()
    session_id = "s_dup"

    e1 = engine._create_event(session_id, 5, 100.0, AttendanceEventType.FIRST_SEEN, "First seen")
    e2 = engine._create_event(session_id, 5, 100.0, AttendanceEventType.FIRST_SEEN, "Duplicate first seen")

    assert e1 is not None
    assert e2 is None  # Suppressed


def test_session_boundaries_and_observation_ratio() -> None:
    """Test observation proportion calculation against strict session bounds."""
    engine = AttendanceEngine()
    session_id = "s_bounds"
    session_start = 0.0
    session_end = 1000.0

    # Participant observed from start to middle (0 to 500)
    for t in range(0, 501, 10):
        engine.record_observation(session_id=session_id, track_id=6, timestamp=float(t), is_visible=True)

    ratio = engine.calculate_observation_ratio(track_id=6, expected_duration_sec=session_end - session_start)
    assert ratio == 0.50


def test_unknown_identities_remain_anonymous() -> None:
    """Test that unassigned tracks strictly retain anonymous verification method and None student ID."""
    engine = AttendanceEngine()
    session_id = "s_anon"

    for t in range(0, 200, 10):
        engine.record_observation(session_id=session_id, track_id=7, timestamp=float(t), is_visible=True)

    evals = engine.evaluate_session_presence(total_session_active_seconds=300.0)
    assert len(evals) == 1
    assert evals[0].assigned_student_id is None
    assert evals[0].verification_method == "ANONYMOUS"


def test_double_counting_prevention() -> None:
    """Test that assigning a student ID already linked to another track in the same session is rejected."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseManager(db_path=Path(tmp.name))
        session_id = "sess_double"
        db.create_session(session_id, "CS101", "Algorithms", "webcam")

        engine = AttendanceEngine()
        # Track 1 claimed for Student STU-1
        claim1 = engine.claim_track_for_student(db, session_id, track_id=1, student_id="STU-001")
        assert claim1 is True

        # Attempt to claim Track 2 for the same Student STU-1
        claim2 = engine.claim_track_for_student(db, session_id, track_id=2, student_id="STU-001")
        assert claim2 is False  # Rejected to prevent double-counting


def test_manual_teacher_correction() -> None:
    """Test authorized teacher manual status override with audit event logging."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseManager(db_path=Path(tmp.name))
        session_id = "sess_override"
        db.create_session(session_id, "MATH101", "Calculus", "webcam")

        engine = AttendanceEngine()
        # Ingest brief passer-by
        engine.record_observation(session_id, track_id=8, timestamp=10.0, is_visible=True)
        engine.record_observation(session_id, track_id=8, timestamp=15.0, is_visible=True)

        evals_initial = engine.evaluate_session_presence(total_session_active_seconds=300.0)
        assert evals_initial[0].status == AttendanceStatus.TRANSIENT
        engine.sync_to_database(db, session_id, evals_initial)

        # Teacher overrides status (e.g. verified student stepped out on permitted errand)
        engine.manual_override_status(
            db=db,
            session_id=session_id,
            track_id=8,
            new_status=AttendanceStatus.SUSTAINED_PRESENT,
            teacher_note="Excused student presentation",
        )

        evals_corrected = engine.evaluate_session_presence(total_session_active_seconds=300.0)
        assert evals_corrected[0].status == AttendanceStatus.SUSTAINED_PRESENT
        assert evals_corrected[0].verification_method == "TEACHER_OVERRIDE"

        # Verify audit event recorded in database
        events = db.get_attendance_events(session_id, track_id=8)
        assert len(events) >= 1
        assert any(e["event_type"] == AttendanceEventType.MANUAL_CORRECTION.value for e in events)


def test_sqlite_sync_and_csv_export() -> None:
    """Test persisting attendance evaluations and exporting to CSV."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseManager(db_path=Path(tmp.name))
        session_id = "sess_export"
        db.create_session(session_id, "ENG101", "Literature", "webcam")

        engine = AttendanceEngine()
        for t in range(0, 400, 10):
            engine.record_observation(session_id, track_id=9, timestamp=float(t), is_visible=True)

        evals = engine.evaluate_session_presence(total_session_active_seconds=500.0)
        engine.sync_to_database(db, session_id, evals)

        with tempfile.TemporaryDirectory() as export_dir:
            csv_path = engine.export_attendance_csv(db, session_id, export_dir=Path(export_dir))
            assert csv_path.exists()
            assert csv_path.suffix == ".csv"
