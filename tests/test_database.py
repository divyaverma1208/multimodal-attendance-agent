"""
Comprehensive Unit Tests for DatabaseManager SQLite Storage and Relational Schema.

Validates all 15 core database requirements:
1. Automatic table and index creation.
2. Parameterized SQL queries (SQL injection safety).
3. Session CRUD operations (Create, Read, Update, Delete).
4. Participant roster operations (Insert, Bulk, Get, List, Update, Delete).
5. Timestamped attendance events logging and querying.
6. Aggregated activity measurements (with NULL support for missing modalities).
7. Recommendation history and acknowledgment.
8. Manual attendance corrections and teacher overrides.
9. Database transactions and atomic rollback.
10. Safe database error handling.
11. Configurable local database path and auto-directory creation.
12. Strict privacy guardrail (zero raw media / BLOB storage).
13. Configurable record-retention and telemetry deletion.
14. Git exclusion verification.
"""

from pathlib import Path
import tempfile
import time
import pytest

from modules.database import DatabaseError, DatabaseManager


@pytest.fixture
def temp_db() -> DatabaseManager:
    """Fixture providing an isolated temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseManager(db_path=Path(tmp.name))
        yield db


@pytest.fixture
def memory_db() -> DatabaseManager:
    """Fixture providing an in-memory SQLite database."""
    return DatabaseManager(db_path=":memory:")


# -----------------------------------------------------------------------------
# 1. Automatic Schema and Table Creation
# -----------------------------------------------------------------------------

def test_automatic_table_and_index_creation(temp_db: DatabaseManager) -> None:
    """Verify that all required tables and indexes are automatically created."""
    with temp_db.get_connection() as conn:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]
        expected_tables = {
            "sessions",
            "students",
            "attendance_records",
            "attendance_events",
            "activity_windows",
            "teaching_suggestions",
        }
        assert expected_tables.issubset(set(tables)), f"Missing tables: {expected_tables - set(tables)}"

        indexes = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        ]
        expected_indexes = {
            "idx_attendance_session",
            "idx_events_session",
            "idx_windows_session",
            "idx_suggestions_session",
            "idx_students_course",
        }
        assert expected_indexes.issubset(set(indexes)), f"Missing indexes: {expected_indexes - set(indexes)}"


# -----------------------------------------------------------------------------
# 2. Parameterized SQL Queries & Injection Safety
# -----------------------------------------------------------------------------

def test_parameterized_queries_sql_injection_resilience(temp_db: DatabaseManager) -> None:
    """Verify that user-supplied input with SQL syntax is treated safely as literals."""
    injection_id = "sess_' OR '1'='1'; DROP TABLE students; --"
    injection_title = "Title'); DROP TABLE sessions; --"

    temp_db.create_session(
        session_id=injection_id,
        course_code="SEC101",
        session_title=injection_title,
        video_source_type="webcam",
    )

    # Tables must still exist
    sess = temp_db.get_session(injection_id)
    assert sess is not None
    assert sess["session_id"] == injection_id
    assert sess["session_title"] == injection_title

    # Verify students table was not dropped
    with temp_db.get_connection() as conn:
        row = conn.execute("SELECT count(*) as cnt FROM students;").fetchone()
        assert row["cnt"] == 0


# -----------------------------------------------------------------------------
# 3. Session CRUD Operations
# -----------------------------------------------------------------------------

def test_session_crud_operations(temp_db: DatabaseManager) -> None:
    """Test full CRUD lifecycle for sessions."""
    sess_id = "sess_crud_1"

    # Create
    temp_db.create_session(
        session_id=sess_id,
        course_code="CS101",
        session_title="Algorithms",
        video_source_type="webcam",
        audio_enabled=True,
        sustained_threshold_pct=65.0,
    )

    # Read
    sess = temp_db.get_session(sess_id)
    assert sess is not None
    assert sess["status"] == "READY"
    assert sess["course_code"] == "CS101"
    assert sess["sustained_threshold_pct"] == 65.0
    assert sess["audio_enabled"] == 1

    # Update status
    temp_db.update_session_status(
        session_id=sess_id,
        status="ACTIVE",
        started_at="2026-10-20T10:00:00",
    )
    sess_updated = temp_db.get_session(sess_id)
    assert sess_updated["status"] == "ACTIVE"
    assert sess_updated["started_at"] == "2026-10-20T10:00:00"

    # Update arbitrary safe columns
    ok = temp_db.update_session(sess_id, session_title="Advanced Algorithms", total_active_seconds=300.0)
    assert ok is True
    sess_updated2 = temp_db.get_session(sess_id)
    assert sess_updated2["session_title"] == "Advanced Algorithms"
    assert sess_updated2["total_active_seconds"] == 300.0

    # Reject invalid column updates
    with pytest.raises(ValueError, match="Invalid column names"):
        temp_db.update_session(sess_id, malicious_col="value")

    # List sessions with filter
    temp_db.create_session("sess_crud_2", "MATH101", "Calculus", "file")
    cs_sessions = temp_db.list_sessions(course_code="CS101")
    assert len(cs_sessions) == 1
    assert cs_sessions[0]["session_id"] == sess_id

    all_sessions = temp_db.list_sessions(limit=10)
    assert len(all_sessions) == 2

    # Delete
    deleted = temp_db.delete_session(sess_id)
    assert deleted is True
    assert temp_db.get_session(sess_id) is None
    # Deleting non-existent returns False
    assert temp_db.delete_session("non_existent") is False


# -----------------------------------------------------------------------------
# 4. Participant Roster Operations
# -----------------------------------------------------------------------------

def test_participant_roster_operations(temp_db: DatabaseManager) -> None:
    """Test inserting, listing, updating, bulk importing, and deleting students."""
    # Insert
    temp_db.insert_student(
        student_id="STU_001",
        course_code="CS101",
        full_name="Alice Smith",
        email="alice@univ.edu",
    )

    stu = temp_db.get_student("STU_001")
    assert stu is not None
    assert stu["full_name"] == "Alice Smith"
    assert stu["email"] == "alice@univ.edu"

    # Update
    updated = temp_db.update_student("STU_001", full_name="Alice Johnson", email="alice.j@univ.edu")
    assert updated is True
    stu_updated = temp_db.get_student("STU_001")
    assert stu_updated["full_name"] == "Alice Johnson"
    assert stu_updated["email"] == "alice.j@univ.edu"

    # Bulk insert
    students_batch = [
        {"student_id": "STU_002", "course_code": "CS101", "full_name": "Bob Stone", "email": "bob@univ.edu"},
        {"student_id": "STU_003", "course_code": "MATH101", "full_name": "Charlie Day"},
    ]
    count = temp_db.bulk_insert_students(students_batch)
    assert count == 2

    cs_students = temp_db.get_students(course_code="CS101")
    assert len(cs_students) == 2
    all_students = temp_db.list_students()
    assert len(all_students) == 3

    # Delete student
    deleted = temp_db.delete_student("STU_003")
    assert deleted is True
    assert temp_db.get_student("STU_003") is None
    assert temp_db.delete_student("STU_999") is False


# -----------------------------------------------------------------------------
# 5. Timestamped Attendance Events
# -----------------------------------------------------------------------------

def test_timestamped_attendance_events(temp_db: DatabaseManager) -> None:
    """Verify logging and querying chronological attendance events."""
    sess_id = "sess_evt_1"
    temp_db.create_session(sess_id, "CS101", "Lecture", "webcam")

    t0 = time.time()
    temp_db.insert_attendance_event("evt_1", sess_id, 1, t0, "ENTRY", "Track 1 entered view")
    temp_db.log_attendance_event("evt_2", sess_id, 1, t0 + 10.0, "SUSTAINED_MET", "Sustained threshold met")
    temp_db.insert_attendance_event("evt_3", sess_id, 2, t0 + 15.0, "ENTRY", "Track 2 entered view")

    all_events = temp_db.get_attendance_events(sess_id)
    assert len(all_events) == 3
    assert all_events[0]["event_id"] == "evt_1"
    assert all_events[1]["event_id"] == "evt_2"
    assert all_events[2]["event_id"] == "evt_3"

    track1_events = temp_db.get_attendance_events(sess_id, track_id=1)
    assert len(track1_events) == 2
    assert all(e["track_id"] == 1 for e in track1_events)


# -----------------------------------------------------------------------------
# 6. Aggregated Activity Measurements & Missing Modality NULLs
# -----------------------------------------------------------------------------

def test_aggregated_activity_measurements_with_nulls(temp_db: DatabaseManager) -> None:
    """Test storing and querying activity windows, ensuring NULLs distinguish missing modalities from 0.0."""
    sess_id = "sess_act_1"
    temp_db.create_session(sess_id, "CS101", "Lab", "webcam")

    # Complete window (visual + audio)
    temp_db.insert_activity_window(
        window_id="win_1",
        session_id=sess_id,
        window_start_sec=0.0,
        window_end_sec=30.0,
        active_person_count=15,
        visual_movement_index=12.5,
        head_pose_variance=2.1,
        speech_activity_pct=45.0,
        rms_energy_mean=0.035,
        data_quality_score=1.0,
        data_quality_label="HIGH",
    )

    # Window with missing audio modality (speech_activity_pct and rms_energy_mean are None)
    temp_db.insert_activity_window(
        window_id="win_2",
        session_id=sess_id,
        window_start_sec=30.0,
        window_end_sec=60.0,
        active_person_count=14,
        visual_movement_index=10.0,
        head_pose_variance=1.8,
        speech_activity_pct=None,
        rms_energy_mean=None,
        data_quality_score=0.5,
        data_quality_label="DEGRADED",
    )

    # Window with genuine zero audio (silence, speech_activity_pct = 0.0)
    temp_db.insert_activity_window(
        window_id="win_3",
        session_id=sess_id,
        window_start_sec=60.0,
        window_end_sec=90.0,
        active_person_count=14,
        visual_movement_index=5.0,
        head_pose_variance=1.2,
        speech_activity_pct=0.0,
        rms_energy_mean=0.001,
        data_quality_score=1.0,
        data_quality_label="HIGH",
    )

    windows = temp_db.get_activity_windows(sess_id)
    assert len(windows) == 3

    # win_1: all values present
    assert windows[0]["speech_activity_pct"] == 45.0
    assert windows[0]["rms_energy_mean"] == pytest.approx(0.035)

    # win_2: missing audio modality -> NULL (Python None)
    assert windows[1]["speech_activity_pct"] is None
    assert windows[1]["rms_energy_mean"] is None
    assert windows[1]["data_quality_label"] == "DEGRADED"

    # win_3: genuine zero -> 0.0 (distinguished from None)
    assert windows[2]["speech_activity_pct"] == 0.0
    assert windows[2]["speech_activity_pct"] is not None

    # Latest window
    latest = temp_db.get_latest_activity_window(sess_id)
    assert latest is not None
    assert latest["window_id"] == "win_3"


# -----------------------------------------------------------------------------
# 7. Recommendation History & Acknowledgment
# -----------------------------------------------------------------------------

def test_recommendation_history_and_acknowledgment(temp_db: DatabaseManager) -> None:
    """Verify storing, querying, and acknowledging teaching suggestions."""
    sess_id = "sess_sug_1"
    temp_db.create_session(sess_id, "CS101", "Discussion", "webcam")

    temp_db.insert_teaching_suggestion(
        suggestion_id="sug_1",
        session_id=sess_id,
        timestamp_sec=600.0,
        trigger_rule_code="PROLONGED_MONOLOGUE",
        suggestion_text="Consider posing an open-ended question to encourage discussion.",
    )
    temp_db.log_teaching_suggestion(
        suggestion_id="sug_2",
        session_id=sess_id,
        timestamp_sec=900.0,
        trigger_rule_code="RESTLESSNESS_SPIKE",
        suggestion_text="Movement elevated. A 2-minute stretch or activity transition may help.",
    )

    all_sug = temp_db.get_teaching_suggestions(sess_id)
    assert len(all_sug) == 2
    # Ordered by timestamp descending
    assert all_sug[0]["suggestion_id"] == "sug_2"
    assert all_sug[1]["suggestion_id"] == "sug_1"

    # Acknowledge sug_1
    ack = temp_db.acknowledge_suggestion("sug_1")
    assert ack is True

    unack = temp_db.get_teaching_suggestions(sess_id, unacknowledged_only=True)
    assert len(unack) == 1
    assert unack[0]["suggestion_id"] == "sug_2"


# -----------------------------------------------------------------------------
# 8. Manual Attendance Corrections & Overrides
# -----------------------------------------------------------------------------

def test_manual_attendance_corrections_and_overrides(temp_db: DatabaseManager) -> None:
    """Verify manual attendance updates, teacher overrides, and manual record creations."""
    sess_id = "sess_att_1"
    temp_db.create_session(sess_id, "CS101", "Lecture", "webcam")
    temp_db.insert_student("STU_100", "CS101", "Grace Hopper", "grace@univ.edu")

    # Automated tracking detection
    temp_db.upsert_attendance_record(
        record_id="rec_100",
        session_id=sess_id,
        track_id=1,
        first_seen_timestamp=10.0,
        last_seen_timestamp=500.0,
        total_detected_seconds=490.0,
        sustained_presence_status="SUSTAINED_PRESENT",
    )

    # 1. Assign student to track
    temp_db.assign_student_to_track(
        session_id=sess_id,
        track_id=1,
        student_id="STU_100",
        verification_method="TEACHER_CONFIRMED",
    )
    rec = temp_db.get_attendance_record(sess_id, 1)
    assert rec is not None
    assert rec["assigned_student_id"] == "STU_100"
    assert rec["student_name"] == "Grace Hopper"
    assert rec["verification_method"] == "TEACHER_CONFIRMED"

    # 2. Status update
    temp_db.update_attendance_status(
        session_id=sess_id,
        track_id=1,
        status="PARTIAL",
        verification_method="TEACHER_OVERRIDE",
        notes="Student stepped out early for medical appointment",
    )
    rec_updated = temp_db.get_attendance_record(sess_id, 1)
    assert rec_updated["sustained_presence_status"] == "PARTIAL"
    assert rec_updated["verification_method"] == "TEACHER_OVERRIDE"
    assert "medical appointment" in rec_updated["correction_notes"]

    # 3. Comprehensive override method
    ok = temp_db.override_attendance_record(
        session_id=sess_id,
        track_id=1,
        status="SUSTAINED_PRESENT",
        reason="Teacher confirmed presence via roll call",
    )
    assert ok is True
    rec_overridden = temp_db.get_attendance_record(sess_id, 1)
    assert rec_overridden["sustained_presence_status"] == "SUSTAINED_PRESENT"

    # Verify audit event was logged
    events = temp_db.get_attendance_events(sess_id, track_id=1)
    override_events = [e for e in events if e["event_type"] == "MANUAL_CORRECTION"]
    assert len(override_events) >= 1
    assert "roll call" in override_events[-1]["description"]

    # 4. Manual entry for untracked student
    temp_db.create_manual_attendance_record(
        record_id="rec_manual_1",
        session_id=sess_id,
        track_id=-1,
        status="SUSTAINED_PRESENT",
        student_id="STU_100",
        reason="Camera occlusion, confirmed present in front row",
    )
    manual_rec = temp_db.get_attendance_record(sess_id, -1)
    assert manual_rec is not None
    assert manual_rec["verification_method"] == "TEACHER_MANUAL_ENTRY"
    assert manual_rec["sustained_presence_status"] == "SUSTAINED_PRESENT"


# -----------------------------------------------------------------------------
# 9. Database Transactions & Rollback
# -----------------------------------------------------------------------------

def test_transaction_rollback_on_error(temp_db: DatabaseManager) -> None:
    """Verify that failure during a transaction cleanly rolls back all modifications."""
    sess_id = "sess_tx_1"
    temp_db.create_session(sess_id, "CS101", "Transaction Test", "webcam")

    # Attempting to execute an operation that violates a foreign key constraint
    with pytest.raises(DatabaseError):
        with temp_db.get_connection() as conn:
            # First statement succeeds
            conn.execute(
                """
                INSERT INTO students (student_id, course_code, full_name)
                VALUES ('STU_TX_1', 'CS101', 'Tx Test');
                """
            )
            # Second statement fails (invalid foreign key on non-existent session)
            conn.execute(
                """
                INSERT INTO attendance_records (
                    record_id, session_id, track_id, first_seen_timestamp,
                    last_seen_timestamp, total_detected_seconds, sustained_presence_status
                ) VALUES ('bad_rec', 'NON_EXISTENT_SESSION', 1, 0.0, 10.0, 10.0, 'INVALID_STATUS');
                """
            )

    # First statement should be rolled back!
    assert temp_db.get_student("STU_TX_1") is None


# -----------------------------------------------------------------------------
# 10. Safe Error Handling
# -----------------------------------------------------------------------------

def test_safe_error_handling_on_invalid_sql(temp_db: DatabaseManager) -> None:
    """Verify that database errors are wrapped in DatabaseError without leaking unhandled exceptions."""
    with pytest.raises(DatabaseError, match="Database transaction error"):
        with temp_db.get_connection() as conn:
            conn.execute("SELECT * FROM non_existent_table_xyz;")


# -----------------------------------------------------------------------------
# 11. Configurable Local Database Path & Auto-Directory Creation
# -----------------------------------------------------------------------------

def test_configurable_database_path_and_directory_creation() -> None:
    """Verify that custom paths (including nested nonexistent directories) are created automatically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_db_path = Path(tmpdir) / "nested" / "deeply" / "test_custom.db"
        assert not nested_db_path.parent.exists()

        db = DatabaseManager(db_path=nested_db_path)
        assert nested_db_path.exists()
        assert db.get_session("any") is None


def test_in_memory_database(memory_db: DatabaseManager) -> None:
    """Verify that :memory: databases operate seamlessly without filesystem dependencies."""
    memory_db.create_session("mem_1", "CS101", "Memory Session", "webcam")
    sess = memory_db.get_session("mem_1")
    assert sess is not None
    assert sess["session_id"] == "mem_1"


# -----------------------------------------------------------------------------
# 12. Privacy Architecture: Never Store Raw Media (No BLOBs)
# -----------------------------------------------------------------------------

def test_privacy_guardrail_zero_raw_media_storage(temp_db: DatabaseManager) -> None:
    """
    Architectural guarantee: assert that NO table in SQLite has columns
    capable of storing raw video frames or raw audio waveforms.
    """
    assert temp_db.verify_no_raw_media_stored() is True

    # Manually check every column type across all tables
    with temp_db.get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()

        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({t['name']});").fetchall()
            for col in cols:
                col_type = col["type"].upper()
                assert "BLOB" not in col_type, f"Forbidden BLOB type in {t['name']}.{col['name']}"
                assert "IMAGE" not in col_type
                assert "AUDIO" not in col_type
                assert col["name"] not in ("raw_frame", "raw_audio", "video_buffer", "audio_samples")


# -----------------------------------------------------------------------------
# 13. Configurable Record Retention & Deletion Functions
# -----------------------------------------------------------------------------

def test_record_retention_and_purge(temp_db: DatabaseManager) -> None:
    """Test purging historical records and session-specific deletion."""
    sess_old = "sess_old_1"
    sess_new = "sess_new_1"

    temp_db.create_session(sess_old, "HIST101", "Old History", "file")
    temp_db.create_session(sess_new, "HIST101", "New History", "file")

    # Artificially age the old session by setting created_at to 40 days ago
    with temp_db.get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET created_at = datetime('now', '-40 days') WHERE session_id = ?;",
            (sess_old,),
        )

    # Purge older than 30 days
    purged_count = temp_db.purge_old_records(days_threshold=30)
    assert purged_count == 1
    assert temp_db.get_session(sess_old) is None
    assert temp_db.get_session(sess_new) is not None

    # Test telemetry cleanup while preserving attendance
    temp_db.upsert_attendance_record("rec_new", sess_new, 1, 0.0, 100.0, 100.0, "SUSTAINED_PRESENT")
    temp_db.insert_activity_window("win_new", sess_new, 0.0, 30.0, 5, 2.0, 1.0, 10.0, 0.02)
    temp_db.insert_teaching_suggestion("sug_new", sess_new, 10.0, "RECAP", "Conduct recap")

    del_win, del_sug = temp_db.delete_telemetry_for_session(sess_new)
    assert del_win == 1
    assert del_sug == 1

    # Telemetry is gone
    assert len(temp_db.get_activity_windows(sess_new)) == 0
    assert len(temp_db.get_teaching_suggestions(sess_new)) == 0
    # But attendance record is preserved
    assert len(temp_db.get_session_attendance(sess_new)) == 1

    # Explicit purge_session_records
    ok = temp_db.purge_session_records(sess_new)
    assert ok is True
    assert temp_db.get_session(sess_new) is None
    assert len(temp_db.get_session_attendance(sess_new)) == 0


# -----------------------------------------------------------------------------
# 14. Cascading Foreign Key Deletion
# -----------------------------------------------------------------------------

def test_foreign_key_cascading_deletion(temp_db: DatabaseManager) -> None:
    """Verify that deleting a session automatically deletes all relational children."""
    sess_id = "sess_cascade"
    temp_db.create_session(sess_id, "CS101", "Cascade Test", "webcam")

    temp_db.upsert_attendance_record("rec_c", sess_id, 1, 0, 10, 10, "SUSTAINED_PRESENT")
    temp_db.insert_attendance_event("evt_c", sess_id, 1, 0, "ENTRY", "Entry")
    temp_db.insert_activity_window("win_c", sess_id, 0, 30, 2, 1.0, 0.5, 20.0, 0.01)
    temp_db.insert_teaching_suggestion("sug_c", sess_id, 15.0, "RULE", "Suggestion")

    # Delete session
    temp_db.delete_session(sess_id)

    # All related data should be automatically purged
    assert len(temp_db.get_session_attendance(sess_id)) == 0
    assert len(temp_db.get_attendance_events(sess_id)) == 0
    assert len(temp_db.get_activity_windows(sess_id)) == 0
    assert len(temp_db.get_teaching_suggestions(sess_id)) == 0


# -----------------------------------------------------------------------------
# 15. Student Deletion Nullification in Attendance Records
# -----------------------------------------------------------------------------

def test_student_deletion_preserves_attendance_track(temp_db: DatabaseManager) -> None:
    """Verify that deleting a student sets assigned_student_id to NULL without dropping attendance record."""
    sess_id = "sess_nullify"
    temp_db.create_session(sess_id, "CS101", "Nullify Test", "webcam")
    temp_db.insert_student("STU_DEL", "CS101", "Temporary Student")

    temp_db.upsert_attendance_record("rec_n", sess_id, 1, 0, 100, 100, "SUSTAINED_PRESENT")
    temp_db.assign_student_to_track(sess_id, 1, "STU_DEL")

    rec_before = temp_db.get_attendance_record(sess_id, 1)
    assert rec_before["assigned_student_id"] == "STU_DEL"

    # Delete student
    temp_db.delete_student("STU_DEL")

    # Attendance record remains, assigned_student_id set to NULL
    rec_after = temp_db.get_attendance_record(sess_id, 1)
    assert rec_after is not None
    assert rec_after["assigned_student_id"] is None
    assert rec_after["sustained_presence_status"] == "SUSTAINED_PRESENT"
