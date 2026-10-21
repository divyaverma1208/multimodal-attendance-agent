"""
Automated Privacy and Security Verification Suite.

Tests local perception, network isolation, raw media persistence prohibitions,
Git exclusion enforcement, configurable data retention, complete session deletion,
anonymity of tracking IDs, path traversal sanitization, and PII log masking.
"""

import os
from pathlib import Path
import re
import sqlite3
from typing import List
import pytest

from config.settings import BASE_DIR, settings
from modules.attendance import AttendanceEngine, AttendanceStatus, mask_identifier
from modules.audio_processor import AudioProcessor
from modules.database import DatabaseManager
from modules.report_generator import ReportGenerator
from modules.video_processor import VideoProcessor


def test_zero_external_network_dependencies() -> None:
    """Requirement 1 & 2: Search for network imports across all module source files."""
    code_dir = BASE_DIR / "modules"
    ui_dir = BASE_DIR / "ui"
    disallowed_network_modules = [
        "requests", "urllib.request", "httpx", "aiohttp", "boto3",
        "google.cloud", "openai", "anthropic", "socket", "ftplib"
    ]

    scanned_files = list(code_dir.glob("*.py")) + list(ui_dir.glob("*.py"))
    assert len(scanned_files) >= 10

    for py_file in scanned_files:
        content = py_file.read_text(encoding="utf-8")
        for bad_mod in disallowed_network_modules:
            pattern = rf"(import\s+{bad_mod}|from\s+{bad_mod})"
            assert not re.search(pattern, content), (
                f"Security violation: Prohibited network module '{bad_mod}' found in {py_file.name}"
            )


def test_raw_media_persistence_guardrail_enforcement(tmp_path: Path) -> None:
    """Requirement 3: Ensure Video and Audio processors block raw media persistence by default."""
    # Video processor guardrail
    vp = VideoProcessor(source="nonexistent.mp4", allow_save_frames=False)
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(PermissionError, match="allow_save_frames"):
        vp.save_frame_diagnostic(dummy_frame, tmp_path / "frame.jpg")

    # Audio processor guardrail
    ap = AudioProcessor(allow_save_audio=False)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(PermissionError, match="allow_save_audio"):
        ap.save_audio_diagnostic(dummy_audio, tmp_path / "audio.wav")


def test_database_zero_media_storage_schema(tmp_path: Path) -> None:
    """Requirement 3 & 4: Ensure SQLite tables contain zero BLOB or media storage columns."""
    db_file = tmp_path / "test_sec.db"
    db = DatabaseManager(db_path=db_file)
    assert db.verify_no_raw_media_stored() is True


def test_git_exclusion_rules() -> None:
    """Requirement 4: Verify that .gitignore rules exclude databases, media, rosters, and logs."""
    gitignore_file = BASE_DIR / ".gitignore"
    assert gitignore_file.exists()
    content = gitignore_file.read_text(encoding="utf-8")

    # Verify critical patterns exist
    assert "*.db" in content
    assert "*.sqlite" in content
    assert "*.mp4" in content
    assert "*.wav" in content
    assert "*.csv" in content
    assert "!data/sample/sample_roster.csv" in content
    assert "data/temp/" in content
    assert "*.log" in content


def test_configurable_retention_and_purge(tmp_path: Path) -> None:
    """Requirement 5: Test configurable historical data retention schedules."""
    db_file = tmp_path / "retention_test.db"
    db = DatabaseManager(db_path=db_file)

    sess_old = "sess_old_test"
    sess_new = "sess_new_test"
    db.create_session(sess_old, "HIST101", "Old Lecture", "file")
    db.create_session(sess_new, "HIST101", "New Lecture", "file")

    # Artificially age the old session to 45 days ago and mark CONCLUDED
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET created_at = datetime('now', '-45 days'), status = 'CONCLUDED' WHERE session_id = ?;",
            (sess_old,),
        )

    # Purge with 30-day retention threshold
    purged = db.purge_old_records(days_threshold=30)
    assert purged == 1
    assert db.get_session(sess_old) is None
    assert db.get_session(sess_new) is not None


def test_session_data_deletion_functionality(tmp_path: Path) -> None:
    """Requirement 6: Test complete session-data deletion (right to erasure)."""
    db_file = tmp_path / "delete_test.db"
    db = DatabaseManager(db_path=db_file)

    sess_id = "sess_to_delete"
    db.create_session(sess_id, "CS101", "Intro to CS", "file")
    db.insert_attendance_event("ev1", sess_id, 1, 100.0, "VERIFIED", "Confirmed")
    db.upsert_attendance_record("rec1", sess_id, 1, 0.0, 100.0, 100.0, "SUSTAINED_PRESENT")
    db.insert_activity_window("win1", sess_id, 0.0, 15.0, 5, 2.0, 1.0, 10.0, 0.02)
    db.insert_teaching_suggestion("sug1", sess_id, 10.0, "RECAP", "Conduct recap")

    # Verify data exists
    assert db.get_session(sess_id) is not None
    assert len(db.get_session_attendance(sess_id)) == 1
    assert len(db.get_activity_windows(sess_id)) == 1

    # Execute complete permanent deletion
    deleted = db.delete_session(sess_id)
    assert deleted is True

    # Assert cascading deletion cleared everything
    assert db.get_session(sess_id) is None
    assert len(db.get_session_attendance(sess_id)) == 0
    assert len(db.get_activity_windows(sess_id)) == 0
    assert len(db.get_teaching_suggestions(sess_id)) == 0


def test_student_identities_not_inferred_from_tracking(tmp_path: Path) -> None:
    """Requirement 7: Tracking IDs must remain anonymous until explicitly linked by teacher."""
    engine = AttendanceEngine()
    db_file = tmp_path / "anon_test.db"
    db = DatabaseManager(db_path=db_file)
    db.create_session("sess_anon", "CS101", "Anonymous Session", "file")

    # Ingest observations for Track-10 and Track-20
    engine.record_observation("sess_anon", track_id=10, timestamp=100.0, is_visible=True)
    engine.record_observation("sess_anon", track_id=20, timestamp=100.0, is_visible=True)

    part10 = engine.participants.get(10)
    assert part10 is not None
    assert part10.assigned_student_id is None
    assert part10.verification_method == "ANONYMOUS"

    part20 = engine.participants.get(20)
    assert part20 is not None
    assert part20.assigned_student_id is None
    assert part20.verification_method == "ANONYMOUS"


def test_teacher_can_correct_attendance(tmp_path: Path) -> None:
    """Requirement 8: Teachers can manually override attendance records and link roster."""
    db_file = tmp_path / "override_test.db"
    db = DatabaseManager(db_path=db_file)
    sess_id = "sess_override"
    db.create_session(sess_id, "CS101", "Override Session", "file")
    db.insert_student("STU_100", "CS101", "Bob Roberts")

    # Create initial provisional record
    db.upsert_attendance_record("rec_ov", sess_id, 1, 0.0, 50.0, 50.0, "PROVISIONAL")

    # Teacher applies manual override
    db.override_attendance_record(
        session_id=sess_id,
        track_id=1,
        status="SUSTAINED_PRESENT",
        verification_method="TEACHER_OVERRIDE",
        reason="Visual confirmation of student attendance",
    )

    records = db.get_session_attendance(sess_id)
    assert len(records) == 1
    assert records[0]["sustained_presence_status"] == "SUSTAINED_PRESENT"
    assert records[0]["verification_method"] == "TEACHER_OVERRIDE"


def test_path_traversal_protection(tmp_path: Path) -> None:
    """Requirement 10: Protection against path traversal in filenames."""
    export_dir = tmp_path / "safe_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame([{"Track ID": "Track-1", "Attendance Status": "PRESENT"}])

    # Path traversal attempt in session_id
    malicious_session_id = "../../etc/cron.d/evil"
    # Should sanitize and remain inside export_dir
    file_path = ReportGenerator.export_attendance_csv(df, session_id=malicious_session_id, export_dir=export_dir)
    assert file_path.exists()
    assert str(file_path.resolve()).startswith(str(export_dir.resolve()))
    assert ".." not in file_path.name


def test_sensitive_record_masking() -> None:
    """Requirement 11: Avoid logging student names or sensitive identifiers in plaintext."""
    assert mask_identifier(None) == "[ANONYMOUS]"
    assert mask_identifier("") == "[ANONYMOUS]"
    assert mask_identifier("AB") == "***"
    assert mask_identifier("STU-10045") == "ST***5"
    assert mask_identifier("Alice Smith") == "Al***h"
