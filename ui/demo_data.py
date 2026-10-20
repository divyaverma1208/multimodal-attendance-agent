"""
Demo Data Module.

Provides synthetic sample data generation for demonstration, UI testing, and inspection
when live camera or microphone hardware is unavailable or in an offline testing environment.

IMPORTANT PRIVACY & TRANSPARENCY GUARDRAIL:
All data generated here is explicitly marked and labelled as '[DEMO / SYNTHETIC DATA]'.
Fake data is never masqueraded as live sensor or model output.
"""

from datetime import datetime, timedelta
import random
from typing import Any, Dict, List, Optional
import uuid

from modules.database import DatabaseManager


DEMO_PREFIX = "[DEMO] "


def is_demo_session(session_or_title: Any) -> bool:
    """Check if a session or session title is a synthetic demo object."""
    if isinstance(session_or_title, dict):
        title = session_or_title.get("session_title", "")
    else:
        title = str(session_or_title)
    return title.startswith(DEMO_PREFIX) or "SYNTHETIC" in title.upper()


def load_synthetic_demo_session(db: DatabaseManager) -> str:
    """
    Populate SQLite database with a realistic, clearly labelled synthetic demo classroom session.

    Args:
        db: Initialized DatabaseManager instance.

    Returns:
        The session_id of the generated demo session.
    """
    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    now = datetime.now()
    started_at = (now - timedelta(minutes=45)).isoformat()
    concluded_at = now.isoformat()

    # 1. Create Demo Session Record
    db.create_session(
        session_id=session_id,
        course_code="CS101",
        session_title=f"{DEMO_PREFIX}Intro to Computer Systems (Synthetic Sample)",
        video_source_type="file",
        audio_enabled=True,
        sustained_threshold_pct=60.0,
    )

    db.update_session_status(
        session_id=session_id,
        status="CONCLUDED",
        total_active_seconds=2700.0,  # 45 minutes
        started_at=started_at,
        concluded_at=concluded_at,
    )

    # 2. Insert Sample Student Roster for CS101
    sample_students = [
        {"student_id": "STU-101", "course_code": "CS101", "full_name": "Alice Chen", "email": "alice.c@univ.edu"},
        {"student_id": "STU-102", "course_code": "CS101", "full_name": "Bob Vance", "email": "bob.v@univ.edu"},
        {"student_id": "STU-103", "course_code": "CS101", "full_name": "Charlie Kim", "email": "charlie.k@univ.edu"},
        {"student_id": "STU-104", "course_code": "CS101", "full_name": "Diana Prince", "email": "diana.p@univ.edu"},
        {"student_id": "STU-105", "course_code": "CS101", "full_name": "Evan Wright", "email": "evan.w@univ.edu"},
    ]
    db.bulk_insert_students(sample_students)

    # 3. Insert Attendance Records (Mix of Sustained Present, Provisional, Partial, Transient)
    attendance_samples = [
        # Track 1: Sustained Present & Teacher-Confirmed (Alice)
        {
            "record_id": f"rec_{session_id}_1",
            "session_id": session_id,
            "track_id": 1,
            "first_seen": 60.0,
            "last_seen": 2680.0,
            "detected_sec": 2450.0,  # > 90% observed
            "status": "SUSTAINED_PRESENT",
            "student_id": "STU-101",
            "method": "TEACHER_CONFIRMED",
            "notes": "Verified present in seat row 2",
        },
        # Track 2: Sustained Present & Teacher-Confirmed (Bob)
        {
            "record_id": f"rec_{session_id}_2",
            "session_id": session_id,
            "track_id": 2,
            "first_seen": 75.0,
            "last_seen": 2650.0,
            "detected_sec": 2200.0,
            "status": "SUSTAINED_PRESENT",
            "student_id": "STU-102",
            "method": "TEACHER_CONFIRMED",
            "notes": None,
        },
        # Track 3: Sustained Present but Unassigned (Anonymous)
        {
            "record_id": f"rec_{session_id}_3",
            "session_id": session_id,
            "track_id": 3,
            "first_seen": 120.0,
            "last_seen": 2600.0,
            "detected_sec": 1950.0,  # ~ 72%
            "status": "SUSTAINED_PRESENT",
            "student_id": None,
            "method": "ANONYMOUS",
            "notes": None,
        },
        # Track 4: Provisional Attendance (In-Progress / Borderline observation)
        {
            "record_id": f"rec_{session_id}_4",
            "session_id": session_id,
            "track_id": 4,
            "first_seen": 300.0,
            "last_seen": 1800.0,
            "detected_sec": 1400.0,  # 51.8% of session (provisional threshold)
            "status": "PROVISIONAL",
            "student_id": "STU-103",
            "method": "TEACHER_CONFIRMED",
            "notes": "Provisional attendance observed; awaiting full threshold",
        },
        # Track 5: Partial Presence (Stepped out halfway)
        {
            "record_id": f"rec_{session_id}_5",
            "session_id": session_id,
            "track_id": 5,
            "first_seen": 100.0,
            "last_seen": 1200.0,
            "detected_sec": 950.0,  # 35% of session
            "status": "PARTIAL",
            "student_id": "STU-104",
            "method": "TEACHER_OVERRIDE",
            "notes": "Student excused early at 20-minute mark",
        },
        # Track 6: Transient passer-by (e.g., student delivering a message)
        {
            "record_id": f"rec_{session_id}_6",
            "session_id": session_id,
            "track_id": 6,
            "first_seen": 900.0,
            "last_seen": 925.0,
            "detected_sec": 25.0,
            "status": "TRANSIENT",
            "student_id": None,
            "method": "ANONYMOUS",
            "notes": "Filtered transient track (under 30s threshold)",
        },
    ]

    for att in attendance_samples:
        db.upsert_attendance_record(
            record_id=att["record_id"],
            session_id=att["session_id"],
            track_id=att["track_id"],
            first_seen_timestamp=att["first_seen"],
            last_seen_timestamp=att["last_seen"],
            total_detected_seconds=att["detected_sec"],
            sustained_presence_status=att["status"],
            assigned_student_id=att["student_id"],
            verification_method=att["method"],
            correction_notes=att["notes"],
        )

    # 4. Insert Timestamped Attendance Events
    db.insert_attendance_event(
        event_id=f"evt_{session_id}_1",
        session_id=session_id,
        track_id=1,
        timestamp=60.0,
        event_type="ENTRY",
        description="[DEMO] Track-1 entered camera FOV",
    )
    db.insert_attendance_event(
        event_id=f"evt_{session_id}_2",
        session_id=session_id,
        track_id=1,
        timestamp=1680.0,
        event_type="SUSTAINED_PRESENCE_MET",
        description="[DEMO] Track-1 met 60% sustained presence threshold",
    )
    db.insert_attendance_event(
        event_id=f"evt_{session_id}_3",
        session_id=session_id,
        track_id=5,
        timestamp=1200.0,
        event_type="MANUAL_CORRECTION",
        description="[DEMO] Teacher marked partial attendance: excused early",
    )

    # 5. Insert Activity Windows (30-sec windows across 45 min = 15 representative windows)
    # Includes clear examples of:
    # - Standard multimodal observation
    # - Lecture monologue block
    # - Discussion / peer-work block
    # - Deliberate MISSING AUDIO window (speech_activity_pct = None) to demonstrate "measurement unavailable"
    # - Deliberate MISSING VIDEO window (active_person_count = None) to demonstrate camera occlusion
    synthetic_windows = [
        # (offset_sec, headcount, movement, pose_var, speech_pct, rms, qual_score, qual_label)
        (0.0, 4, 18.0, 1.2, 35.0, 0.022, 1.0, "HIGH"),
        (180.0, 5, 12.0, 0.8, 85.0, 0.045, 1.0, "HIGH"),  # Lecture segment
        (360.0, 5, 10.0, 0.7, 88.0, 0.048, 1.0, "HIGH"),
        (540.0, 5, 14.0, 0.9, 92.0, 0.051, 1.0, "HIGH"),
        (720.0, 5, 22.0, 1.5, 60.0, 0.038, 1.0, "HIGH"),  # Transition / questions
        (900.0, 5, 65.0, 3.8, 15.0, 0.015, 1.0, "HIGH"),  # Restlessness spike
        (1080.0, 5, 25.0, 1.6, 40.0, 0.028, 1.0, "HIGH"),
        (1260.0, 4, 30.0, 2.0, 75.0, 0.042, 1.0, "HIGH"),
        # Unavailable audio window (simulating temporary mic disconnection)
        (1440.0, 4, 28.0, 1.8, None, None, 0.5, "PARTIAL_VISUAL_ONLY"),
        (1620.0, 4, 32.0, 2.2, 55.0, 0.034, 1.0, "HIGH"),
        (1800.0, 4, 45.0, 2.8, 62.0, 0.039, 1.0, "HIGH"),  # Group activity
        (1980.0, 4, 40.0, 2.6, 58.0, 0.036, 1.0, "HIGH"),
        # Unavailable visual window (simulating projector mode / camera obstruction)
        (2160.0, None, None, None, 70.0, 0.044, 0.5, "PARTIAL_AUDIO_ONLY"),
        (2340.0, 4, 15.0, 1.1, 45.0, 0.030, 1.0, "HIGH"),
        (2520.0, 4, 35.0, 2.4, 20.0, 0.018, 1.0, "HIGH"),  # Wrap-up
    ]

    for start_sec, count, mov, pose, speech, rms, qual_s, qual_l in synthetic_windows:
        end_sec = start_sec + 30.0
        win_id = f"win_{session_id}_{int(start_sec)}"
        db.insert_activity_window(
            window_id=win_id,
            session_id=session_id,
            window_start_sec=start_sec,
            window_end_sec=end_sec,
            active_person_count=count,
            visual_movement_index=mov,
            head_pose_variance=pose,
            speech_activity_pct=speech,
            rms_energy_mean=rms,
            data_quality_score=qual_s,
            data_quality_label=qual_l,
        )

    # 6. Insert Teaching Suggestions (Pedagogical Decision Support)
    db.insert_teaching_suggestion(
        suggestion_id=f"sug_{session_id}_1",
        session_id=session_id,
        timestamp_sec=600.0,
        trigger_rule_code="RULE_PROLONGED_MONOLOGUE",
        suggestion_text="[DEMO] Extended presentation detected (speech > 85% for 10 min). Consider introducing a quick recap or peer Q&A.",
    )
    db.insert_teaching_suggestion(
        suggestion_id=f"sug_{session_id}_2",
        session_id=session_id,
        timestamp_sec=960.0,
        trigger_rule_code="RULE_RESTLESSNESS_SPIKE",
        suggestion_text="[DEMO] Elevated movement index detected. Students may be ready for an activity transition or 60-second stretch.",
    )
    # Acknowledge the first one to show acknowledged vs pending states
    db.acknowledge_suggestion(f"sug_{session_id}_1")

    return session_id


def clear_demo_sessions(db: DatabaseManager) -> int:
    """Purge all synthetic demo sessions from the database."""
    sessions = db.list_sessions(limit=100)
    purged = 0
    for s in sessions:
        if is_demo_session(s):
            db.delete_session(s["session_id"])
            purged += 1
    return purged
