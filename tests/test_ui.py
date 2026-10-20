"""
Automated Unit Tests for Streamlit Dashboard, Navigation, and UI Pages.

Tests:
1. Module imports and component exports.
2. Synthetic demo data generation, labeling, and cleanup.
3. Streamlit AppTest headless execution across all 6 pages (empty states).
4. Streamlit AppTest headless execution across all 6 pages (populated states).
5. Separation of verified and provisional attendance ledgers.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from config.settings import settings
from modules.database import DatabaseManager
from ui import (
    clear_demo_sessions,
    is_demo_session,
    load_synthetic_demo_session,
    render_analytics_page,
    render_attendance_page,
    render_dashboard,
    render_home_page,
    render_recommendations_page,
    render_session_page,
    render_settings_page,
)


def test_ui_exports() -> None:
    """Verify that all 6 page renderers and demo helpers are exported from ui package."""
    assert callable(render_dashboard)
    assert callable(render_home_page)
    assert callable(render_session_page)
    assert callable(render_attendance_page)
    assert callable(render_analytics_page)
    assert callable(render_recommendations_page)
    assert callable(render_settings_page)
    assert callable(load_synthetic_demo_session)
    assert callable(clear_demo_sessions)
    assert callable(is_demo_session)


def test_demo_data_lifecycle() -> None:
    """Test generating, identifying, and clearing synthetic demo sessions."""
    db = DatabaseManager(":memory:")

    # Generate demo session
    demo_id = load_synthetic_demo_session(db)
    assert demo_id.startswith("demo_")

    sess = db.get_session(demo_id)
    assert sess is not None
    assert is_demo_session(sess)
    assert "[DEMO]" in sess["session_title"]

    # Verify relational children
    records = db.get_session_attendance(demo_id)
    assert len(records) >= 5

    # Check separation of verified vs provisional in demo data
    verified = [r for r in records if r["sustained_presence_status"] == "SUSTAINED_PRESENT"]
    provisional = [r for r in records if r["sustained_presence_status"] != "SUSTAINED_PRESENT"]
    assert len(verified) >= 2
    assert len(provisional) >= 2

    # Check activity windows
    windows = db.get_activity_windows(demo_id)
    assert len(windows) >= 10

    # Verify NULL modality handling in demo data (unavailable measurements)
    missing_audio_wins = [w for w in windows if w["speech_activity_pct"] is None]
    assert len(missing_audio_wins) >= 1

    # Check suggestions
    suggestions = db.get_teaching_suggestions(demo_id)
    assert len(suggestions) >= 2

    # Purge demo data
    purged = clear_demo_sessions(db)
    assert purged == 1
    assert db.get_session(demo_id) is None


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_navigation_empty_state() -> None:
    """Run Streamlit AppTest across all 6 navigation views in an empty state."""
    at = AppTest.from_file(APP_PATH, default_timeout=10)
    at.run()
    assert not at.exception

    nav_pages = [
        "Home",
        "Classroom Sessions",
        "Attendance",
        "Analytics",
        "Teaching Recommendations",
        "Privacy and Settings",
    ]

    for page in nav_pages:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"Navigation failure on page '{page}': {at.exception}"


def test_app_navigation_with_demo_data() -> None:
    """Run Streamlit AppTest across all 6 navigation views with populated demo session."""
    db = DatabaseManager(settings.db.db_path)
    demo_id = load_synthetic_demo_session(db)

    try:
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()
        assert not at.exception

        nav_pages = [
            "Home",
            "Classroom Sessions",
            "Attendance",
            "Analytics",
            "Teaching Recommendations",
            "Privacy and Settings",
        ]

        for page in nav_pages:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, f"Navigation failure on page '{page}' with demo data: {at.exception}"
    finally:
        clear_demo_sessions(db)


def test_sample_recorded_video_asset() -> None:
    """Verify that the pre-recorded demonstration video asset exists and is decodable."""
    from modules.video_processor import VideoProcessor, VideoStatus
    demo_video = Path("data/sample/classroom_demo.mp4")
    assert demo_video.exists(), "Sample demo video data/sample/classroom_demo.mp4 must exist for offline evaluation"
    assert demo_video.stat().st_size > 1000

    vp = VideoProcessor(source=str(demo_video), target_fps=10)
    assert vp.open_source() is True
    assert vp.metadata.width == 640
    assert vp.metadata.height == 480
    assert vp.metadata.total_frames >= 50
    ret, frame = vp.get_next_frame(apply_pacing=False)
    assert ret is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    vp.release()


def test_attendance_events_and_csv_generation() -> None:
    """Test that attendance events and CSV byte exports are generated properly."""
    from modules.report_generator import ReportGenerator
    import pandas as pd

    db = DatabaseManager(":memory:")
    demo_id = load_synthetic_demo_session(db)

    try:
        events = db.get_attendance_events(demo_id)
        assert len(events) >= 3
        event_types = {e["event_type"] for e in events}
        assert "ENTRY" in event_types

        # Verify CSV conversion of events
        df_events = pd.DataFrame(events)
        csv_bytes = df_events.to_csv(index=False).encode("utf-8")
        assert len(csv_bytes) > 50
        assert b"ENTRY" in csv_bytes

        # Verify attendance record CSV export
        records = db.get_session_attendance(demo_id)
        df_att = ReportGenerator.generate_attendance_dataframe(records, "CS101", "Demo Session")
        att_csv = ReportGenerator.to_csv_bytes(df_att)
        assert len(att_csv) > 50
        assert b"Track ID" in att_csv
    finally:
        clear_demo_sessions(db)
