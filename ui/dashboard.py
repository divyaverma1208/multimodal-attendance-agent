"""
Main Dashboard View.

Coordinates top-level sidebar navigation, operational status indicators,
and page routing across the 6 core application views:
1. Home
2. Classroom Sessions
3. Attendance
4. Analytics
5. Teaching Recommendations
6. Privacy and Settings
"""

import logging
import streamlit as st

from modules.database import DatabaseManager
from ui.home_page import render_home_page
from ui.session_page import render_session_page
from ui.attendance_page import render_attendance_page
from ui.analytics_page import render_analytics_page
from ui.recommendations_page import render_recommendations_page
from ui.settings_page import render_settings_page

logger = logging.getLogger(__name__)


@st.fragment(run_every=2)
def render_system_health(db: DatabaseManager) -> None:
    """
    Render live dynamic system health & operational status indicators.
    Periodically refreshes via Streamlit fragment to reflect hardware state changes.
    """
    st.markdown("#### System Health Status")

    # 1. Perception Status
    pipe = getattr(st.session_state, "pipeline", None)
    vp = getattr(st.session_state, "video_processor", None)
    cam_enabled = st.session_state.get("camera_enabled", True)
    if not cam_enabled:
        st.markdown("⚪ **Perception Core:** `Disabled by user`")
    elif vp and vp.is_opened():
        st.markdown("🟢 **Perception Core:** `Processing (Active)`")
    elif vp and getattr(vp, "status", None) and vp.status.value == "ERROR":
        st.markdown("🔴 **Perception Core:** `Error`")
    else:
        st.markdown("🔵 **Perception Core:** `Ready (OpenCV)`")

    # 2. Acoustic Status
    mic_enabled = st.session_state.get("mic_enabled", True)
    if not mic_enabled:
        st.markdown("⚪ **Acoustic VAD:** `Disabled by user`")
    elif pipe and pipe.audio_processor and pipe.mic_started:
        st.markdown("🟢 **Acoustic VAD:** `Processing (Streaming)`")
    elif pipe and pipe.audio_processor and pipe.audio_processor.error_message:
        st.markdown("🟡 **Acoustic VAD:** `Device Unavailable`")
    else:
        st.markdown("🔵 **Acoustic VAD:** `Ready (Local)`")

    # 3. Database Status (Dynamically read PRAGMA journal_mode)
    try:
        with db.get_connection() as conn:
            row = conn.execute("PRAGMA journal_mode;").fetchone()
            jmode = (row[0] if row else "WAL").upper()
        st.markdown(f"🟢 **Database:** `Connected ({jmode} Mode)`")
    except Exception as exc:
        logger.exception("Database health check failed: %s", exc)
        st.markdown("🔴 **Database:** `Connection Error`")

    # Active Session Indicator if running
    sm = getattr(st.session_state, "session_manager", None)
    if sm and sm.current_session and sm.current_session.status.value in ("ACTIVE", "PAUSED"):
        st.markdown("---")
        st.markdown(f"🔴 **Active Session:** `{sm.current_session.status.value}`")
        st.caption(f"{sm.current_session.session_title} ({sm.current_session.course_code})")

    # Persistent Privacy Indicator in sidebar
    st.markdown("---")
    st.caption("🔒 **Offline Privacy Mode Active**\n\n100% Local Processing • Zero Facial Recognition • Ephemeral RAM Buffers")


def render_dashboard(db: DatabaseManager) -> None:
    """
    Render main application layout and handle page routing.

    Args:
        db: Initialized DatabaseManager instance.
    """

    # Sidebar Navigation
    with st.sidebar:
        st.title("🎓 Classroom Agent")
        st.caption("Multimodal Attendance & Engagement")
        st.markdown("---")

        nav_options = [
            "Home",
            "Classroom Sessions",
            "Attendance",
            "Analytics",
            "Teaching Recommendations",
            "Privacy and Settings",
        ]

        # Check if programmatic navigation was requested (e.g. from quick navigation buttons)
        if "pending_nav" in st.session_state:
            target = st.session_state.pop("pending_nav")
            if target in nav_options:
                st.session_state["main_nav_radio"] = target

        if "main_nav_radio" not in st.session_state:
            st.session_state["main_nav_radio"] = "Home"

        menu = st.radio(
            "Navigation Menu",
            options=nav_options,
            key="main_nav_radio",
        )
        st.session_state["main_nav_selection"] = menu

        st.markdown("---")

        # Dynamic System Health & Operational Status Indicators
        render_system_health(db)

    # Route to specific views
    if menu == "Home":
        render_home_page(db)
    elif menu == "Classroom Sessions":
        render_session_page(db)
    elif menu == "Attendance":
        render_attendance_page(db)
    elif menu == "Analytics":
        render_analytics_page(db)
    elif menu == "Teaching Recommendations":
        render_recommendations_page(db)
    elif menu == "Privacy and Settings":
        render_settings_page(db)
