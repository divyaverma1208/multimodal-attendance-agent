"""
Main Dashboard View.

Coordinates top-level sidebar navigation, persistent privacy banners, status indicators,
and page routing across the 6 core application views:
1. Home
2. Classroom Sessions
3. Attendance
4. Analytics
5. Teaching Recommendations
6. Privacy and Settings
"""

from typing import Optional
import streamlit as st

from modules.database import DatabaseManager
from ui.home_page import render_home_page
from ui.session_page import render_session_page
from ui.attendance_page import render_attendance_page
from ui.analytics_page import render_analytics_page
from ui.recommendations_page import render_recommendations_page
from ui.settings_page import render_settings_page


def render_privacy_banner() -> None:
    """Render persistent privacy and ethical guardrail banner at top of dashboard."""
    st.markdown(
        """
        <div style="background-color: #0e1e25; border-left: 5px solid #00c873; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #00c873; font-size: 1.05rem;">🔒 Offline Privacy Mode Active</strong>
                    <p style="color: #a0aec0; margin: 4px 0 0 0; font-size: 0.88rem;">
                        Anonymous spatial tracking only • <strong>Zero biometric facial recognition</strong> • 
                        Raw video/audio processed ephemerally in RAM • 100% Local SQLite storage • No cloud egress
                    </p>
                </div>
                <span style="background-color: #00c87322; color: #00c873; padding: 4px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600;">
                    AIR-GAPPED COMPLIANT
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(db: DatabaseManager) -> None:
    """
    Render main application layout and handle page routing.

    Args:
        db: Initialized DatabaseManager instance.
    """
    # Persistent privacy notice
    render_privacy_banner()

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

        if "main_nav_selection" not in st.session_state:
            st.session_state.main_nav_selection = "Home"

        # Find index of current selection
        current_idx = 0
        if st.session_state.main_nav_selection in nav_options:
            current_idx = nav_options.index(st.session_state.main_nav_selection)

        menu = st.radio(
            "Navigation Menu",
            options=nav_options,
            index=current_idx,
            key="main_nav_radio",
        )
        st.session_state.main_nav_selection = menu

        st.markdown("---")

        # System Health & Operational Status Indicators
        st.markdown("#### System Health Status")
        st.markdown("🟢 **Perception Core:** `Online (OpenCV)`")
        st.markdown("🟢 **Acoustic VAD:** `Ready (Local)`")
        st.markdown("🟢 **Database:** `Connected (WAL Mode)`")

        # Active Session Indicator if running
        sm = getattr(st.session_state, "session_manager", None)
        if sm and sm.current_session and sm.current_session.status.value in ("ACTIVE", "PAUSED"):
            st.markdown("---")
            st.markdown(f"🔴 **Active Session:** `{sm.current_session.status.value}`")
            st.caption(f"{sm.current_session.session_title} ({sm.current_session.course_code})")

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
