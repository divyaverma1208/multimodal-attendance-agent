"""
UI Module Package.

Provides modular Streamlit page views for the teacher dashboard, live session monitoring,
attendance roster claims, multimodal analytics visualizations, and privacy settings.
"""

from .dashboard import render_dashboard
from .home_page import render_home_page
from .session_page import render_session_page
from .attendance_page import render_attendance_page
from .analytics_page import render_analytics_page
from .recommendations_page import render_recommendations_page
from .settings_page import render_settings_page
from .demo_data import clear_demo_sessions, is_demo_session, load_synthetic_demo_session

__all__ = [
    "render_dashboard",
    "render_home_page",
    "render_session_page",
    "render_attendance_page",
    "render_analytics_page",
    "render_recommendations_page",
    "render_settings_page",
    "load_synthetic_demo_session",
    "clear_demo_sessions",
    "is_demo_session",
]
