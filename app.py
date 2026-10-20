"""
Multimodal Attendance & Engagement Agent - Streamlit Application Entrypoint.

Initializes application configuration, logging, database connections, and routes
to the modular presentation dashboard.
"""

import logging
import streamlit as st

from config.settings import settings
from modules.database import DatabaseManager
from ui.dashboard import render_dashboard

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("attendance_agent")


def main() -> None:
    """Main application lifecycle runner."""
    # Set up Streamlit page configuration
    st.set_page_config(
        page_title="Classroom Intelligence Agent",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize local SQLite storage
    db = DatabaseManager(db_path=settings.db.db_path)

    # Render presentation dashboard
    render_dashboard(db)


if __name__ == "__main__":
    main()
