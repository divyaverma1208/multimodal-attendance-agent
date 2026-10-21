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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("attendance_agent")


@st.cache_resource
def get_db() -> DatabaseManager:
    """Create the database manager once per server process."""
    logger.info("Initializing database at %s", settings.db.db_path)
    return DatabaseManager(db_path=settings.db.db_path)


def main() -> None:
    st.set_page_config(
        page_title="Classroom Intelligence Agent",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        db = get_db()
    except Exception as exc:
        logger.exception("Database initialization failed")
        st.error(f"Could not open the database: {exc}")
        st.stop()

    render_dashboard(db)


if __name__ == "__main__":
    main()
