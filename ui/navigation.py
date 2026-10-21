"""
Navigation helper utilities for Streamlit dashboard.

Ensures consistent programmatic page transitions by queuing target page in session state
and triggering a rerun. On rerun, ui/dashboard.py assigns the target to main_nav_radio
BEFORE the sidebar radio widget is instantiated.
"""

import streamlit as st


def navigate_to(page_name: str) -> None:
    """
    Programmatically switch the active dashboard page and trigger rerun.

    Queues the navigation target in session state and triggers an immediate rerun.
    The dashboard router updates the sidebar radio widget before instantiation.

    Args:
        page_name: Name of the destination page in the sidebar navigation menu.
    """
    st.session_state["pending_nav"] = page_name
    st.session_state["main_nav_selection"] = page_name
    st.rerun()
