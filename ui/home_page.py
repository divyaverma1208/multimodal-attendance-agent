"""
Home Page Module.

Provides executive overview, system status, quick navigation launchers,
high-level statistics, core capability cards, and a clearly labelled demo session loader.
"""

from pathlib import Path
from typing import Any, Dict, List
import streamlit as st

from modules.database import DatabaseManager
from ui.demo_data import clear_demo_sessions, is_demo_session, load_synthetic_demo_session
from ui.navigation import navigate_to


def render_home_page(db: DatabaseManager) -> None:
    """
    Render executive dashboard home view.

    Args:
        db: Initialized DatabaseManager instance.
    """
    st.header("🏫 Classroom Intelligence Agent")
    st.markdown(
        "A multimodal, privacy-preserving assistant for classroom attendance verification, "
        "group acoustic dynamics, and real-time pedagogical decision support."
    )

    # 1. System Key Metrics Summary
    sessions = db.list_sessions(limit=100)
    total_sessions = len(sessions)
    active_sessions = sum(1 for s in sessions if s["status"] in ("ACTIVE", "PAUSED"))
    students = db.get_students()
    total_students = len(students)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Sessions", total_sessions)
    kpi2.metric("Active Sessions", active_sessions, delta="Live" if active_sessions > 0 else "Idle")
    kpi3.metric("Enrolled Students", total_students)
    kpi4.metric("Privacy Status", "Air-Gapped", delta="Local SQLite")

    st.markdown("---")

    # 2. Active Session Alert (if any session is active/paused)
    active_sess_list = [s for s in sessions if s["status"] in ("ACTIVE", "PAUSED")]
    if active_sess_list:
        active_sess = active_sess_list[0]
        st.warning(
            f"⚠️ **Active Session in Progress:** `{active_sess['course_code']} - {active_sess['session_title']}` "
            f"(Status: `{active_sess['status']}`)"
        )
        col_act1, col_act2 = st.columns([1, 4])
        with col_act1:
            if st.button("👉 Go to Active Session", type="primary", width="stretch"):
                navigate_to("Classroom Sessions")

    # 3. Quick Navigation Launchers
    st.subheader("⚡ Quick Navigation")
    q1, q2, q3, q4 = st.columns(4)

    nav_items = [
        (q1, "📹 Sessions", "Start or configure live camera / video tracking.", "Open Sessions", "btn_nav_sess", "Classroom Sessions"),
        (q2, "📋 Attendance", "Inspect verified vs provisional presence and export CSV.", "Open Attendance", "btn_nav_att", "Attendance"),
        (q3, "📊 Analytics", "Review Plotly time-series charts and group activity trends.", "Open Analytics", "btn_nav_anl", "Analytics"),
        (q4, "💡 Recommendations", "Inspect non-judgmental pedagogical teaching cues.", "Open Recommendations", "btn_nav_rec", "Teaching Recommendations"),
    ]

    for col, title, desc, btn_label, btn_key, page_target in nav_items:
        with col.container(border=True, height="stretch"):
            st.markdown(f"##### {title}")
            st.caption(desc)
            st.space("stretch")
            if st.button(btn_label, key=btn_key, width="stretch"):
                navigate_to(page_target)

    st.markdown("---")

    # 4. Core System Pillars
    st.subheader("🛡️ Core Principles & Architecture")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown("##### 👤 Zero Facial Recognition")
        st.write(
            "Tracking uses anonymous spatial centroids (`Track-01`, `Track-02`). "
            "Biometric facial recognition is never used; roster names are bound strictly through "
            "explicit teacher confirmation."
        )

    with p2:
        st.markdown("##### 🔒 Ephemeral RAM Media")
        st.write(
            "Raw camera frames and microphone audio waveforms exist ephemerally in RAM "
            "during feature extraction and are immediately discarded. Never stored to disk."
        )

    with p3:
        st.markdown("##### 🎯 Non-Evaluative Support")
        st.write(
            "Metrics describe observable physical movement and speech ratios only. "
            "The system never infers attention, emotion, or comprehension, and never penalizes students."
        )

    with st.expander("📜 Institutional Consent & Data-Use Notice", expanded=False):
        st.markdown(
            """
            **Classroom Observation, Privacy & Compliance Overview**
            
            - **Local Execution:** 100% offline perception. No network sockets, cloud services, or remote telemetry.
            - **Data Minimization:** No raw video or audio files are recorded or persisted. Visual and acoustic features are processed ephemerally in RAM.
            - **Identity Protection:** Tracking operates strictly on anonymous spatial tokens (`Track-01`, `Track-02`). Roster names are bound only with teacher confirmation.
            - **Data Governance:** Records are retained in local SQLite and subject to configurable institutional purge schedules. Individual sessions can be permanently deleted at any time under institutional right-to-erasure policies.
            """
        )

    st.markdown("---")

    # 5. Clearly Labelled Demo Mode Section
    st.subheader("🧪 Synthetic Demo Tour Mode")
    st.info(
        "**Offline Testing & Inspection:** If live camera or microphone hardware is unavailable, "
        "you can populate a realistic classroom session using synthetic sample data. "
        "All generated data is strictly marked with `[DEMO]` to preserve audit integrity."
    )

    d_col1, d_col2 = st.columns([1, 1])
    with d_col1:
        if st.button("📥 Load Synthetic Sample Session [DEMO]", type="secondary"):
            with st.spinner("Generating synthetic classroom telemetry..."):
                demo_id = load_synthetic_demo_session(db)
            st.success(f"Loaded synthetic demo session (`{demo_id}`). You can now view Attendance, Analytics, and Recommendations!")
            st.rerun()

    with d_col2:
        demo_sessions = [s for s in sessions if is_demo_session(s)]
        if demo_sessions:
            if st.button(f"🗑️ Clean Up Demo Data ({len(demo_sessions)} sessions)", type="secondary"):
                count = clear_demo_sessions(db)
                st.success(f"Purged {count} synthetic demo session(s).")
                st.rerun()

    # 6. Recent Sessions Overview Table
    st.markdown("---")
    st.subheader("📁 Recent Classroom Sessions")
    if sessions:
        table_rows = []
        for s in sessions[:8]:
            active_min = round(float(s.get("total_active_seconds", 0.0)) / 60.0, 1)
            is_demo = is_demo_session(s)
            label = "🧪 DEMO" if is_demo else "LIVE"
            table_rows.append({
                "Type": label,
                "Course": s["course_code"],
                "Session Title": s["session_title"],
                "Status": s["status"],
                "Created": s["created_at"][:16] if s.get("created_at") else "-",
                "Active Duration": f"{active_min} min",
                "Video Source": s["video_source_type"],
                "Audio": "Enabled" if s["audio_enabled"] else "Disabled",
            })
        st.dataframe(table_rows, hide_index=True)
    else:
        st.info("No recorded sessions found in local SQLite database. Use 'Classroom Sessions' to initialize your first session.")
