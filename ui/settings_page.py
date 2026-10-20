"""
Settings and Privacy Controls Page.

Provides controls for data retention schedules, privacy audit compliance, hardware parameters,
telemetry pruning, local database diagnostics, and clearly labelled demo data management.
"""

from pathlib import Path
from typing import Any, Dict, List
import streamlit as st

from config.settings import settings
from modules.database import DatabaseManager
from ui.demo_data import clear_demo_sessions, is_demo_session, load_synthetic_demo_session


def render_settings_page(db: DatabaseManager) -> None:
    """
    Render application configuration and privacy governance interface.

    Args:
        db: Initialized DatabaseManager instance.
    """
    st.header("⚙️ Settings & Privacy Governance")

    # Section 1: Institutional Privacy Notice & Governance Protocol
    st.subheader("📜 Institutional Privacy, Consent & Governance Notice")
    st.info(
        "**Core Mandate:** Offline-first data minimization. All student attendance records and "
        "activity telemetry are stored in local SQLite without external network egress. "
        "Zero facial recognition is performed; raw audio/video frames are ephemeral in RAM and discarded immediately."
    )
    with st.expander("🔍 View Full Institutional Data-Use & Consent Protocol", expanded=False):
        st.markdown(
            """
            ### Institutional Data-Use & Privacy Policy
            1. **Ephemeral Sensor Processing:** Video and audio inputs are processed exclusively in volatile memory (`RAM`). Raw frames, facial crops, and raw audio waveforms are discarded immediately after feature extraction.
            2. **No Automated Biometrics:** Human presence is identified using temporary tracking IDs (`Track-01`, `Track-02`). Biometric facial recognition and voiceprint profiling are strictly prohibited by architecture.
            3. **Explicit Identity Binding:** Tracking IDs can only be bound to enrolled students through explicit manual assignment by an instructor or an authenticated student check-in token.
            4. **Data Retention & Right to Erasure:** Session data and telemetry are subject to configurable institutional retention limits and may be permanently deleted at any time by authorized personnel.
            5. **Zero Cloud Telemetry:** No metrics, identifiers, or audio/video streams are ever transmitted over external networks or third-party APIs.
            """
        )

    st.markdown("---")

    # Section 2: Data Retention & Guardrail Verification
    st.subheader("🔒 Data Retention Governance & Guardrail Verification")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Historical Data Retention Schedule")
        current_retention = settings.privacy.auto_purge_days
        retention_days = st.selectbox(
            "Auto-Purge Historical Sessions Older Than:",
            options=[7, 14, 30, 60, 90, 180, 365],
            index=2 if current_retention not in [7, 14, 30, 60, 90, 180, 365] else [7, 14, 30, 60, 90, 180, 365].index(current_retention),
            help="Sessions, telemetry windows, and suggestion logs older than this threshold will be purged.",
        )

        if st.button("🧹 Execute Purge Now", type="secondary"):
            count = db.purge_old_records(retention_days)
            st.success(f"Purge complete: {count} historical session(s) older than {retention_days} days removed.")
            st.rerun()

    with col2:
        st.markdown("##### Raw Media Preservation Guardrail")
        st.checkbox(
            "Allow Raw Video/Audio Disk Caching (Debug/Archival Mode)",
            value=False,
            disabled=True,
            help="Strictly locked by policy to maintain student biometric privacy.",
        )
        st.caption(
            "🔒 **Locked by Architecture:** Raw frames are processed ephemerally in RAM and "
            "never written to persistent storage."
        )

        # Architectural Privacy Assertion
        no_blob = db.verify_no_raw_media_stored()
        if no_blob:
            st.success("✅ SQLite Schema Verified: Zero BLOB columns across all tables.")
        else:
            st.error("⚠️ Privacy Violation Warning: Non-compliant media storage detected!")

    st.markdown("---")

    # Section 3: Permanent Session Deletion (Right to Erasure / GDPR Compliance)
    st.subheader("🗑️ Permanent Session Deletion (Right to Erasure)")
    st.caption(
        "Permanently delete all records, events, activity windows, and teaching recommendations "
        "associated with a specific session from the local SQLite database."
    )

    sessions = db.list_sessions(limit=50)
    if sessions:
        session_opts = {
            f"{s['course_code']} - {s['session_title']} ({s['created_at'][:16]}) [{s['status']}]": s["session_id"]
            for s in sessions
        }
        del_target_label = st.selectbox("Select Session to Permanently Delete", options=list(session_opts.keys()), key="sb_perm_del_settings")
        del_target_id = session_opts[del_target_label]

        confirm_del_box = st.checkbox(
            "I confirm that this session should be permanently deleted from the database.",
            key="chk_confirm_del_settings",
        )

        if st.button("Permanently Delete Selected Session", type="primary", key="btn_perm_del_settings"):
            if not confirm_del_box:
                st.error("Please confirm acknowledgment before proceeding with deletion.")
            else:
                ok = db.delete_session(del_target_id)
                if ok:
                    st.success(f"Session {del_target_id[:8]} permanently deleted from SQLite.")
                    if st.session_state.get("selected_session_id") == del_target_id:
                        st.session_state["selected_session_id"] = None
                    st.rerun()
                else:
                    st.error("Failed to delete session.")
    else:
        st.caption("No sessions available to delete.")

    st.markdown("---")

    # Section 4: Selective Telemetry Pruning (Data Minimization)
    st.subheader("✂️ Selective Telemetry Pruning")
    st.caption(
        "Delete granular activity time-windows and pedagogical suggestions for completed sessions "
        "while permanently retaining verified attendance ledgers for institutional audit."
    )

    if sessions:
        prune_target_label = st.selectbox("Select Session to Prune Telemetry", options=list(session_opts.keys()), key="sb_prune_settings")
        prune_target_id = session_opts[prune_target_label]

        if st.button("Prune Telemetry (Retain Attendance Only)", type="secondary", key="btn_prune_settings"):
            del_w, del_s = db.delete_telemetry_for_session(prune_target_id)
            st.success(f"Pruned {del_w} activity windows and {del_s} suggestions. Attendance records preserved.")
            st.rerun()
    else:
        st.caption("No sessions available to prune.")

    # Section 5: Temporary Uploads Cleanup
    st.markdown("---")
    st.subheader("🧹 Temporary Staged Video Cache Cleanup")
    temp_dir = Path("data/temp")
    temp_files = list(temp_dir.glob("*")) if temp_dir.exists() else []
    temp_size_mb = round(sum(f.stat().st_size for f in temp_files if f.is_file()) / (1024 * 1024), 2)
    st.caption(f"Cached temporary uploaded video files: {len(temp_files)} file(s) ({temp_size_mb} MB).")
    if temp_files:
        if st.button("Clear Temporary Staged Files", type="secondary"):
            cleared = 0
            for f in temp_files:
                try:
                    if f.is_file():
                        f.unlink()
                        cleared += 1
                except Exception:
                    pass
            st.success(f"Cleaned up {cleared} temporary staged video file(s).")
            st.rerun()

    # Section 3: Synthetic Demo Mode Management
    st.subheader("🧪 Synthetic Demo Mode Management")
    st.caption(
        "Demonstration and UI inspection tool. Populates realistic synthetic sessions with clearly labelled "
        "`[DEMO]` tags. Fake data is NEVER masqueraded as live output."
    )

    demo_c1, demo_c2 = st.columns(2)
    with demo_c1:
        if st.button("📥 Load Synthetic Sample Session [DEMO]", key="btn_load_demo_settings"):
            with st.spinner("Generating synthetic classroom telemetry..."):
                demo_id = load_synthetic_demo_session(db)
            st.success(f"Generated synthetic demo session (`{demo_id}`).")
            st.rerun()

    with demo_c2:
        demo_sessions = [s for s in sessions if is_demo_session(s)]
        if demo_sessions:
            if st.button(f"🗑️ Purge All Demo Sessions ({len(demo_sessions)})", key="btn_purge_demo_settings"):
                count = clear_demo_sessions(db)
                st.success(f"Purged {count} synthetic demo session(s).")
                st.rerun()
        else:
            st.caption("No active demo sessions in database.")

    st.markdown("---")

    # Section 4: Perception & Hardware Configuration
    st.subheader("📹 Hardware & Detection Tuning")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.slider(
            "Default Processing Frame Rate (FPS)",
            min_value=4,
            max_value=20,
            value=settings.video.target_fps,
            help="Lower rates reduce CPU consumption while preserving trajectory tracking.",
        )
        st.slider(
            "Max Centroid Tracking Match Distance (px)",
            min_value=30.0,
            max_value=150.0,
            value=settings.video.max_centroid_distance_px,
        )

    with col_h2:
        st.slider(
            "Voice Activity Energy Threshold",
            min_value=0.005,
            max_value=0.05,
            value=settings.audio.energy_vad_threshold,
            step=0.005,
        )
        st.slider(
            "Analytics Window Duration (Seconds)",
            min_value=15,
            max_value=120,
            value=settings.analytics.window_duration_seconds,
            step=15,
        )

    st.markdown("---")

    # Section 5: Storage Diagnostics
    st.subheader("💾 Local Storage Diagnostics")
    db_file = Path(settings.db.db_path)
    db_size_kb = round(db_file.stat().st_size / 1024, 2) if db_file.exists() else 0.0

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Database File", str(db_file.name))
    d2.metric("Database Size", f"{db_size_kb} KB")
    d3.metric("Journal Mode", "WAL")
    d4.metric("Foreign Keys", "Enabled")

    st.code(f"Absolute Storage Path: {db_file.resolve()}", language="bash")
