"""
Teaching Recommendations Page Module.

Provides transparent pedagogical decision support cues for educators based on multi-window
trend analytics. Enables educators to review, acknowledge, or dismiss contextual suggestions.
"""

from typing import Any, Dict, List
import streamlit as st

from modules.database import DatabaseManager
from ui.demo_data import is_demo_session


def render_recommendations_page(db: DatabaseManager) -> None:
    """
    Render pedagogical suggestions view, rule references, and acknowledgment controls.

    Args:
        db: Initialized DatabaseManager instance.
    """
    st.header("💡 Teaching Recommendations & Decision Support")
    st.markdown(
        "Non-evaluative, voluntary pedagogical cues derived from group dynamics trends. "
        "Intended solely to assist educator pacing and instructional reflection."
    )

    sessions = db.list_sessions(limit=50)
    if not sessions:
        st.info("No recorded sessions found. Complete a session or load a demo session to view recommendations.")
        return

    session_options = {
        f"{s['course_code']} - {s['session_title']} ({s['created_at'][:16]}) [{s['status']}]": s["session_id"]
        for s in sessions
    }

    # Pre-select active session or previously selected session
    preferred_id = st.session_state.get("selected_session_id")
    if not preferred_id:
        sm = getattr(st.session_state, "session_manager", None)
        if sm and sm.current_session:
            preferred_id = sm.current_session.session_id

    default_idx = 0
    if preferred_id:
        for idx, sid in enumerate(session_options.values()):
            if sid == preferred_id:
                default_idx = idx
                break

    selected_label = st.selectbox("Select Session for Teaching Cues", options=list(session_options.keys()), index=default_idx)
    selected_session_id = session_options[selected_label]
    st.session_state["selected_session_id"] = selected_session_id
    session_obj = db.get_session(selected_session_id)

    if not session_obj:
        st.error("Session details not found.")
        return

    if is_demo_session(session_obj):
        st.warning("🧪 **Notice: [DEMO / SYNTHETIC DATA]** — Pedagogical suggestions below are derived from synthetic demo records. They are strictly isolated and never confused with live sensor output.")

    # Load Teaching Suggestions
    suggestions = db.get_teaching_suggestions(selected_session_id)

    total_sug = len(suggestions)
    acknowledged_sug = sum(1 for s in suggestions if s.get("acknowledged") == 1)
    pending_sug = total_sug - acknowledged_sug

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Generated Cues", total_sug)
    k2.metric("Pending Review", pending_sug, delta="Needs Review" if pending_sug > 0 else "All Handled")
    k3.metric("Acknowledged Cues", acknowledged_sug)
    k4.metric("Rule Engine Mode", "Deterministic", delta="No Black-Box ML")

    st.markdown("---")

    # Section 1: Pedagogical Suggestions Ledger
    st.subheader("Historical Instructional Cues")

    if not suggestions:
        st.info(
            f"No pedagogical cues were triggered for '{session_obj['session_title']}'. "
            "Pacing and acoustic levels remained within standard balanced bounds throughout observation."
        )
    else:
        filter_mode = st.radio(
            "Filter Cues",
            options=["All Suggestions", "Pending Review Only", "Acknowledged Only"],
            horizontal=True,
        )

        filtered = suggestions
        if filter_mode == "Pending Review Only":
            filtered = [s for s in suggestions if s.get("acknowledged") != 1]
        elif filter_mode == "Acknowledged Only":
            filtered = [s for s in suggestions if s.get("acknowledged") == 1]

        if not filtered:
            st.caption(f"No suggestions matching '{filter_mode}'.")
        else:
            for s in filtered:
                is_ack = s.get("acknowledged") == 1
                status_badge = "✅ Acknowledged" if is_ack else "⏳ Pending Review"
                rel_min = round(float(s["timestamp_sec"]) / 60.0, 1)

                with st.container():
                    st.markdown(
                        f"""
                        <div style="background-color: #1a202c; border-left: 4px solid {'#00c873' if is_ack else '#f59e0b'}; padding: 14px 18px; border-radius: 4px; margin-bottom: 12px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <strong style="color: {'#00c873' if is_ack else '#f59e0b'}; font-size: 1.0rem;">
                                    📌 {s['trigger_rule_code']}
                                </strong>
                                <span style="font-size: 0.85rem; color: #a0aec0;">
                                    Timeline: {rel_min:.1f}m | {status_badge}
                                </span>
                            </div>
                            <p style="margin: 0 0 8px 0; color: #e2e8f0; font-size: 0.95rem;">
                                {s['suggestion_text']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if not is_ack:
                        c_act1, c_act2 = st.columns([1, 5])
                        with c_act1:
                            if st.button("Mark Acknowledged", key=f"ack_{s['suggestion_id']}", type="secondary"):
                                db.acknowledge_suggestion(s["suggestion_id"])
                                st.success("Suggestion marked as acknowledged.")
                                st.rerun()

    st.markdown("---")

    # Section 2: Pedagogical Rule Inventory & Transparency Reference
    st.subheader("📚 Pedagogical Rules Inventory & Decision Criteria")
    st.caption("All suggestions fire from deterministic, multi-window heuristics with cooldown anti-spam protection.")

    rules_inventory = [
        {
            "Rule Code": "RULE_PROLONGED_MONOLOGUE",
            "Trigger Condition": "Continuous teacher speech ratio > 80% for ≥ 10 minutes",
            "Pedagogical Intervention": "Introduce a quick Q&A activity, short poll, or peer discussion.",
            "Multi-Window Anti-Spike": "Requires 20 consecutive windows (10 min)",
        },
        {
            "Rule Code": "RULE_EXTENDED_SILENCE",
            "Trigger Condition": "Speech ratio < 5% for ≥ 3 minutes (genuine silence verified)",
            "Pedagogical Intervention": "Invite voluntary questions or provide a structured prompt.",
            "Multi-Window Anti-Spike": "Requires 6 consecutive windows (3 min)",
        },
        {
            "Rule Code": "RULE_CONDUCT_RECAP",
            "Trigger Condition": "High speech block (> 65%) followed by quiet lull (< 20%)",
            "Pedagogical Intervention": "Conduct a 2-minute synthesis recap of core concepts.",
            "Multi-Window Anti-Spike": "Evaluates 13-minute sequence transition",
        },
        {
            "Rule Code": "RULE_RESTLESSNESS_SPIKE",
            "Trigger Condition": "Visual movement > 70% and speech < 15% for ≥ 3 minutes",
            "Pedagogical Intervention": "Offer a 60-second stretch or topic transition.",
            "Multi-Window Anti-Spike": "Requires 6 consecutive windows (3 min)",
        },
        {
            "Rule Code": "RULE_CHECK_AUDIO_QUALITY",
            "Trigger Condition": "Audio stream unavailable or persistent buffer dropping",
            "Pedagogical Intervention": "Check microphone placement, connection, or permissions.",
            "Multi-Window Anti-Spike": "Requires 3 consecutive degraded windows",
        },
        {
            "Rule Code": "RULE_CHECK_CAMERA_ANGLE",
            "Trigger Condition": "Zero headcount detected during active session",
            "Pedagogical Intervention": "Review camera tilt, FOV, or lighting glare.",
            "Multi-Window Anti-Spike": "Requires 3 consecutive zero-headcount windows",
        },
        {
            "Rule Code": "RULE_PRESENCE_DROP",
            "Trigger Condition": "Detected headcount drops > 30% vs session baseline",
            "Pedagogical Intervention": "Check if camera was bumped or students stepped out for break.",
            "Multi-Window Anti-Spike": "Requires 5-window comparative horizon",
        },
        {
            "Rule Code": "RULE_MAINTAIN_ACTIVITY",
            "Trigger Condition": "Balanced speech (20-70%) and movement (10-60%) for ≥ 5 min",
            "Pedagogical Intervention": "Pacing is balanced; continue current activity as planned.",
            "Multi-Window Anti-Spike": "Requires 10 consecutive windows (5 min)",
        },
    ]

    st.dataframe(rules_inventory, hide_index=True)

    st.markdown("---")

    # Section 3: Ethical Boundaries & Limitations of Automated Observations (Requirement 9)
    st.subheader("⚠️ Ethical Boundaries & Observation Limitations")
    st.info(
        "🛡️ **Institutional Safeguards & Epistemic Boundaries:**\n"
        "- **Advisory Decision Support:** All recommendations are advisory and optional decision-support tools for the human educator.\n"
        "- **Non-Punitive & Non-Disciplinary:** The system never executes disciplinary actions, penalizes students, or evaluates teacher performance.\n"
        "- **No Psychological or Cognitive Inferences:** Suggestions trigger solely on deterministic multi-window acoustic ratios and physical motion dynamics. The engine does NOT infer student emotion, attentiveness, learning ability, or comprehension.\n"
        "- **Sensor Outage Resilience:** Missing audio or camera feeds trigger diagnostics (e.g. `RULE_CHECK_AUDIO_QUALITY`) rather than generating false instructional suggestions."
    )
