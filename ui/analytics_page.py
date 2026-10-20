"""
Analytics Page Module.

Renders interactive Plotly time-series visualizations for observable classroom activity,
acoustic speech activity ratios, headcount variations, and trend detections.
Explicitly identifies when sensor measurements are unavailable versus genuine zero values.
"""

from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from modules.database import DatabaseManager
from modules.engagement_analytics import EngagementAnalytics
from ui.demo_data import is_demo_session


def render_analytics_page(db: DatabaseManager) -> None:
    """
    Render multimodal analytics charts, data quality indicators, and activity timelines.

    Args:
        db: Initialized DatabaseManager instance.
    """
    st.header("📊 Classroom Activity Analytics")

    sessions = db.list_sessions(limit=50)
    if not sessions:
        st.info("No recorded sessions found. Complete a session or load a demo session to generate analytics.")
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

    selected_label = st.selectbox("Select Session for Analysis", options=list(session_options.keys()), index=default_idx)
    selected_session_id = session_options[selected_label]
    st.session_state["selected_session_id"] = selected_session_id
    session_obj = db.get_session(selected_session_id)

    if not session_obj:
        st.error("Session details not found.")
        return

    # Indicate if viewing a synthetic demo session
    if is_demo_session(session_obj):
        st.warning("🧪 **Notice: [DEMO / SYNTHETIC DATA]** — Analytics displayed below are generated from synthetic demo records. It is strictly isolated and never confused with live sensor output.")

    # Load Activity Windows
    windows_raw = db.get_activity_windows(selected_session_id)

    if not windows_raw:
        st.info(
            f"No activity telemetry windows recorded for '{session_obj['session_title']}'. "
            "Telemetry windows are generated during active monitoring sessions."
        )
        return

    df = EngagementAnalytics.to_dataframe(windows_raw)
    if df.empty:
        st.info("Telemetry data could not be parsed into time-series.")
        return

    # Section 1: Sensor Health & Data Quality Summary
    st.subheader("📡 Sensor Availability & Data Quality")
    total_windows = len(windows_raw)
    valid_video_windows = sum(1 for w in windows_raw if w.get("visual_movement_index") is not None)
    valid_audio_windows = sum(1 for w in windows_raw if w.get("speech_activity_pct") is not None)
    high_qual_windows = sum(1 for w in windows_raw if w.get("data_quality_label") == "HIGH")

    video_coverage_pct = round((valid_video_windows / total_windows) * 100, 1) if total_windows else 0.0
    audio_coverage_pct = round((valid_audio_windows / total_windows) * 100, 1) if total_windows else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Observed Windows", f"{total_windows} ({total_windows * 0.5:.1f} min)")
    kpi2.metric(
        "Camera Availability",
        f"{video_coverage_pct}%",
        delta="Normal" if video_coverage_pct >= 90 else "Gaps Detected",
    )
    kpi3.metric(
        "Microphone Availability",
        f"{audio_coverage_pct}%",
        delta="Normal" if audio_coverage_pct >= 90 else "Gaps Detected",
    )
    kpi4.metric("High-Quality Windows", f"{high_qual_windows} / {total_windows}")

    # Notice if measurements were unavailable
    missing_audio_count = total_windows - valid_audio_windows
    missing_video_count = total_windows - valid_video_windows
    if missing_audio_count > 0 or missing_video_count > 0:
        notices = []
        if missing_audio_count > 0:
            notices.append(f"🎙️ **{missing_audio_count} window(s)** missing audio input (shown as gaps, NOT zero speech).")
        if missing_video_count > 0:
            notices.append(f"📷 **{missing_video_count} window(s)** missing camera input (shown as gaps, NOT zero headcount).")
        st.warning(" • ".join(notices))

    st.markdown("---")

    # Section 2: Summary Averages (excluding unavailable measurements)
    st.subheader("Observable Activity Highlights")
    mean_headcount = df["headcount"].dropna().mean() if not df["headcount"].dropna().empty else None
    peak_headcount = df["headcount"].dropna().max() if not df["headcount"].dropna().empty else None
    mean_motion = df["visual_movement"].dropna().mean() if not df["visual_movement"].dropna().empty else None
    mean_speech = df["speech_pct"].dropna().mean() if not df["speech_pct"].dropna().empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Headcount", f"{mean_headcount:.1f} persons" if mean_headcount is not None else "Unavailable")
    c2.metric("Peak Headcount", f"{int(peak_headcount)} persons" if peak_headcount is not None else "Unavailable")
    c3.metric("Average Movement", f"{mean_motion:.1f}%" if mean_motion is not None else "Unavailable")
    c4.metric("Average Speech Ratio", f"{mean_speech:.1f}%" if mean_speech is not None else "Unavailable")

    st.markdown("---")

    # Section 3: Interactive Plotly Time-Series Visualizations
    st.subheader("Multimodal Time-Series Timeline")
    st.caption("Acoustic speech ratios and visual movement index plotted across 30-second windows.")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Observable Classroom Dynamics (Speech Ratio % vs Visual Movement %)",
            "Observed Attendee Headcount",
        ),
    )

    # Trace 1: Speech Activity Ratio %
    fig.add_trace(
        go.Scatter(
            x=df["timeline_min"],
            y=df["speech_pct"],
            name="Speech Ratio (%)",
            line=dict(color="#00c873", width=2.5),
            mode="lines+markers",
            connectgaps=False,  # REQUIREMENT: Do NOT bridge across unavailable periods
            hovertemplate="Time: %{x:.1f}m<br>Speech Ratio: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Trace 2: Visual Movement Index %
    fig.add_trace(
        go.Scatter(
            x=df["timeline_min"],
            y=df["visual_movement"],
            name="Visual Movement (%)",
            line=dict(color="#3b82f6", width=2, dash="dash"),
            mode="lines",
            connectgaps=False,  # REQUIREMENT: Do NOT bridge across unavailable periods
            hovertemplate="Time: %{x:.1f}m<br>Movement Index: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Trace 3: Headcount Bar Chart
    fig.add_trace(
        go.Bar(
            x=df["timeline_min"],
            y=df["headcount"],
            name="Observed Headcount",
            marker=dict(color="#f59e0b", opacity=0.75),
            hovertemplate="Time: %{x:.1f}m<br>Headcount: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=540,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
    )
    fig.update_xaxes(title_text="Session Timeline (Elapsed Minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Percentage (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Headcount (Persons)", row=2, col=1)

    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Section 4: Consecutive Window Deltas & Macro Dynamics
    st.subheader("Dynamic Shifts & Group Transitions")
    st.caption("Consecutive window deltas highlight classroom activity ramp-ups, pauses, and phase shifts.")

    summary = EngagementAnalytics.summarize_windows(windows_raw)
    trends = EngagementAnalytics.detect_sustained_trends(windows_raw)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("##### Consecutive Window Transition Trends")
        if trends:
            trend_data = []
            for t in trends:
                trend_data.append({
                    "Trend": t.trend_type,
                    "Start (Min)": f"{t.start_sec / 60.0:.1f}m",
                    "End (Min)": f"{t.end_sec / 60.0:.1f}m",
                    "Net Delta": f"{t.delta_value:+.1f} pts",
                    "Pedagogical Note": t.explanation,
                })
            st.dataframe(trend_data, hide_index=True)
        else:
            st.info("No sustained macro shifts detected. Group dynamics remained steady.")

    with t_col2:
        st.markdown("##### Instructional Dynamic Profile")
        st.write(f"🗣️ **Estimated Monologue Duration:** {summary.estimated_monologue_minutes:.1f} min")
        st.write(f"👥 **Interactive Dialogue Duration:** {summary.estimated_discussion_minutes:.1f} min")
        st.write(f"🤫 **Quiet / Individual Work:** {summary.silent_period_minutes:.1f} min")
        st.write(f"📈 **Composite Activity Index:** {summary.mean_composite_activity:.1f} / 100")
        st.write(f"🛡️ **Average Data Quality:** {summary.average_data_quality * 100:.1f}%")

    st.markdown("---")

    # Section 5: Epistemic Modesty & Limitations of Automated Observation (Requirement 9)
    st.subheader("⚠️ Limitations of Automated Classroom Observation")
    st.caption("Ethical guidelines and epistemological constraints governing observational telemetry.")

    st.warning(
        "**Epistemic Modesty Notice:** Automated observational telemetry reflects gross physical pixel displacement and acoustic short-time energy only. "
        "It does NOT measure or infer internal psychological phenomena such as student attentiveness, engagement, emotional states, cognitive comprehension, or intelligence."
    )

    with st.expander("📖 Comprehensive Observation Limitations & Measurement Boundaries", expanded=True):
        st.markdown(
            """
            1. **Physical Proxies vs. Psychological States:**
               - **Visual Movement Index (%):** Quantifies relative pixel displacement between consecutive video frames across anonymous bounding boxes. It reflects bodily gestures, posture shifts, or board work. A still student may be deeply thinking or disengaged; a moving student may be taking notes or restless. The system makes zero inferences regarding cognitive state.
               - **Speech Activity Ratio (%):** Proportion of short-time audio frames classified as vocal energy via Voice Activity Detection (VAD). It reflects acoustic room energy and does not perform speech-to-text, sentiment analysis, or individual speaker diarization.
            2. **Sensor Occlusion & Hardware Failure Handling:**
               - When a microphone is disconnected or camera view is obstructed, measurements are logged as `NULL / Unavailable` rather than artificially coerced to $0.0$.
               - Missing data points appear as discontinuities (gaps) in the time-series charts. Educators should not interpret sensor gaps as student inactivity.
            3. **Strict Non-Punitive & Non-Disciplinary Rule:**
               - Automated observations must never be used to enforce disciplinary penalties, dock participation grades, or evaluate educator competence.
               - All analytics are advisory cues intended solely for educator self-reflection and lesson pacing.
            4. **Human Verification Primacy:**
               - Algorithmic presence counts and activity trends are provisional hypotheses. The human educator retains full authority to override, correct, or discard observations at any time.
            """
        )
