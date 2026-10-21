"""
Attendance Page Module.

Displays verified sustained-presence records and provisional observations separately,
enables teacher roster claims and manual status overrides, and provides CSV report downloads.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from modules.attendance import AttendanceEngine, AttendanceStatus
from modules.database import DatabaseManager
from modules.report_generator import ReportGenerator
from ui.demo_data import is_demo_session
from ui.navigation import navigate_to


def render_attendance_page(db: DatabaseManager) -> None:
    """
    Render attendance ledger, separating verified and provisional records.

    Args:
        db: Initialized DatabaseManager instance.
    """
    st.header("📋 Attendance Ledger & Verification")

    sessions = db.list_sessions(limit=50)
    if not sessions:
        st.info("No classroom sessions found in database. Create a session or load a demo session to view attendance.")
        col_c1, col_c2 = st.columns([1, 4])
        with col_c1:
            if st.button("➕ Go to Sessions", type="primary"):
                navigate_to("Classroom Sessions")
        return

    # Select Session
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

    selected_label = st.selectbox("Select Session", options=list(session_options.keys()), index=default_idx)
    selected_session_id = session_options[selected_label]
    st.session_state["selected_session_id"] = selected_session_id
    session_obj = db.get_session(selected_session_id)

    if not session_obj:
        st.error("Session details not found.")
        return

    # Indicate if viewing a synthetic demo session
    if is_demo_session(session_obj):
        st.warning("🧪 **Notice: [DEMO / SYNTHETIC DATA]** — This session contains pre-generated synthetic demo records. It is strictly isolated and never confused with live sensor output.")

    # Load Attendance Records
    records = db.get_session_attendance(selected_session_id)

    if not records:
        st.info("No attendance records have been logged for this session yet. Start the session and observe participants.")
        return

    # Generate master DataFrame
    df = ReportGenerator.generate_attendance_dataframe(
        records=records,
        session_title=session_obj["session_title"],
        course_code=session_obj["course_code"],
    )

    # Compute KPI statistics
    total_tracks = len(records)
    sustained_present = sum(1 for r in records if r["sustained_presence_status"] == "SUSTAINED_PRESENT")
    provisional = sum(1 for r in records if r["sustained_presence_status"] == "PROVISIONAL")
    partial = sum(1 for r in records if r["sustained_presence_status"] == "PARTIAL")
    transient = sum(1 for r in records if r["sustained_presence_status"] == "TRANSIENT")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observed Tracks", total_tracks)
    col2.metric("Verified Present", sustained_present, f"{session_obj['sustained_threshold_pct']}% req.")
    col3.metric("Provisional / In-Progress", provisional)
    col4.metric("Partial / Transient", partial + transient)

    st.markdown("---")

    # Load Attendance Events
    events = db.get_attendance_events(selected_session_id)

    # REQUIREMENT: Display provisional and verified attendance separately, plus timestamped events
    tab_verified, tab_provisional, tab_events = st.tabs([
        f"✅ Verified Attendance ({sustained_present})",
        f"⏳ Provisional & In-Progress ({total_tracks - sustained_present})",
        f"📜 Attendance Events Log ({len(events)})",
    ])

    with tab_verified:
        st.subheader("Verified Attendance Ledger")
        st.caption(
            "Participants who have met the sustained presence threshold or were confirmed by an authorized teacher. "
            "Only verified attendees are included on official roll sheets."
        )

        verified_records = [r for r in records if r["sustained_presence_status"] == "SUSTAINED_PRESENT"]

        if verified_records:
            df_verified = ReportGenerator.generate_attendance_dataframe(
                records=verified_records,
                session_title=session_obj["session_title"],
                course_code=session_obj["course_code"],
            )
            st.dataframe(df_verified, hide_index=True)

            csv_verified = ReportGenerator.to_csv_bytes(df_verified)
            st.download_button(
                label="📥 Download Verified Attendance CSV",
                data=csv_verified,
                file_name=f"verified_attendance_{session_obj['course_code']}_{session_obj['session_id'][:8]}.csv",
                mime="text/csv",
                type="primary",
            )
        else:
            st.info("No participants have met the sustained attendance threshold yet.")

    with tab_provisional:
        st.subheader("Provisional & In-Progress Observations")
        st.caption(
            "Tentative detections, transient visitors, or participants who have not yet satisfied the sustained observation window. "
            "These are NOT recorded as final attendance."
        )

        provisional_records = [r for r in records if r["sustained_presence_status"] != "SUSTAINED_PRESENT"]

        if provisional_records:
            df_provisional = ReportGenerator.generate_attendance_dataframe(
                records=provisional_records,
                session_title=session_obj["session_title"],
                course_code=session_obj["course_code"],
            )
            st.dataframe(df_provisional, hide_index=True)

            csv_provisional = ReportGenerator.to_csv_bytes(df_provisional)
            st.download_button(
                label="📥 Download Provisional Audit CSV",
                data=csv_provisional,
                file_name=f"provisional_attendance_{session_obj['course_code']}_{session_obj['session_id'][:8]}.csv",
                mime="text/csv",
            )
        else:
            st.info("No provisional or transient observations recorded.")

    with tab_events:
        st.subheader("Timestamped Attendance Events Log")
        st.caption(
            "Audit log of discrete perception events: subject entry, sustained presence qualification, "
            "temporary departures, and instructor manual status overrides."
        )
        if events:
            event_rows = []
            for ev in events:
                rel_sec = float(ev.get("timestamp", 0.0))
                rel_min = int(rel_sec // 60)
                rel_rem = int(rel_sec % 60)
                event_rows.append({
                    "Timestamp": f"{rel_min:02d}:{rel_rem:02d} ({rel_sec:.1f}s)",
                    "Track ID": f"Track-{ev.get('track_id', '?')}",
                    "Event Type": ev.get("event_type", "UNKNOWN"),
                    "Description": ev.get("description", "-"),
                    "Logged At": ev.get("created_at", "-")[:19] if ev.get("created_at") else "-",
                })
            st.dataframe(event_rows, hide_index=True)

            df_events = pd.DataFrame(event_rows)
            csv_events = df_events.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Attendance Events Log CSV",
                data=csv_events,
                file_name=f"attendance_events_{session_obj['course_code']}_{session_obj['session_id'][:8]}.csv",
                mime="text/csv",
            )
        else:
            st.info("No attendance events logged for this session yet.")

    st.markdown("---")

    # Manual Attendance Corrections & Teacher Roster Claims
    st.subheader("✍️ Teacher Manual Corrections & Roster Binding")
    st.caption(
        "Ethical Constraint: Tracks are anonymous by default. Teachers can bind tracks to enrolled students "
        "or perform manual status overrides with audit logging."
    )

    c_corr1, c_corr2 = st.columns(2)

    with c_corr1:
        st.markdown("##### Link Track to Enrolled Student")
        track_choices = [
            f"Track-{r['track_id']} [{r['sustained_presence_status']}] (Seen: {round(r['total_detected_seconds']/60, 1)}m)"
            for r in records
        ]
        students = db.get_students(course_code=session_obj["course_code"])
        student_options: Dict[str, Optional[str]] = {
            f"{s['full_name']} ({s['student_id']})": s["student_id"] for s in students
        }
        student_options["[Remove Assignment / Anonymize]"] = None

        if track_choices and student_options:
            selected_track_str = st.selectbox("Select Track", options=track_choices)
            selected_student_str = st.selectbox("Assign Enrolled Student", options=list(student_options.keys()))

            if st.button("Confirm Student Assignment", type="secondary"):
                tid = int(selected_track_str.split()[0].replace("Track-", ""))
                stu_id = student_options[selected_student_str]
                engine = AttendanceEngine()
                ok = engine.claim_track_for_student(
                    db=db,
                    session_id=selected_session_id,
                    track_id=tid,
                    student_id=stu_id,
                    method="TEACHER_CONFIRMED" if stu_id else "ANONYMOUS",
                )
                if ok:
                    st.success(f"Track-{tid} assigned to {selected_student_str}.")
                    st.rerun()
                else:
                    st.error("Double-counting guardrail prevented duplicate student assignment.")
        else:
            st.info("No enrolled students found for course. Add students to the roster below.")

    with c_corr2:
        st.markdown("##### Manual Status Override")
        if records:
            override_track_str = st.selectbox(
                "Select Track to Override",
                options=[f"Track-{r['track_id']} (Current: {r['sustained_presence_status']})" for r in records],
                key="sb_override_track",
            )
            new_status_str = st.selectbox(
                "New Attendance Status",
                options=["SUSTAINED_PRESENT", "PROVISIONAL", "PARTIAL", "TRANSIENT", "ABSENT"],
                index=0,
            )
            override_reason = st.text_input("Override Reason / Note", value="Teacher manual verification")

            if st.button("Apply Status Override", type="secondary"):
                tid = int(override_track_str.split()[0].replace("Track-", ""))
                db.override_attendance_record(
                    session_id=selected_session_id,
                    track_id=tid,
                    status=new_status_str,
                    verification_method="TEACHER_OVERRIDE",
                    reason=override_reason,
                )
                st.success(f"Track-{tid} updated to {new_status_str}.")
                st.rerun()

    # Roster Management: Quick Add Student
    with st.expander("➕ Add Enrolled Student to Course Roster"):
        with st.form("add_student_form"):
            s_id = st.text_input("Student ID", placeholder="STU-001")
            s_name = st.text_input("Full Name", placeholder="e.g. Alice Smith")
            s_email = st.text_input("Email (Optional)", placeholder="alice@university.edu")
            add_sub = st.form_submit_button("Add to Course Roster")

            if add_sub and s_id and s_name:
                db.insert_student(
                    student_id=s_id,
                    course_code=session_obj["course_code"],
                    full_name=s_name,
                    email=s_email,
                )
                st.success(f"Added {s_name} ({s_id}) to {session_obj['course_code']} roster.")
                st.rerun()
