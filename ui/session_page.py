"""
Classroom Sessions Management and Live View Module.

Provides UI for session creation, hardware source selection (live webcams and recorded video files),
hardware toggles to disable camera or microphone processing, video metadata inspection,
configurable processing FPS, graceful error handling for missing devices, and real-time anonymous perception overlays.
"""

from pathlib import Path
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import streamlit as st

from config.settings import settings
from modules.attendance import AttendanceEngine
from modules.database import DatabaseManager
from modules.person_tracker import CentroidTracker
from modules.pipeline import ClassroomPipeline, PipelineStepResult
from modules.session_manager import SessionManager, SessionState
from modules.video_processor import VideoProcessor, VideoStatus


def init_session_state(db: DatabaseManager) -> None:
    """Initialize Streamlit session state objects if not already present."""
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager(db)
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "latest_step_result" not in st.session_state:
        st.session_state.latest_step_result = None
    if "video_processor" not in st.session_state:
        st.session_state.video_processor = None
    if "active_tracks" not in st.session_state:
        st.session_state.active_tracks = {}
    if "attendance_engine" not in st.session_state:
        st.session_state.attendance_engine = AttendanceEngine()
    if "video_error_msg" not in st.session_state:
        st.session_state.video_error_msg = None
    if "camera_enabled" not in st.session_state:
        st.session_state.camera_enabled = True
    if "mic_enabled" not in st.session_state:
        st.session_state.mic_enabled = True


def render_source_selection_form() -> Tuple[Union[int, str], int, bool, bool, float, str, str]:
    """
    Render video and audio source configuration controls and return parsed configuration.

    Returns:
        Tuple of (resolved_source, target_fps, camera_enabled, audio_enabled, threshold_pct, course_code, session_title)
    """
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        course_code = st.text_input("Course Code", value="CS101", placeholder="e.g. CS101")
        session_title = st.text_input("Session Title", value="Introduction to Algorithms")
    with col_meta2:
        source_mode = st.radio(
            "Video Input Source Mode",
            options=["Live Webcam", "Recorded Video File"],
            horizontal=True,
        )

    resolved_source: Union[int, str] = 0
    if source_mode == "Live Webcam":
        c_cam1, c_cam2 = st.columns([2, 1])
        with c_cam1:
            cam_idx = st.selectbox(
                "Select Camera Device Index",
                options=[0, 1, 2, 3],
                index=0,
                help="0 is typically the integrated camera; 1 or 2 correspond to external USB cameras.",
            )
            resolved_source = cam_idx
        with c_cam2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔍 Test Camera"):
                test_cap = cv2.VideoCapture(cam_idx)
                if test_cap.isOpened():
                    w = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    test_cap.release()
                    st.success(f"Camera {cam_idx} online ({w}x{h})")
                else:
                    st.error(f"Camera {cam_idx} unavailable or permission denied.")
    else:
        # Recorded video mode
        file_mode = st.radio(
            "Video File Method",
            options=["Sample Demo Video", "Enter File Path", "Upload Video File"],
            horizontal=True,
        )
        if file_mode == "Sample Demo Video":
            demo_video_path = "data/sample/classroom_demo.mp4"
            st.success("Selected pre-recorded sample video: `data/sample/classroom_demo.mp4` (10s offline video, 3 tracked participants, 10 FPS).")
            resolved_source = demo_video_path
        elif file_mode == "Enter File Path":
            video_path_str = st.text_input(
                "Absolute or Relative File Path",
                placeholder="data/sample/classroom_demo.mp4",
                help="Supported formats: .mp4, .avi, .mkv, .mov",
            )
            cleaned_path = video_path_str.strip()
            if cleaned_path:
                allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
                file_ext = Path(cleaned_path).suffix.lower()
                if file_ext and file_ext not in allowed_extensions:
                    st.warning(f"File extension '{file_ext}' may not be supported. Recommended: .mp4, .avi, .mov, .mkv")
            resolved_source = cleaned_path
        else:
            uploaded_file = st.file_uploader(
                "Upload Classroom Video",
                type=["mp4", "avi", "mov", "mkv"],
                help="Video is processed locally in temporary storage and never transmitted.",
            )
            if uploaded_file is not None:
                # Sanitize filename and prevent directory traversal
                raw_filename = Path(uploaded_file.name).name
                sanitized_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", raw_filename)
                allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
                file_ext = Path(sanitized_name).suffix.lower()
                if file_ext not in allowed_extensions:
                    st.error(f"Disallowed file extension '{file_ext}'. Allowed: {', '.join(allowed_extensions)}")
                    resolved_source = ""
                else:
                    temp_dir = Path("data/temp").resolve()
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file = (temp_dir / f"uploaded_{sanitized_name}").resolve()
                    # Verify path does not escape temp_dir
                    if not str(temp_file).startswith(str(temp_dir)):
                        st.error("Invalid file path: path traversal attempt detected.")
                        resolved_source = ""
                    else:
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.read())
                        resolved_source = str(temp_file)
                        st.info(f"File safely staged in local cache: {temp_file.name}")
            else:
                resolved_source = ""

    # Hardware Enable / Disable Toggles
    st.markdown("##### Sensor Modality Controls")
    c_tog1, c_tog2 = st.columns(2)
    with c_tog1:
        camera_enabled = st.checkbox(
            "Enable Camera Stream Processing (Visual Presence)",
            value=st.session_state.camera_enabled,
            help="Uncheck to operate in audio-only mode or disable camera hardware access.",
        )
        st.session_state.camera_enabled = camera_enabled
    with c_tog2:
        audio_enabled = st.checkbox(
            "Enable Microphone / Audio Stream Processing (Speech Ratio)",
            value=st.session_state.mic_enabled,
            help="Uncheck to operate in video-only mode or disable acoustic feature extraction.",
        )
        st.session_state.mic_enabled = audio_enabled

    # Processing Tuning
    c_tune1, c_tune2 = st.columns(2)
    with c_tune1:
        target_fps = st.slider(
            "Target Processing Frame Rate (FPS)",
            min_value=2,
            max_value=30,
            value=8,
            help="Subsampling incoming video reduces CPU usage while preserving accurate person tracking.",
        )
    with c_tune2:
        threshold_pct = st.slider(
            "Sustained Presence Threshold (%)",
            min_value=30.0,
            max_value=90.0,
            value=60.0,
            step=5.0,
            help="Percentage of active session duration required for verified attendance.",
        )

    return resolved_source, target_fps, camera_enabled, audio_enabled, threshold_pct, course_code, session_title


def render_session_page(db: DatabaseManager) -> None:
    """
    Render session management and live monitoring interface.

    Args:
        db: DatabaseManager instance.
    """
    init_session_state(db)
    sm: SessionManager = st.session_state.session_manager

    st.header("📹 Classroom Sessions")

    current = sm.current_session

    # Display any persistent video hardware/file errors
    if st.session_state.video_error_msg:
        st.error(f"Hardware/Stream Notice: {st.session_state.video_error_msg}")

    # Section 1: Session Setup when no session is active
    if current is None or current.status == SessionState.CONCLUDED:
        with st.expander("🧪 Demonstration Options: Pre-recorded Video & Synthetic Demo Data", expanded=False):
            st.markdown(
                """
                **Two Air-Gapped Demonstration Paths:**
                1. **Pre-recorded Video Pipeline:** Select *Recorded Video File* -> *Sample Demo Video* (`data/sample/classroom_demo.mp4`) below. Check the notice checkbox and click **Create Session** -> **▶️ Start Session** -> **⚡ Fast-Process Entire Video** to observe full end-to-end detection, tracking, and attendance calculation in seconds without webcams.
                2. **Instant Synthetic Demo Session:** Click below to immediately load a comprehensive 45-minute multi-modal classroom session (`[DEMO]`) to inspect rich analytics charts, trend detectors, and pedagogical decision support.
                """
            )
            if st.button("📥 Load Synthetic Sample Session [DEMO]", key="btn_load_demo_session_top"):
                from ui.demo_data import load_synthetic_demo_session
                demo_id = load_synthetic_demo_session(db)
                st.session_state["selected_session_id"] = demo_id
                st.success(f"Loaded synthetic demo session ({demo_id}). View Attendance, Analytics, or Recommendations!")
                st.rerun()

        st.subheader("Initialize New Classroom Session")

        resolved_source, target_fps, camera_enabled, audio_enabled, threshold_pct, course_code, session_title = (
            render_source_selection_form()
        )

        with st.expander("📜 Institutional Privacy, Consent & Data-Use Notice (Mandatory)", expanded=True):
            st.markdown(
                """
                **Classroom Observation & Activity Telemetry Protocol**
                
                - **Scope of Perception:** Visual presence is analyzed via anonymous spatial bounding boxes (`Track-01`, `Track-02`) and aggregate movement variance. Acoustic dynamics measure group speech activity ratios and volume variance.
                - **Prohibited Processing:** Zero automated facial recognition or biometric template matching is performed. Zero audio transcription or speaker voiceprint identification is executed.
                - **Data Storage:** All video frames and raw audio streams are ephemeral in volatile RAM and discarded immediately after feature extraction. No raw recordings are saved to disk. Verified attendance summaries and aggregate activity time-windows are stored locally in SQLite (`attendance_agent.db`) with zero external network egress.
                - **Participant Rights:** Track IDs remain completely anonymous unless an instructor manually assigns a student name through the verification roster. Participants may request manual attendance adjustments or complete record purging from the instructor.
                """
            )

        consent_agreed = st.checkbox(
            "I acknowledge the Institutional Privacy & Data-Use Protocol and confirm that all attendees have received observation notice.",
            value=False,
            help="Mandatory confirmation required prior to initializing session data processing.",
        )

        c_btn1, c_btn2 = st.columns([1, 4])
        with c_btn1:
            if st.button("Create Session", type="primary", width="stretch"):
                if not consent_agreed:
                    st.error("Classroom notice confirmation is required before proceeding.")
                elif camera_enabled and isinstance(resolved_source, str) and not resolved_source:
                    st.error("Please provide a valid video file path or upload a video.")
                else:
                    # Test opening the source immediately if camera is enabled
                    if camera_enabled:
                        vp = VideoProcessor(source=resolved_source, target_fps=target_fps)
                        if not vp.open_source():
                            st.session_state.video_error_msg = vp.error_message
                            st.error(vp.error_message)
                            return
                        else:
                            st.session_state.video_error_msg = None
                            vp.release()  # Will be reopened when user clicks Start

                    pipeline = ClassroomPipeline(
                        db=db,
                        course_code=course_code,
                        session_title=session_title,
                        video_source=resolved_source,
                        target_fps=target_fps,
                        camera_enabled=camera_enabled,
                        audio_enabled=audio_enabled,
                        sustained_threshold_pct=threshold_pct,
                        window_duration_sec=30,
                    )
                    sess = pipeline.create_session()
                    st.session_state.pipeline = pipeline
                    st.session_state.session_manager = pipeline.session_manager
                    st.session_state.video_processor = pipeline.video_processor
                    st.session_state.attendance_engine = pipeline.attendance_engine

                    st.success(f"Session '{sess.session_title}' initialized in READY state.")
                    st.rerun()

        # Existing Sessions Table
        st.markdown("---")
        st.subheader("Historical & Stored Sessions")
        recent = db.list_sessions(limit=20)
        if recent:
            table_rows = []
            for s in recent:
                dur_min = round(float(s.get("total_active_seconds", 0.0)) / 60.0, 1)
                table_rows.append({
                    "Session ID": s["session_id"][:12] + "...",
                    "Course": s["course_code"],
                    "Title": s["session_title"],
                    "Status": s["status"],
                    "Active (Min)": f"{dur_min}m",
                    "Video Source": s["video_source_type"],
                    "Created": s["created_at"][:16] if s.get("created_at") else "-",
                })
            st.dataframe(table_rows, hide_index=True)

            # Session Data Deletion Section (Requirement 6)
            with st.expander("🗑️ Delete Session Records (Permanent Deletion)"):
                st.caption(
                    "Permanently delete a session and all its associated attendance events, "
                    "activity windows, and recommendations from the local SQLite database."
                )
                del_options = {
                    f"{s['course_code']} - {s['session_title']} ({s['created_at'][:16]}) [{s['status']}]": s["session_id"]
                    for s in recent
                }
                if del_options:
                    target_del_label = st.selectbox("Select Session to Permanently Delete", options=list(del_options.keys()), key="sb_del_sess_page")
                    target_del_id = del_options[target_del_label]
                    confirm_del = st.checkbox(
                        "I understand this action is irreversible and permanently deletes all records for this session.",
                        key="chk_confirm_del_sess_page",
                    )
                    if st.button("Permanently Delete Session", type="primary", key="btn_confirm_del_sess_page"):
                        if not confirm_del:
                            st.error("Please confirm acknowledgment before deleting.")
                        else:
                            ok = db.delete_session(target_del_id)
                            if ok:
                                st.success(f"Session {target_del_id[:8]} permanently deleted from SQLite.")
                                if st.session_state.get("selected_session_id") == target_del_id:
                                    st.session_state["selected_session_id"] = None
                                st.rerun()
                            else:
                                st.error("Failed to delete session.")
                else:
                    st.caption("No historical sessions available to delete.")
        else:
            st.info("No classroom sessions found. Configure the form above to start your first session.")

    else:
        # Active or Paused Session View
        col_info, col_controls = st.columns([2, 1])
        vp: Optional[VideoProcessor] = st.session_state.video_processor

        with col_info:
            st.markdown(f"### {current.session_title} (`{current.course_code}`)")
            if "[DEMO]" in current.session_title or "SYNTHETIC" in current.session_title.upper():
                st.warning("🧪 **Notice:** This session contains **[DEMO / SYNTHETIC DATA]**. All data is synthetic and clearly isolated from live sensor telemetry.")
            elapsed_sec = sm.get_active_elapsed_seconds()
            elapsed_min = int(elapsed_sec // 60)
            elapsed_rem_sec = int(elapsed_sec % 60)

            m1, m2, m3 = st.columns(3)
            m1.metric(
                label="Elapsed Active Time",
                value=f"{elapsed_min:02d}:{elapsed_rem_sec:02d}",
            )
            m2.metric(
                label="Session State",
                value=current.status.value,
            )
            m3.metric(
                label="Presence Threshold",
                value=f"{current.sustained_threshold_pct}%",
            )

        with col_controls:
            st.markdown("#### Controls")
            c1, c2 = st.columns(2)

            if current.status == SessionState.READY:
                with c1:
                    if st.button("▶️ Start Session", type="primary", width="stretch"):
                        if st.session_state.pipeline:
                            st.session_state.pipeline.start()
                        else:
                            sm.start_session()
                            if vp:
                                opened = vp.open_source()
                                if not opened:
                                    st.session_state.video_error_msg = vp.error_message
                                    st.error(vp.error_message)
                                    sm.pause_session()
                        st.rerun()

            elif current.status == SessionState.ACTIVE:
                with c1:
                    if st.button("⏸️ Pause", width="stretch"):
                        if st.session_state.pipeline:
                            st.session_state.pipeline.pause()
                        else:
                            sm.pause_session()
                        st.rerun()
                with c2:
                    if st.button("⏹️ Conclude", type="secondary", width="stretch"):
                        if st.session_state.pipeline:
                            summary = st.session_state.pipeline.conclude()
                            st.session_state["selected_session_id"] = current.session_id
                            st.session_state.pipeline = None
                            st.session_state.video_processor = None
                            st.session_state.active_tracks = {}
                            st.session_state.latest_step_result = None
                            st.success(f"Session concluded. {summary.get('total_observed_tracks', 0)} participants evaluated.")
                        else:
                            total_sec = sm.conclude_session()
                            st.session_state["selected_session_id"] = current.session_id
                            if vp:
                                tracks = vp.tracker.get_all_session_tracks()
                                evals = st.session_state.attendance_engine.evaluate_session_presence(
                                    tracks=tracks,
                                    total_session_active_seconds=total_sec,
                                    threshold_pct=current.sustained_threshold_pct,
                                    session_id=current.session_id,
                                )
                                st.session_state.attendance_engine.sync_to_database(
                                    db=db,
                                    session_id=current.session_id,
                                    evaluations=evals,
                                )
                                vp.release()
                            st.session_state.video_processor = None
                            st.session_state.active_tracks = {}
                            st.session_state.latest_step_result = None
                            st.session_state.continuous_playback = False
                            st.success("Session concluded and attendance records synced.")
                        st.rerun()

                # Stepping and Continuous Playback Controls
                c_step1, c_step2 = st.columns(2)
                with c_step1:
                    if st.button("⏭️ Next Frame", width="stretch", help="Advance stream by one frame"):
                        st.rerun()
                with c_step2:
                    continuous = st.checkbox(
                        "🔁 Live Playback / Stream",
                        value=st.session_state.get("continuous_playback", False),
                        key="chk_continuous_playback",
                    )
                    st.session_state.continuous_playback = continuous

                # Fast Batch Processing for Recorded Video Files
                if current.video_source_type == "file" and st.session_state.pipeline:
                    if st.button("⚡ Fast-Process Entire Video", type="secondary", width="stretch"):
                        pipe = st.session_state.pipeline
                        vp_proc = pipe.video_processor
                        total_f = vp_proc.metadata.total_frames if (vp_proc and vp_proc.metadata and vp_proc.metadata.total_frames > 0) else 100
                        with st.spinner("Processing video frames..."):
                            prog_bar = st.progress(0, text="Processing video frames...")
                            f_count = 0
                            while True:
                                step_res = pipe.step(apply_pacing=False)
                                if not step_res.success:
                                    break
                                f_count += 1
                                if total_f > 0:
                                    prog_val = min(1.0, f_count / float(total_f))
                                    prog_bar.progress(prog_val, text=f"Processed {f_count}/{total_f} frames...")
                            summary = pipe.conclude()
                            st.session_state["selected_session_id"] = current.session_id
                            st.session_state.pipeline = None
                            st.session_state.video_processor = None
                            st.session_state.active_tracks = {}
                            st.session_state.latest_step_result = None
                            st.session_state.continuous_playback = False
                            st.success(f"Video concluded. {f_count} frames processed. {summary.get('total_observed_tracks', 0)} tracks evaluated.")
                            st.rerun()

            elif current.status == SessionState.PAUSED:
                with c1:
                    if st.button("▶️ Resume", type="primary", width="stretch"):
                        if st.session_state.pipeline:
                            st.session_state.pipeline.resume()
                        else:
                            sm.start_session()
                            if vp and not vp.is_opened():
                                vp.open_source()
                        st.rerun()
                with c2:
                    if st.button("⏹️ Conclude", type="secondary", width="stretch"):
                        if st.session_state.pipeline:
                            summary = st.session_state.pipeline.conclude()
                            st.session_state.pipeline = None
                            st.session_state.video_processor = None
                            st.session_state.active_tracks = {}
                            st.session_state.latest_step_result = None
                            st.success(f"Session concluded. {summary.get('total_observed_tracks', 0)} participants evaluated.")
                        else:
                            total_sec = sm.conclude_session()
                            if vp:
                                tracks = vp.tracker.get_all_session_tracks()
                                evals = st.session_state.attendance_engine.evaluate_session_presence(
                                    tracks=tracks,
                                    total_session_active_seconds=total_sec,
                                    threshold_pct=current.sustained_threshold_pct,
                                )
                                st.session_state.attendance_engine.sync_to_database(
                                    db=db,
                                    session_id=current.session_id,
                                    evaluations=evals,
                                )
                                vp.release()
                            st.session_state.video_processor = None
                            st.session_state.active_tracks = {}
                            st.session_state.latest_step_result = None
                            st.success("Session concluded.")
                        st.rerun()

        # Hardware & Stream Status Card
        cam_status_str = "Disabled by user" if not st.session_state.camera_enabled else ("Online" if vp and vp.is_opened() else "Idle / Paused")
        mic_status_str = "Online (VAD)" if current.audio_enabled else "Disabled by user"

        st.markdown(
            f"""
            <div style="background-color: #1a202c; padding: 10px 16px; border-radius: 6px; margin: 12px 0; font-size: 0.9rem; display: flex; justify-content: space-between; align-items: center;">
                <span>📷 <strong>Camera:</strong> {cam_status_str}</span>
                <span>🎙️ <strong>Microphone:</strong> {mic_status_str}</span>
                <span>📐 <strong>Resolution:</strong> {vp.metadata.resolution_str if vp else 'N/A'}</span>
                <span>⚡ <strong>Target FPS:</strong> {vp.target_fps if vp else 'N/A'}</span>
                <span>🚦 <strong>Stream Status:</strong> <code>{vp.status.value if vp else 'DISABLED'}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Section 2: Perception Preview & Live Metrics
        feed_col, metrics_col = st.columns([3, 2])

        with feed_col:
            st.markdown("#### Live Perception Stream")
            frame_placeholder = st.empty()

            if not st.session_state.camera_enabled:
                frame_placeholder.info("📷 **Camera processing is disabled.** Running in audio-only / low-resource mode.")
                if st.session_state.pipeline and current.status == SessionState.ACTIVE:
                    step_res = st.session_state.pipeline.step(apply_pacing=False)
                    st.session_state.latest_step_result = step_res
                    if step_res.completed_window:
                        st.toast(f"Window fused: Activity {step_res.composite_activity or 0.0:.1f}%")
                    if step_res.new_suggestions:
                        for sug in step_res.new_suggestions:
                            st.toast(f"💡 Suggestion: {sug.title}")
            elif current.status == SessionState.ACTIVE:
                if st.session_state.pipeline:
                    step_res = st.session_state.pipeline.step(apply_pacing=False)
                    st.session_state.latest_step_result = step_res
                    if step_res.success and step_res.annotated_frame is not None:
                        st.session_state.active_tracks = step_res.active_tracks
                        st.session_state.current_motion = step_res.motion_index
                        rgb_frame = cv2.cvtColor(step_res.annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", width="stretch", alt="Live Perception Stream")
                        if step_res.completed_window:
                            st.toast(f"Window fused: Activity {step_res.composite_activity or 0.0:.1f}%")
                        if step_res.new_suggestions:
                            for sug in step_res.new_suggestions:
                                st.toast(f"💡 Suggestion: {sug.title}")
                        if st.session_state.get("continuous_playback", False):
                            pipe_vp = st.session_state.pipeline.video_processor
                            t_fps = pipe_vp.target_fps if pipe_vp else (vp.target_fps if vp else 8)
                            time.sleep(1.0 / max(1, t_fps))
                            st.rerun()
                    else:
                        st.session_state.continuous_playback = False
                        pipe_vp = st.session_state.pipeline.video_processor
                        if pipe_vp and pipe_vp.status == VideoStatus.COMPLETED:
                            frame_placeholder.success("🎉 Video playback completed (reached end of file).")
                        elif pipe_vp and pipe_vp.status == VideoStatus.ERROR:
                            frame_placeholder.error(f"Stream Error: {pipe_vp.error_message}")
                        else:
                            frame_placeholder.info("Awaiting video stream frames...")
                elif vp:
                    success, frame = vp.get_next_frame(apply_pacing=False)
                    if success and frame is not None:
                        annotated_frame, active_tracks, motion_index = vp.process_frame(frame)
                        st.session_state.active_tracks = active_tracks
                        st.session_state.current_motion = motion_index
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(rgb_frame, channels="RGB", width="stretch", alt="Live Perception Stream")
                        if st.session_state.get("continuous_playback", False):
                            time.sleep(1.0 / max(1, vp.target_fps))
                            st.rerun()
                    else:
                        st.session_state.continuous_playback = False
                        if vp.status == VideoStatus.COMPLETED:
                            frame_placeholder.success("🎉 Video playback completed (reached end of file).")
                        elif vp.status == VideoStatus.ERROR:
                            frame_placeholder.error(f"Stream Error: {vp.error_message}")
                        else:
                            frame_placeholder.info("Awaiting video stream frames...")
            else:
                frame_placeholder.info("Stream is currently idle or paused. Click 'Start Session' or 'Resume' to begin.")

        with metrics_col:
            st.markdown("#### Live Telemetry")
            active_count = len(st.session_state.active_tracks) if st.session_state.camera_enabled else None
            motion_val = getattr(st.session_state, "current_motion", None) if st.session_state.camera_enabled else None
            latest_res = getattr(st.session_state, "latest_step_result", None)

            m1, m2 = st.columns(2)
            if active_count is not None:
                m1.metric("Observed Headcount", f"{active_count} persons")
            else:
                m1.metric("Observed Headcount", "Unavailable", delta="Camera Disabled")

            if motion_val is not None:
                m2.metric("Motion Intensity", f"{motion_val:.1f}%")
            else:
                m2.metric("Motion Intensity", "Unavailable", delta="Camera Disabled")

            m3, m4 = st.columns(2)
            if current.audio_enabled:
                speech_ratio = latest_res.audio_features.speech_ratio if (latest_res and latest_res.audio_features) else None
                if speech_ratio is not None:
                    m3.metric("Speech Ratio", f"{speech_ratio * 100:.1f}%")
                else:
                    m3.metric("Speech Ratio", "Awaiting audio")
            else:
                m3.metric("Speech Ratio", "Unavailable", delta="Mic Disabled")

            comp_act = latest_res.composite_activity if latest_res else None
            if comp_act is not None:
                m4.metric("Activity Index", f"{comp_act:.1f}%")
            else:
                m4.metric("Activity Index", "Pending Window")

            st.markdown("##### Real-Time Anonymous Tracks")
            if st.session_state.active_tracks and st.session_state.camera_enabled:
                track_data = []
                for tid, tr in st.session_state.active_tracks.items():
                    track_data.append({
                        "Track ID": f"Track-{tid}",
                        "Active Duration": f"{tr.total_active_seconds:.0f}s",
                        "Frames Observed": tr.total_frames_detected,
                    })
                st.dataframe(track_data, hide_index=True)
            else:
                if not st.session_state.camera_enabled:
                    st.caption("Visual tracking is disabled.")
                else:
                    st.caption("No human subjects currently detected in view.")

            # Live Teaching Suggestions Box
            st.markdown("##### Pedagogical Assistant Cues")
            suggestions = db.get_teaching_suggestions(current.session_id)
            if suggestions:
                for s in suggestions[:3]:
                    st.warning(f"💡 **{s['trigger_rule_code']}**: {s['suggestion_text']}")
            else:
                st.info("No active pedagogical alerts. Pacing and dynamics within standard ranges.")
