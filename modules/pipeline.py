"""
Classroom Session Processing Pipeline.

Orchestrates the complete 11-stage classroom intelligence lifecycle:
1. Create Session
2. Open Video / Optional Audio
3. Extract Frames and Audio Windows
4. Detect and Track People
5. Calculate Attendance Observations
6. Extract Visual and Audio Features
7. Aggregate Features by Time Window
8. Calculate Activity Indicators
9. Generate Teaching Suggestions
10. Save Results in SQLite
11. Update Dashboard and Reports
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from config.settings import settings
from modules.attendance import AttendanceEngine, AttendanceEvaluation, AttendanceEvent
from modules.audio_processor import AudioFeatures, AudioProcessor
from modules.database import DatabaseManager
from modules.engagement_analytics import EngagementAnalytics
from modules.feature_fusion import FeatureFusionEngine, MultimodalWindow, VisualFeatures
from modules.person_tracker import Track
from modules.recommendation_engine import RecommendationEngine, Suggestion
from modules.report_generator import ReportGenerator
from modules.session_manager import SessionManager, SessionMetadata, SessionState
from modules.video_processor import VideoProcessor, VideoStatus

logger = logging.getLogger(__name__)


@dataclass
class PipelineStepResult:
    """Output metrics and state produced by a single discrete step of the pipeline."""
    success: bool
    frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    active_tracks: Dict[int, Track] = field(default_factory=dict)
    motion_index: float = 0.0
    headcount: int = 0
    audio_features: Optional[AudioFeatures] = None
    completed_window: Optional[MultimodalWindow] = None
    new_suggestions: List[Suggestion] = field(default_factory=list)
    new_attendance_events: List[AttendanceEvent] = field(default_factory=list)
    composite_activity: Optional[float] = None
    elapsed_active_seconds: float = 0.0
    session_status: SessionState = SessionState.READY
    error_message: Optional[str] = None
    message: str = ""
    window_progress_str: str = ""


class ClassroomPipeline:
    """
    End-to-End Orchestrator for multimodal classroom observation, attendance ledgering,
    activity dynamics extraction, and pedagogical decision support.
    """

    def __init__(
        self,
        db: DatabaseManager,
        course_code: str,
        session_title: str,
        video_source: Union[int, str] = 0,
        target_fps: int = 8,
        camera_enabled: bool = True,
        audio_enabled: bool = True,
        sustained_threshold_pct: float = 60.0,
        window_duration_sec: int = 30,
        custom_detector: Optional[Callable[[np.ndarray], List[Tuple[int, int, int, int]]]] = None,
        audio_device_index: Optional[int] = None,
    ) -> None:
        """
        Initialize the end-to-end classroom pipeline.

        Args:
            db: Initialized DatabaseManager instance.
            course_code: Course identifier (e.g. 'CS101').
            session_title: Descriptive session topic.
            video_source: Camera index (int) or video file path (str).
            target_fps: Processing subsample frame rate.
            camera_enabled: Toggle visual tracking.
            audio_enabled: Toggle acoustic feature extraction.
            sustained_threshold_pct: Percentage of active session required for sustained presence.
            window_duration_sec: Time-window aggregation duration (default: 30s).
            custom_detector: Optional mock/custom person detector for testing or specialized cameras.
            audio_device_index: Optional microphone hardware device index.
        """
        self.db = db
        self.course_code = course_code
        self.session_title = session_title
        self.video_source = video_source
        self.target_fps = target_fps
        self.camera_enabled = camera_enabled
        self.audio_enabled = audio_enabled
        self.sustained_threshold_pct = sustained_threshold_pct
        self.window_duration_sec = window_duration_sec
        self.custom_detector = custom_detector
        self.audio_device_index = audio_device_index

        # 1. Session Manager
        self.session_manager = SessionManager(db_manager=self.db)
        self.session_metadata: Optional[SessionMetadata] = None

        # 2. Perception Modules
        self.video_processor: Optional[VideoProcessor] = None
        if self.camera_enabled:
            self.video_processor = VideoProcessor(
                source=self.video_source,
                target_fps=self.target_fps,
                custom_detector=self.custom_detector,
            )

        self.audio_processor: Optional[AudioProcessor] = None
        if self.audio_enabled:
            self.audio_processor = AudioProcessor()

        # 3. Attendance Engine
        self.attendance_engine = AttendanceEngine()

        # 4. Multimodal Feature Fusion Engine
        self.fusion_engine = FeatureFusionEngine(
            window_duration_sec=self.window_duration_sec,
            db_manager=self.db,
            expected_video_fps=float(self.target_fps),
        )

        # 5. Recommendation Engine
        self.recommendation_engine = RecommendationEngine(db_manager=self.db)

        # Pipeline runtime state
        self.windows_history: List[MultimodalWindow] = []
        self._last_step_time: Optional[float] = None
        self._mic_started: bool = False
        self.latest_audio_features: Optional[AudioFeatures] = None
        self.latest_composite_activity: Optional[float] = None

    @property
    def session_id(self) -> Optional[str]:
        """Return the active session ID if initialized."""
        return self.session_metadata.session_id if self.session_metadata else None

    @property
    def mic_started(self) -> bool:
        """Public indicator of whether microphone streaming is active."""
        return self._mic_started

    # =========================================================================
    # Step 1: Create Session
    # =========================================================================

    def create_session(self) -> SessionMetadata:
        """
        Create a new session record in SQLite in READY status.

        Returns:
            Configured SessionMetadata.
        """
        source_type = "file" if isinstance(self.video_source, str) else "webcam"
        if not self.camera_enabled:
            source_type = "audio_only"

        self.session_metadata = self.session_manager.create_session(
            course_code=self.course_code,
            session_title=self.session_title,
            video_source_type=source_type,
            audio_enabled=self.audio_enabled,
            sustained_threshold_pct=self.sustained_threshold_pct,
        )
        return self.session_metadata

    # =========================================================================
    # Step 2: Open Video / Optional Audio
    # =========================================================================

    def open_sources(self) -> bool:
        """
        Initialize and open physical or recorded hardware inputs.

        Returns:
            True if all required enabled sources were opened successfully.
        """
        if self.session_metadata is None:
            self.create_session()

        # Open video source if camera is enabled
        if self.camera_enabled and self.video_processor:
            opened = self.video_processor.open_source()
            if not opened:
                logger.error("Failed to open video source: %s", self.video_processor.error_message)
                return False

        # Open microphone audio stream if enabled
        if self.audio_enabled and self.audio_processor:
            try:
                self._mic_started = self.audio_processor.start_microphone_stream(
                    device_index=self.audio_device_index
                )
            except Exception as e:
                logger.warning("Microphone stream unavailable (%s); proceeding with audio disabled", e)
                self._mic_started = False

        return True

    # =========================================================================
    # Lifecycle Control
    # =========================================================================

    def start(self) -> None:
        """Start or transition the session into ACTIVE state."""
        if not self.session_metadata:
            self.create_session()
        self.open_sources()
        self.session_manager.start_session()
        self.fusion_engine.start_window(time.time())

    def pause(self) -> None:
        """Pause active session."""
        self.session_manager.pause_session()

    def reconnect_sources(self) -> bool:
        """Force restart and reconnect of both video capture and audio stream."""
        if self.video_processor:
            self.video_processor.release()
            self.video_processor.open_source()
        if self.audio_enabled and self.audio_processor:
            self.audio_processor.stop_mic_stream()
            try:
                self._mic_started = self.audio_processor.start_microphone_stream(
                    device_index=self.audio_device_index
                )
            except Exception as e:
                logger.warning("Microphone stream unavailable on reconnect: %s", e)
                self._mic_started = False
        return True

    def resume(self) -> None:
        """Resume paused session."""
        self.session_manager.start_session()
        if self.camera_enabled and self.video_processor and not self.video_processor.is_opened():
            self.video_processor.open_source()
        if self.audio_enabled and self.audio_processor and not self._mic_started:
            try:
                self._mic_started = self.audio_processor.start_microphone_stream(
                    device_index=self.audio_device_index
                )
            except Exception:
                pass

    # =========================================================================
    # Steps 3 through 10: Discrete Pipeline Execution Step
    # =========================================================================

    def step(self, apply_pacing: bool = False) -> PipelineStepResult:
        """
        Execute one discrete iteration of the 11-stage pipeline.

        1. Extract frame and audio sample.
        2. Detect and track people.
        3. Calculate attendance observations.
        4. Extract visual & audio features.
        5. Aggregate features by time window.
        6. Calculate activity indicators.
        7. Generate teaching suggestions.
        8. Save results in SQLite.

        Returns:
            PipelineStepResult with live outputs.
        """
        if not self.session_metadata:
            return PipelineStepResult(success=False, error_message="Session not initialized.")

        now = time.time()
        elapsed_sec = self.session_manager.get_active_elapsed_seconds()
        status = self.session_metadata.status

        if status != SessionState.ACTIVE:
            return PipelineStepResult(
                success=True,
                elapsed_active_seconds=elapsed_sec,
                session_status=status,
                message=f"Pipeline is idle (session is {status.value}).",
            )

        # ---------------------------------------------------------------------
        # 3. Extract Frames and Audio Windows
        # ---------------------------------------------------------------------
        raw_frame: Optional[np.ndarray] = None
        annotated_frame: Optional[np.ndarray] = None
        active_tracks: Dict[int, Track] = {}
        motion_idx: float = 0.0

        if self.camera_enabled and self.video_processor:
            success, raw_frame = self.video_processor.get_next_frame(apply_pacing=apply_pacing)
            if not success or raw_frame is None:
                if self.video_processor.status == VideoStatus.COMPLETED:
                    return PipelineStepResult(
                        success=False,
                        elapsed_active_seconds=elapsed_sec,
                        session_status=status,
                        error_message="Video playback completed (end of file).",
                    )
                return PipelineStepResult(
                    success=False,
                    elapsed_active_seconds=elapsed_sec,
                    session_status=status,
                    error_message=self.video_processor.error_message or "Unable to retrieve video frame.",
                )

            # -----------------------------------------------------------------
            # 4. Detect and Track People
            # -----------------------------------------------------------------
            annotated_frame, active_tracks, motion_idx = self.video_processor.process_frame(
                raw_frame, timestamp=now
            )

        # Extract audio chunk if active
        new_audio_feature: Optional[AudioFeatures] = None
        if self.audio_enabled and self.audio_processor:
            if not self._mic_started:
                try:
                    self._mic_started = self.audio_processor.start_microphone_stream(
                        device_index=self.audio_device_index
                    )
                except Exception as e:
                    logger.debug("Auto-reconnection of mic stream failed: %s", e)
                    self._mic_started = False

            if self._mic_started:
                try:
                    while True:
                        chunk = self.audio_processor.read_stream_chunk()
                        if chunk is None or len(chunk) == 0:
                            break
                        feat = self.audio_processor.process_audio_chunk(chunk, timestamp=now)
                        self.latest_audio_features = feat
                        new_audio_feature = feat
                except Exception as e:
                    logger.debug("Audio chunk read failed: %s", e)
                    self.latest_audio_features = None
                    new_audio_feature = None

        # ---------------------------------------------------------------------
        # 5. Calculate Attendance Observations
        # ---------------------------------------------------------------------
        new_events: List[AttendanceEvent] = []
        if self.camera_enabled and active_tracks:
            new_events = self.attendance_engine.record_frame_observations(
                session_id=self.session_id,  # type: ignore
                timestamp=now,
                active_tracks=active_tracks,
            )
            # Persist fresh attendance events immediately
            for evt in new_events:
                self.db.insert_attendance_event(
                    event_id=evt.event_id,
                    session_id=self.session_id,  # type: ignore
                    track_id=evt.track_id,
                    timestamp=evt.timestamp,
                    event_type=evt.event_type.value,
                    description=evt.description,
                )

        # ---------------------------------------------------------------------
        # 6. Extract Visual and Audio Features & 7. Aggregate by Time Window
        # ---------------------------------------------------------------------
        visible_tracks = {
            tid: tr for tid, tr in active_tracks.items()
            if tr.is_visible or tr.consecutive_missing_frames <= 3
        }
        headcount = len(visible_tracks) if self.camera_enabled else 0
        head_pose_var = 0.0
        if visible_tracks:
            poses = [t.head_orientation.yaw for t in visible_tracks.values() if t.head_orientation]
            if len(poses) >= 2:
                head_pose_var = float(np.var(poses))

        self.fusion_engine.record_sample(
            active_persons=headcount,
            movement_index=motion_idx,
            audio_features=new_audio_feature,
            timestamp=now,
            head_pose_variance=head_pose_var,
        )

        completed_window: Optional[MultimodalWindow] = None
        new_suggestions: List[Suggestion] = []
        composite_score: Optional[float] = None

        if self.fusion_engine.is_window_ready(now):
            # Window completed: fuse and save to SQLite
            completed_window = self.fusion_engine.close_and_fuse_window(
                session_id=self.session_id,  # type: ignore
                current_timestamp=now,
            )

            if completed_window:
                self.windows_history.append(completed_window)

                # -------------------------------------------------------------
                # 8. Calculate Activity Indicators
                # -------------------------------------------------------------
                composite_score = EngagementAnalytics.calculate_composite_activity(
                    visual_movement=completed_window.visual_movement_index,
                    speech_pct=completed_window.speech_activity_pct,
                    head_pose_variance=completed_window.head_pose_variance,
                )
                if composite_score is not None:
                    self.latest_composite_activity = composite_score

                # -------------------------------------------------------------
                # 9. Generate Teaching Suggestions & 10. Save in SQLite
                # -------------------------------------------------------------
                new_suggestions = self.recommendation_engine.evaluate_rules(
                    session_id=self.session_id,  # type: ignore
                    recent_windows=self.windows_history,
                    current_time=now,
                )

        # Window accumulation progress
        w_start = self.fusion_engine.current_window_start or now
        w_dur = self.fusion_engine.window_duration_sec
        w_elapsed = max(0.0, now - w_start)
        progress_str = f"Accumulating ({w_elapsed:.0f}s/{w_dur}s)"

        # Persistent metrics to avoid sub-frame flickering on dashboard
        effective_audio = new_audio_feature or self.latest_audio_features
        effective_composite = self.latest_composite_activity
        if effective_composite is None:
            effective_composite = self.fusion_engine.get_current_instantaneous_activity()
        if effective_composite is None:
            effective_composite = 0.0

        return PipelineStepResult(
            success=True,
            frame=raw_frame,
            annotated_frame=annotated_frame,
            active_tracks=active_tracks,
            motion_index=motion_idx,
            headcount=headcount,
            audio_features=effective_audio,
            completed_window=completed_window,
            new_suggestions=new_suggestions,
            new_attendance_events=new_events,
            composite_activity=effective_composite,
            elapsed_active_seconds=elapsed_sec,
            session_status=status,
            window_progress_str=progress_str,
        )

    # =========================================================================
    # Step 11: Conclude Session, Save Results & Update Reports
    # =========================================================================

    def conclude(self) -> Dict[str, Any]:
        """
        Conclude the session:
        - Freezes active duration.
        - Flushes any open aggregation window.
        - Calculates verified sustained attendance evaluations.
        - Persists all final attendance records to SQLite.
        - Generates attendance CSV export.
        - Releases camera and audio resources.

        Returns:
            Dictionary summary with total duration, attendance counts, and export paths.
        """
        total_sec = self.session_manager.conclude_session()

        # Flush final partial window if exists
        now = time.time()
        final_window = self.fusion_engine.close_and_fuse_window(
            session_id=self.session_id,  # type: ignore
            current_timestamp=now,
        )
        if final_window:
            self.windows_history.append(final_window)

        # Evaluate sustained presence for all tracks observed throughout the session
        all_tracks = {}
        if self.video_processor:
            all_tracks = self.video_processor.tracker.get_all_session_tracks()

        evaluations: List[AttendanceEvaluation] = self.attendance_engine.evaluate_session_presence(
            tracks=all_tracks,
            total_session_active_seconds=total_sec,
            threshold_pct=self.sustained_threshold_pct,
        )

        # Persist final evaluations to SQLite
        self.attendance_engine.sync_to_database(
            db=self.db,
            session_id=self.session_id,  # type: ignore
            evaluations=evaluations,
        )

        # Generate attendance CSV report
        csv_path: Optional[Path] = None
        try:
            records = self.db.get_session_attendance(self.session_id)  # type: ignore
            df = ReportGenerator.generate_attendance_dataframe(
                records=records,
                session_title=self.session_title,
                course_code=self.course_code,
            )
            csv_path = ReportGenerator.export_attendance_csv(
                df=df,
                session_id=self.session_id,
            )
        except Exception as e:
            logger.error("Failed to generate final attendance CSV: %s", e)

        # Release resources
        if self.video_processor:
            self.video_processor.release()

        if self.audio_processor and self._mic_started:
            try:
                self.audio_processor.stop_microphone_stream()
            except Exception:
                pass
            self._mic_started = False

        verified_count = sum(1 for e in evaluations if e.status.value == "SUSTAINED_PRESENT")
        provisional_count = len(evaluations) - verified_count

        return {
            "session_id": self.session_id,
            "course_code": self.course_code,
            "session_title": self.session_title,
            "total_active_seconds": total_sec,
            "total_observed_tracks": len(evaluations),
            "verified_attendance_count": verified_count,
            "provisional_count": provisional_count,
            "activity_windows_count": len(self.windows_history),
            "csv_export_path": str(csv_path) if csv_path else None,
        }

    # =========================================================================
    # Headless Batch Processing Mode (for Files & Automated Testing)
    # =========================================================================

    def run_batch(
        self,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute the entire pipeline in an automated, headless batch loop.
        Processes incoming video frames until completion, max_frames, or max_seconds.

        Returns:
            Concluded session summary dictionary.
        """
        self.start()
        start_time = time.time()
        frame_idx = 0

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                break

            result = self.step(apply_pacing=False)
            if not result.success:
                # Video file reached EOF or error
                break

            frame_idx += 1

        return self.conclude()
