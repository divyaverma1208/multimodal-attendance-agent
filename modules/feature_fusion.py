"""
Feature Fusion Module.

Synchronizes visual metrics (active headcount, movement intensity, head pose proxy)
and acoustic features (RMS energy, speech activity ratio) into standardized time-window buckets.

EPISTEMIC & ETHICAL GUARDRAILS:
- Fuses observable physical metrics (pixel movement, sound pressure levels, voice activity ratios).
- MUST NEVER be labeled as, or presumed to measure, ground-truth cognitive attention,
  psychological engagement, emotional valence, or student comprehension.
- Distinguishes genuine zero measurements (e.g., quiet room with active microphone) from
  unavailable measurements (missing/failed sensor input).
- Prevents treating collinear acoustic features (RMS and Speech Ratio) or collinear visual
  features (Movement Index and Pose Variance) as independent additive evidence.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np

from config.settings import settings
from modules.audio_processor import AudioFeatures
from modules.database import DatabaseManager

logger = logging.getLogger(__name__)


class DataQualityLabel(str, Enum):
    """Categorical label indicating multimodal observation completeness."""
    HIGH = "HIGH"
    DEGRADED_SAMPLE_RATE = "DEGRADED_SAMPLE_RATE"
    PARTIAL_VISUAL_ONLY = "PARTIAL_VISUAL_ONLY"
    PARTIAL_AUDIO_ONLY = "PARTIAL_AUDIO_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class VisualFeatures:
    """Timestamped visual activity features extracted from camera or video frames."""
    timestamp: float
    active_person_count: int
    visual_movement_index: float
    head_pose_variance: float = 0.0
    detection_confidence: float = 1.0


@dataclass
class MultimodalWindow:
    """
    Aggregated multimodal metrics across a single time window.

    Fields 1-9 maintain strict positional compatibility with existing database
    and downstream analytics signatures.
    """
    window_id: str
    session_id: str
    window_start_sec: float
    window_end_sec: float
    active_person_count: Optional[int]
    visual_movement_index: Optional[float]
    head_pose_variance: Optional[float]
    speech_activity_pct: Optional[float]
    rms_energy_mean: Optional[float]

    # Data Quality, Availability, & Alignment Metadata
    visual_available: bool = True
    audio_available: bool = True
    visual_sample_count: int = 0
    audio_sample_count: int = 0
    data_quality_score: float = 1.0  # Normalized 0.0 to 1.0
    data_quality_label: str = DataQualityLabel.HIGH.value
    is_genuine_silence: bool = False  # True ONLY when audio was actively recorded and measured 0.0% speech
    correlation_adjusted_activity_score: float = 0.0  # Decoupled non-redundant activity proxy (0.0 to 100.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation suitable for JSON export and Plotly visual rendering."""
        return {
            "window_id": self.window_id,
            "session_id": self.session_id,
            "start_sec": round(self.window_start_sec, 2),
            "end_sec": round(self.window_end_sec, 2),
            "duration_sec": round(self.window_end_sec - self.window_start_sec, 2),
            "headcount": self.active_person_count,
            "visual_movement": self.visual_movement_index,
            "head_pose_variance": self.head_pose_variance,
            "speech_pct": self.speech_activity_pct,
            "rms_energy": self.rms_energy_mean,
            "visual_available": self.visual_available,
            "audio_available": self.audio_available,
            "visual_sample_count": self.visual_sample_count,
            "audio_sample_count": self.audio_sample_count,
            "is_genuine_silence": self.is_genuine_silence,
            "data_quality_score": self.data_quality_score,
            "data_quality_label": self.data_quality_label,
            "composite_activity_score": self.correlation_adjusted_activity_score,
        }


class FeatureFusionEngine:
    """
    Aggregates sub-second visual and acoustic measurements into time-window intervals.
    Aligns asynchronous observation streams, diagnoses sensor degradation, and produces
    orthogonal, correlation-adjusted activity indices.
    """

    def __init__(
        self,
        window_duration_sec: Optional[int] = None,
        db_manager: Optional[DatabaseManager] = None,
        expected_video_fps: float = 10.0,
        expected_audio_hz: float = 2.0,
        window_duration_seconds: Optional[int] = None,
        db: Optional[DatabaseManager] = None,
    ) -> None:
        """
        Initialize FeatureFusionEngine.

        Args:
            window_duration_sec: Window length in seconds (default: settings.analytics.window_duration_seconds).
            db_manager: Optional DatabaseManager to persist completed windows.
            expected_video_fps: Target visual frame rate for data-quality density estimation.
            expected_audio_hz: Target audio chunk frequency for data-quality density estimation.
            window_duration_seconds: Alias for window_duration_sec.
            db: Alias for db_manager.
        """
        raw_dur = window_duration_sec or window_duration_seconds or settings.analytics.window_duration_seconds
        self.window_duration_sec = raw_dur
        self.db = db_manager or db
        self.expected_video_fps = expected_video_fps
        self.expected_audio_hz = expected_audio_hz

        # Buffers for current streaming window
        self.current_window_start: Optional[float] = None
        self._visual_buffer: List[VisualFeatures] = []
        self._audio_buffer: List[Tuple[float, AudioFeatures]] = []
        self.window_counter: int = 0

    def start_window(self, start_timestamp: float) -> None:
        """Begin a new aggregation window at the given timestamp."""
        self.current_window_start = start_timestamp
        self._visual_buffer.clear()
        self._audio_buffer.clear()

    def record_visual_sample(self, visual_features: VisualFeatures) -> None:
        """Ingest a standalone timestamped visual observation."""
        if self.current_window_start is None:
            self.start_window(visual_features.timestamp)
        self._visual_buffer.append(visual_features)

    def record_audio_sample(
        self,
        audio_features: AudioFeatures,
        timestamp: Optional[float] = None,
    ) -> None:
        """Ingest a standalone timestamped acoustic observation."""
        ts = timestamp if timestamp is not None else audio_features.timestamp
        if self.current_window_start is None:
            self.start_window(ts)
        self._audio_buffer.append((ts, audio_features))

    def ingest_visual_observation(
        self,
        timestamp: float,
        active_people_count: int = 0,
        motion_energy: float = 0.0,
        head_pose_variance: float = 0.0,
    ) -> None:
        """Alias for visual sample ingestion."""
        self.record_sample(
            active_persons=active_people_count,
            movement_index=motion_energy,
            timestamp=timestamp,
            head_pose_variance=head_pose_variance,
        )

    def ingest_audio_observation(
        self,
        timestamp: float,
        audio_features: AudioFeatures,
    ) -> None:
        """Alias for acoustic sample ingestion."""
        self.record_audio_sample(audio_features=audio_features, timestamp=timestamp)

    def record_sample(
        self,
        active_persons: int,
        movement_index: float,
        audio_features: Optional[AudioFeatures] = None,
        timestamp: Optional[float] = None,
        head_pose_variance: float = 0.0,
    ) -> None:
        """
        Convenience method to ingest a synchronous or paired multimodal observation.

        Args:
            active_persons: Current count of detected persons.
            movement_index: Current visual motion score.
            audio_features: Audio features, or None if audio is unavailable/disabled.
            timestamp: Observation timestamp. Defaults to current time.
            head_pose_variance: Angular variance or proxy of head pose.
        """
        now = timestamp if timestamp is not None else time.time()
        if self.current_window_start is None:
            self.start_window(now)

        self._visual_buffer.append(
            VisualFeatures(
                timestamp=now,
                active_person_count=active_persons,
                visual_movement_index=movement_index,
                head_pose_variance=head_pose_variance,
            )
        )

        if audio_features is not None:
            self._audio_buffer.append((now, audio_features))

    def get_current_instantaneous_activity(self) -> Optional[float]:
        """
        Compute real-time estimated activity score from samples currently accumulated
        in the open window, avoiding empty or pending display states during accumulation.
        """
        if not self._visual_buffer and not self._audio_buffer:
            return None

        # Mean visual movement
        avg_motion: Optional[float] = None
        avg_head_var: Optional[float] = None
        if self._visual_buffer:
            avg_motion = float(np.mean([v.visual_movement_index for v in self._visual_buffer]))
            avg_head_var = float(np.mean([v.head_pose_variance for v in self._visual_buffer]))

        # Mean acoustic features
        avg_speech: Optional[float] = None
        avg_rms: Optional[float] = None
        if self._audio_buffer:
            speech_bools = [100.0 if a.is_speech else 0.0 for _, a in self._audio_buffer]
            avg_speech = float(np.mean(speech_bools))
            avg_rms = float(np.mean([a.rms_energy for _, a in self._audio_buffer]))

        return self.compute_composite_activity_score(
            visual_movement=avg_motion,
            head_pose_variance=avg_head_var,
            speech_activity_pct=avg_speech,
            rms_energy=avg_rms,
        )

    def is_window_ready(self, current_timestamp: float) -> bool:
        """Check if elapsed time exceeds the configured window duration."""
        if self.current_window_start is None:
            return False
        return (current_timestamp - self.current_window_start) >= self.window_duration_sec

    @staticmethod
    def compute_composite_activity_score(
        visual_movement: Optional[float],
        head_pose_variance: Optional[float],
        speech_activity_pct: Optional[float],
        rms_energy: Optional[float],
    ) -> float:
        """
        Compute correlation-adjusted non-redundant activity score (0.0 to 100.0).

        AVOIDING MULTI-COLLINEARITY ARTIFACTS:
        1. Acoustic Collinearity: RMS energy and speech activity are collinear representations
           of acoustic amplitude. We use speech ratio as the primary indicator, modulating it by
           energy confidence, rather than adding them as independent additive evidence.
        2. Visual Collinearity: Head pose variance is collinear with spatial centroid movement.
           We scale visual movement and use head variance only as a bounded damping factor.
        3. Cross-Modal Redundancy: Simultaneous talking and gesturing is a single communicative
           act. We combine visual and acoustic factors via bounded probabilistic saturation:
           S = 1.0 - (1.0 - S_acoustic) * (1.0 - S_visual)
           rather than unbounded linear summation (which artificially doubles during joint speech/motion).
        """
        # 1. Acoustic component
        if speech_activity_pct is not None and rms_energy is not None:
            norm_speech = min(1.0, max(0.0, speech_activity_pct / 100.0))
            norm_rms = min(1.0, max(0.0, rms_energy / 0.10))
            # RMS modulates speech confidence rather than acting as independent additive evidence
            s_acoustic = norm_speech * (0.75 + 0.25 * norm_rms)
        elif speech_activity_pct is not None:
            s_acoustic = min(1.0, max(0.0, speech_activity_pct / 100.0))
        else:
            s_acoustic = None

        # 2. Visual component
        if visual_movement is not None:
            norm_motion = min(1.0, max(0.0, visual_movement / 50.0))
            var = head_pose_variance if head_pose_variance is not None else 0.0
            norm_var = min(1.0, max(0.0, var / 10.0))
            # Bounded modulation to avoid double-counting motion
            s_visual = norm_motion * (0.80 + 0.20 * norm_var)
        else:
            s_visual = None

        # 3. Cross-modal combination
        if s_acoustic is not None and s_visual is not None:
            # Bounded interaction preventing collinear runaway sum
            s_composite = 1.0 - (1.0 - s_acoustic) * (1.0 - s_visual)
        elif s_acoustic is not None:
            s_composite = s_acoustic
        elif s_visual is not None:
            s_composite = s_visual
        else:
            return 0.0

        return round(float(s_composite * 100.0), 1)

    def close_window(
        self,
        timestamp: float,
        session_id: str = "default_session",
    ) -> Optional[MultimodalWindow]:
        """Convenience alias for close_and_fuse_window."""
        return self.close_and_fuse_window(session_id=session_id, current_timestamp=timestamp)

    def close_and_fuse_window(
        self,
        session_id: str,
        current_timestamp: float,
    ) -> Optional[MultimodalWindow]:
        """
        Close the current window, compute aggregate metrics, persist to DB, and return MultimodalWindow.

        Returns:
            MultimodalWindow object or None if no samples were recorded across either modality.
        """
        if self.current_window_start is None:
            return None

        has_visual = len(self._visual_buffer) > 0
        has_audio = len(self._audio_buffer) > 0

        if not has_visual and not has_audio:
            # Empty window, reset start time
            self.start_window(current_timestamp)
            return None

        start_sec = self.current_window_start
        end_sec = current_timestamp
        duration = max(0.001, end_sec - start_sec)

        # 1. Visual aggregation
        if has_visual:
            counts = [v.active_person_count for v in self._visual_buffer]
            motions = [v.visual_movement_index for v in self._visual_buffer]
            poses = [v.head_pose_variance for v in self._visual_buffer]

            mean_count: Optional[int] = int(round(float(np.mean(counts))))
            mean_motion: Optional[float] = round(float(np.mean(motions)), 2)
            # Use pose variance from samples, or fall back to standard deviation of motion if pose is 0
            raw_pose_var = float(np.mean(poses))
            mean_pose_var: Optional[float] = (
                round(raw_pose_var, 2)
                if raw_pose_var > 0.0
                else round(float(np.std(motions)), 2)
            )

            expected_visual_samples = max(1.0, duration * (self.expected_video_fps * 0.5))
            visual_quality = min(1.0, len(self._visual_buffer) / expected_visual_samples)
        else:
            mean_count = None
            mean_motion = None
            mean_pose_var = None
            visual_quality = 0.0

        # 2. Audio aggregation
        if has_audio:
            speeches = [a[1].speech_ratio_pct for a in self._audio_buffer]
            energies = [a[1].rms_energy for a in self._audio_buffer]

            mean_speech: Optional[float] = round(float(np.mean(speeches)), 1)
            mean_rms: Optional[float] = round(float(np.mean(energies)), 4)
            is_genuine_silence = (mean_speech == 0.0)

            expected_audio_samples = max(1.0, duration * (self.expected_audio_hz * 0.5))
            audio_quality = min(1.0, len(self._audio_buffer) / expected_audio_samples)
        else:
            mean_speech = None
            mean_rms = None
            is_genuine_silence = False
            audio_quality = 0.0

        # 3. Overall Data Quality Evaluation
        if has_visual and has_audio:
            overall_quality = round(0.5 * visual_quality + 0.5 * audio_quality, 2)
            quality_label = (
                DataQualityLabel.HIGH.value
                if overall_quality >= 0.7
                else DataQualityLabel.DEGRADED_SAMPLE_RATE.value
            )
        elif has_visual:
            overall_quality = round(visual_quality * 0.6, 2)
            quality_label = DataQualityLabel.PARTIAL_VISUAL_ONLY.value
        elif has_audio:
            overall_quality = round(audio_quality * 0.4, 2)
            quality_label = DataQualityLabel.PARTIAL_AUDIO_ONLY.value
        else:
            overall_quality = 0.0
            quality_label = DataQualityLabel.UNAVAILABLE.value

        # 4. Correlation-adjusted activity calculation
        composite_score = self.compute_composite_activity_score(
            visual_movement=mean_motion,
            head_pose_variance=mean_pose_var,
            speech_activity_pct=mean_speech,
            rms_energy=mean_rms,
        )

        window_id = f"win_{session_id}_{self.window_counter}_{int(start_sec * 1000)}"
        self.window_counter += 1

        window = MultimodalWindow(
            window_id=window_id,
            session_id=session_id,
            window_start_sec=start_sec,
            window_end_sec=end_sec,
            active_person_count=mean_count,
            visual_movement_index=mean_motion,
            head_pose_variance=mean_pose_var,
            speech_activity_pct=mean_speech,
            rms_energy_mean=mean_rms,
            visual_available=has_visual,
            audio_available=has_audio,
            visual_sample_count=len(self._visual_buffer),
            audio_sample_count=len(self._audio_buffer),
            data_quality_score=overall_quality,
            data_quality_label=quality_label,
            is_genuine_silence=is_genuine_silence,
            correlation_adjusted_activity_score=composite_score,
        )

        # 5. Persist to database if configured
        if self.db is not None:
            self.db.insert_activity_window(
                window_id=window.window_id,
                session_id=window.session_id,
                window_start_sec=window.window_start_sec,
                window_end_sec=window.window_end_sec,
                active_person_count=window.active_person_count,
                visual_movement_index=window.visual_movement_index,
                head_pose_variance=window.head_pose_variance,
                speech_activity_pct=window.speech_activity_pct,
                rms_energy_mean=window.rms_energy_mean,
                data_quality_score=window.data_quality_score,
                data_quality_label=window.data_quality_label,
            )

        # Reset state for next window
        self.start_window(current_timestamp)
        return window

    def fuse_streams(
        self,
        session_id: str,
        visual_samples: List[VisualFeatures],
        audio_samples: List[AudioFeatures],
        window_duration_sec: Optional[int] = None,
    ) -> List[MultimodalWindow]:
        """
        Batch align and aggregate asynchronous visual and acoustic timeseries into consecutive windows.

        Args:
            session_id: Session identifier.
            visual_samples: Chronologically ordered list of VisualFeatures.
            audio_samples: Chronologically ordered list of AudioFeatures.
            window_duration_sec: Aggregation window length in seconds.

        Returns:
            List of fused MultimodalWindow records.
        """
        w_dur = window_duration_sec or self.window_duration_sec

        all_timestamps = [v.timestamp for v in visual_samples] + [a.timestamp for a in audio_samples]
        if not all_timestamps:
            return []

        t_min = min(all_timestamps)
        t_max = max(all_timestamps)

        if t_max <= t_min:
            t_max = t_min + w_dur

        windows: List[MultimodalWindow] = []
        w_start = t_min

        while w_start < t_max:
            w_end = w_start + w_dur

            # Extract samples within [w_start, w_end)
            w_visual = [v for v in visual_samples if w_start <= v.timestamp < w_end]
            w_audio = [(a.timestamp, a) for a in audio_samples if w_start <= a.timestamp < w_end]

            # Populate buffers and close
            self.start_window(w_start)
            self._visual_buffer = list(w_visual)
            self._audio_buffer = list(w_audio)

            window = self.close_and_fuse_window(session_id=session_id, current_timestamp=w_end)
            if window is not None:
                windows.append(window)

            w_start = w_end

        return windows
