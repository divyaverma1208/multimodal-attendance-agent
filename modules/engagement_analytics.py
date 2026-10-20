"""
Engagement Analytics Module.

Performs transparent, interpretable time-windowed aggregation and descriptive
analytics on observable classroom dynamics (physical kinetic movement, sound volume,
speech activity ratios).

EPISTEMIC & ETHICAL GUARDRAILS:
- Strictly quantifies observable physical signals (pixel displacement, sound pressure,
  vocalization ratios).
- Does NOT claim to measure true cognitive attention, emotional states, mental engagement,
  comprehension, motivation, or learning outcomes.
- Distinguishes sensor absence/unavailability from true zero values.
- Accompanies every calculation with clear assumptions and human-interpretable explanations.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from modules.feature_fusion import DataQualityLabel, MultimodalWindow

logger = logging.getLogger(__name__)


@dataclass
class ActivityWeights:
    """
    Configurable weights for composite activity index.
    Weights are non-negative and dynamically re-normalized across available modalities.
    """
    visual_movement: float = 0.50
    speech_activity: float = 0.50
    head_pose_variance: float = 0.0

    def __post_init__(self) -> None:
        if self.visual_movement < 0 or self.speech_activity < 0 or self.head_pose_variance < 0:
            raise ValueError("Activity weights must be non-negative.")
        total = self.visual_movement + self.speech_activity + self.head_pose_variance
        if total <= 0:
            raise ValueError("At least one activity weight must be greater than zero.")

    def get_normalized(
        self,
        visual_available: bool = True,
        audio_available: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Dynamically re-normalize weights to sum to 1.0 based on available sensor modalities.
        Prevents composite activity drops caused by missing hardware.
        """
        w_v = self.visual_movement if visual_available else 0.0
        w_p = self.head_pose_variance if visual_available else 0.0
        w_s = self.speech_activity if audio_available else 0.0

        total = w_v + w_s + w_p
        if total <= 0:
            return 0.0, 0.0, 0.0
        return w_v / total, w_s / total, w_p / total


@dataclass
class TrendEvent:
    """Structured record of a sustained increase or decrease in measured activity."""
    trend_type: str  # "SUSTAINED_INCREASE" or "SUSTAINED_DECREASE"
    start_sec: float
    end_sec: float
    start_window_id: str
    end_window_id: str
    initial_value: float
    final_value: float
    delta_value: float
    duration_sec: float
    explanation: str


@dataclass
class SessionAnalyticsSummary:
    """
    High-level summary of observable classroom dynamics for a session.
    Fields 1-8 preserve backwards compatibility with existing UI and testing callers.
    """
    total_windows: int
    avg_headcount: float
    peak_headcount: int
    mean_visual_movement: float
    mean_speech_activity_pct: float
    estimated_monologue_minutes: float
    estimated_discussion_minutes: float
    silent_period_minutes: float

    # Enhanced transparency and data-quality diagnostics
    valid_windows: int = 0
    min_headcount: int = 0
    mean_composite_activity: float = 0.0
    sensor_outage_minutes: float = 0.0
    average_data_quality: float = 1.0
    sustained_increases_count: int = 0
    sustained_decreases_count: int = 0
    metric_explanations: Dict[str, str] = field(default_factory=dict)


class EngagementAnalytics:
    """
    Computes transparent time-series analytics and classroom activity dynamics.
    All calculations are deterministic, inspectable, and documented with assumptions.
    """

    METRIC_ASSUMPTIONS: Dict[str, str] = {
        "visual_movement_index": (
            "Assumption: Inter-frame pixel differencing correlates with physical macroscopic movement "
            "(e.g., student gestures, standing, posture shifts, teacher pacing). "
            "Limitation: Lighting shifts, projector refreshes, or camera vibrations can introduce noise. "
            "Strictly physical: Does NOT measure cognitive attentiveness or boredom."
        ),
        "speech_activity_pct": (
            "Assumption: Short-time energy, zero-crossing rate, and spectral centroid indicate human vocalization. "
            "Limitation: Single-channel audio measures room-wide sound without speaker separation or diarization. "
            "Strictly physical: Loudness or vocal duration does NOT measure comprehension, motivation, or learning."
        ),
        "composite_activity_index": (
            "Assumption: A normalized weighted combination of kinetic movement and acoustic vocalization "
            "summarizes overall room-level physical activity. "
            "Safeguard: Weights are dynamically re-balanced if a modality is unavailable, avoiding artificial penalties."
        ),
        "is_genuine_silence": (
            "Assumption: Adaptive noise calibration establishes an ambient noise floor. Zero voice activity "
            "during active recording indicates quiet room conditions (e.g., individual reading/writing/tests). "
            "Safeguard: Distinguished from missing/failed audio streams which yield None rather than 0%."
        ),
        "monologue_vs_discussion": (
            "Assumption: Continuous vocalization (>=80% speech) over extended windows typically reflects "
            "unbroken presentation/lecturing, while intermediate speech (30-80%) reflects interactive dialogue. "
            "Limitation: Heuristic proxy based on acoustic activity, not semantic conversational analysis."
        ),
        "sustained_trends": (
            "Assumption: Monotonic or cumulative activity shifts across consecutive windows (>= 3 windows) "
            "indicate observable pedagogical transitions in room dynamics (e.g., lecture to group discussion). "
            "Limitation: Reflects physical activity progression, not student emotional state or focus."
        ),
    }

    @classmethod
    def get_metric_explanations(cls) -> Dict[str, str]:
        """Return human-readable explanations and assumptions for all computed metrics."""
        return dict(cls.METRIC_ASSUMPTIONS)

    @staticmethod
    def sanitize_metric(
        value: Optional[float],
        min_val: float = 0.0,
        max_val: float = 100.0,
    ) -> Optional[float]:
        """
        Validate, sanitize, and bound metric values into safe numerical ranges.
        Handles None, NaNs, infinities, and out-of-range floats gracefully.
        """
        if value is None:
            return None
        try:
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return None
            return float(np.clip(val, min_val, max_val))
        except (ValueError, TypeError):
            return None

    @classmethod
    def calculate_composite_activity(
        cls,
        visual_movement: Optional[float] = None,
        speech_pct: Optional[float] = None,
        head_pose_variance: Optional[float] = None,
        weights: Optional[ActivityWeights] = None,
        movement_index: Optional[float] = None,
        speech_activity_pct: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate normalized composite activity indicator (0.0 to 100.0).

        Handles missing modalities by dynamically re-balancing weights across available inputs,
        preventing artificial score drops.

        Args:
            visual_movement: Visual motion score (0.0 to 100.0), or None if unavailable.
            speech_pct: Speech activity percentage (0.0 to 100.0), or None if unavailable.
            head_pose_variance: Head pose variance score, or None.
            weights: Optional ActivityWeights instance. Defaults to equal 50/50 weighting.
            movement_index: Alias for visual_movement.
            speech_activity_pct: Alias for speech_pct.

        Returns:
            Composite activity score (0.0 to 100.0), or None if neither modality is available.
        """
        cfg_weights = weights or ActivityWeights()

        eff_visual = visual_movement if visual_movement is not None else movement_index
        eff_speech = speech_pct if speech_pct is not None else speech_activity_pct

        v_clean = cls.sanitize_metric(eff_visual, 0.0, 100.0)
        s_clean = cls.sanitize_metric(eff_speech, 0.0, 100.0)
        p_clean = cls.sanitize_metric(head_pose_variance, 0.0, 20.0)

        has_v = v_clean is not None
        has_s = s_clean is not None

        if not has_v and not has_s:
            return None

        w_v, w_s, w_p = cfg_weights.get_normalized(
            visual_available=has_v,
            audio_available=has_s,
        )

        score = 0.0
        if has_v and v_clean is not None:
            score += w_v * v_clean
            if p_clean is not None:
                # Scale 0-20 pose variance to 0-100 range
                score += w_p * (p_clean * 5.0)
        if has_s and s_clean is not None:
            score += w_s * s_clean

        return round(float(np.clip(score, 0.0, 100.0)), 1)

    @classmethod
    def detect_sustained_trends(
        cls,
        windows: List[Union[MultimodalWindow, Dict[str, Any]]],
        min_consecutive_windows: int = 3,
        min_delta_threshold: float = 10.0,
        weights: Optional[ActivityWeights] = None,
    ) -> List[TrendEvent]:
        """
        Identify sustained increases or decreases in measured activity over rolling horizons.

        Args:
            windows: Chronologically ordered list of MultimodalWindow or dict records.
            min_consecutive_windows: Minimum window sequence length to constitute a sustained trend.
            min_delta_threshold: Minimum total net change in points to qualify as significant.
            weights: Optional ActivityWeights.

        Returns:
            List of detected TrendEvent objects with explanations.
        """
        if len(windows) < min_consecutive_windows:
            return []

        # Parse windows to extract scores and timing
        parsed = []
        for w in windows:
            if isinstance(w, MultimodalWindow):
                w_id = w.window_id
                t_start = w.window_start_sec
                t_end = w.window_end_sec
                score = cls.calculate_composite_activity(
                    visual_movement=w.visual_movement_index,
                    speech_pct=w.speech_activity_pct,
                    head_pose_variance=w.head_pose_variance,
                    weights=weights,
                )
            else:
                w_id = w.get("window_id", "")
                t_start = float(w.get("window_start_sec", 0.0))
                t_end = float(w.get("window_end_sec", 0.0))
                score = cls.calculate_composite_activity(
                    visual_movement=w.get("visual_movement_index"),
                    speech_pct=w.get("speech_activity_pct"),
                    head_pose_variance=w.get("head_pose_variance"),
                    weights=weights,
                )
            parsed.append((w_id, t_start, t_end, score))

        # Sort chronologically by start timestamp
        parsed.sort(key=lambda x: x[1])

        trend_events: List[TrendEvent] = []
        i = 0
        n = len(parsed)

        while i <= n - min_consecutive_windows:
            slice_items = parsed[i : i + min_consecutive_windows]
            scores = [item[3] for item in slice_items]

            # Require all scores in window to be available
            if any(s is None for s in scores):
                i += 1
                continue

            first_score = float(scores[0])  # type: ignore
            last_score = float(scores[-1])  # type: ignore
            net_delta = round(last_score - first_score, 1)

            # Check for monotonic increase or cumulative ramp-up
            is_increasing = all(scores[k] <= scores[k + 1] for k in range(len(scores) - 1))  # type: ignore
            is_decreasing = all(scores[k] >= scores[k + 1] for k in range(len(scores) - 1))  # type: ignore

            if is_increasing and net_delta >= min_delta_threshold:
                start_w, start_t, _, _ = slice_items[0]
                end_w, _, end_t, _ = slice_items[-1]
                event = TrendEvent(
                    trend_type="SUSTAINED_INCREASE",
                    start_sec=start_t,
                    end_sec=end_t,
                    start_window_id=start_w,
                    end_window_id=end_w,
                    initial_value=first_score,
                    final_value=last_score,
                    delta_value=net_delta,
                    duration_sec=round(end_t - start_t, 1),
                    explanation=(
                        f"Sustained activity ramp-up (+{net_delta} pts) observed across {min_consecutive_windows} "
                        f"consecutive windows from t={int(start_t)}s to t={int(end_t)}s. "
                        "Indicates an increase in observable physical dynamics (e.g. shift to group discussion)."
                    ),
                )
                trend_events.append(event)
                i += min_consecutive_windows  # Step forward past this trend
                continue

            elif is_decreasing and net_delta <= -min_delta_threshold:
                start_w, start_t, _, _ = slice_items[0]
                end_w, _, end_t, _ = slice_items[-1]
                event = TrendEvent(
                    trend_type="SUSTAINED_DECREASE",
                    start_sec=start_t,
                    end_sec=end_t,
                    start_window_id=start_w,
                    end_window_id=end_w,
                    initial_value=first_score,
                    final_value=last_score,
                    delta_value=net_delta,
                    duration_sec=round(end_t - start_t, 1),
                    explanation=(
                        f"Sustained activity wind-down ({net_delta} pts) observed across {min_consecutive_windows} "
                        f"consecutive windows from t={int(start_t)}s to t={int(end_t)}s. "
                        "Indicates a decrease in observable room dynamics (e.g. transition to silent reading/pause)."
                    ),
                )
                trend_events.append(event)
                i += min_consecutive_windows
                continue

            i += 1

        return trend_events

    @classmethod
    def summarize_windows(
        cls,
        windows: List[Union[MultimodalWindow, Dict[str, Any]]],
        weights: Optional[ActivityWeights] = None,
    ) -> SessionAnalyticsSummary:
        """
        Produce summary statistics from a series of multimodal windows.

        Args:
            windows: Chronologically ordered list of MultimodalWindow or dict instances.
            weights: Optional ActivityWeights for composite activity calculation.

        Returns:
            SessionAnalyticsSummary object with transparency explanations.
        """
        explanations = cls.get_metric_explanations()

        if not windows:
            return SessionAnalyticsSummary(
                total_windows=0,
                avg_headcount=0.0,
                peak_headcount=0,
                mean_visual_movement=0.0,
                mean_speech_activity_pct=0.0,
                estimated_monologue_minutes=0.0,
                estimated_discussion_minutes=0.0,
                silent_period_minutes=0.0,
                valid_windows=0,
                min_headcount=0,
                mean_composite_activity=0.0,
                sensor_outage_minutes=0.0,
                average_data_quality=0.0,
                sustained_increases_count=0,
                sustained_decreases_count=0,
                metric_explanations=explanations,
            )

        valid_headcounts: List[int] = []
        valid_motions: List[float] = []
        valid_speeches: List[float] = []
        composites: List[float] = []
        qualities: List[float] = []

        monologue_sec = 0.0
        discussion_sec = 0.0
        silent_sec = 0.0
        outage_sec = 0.0

        for w in windows:
            if isinstance(w, MultimodalWindow):
                dur = max(0.001, w.window_end_sec - w.window_start_sec)
                h = w.active_person_count
                m = cls.sanitize_metric(w.visual_movement_index, 0.0, 100.0)
                s = cls.sanitize_metric(w.speech_activity_pct, 0.0, 100.0)
                q = float(w.data_quality_score)
                is_silence = w.is_genuine_silence
                v_avail = w.visual_available
                a_avail = w.audio_available
            else:
                t_start = float(w.get("window_start_sec", 0.0))
                t_end = float(w.get("window_end_sec", 0.0))
                dur = max(0.001, t_end - t_start)
                h = w.get("active_person_count")
                m = cls.sanitize_metric(w.get("visual_movement_index"), 0.0, 100.0)
                s = cls.sanitize_metric(w.get("speech_activity_pct"), 0.0, 100.0)
                q = float(w.get("data_quality_score", 1.0))
                is_silence = bool(w.get("is_genuine_silence", False)) or (s == 0.0)
                v_avail = m is not None
                a_avail = s is not None

            qualities.append(q)

            if h is not None:
                valid_headcounts.append(int(h))
            if m is not None:
                valid_motions.append(float(m))
            if s is not None:
                valid_speeches.append(float(s))
                # Acoustic mode categorization (only valid when audio was recorded)
                if s >= 80.0:
                    monologue_sec += dur
                elif 30.0 <= s < 80.0:
                    discussion_sec += dur
                elif s < 10.0 or is_silence:
                    silent_sec += dur
            elif not v_avail:
                outage_sec += dur

            comp = cls.calculate_composite_activity(m, s, weights=weights)
            if comp is not None:
                composites.append(comp)

        # Detect sustained trends
        trends = cls.detect_sustained_trends(windows, weights=weights)
        inc_count = sum(1 for t in trends if t.trend_type == "SUSTAINED_INCREASE")
        dec_count = sum(1 for t in trends if t.trend_type == "SUSTAINED_DECREASE")

        return SessionAnalyticsSummary(
            total_windows=len(windows),
            avg_headcount=round(float(np.mean(valid_headcounts)), 1) if valid_headcounts else 0.0,
            peak_headcount=int(max(valid_headcounts)) if valid_headcounts else 0,
            mean_visual_movement=round(float(np.mean(valid_motions)), 1) if valid_motions else 0.0,
            mean_speech_activity_pct=round(float(np.mean(valid_speeches)), 1) if valid_speeches else 0.0,
            estimated_monologue_minutes=round(monologue_sec / 60.0, 1),
            estimated_discussion_minutes=round(discussion_sec / 60.0, 1),
            silent_period_minutes=round(silent_sec / 60.0, 1),
            valid_windows=len(composites),
            min_headcount=int(min(valid_headcounts)) if valid_headcounts else 0,
            mean_composite_activity=round(float(np.mean(composites)), 1) if composites else 0.0,
            sensor_outage_minutes=round(outage_sec / 60.0, 1),
            average_data_quality=round(float(np.mean(qualities)), 2) if qualities else 0.0,
            sustained_increases_count=inc_count,
            sustained_decreases_count=dec_count,
            metric_explanations=explanations,
        )

    @classmethod
    def to_dataframe(
        cls,
        windows: List[Union[MultimodalWindow, Dict[str, Any]]],
        weights: Optional[ActivityWeights] = None,
    ) -> pd.DataFrame:
        """
        Convert windows to a comprehensive, sorted Pandas DataFrame for Plotly charting
        and tabular inspection.

        Calculates deltas between consecutive windows, trend directions, and relative
        timeline minutes. Properly handles unordered timestamps and missing modalities.

        Args:
            windows: List of MultimodalWindow or dict records.
            weights: Optional ActivityWeights.

        Returns:
            Formatted Pandas DataFrame.
        """
        if not windows:
            return pd.DataFrame()

        rows = []
        for w in windows:
            if isinstance(w, MultimodalWindow):
                start = float(w.window_start_sec)
                end = float(w.window_end_sec)
                head = w.active_person_count
                motion = cls.sanitize_metric(w.visual_movement_index, 0.0, 100.0)
                speech = cls.sanitize_metric(w.speech_activity_pct, 0.0, 100.0)
                rms = w.rms_energy_mean
                v_avail = w.visual_available
                a_avail = w.audio_available
                is_silence = w.is_genuine_silence
                q_score = w.data_quality_score
                q_label = w.data_quality_label
                win_id = w.window_id
            else:
                start = float(w.get("window_start_sec", 0.0))
                end = float(w.get("window_end_sec", 0.0))
                head = w.get("active_person_count")
                motion = cls.sanitize_metric(w.get("visual_movement_index"), 0.0, 100.0)
                speech = cls.sanitize_metric(w.get("speech_activity_pct"), 0.0, 100.0)
                rms = w.get("rms_energy_mean")
                v_avail = motion is not None
                a_avail = speech is not None
                is_silence = bool(w.get("is_genuine_silence", False)) or (speech == 0.0)
                q_score = float(w.get("data_quality_score", 1.0))
                q_label = str(w.get("data_quality_label", DataQualityLabel.HIGH.value))
                win_id = str(w.get("window_id", ""))

            composite = cls.calculate_composite_activity(motion, speech, weights=weights)

            rows.append({
                "window_id": win_id,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 2),
                "headcount": head if head is not None else 0,
                "visual_movement": motion if motion is not None else 0.0,
                "speech_pct": speech if speech is not None else 0.0,
                "rms_energy": rms if rms is not None else 0.0,
                "composite_activity": composite if composite is not None else 0.0,
                "visual_available": v_avail,
                "audio_available": a_avail,
                "is_genuine_silence": is_silence,
                "data_quality_score": q_score,
                "data_quality_label": q_label,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Sort chronologically by start timestamp to ensure proper delta calculation
        df = df.sort_values(by="start_sec").reset_index(drop=True)

        base_time = df["start_sec"].min()
        df["timeline_min"] = ((df["start_sec"] - base_time) / 60.0).round(2)

        # Consecutive window changes (deltas)
        df["activity_delta"] = df["composite_activity"].diff().round(2).fillna(0.0)
        df["headcount_delta"] = df["headcount"].diff().fillna(0).astype(int)
        df["visual_movement_delta"] = df["visual_movement"].diff().round(2).fillna(0.0)
        df["speech_pct_delta"] = df["speech_pct"].diff().round(1).fillna(0.0)

        # Trend direction label per window
        def _classify_trend(delta: float) -> str:
            if delta > 5.0:
                return "INCREASING"
            elif delta < -5.0:
                return "DECREASING"
            return "STABLE"

        df["trend_direction"] = df["activity_delta"].apply(_classify_trend)

        return df

    @classmethod
    def generate_analytical_report(
        cls,
        windows: List[Union[MultimodalWindow, Dict[str, Any]]],
        weights: Optional[ActivityWeights] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete analytical session bundle with metrics, trend events,
        data-quality metrics, and epistemic explanations.
        """
        summary = cls.summarize_windows(windows, weights=weights)
        trends = cls.detect_sustained_trends(windows, weights=weights)

        # Categorize window quality distribution
        quality_counts = {label.value: 0 for label in DataQualityLabel}
        for w in windows:
            label = (
                w.data_quality_label
                if isinstance(w, MultimodalWindow)
                else w.get("data_quality_label", DataQualityLabel.HIGH.value)
            )
            if label in quality_counts:
                quality_counts[label] += 1

        return {
            "summary": summary,
            "trends": [
                {
                    "trend_type": t.trend_type,
                    "start_sec": t.start_sec,
                    "end_sec": t.end_sec,
                    "delta_value": t.delta_value,
                    "duration_sec": t.duration_sec,
                    "explanation": t.explanation,
                }
                for t in trends
            ],
            "data_quality_distribution": quality_counts,
            "metric_explanations": cls.get_metric_explanations(),
            "total_windows": len(windows),
        }
