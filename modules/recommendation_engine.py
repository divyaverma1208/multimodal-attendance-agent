"""
Recommendation Engine Module.

Provides offline, transparent, context-aware teaching decision support based on
time-windowed classroom activity metrics.

EPISTEMIC & ETHICAL BOUNDARIES:
- Operates strictly on deterministic, interpretable heuristic rules.
- Analyzes observable macro dynamics across multiple consecutive windows (never a single noisy spike).
- All suggestions are optional, non-evaluative pedagogical decision support for human educators.
- Does NOT infer student emotions, intelligence, comprehension, or psychological focus.
- Does NOT make disciplinary evaluations or penalize students.
- Explicitly accounts for sensor availability and data-quality limitations.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Union
import uuid

import numpy as np

from config.settings import settings
from modules.database import DatabaseManager
from modules.feature_fusion import DataQualityLabel, MultimodalWindow

logger = logging.getLogger(__name__)


class RuleCode(str, Enum):
    """Categorical identifiers for deterministic pedagogical rules."""
    PROLONGED_MONOLOGUE = "RULE_PROLONGED_MONOLOGUE"
    EXTENDED_SILENCE = "RULE_EXTENDED_SILENCE"
    CONDUCT_RECAP = "RULE_CONDUCT_RECAP"
    RESTLESSNESS_SPIKE = "RULE_RESTLESSNESS_SPIKE"
    CHECK_AUDIO_QUALITY = "RULE_CHECK_AUDIO_QUALITY"
    CHECK_CAMERA_ANGLE = "RULE_CHECK_CAMERA_ANGLE"
    PRESENCE_DROP = "RULE_PRESENCE_DROP"
    MAINTAIN_ACTIVITY = "RULE_MAINTAIN_ACTIVITY"


@dataclass
class RecommendationEngineConfig:
    """Configurable thresholds and evaluation windows for the recommendation rules."""
    cooldown_seconds: float = 300.0  # 5-minute cooldown per rule
    monologue_speech_pct: float = 80.0
    monologue_motion_max: float = 20.0
    monologue_duration_min: float = 10.0
    silence_speech_pct: float = 5.0
    silence_duration_min: float = 3.0
    restlessness_movement_min: float = 70.0
    restlessness_speech_max: float = 15.0
    restlessness_duration_min: float = 3.0
    recap_prior_speech_min: float = 65.0
    recap_current_speech_max: float = 20.0
    presence_drop_ratio: float = 0.30
    hardware_warning_windows: int = 3
    stable_pacing_windows: int = 10


@dataclass
class Suggestion:
    """
    Structured teaching decision support record.

    Fields 1-5 preserve backwards compatibility with existing database
    and UI callers.
    """
    suggestion_id: str
    session_id: str
    timestamp_sec: float
    trigger_rule_code: str
    suggestion_text: str
    acknowledged: bool = False

    # Required structured recommendation fields
    title: str = ""
    explanation: str = ""
    triggering_measurements: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""
    data_quality_limitations: str = ""
    confidence_score: float = 1.0

    def __post_init__(self) -> None:
        """Populate default titles and actions if not provided."""
        if not self.title:
            self.title = self.trigger_rule_code.replace("RULE_", "").replace("_", " ").title()
        if not self.explanation:
            self.explanation = self.suggestion_text
        if not self.suggested_action:
            self.suggested_action = self.suggestion_text

    def to_dict(self) -> Dict[str, Any]:
        """Convert suggestion into a dictionary suitable for JSON serialization and UI display."""
        return {
            "suggestion_id": self.suggestion_id,
            "session_id": self.session_id,
            "timestamp_sec": self.timestamp_sec,
            "trigger_rule_code": self.trigger_rule_code,
            "title": self.title,
            "explanation": self.explanation,
            "triggering_measurements": self.triggering_measurements,
            "suggested_action": self.suggested_action,
            "data_quality_limitations": self.data_quality_limitations,
            "confidence_score": round(self.confidence_score, 2),
            "acknowledged": self.acknowledged,
            "suggestion_text": self.suggestion_text,
        }


class RecommendationEngine:
    """
    Evaluates sliding-window multimodal timeseries against transparent pedagogical heuristics.
    Never evaluates single noisy spikes; requires sustained multi-window observations.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        config: Optional[RecommendationEngineConfig] = None,
        db: Optional[DatabaseManager] = None,
    ) -> None:
        """
        Initialize RecommendationEngine.

        Args:
            db_manager: Optional DatabaseManager to persist triggered suggestions.
            config: Optional RecommendationEngineConfig instance with custom thresholds.
            db: Alias for db_manager.
        """
        self.db = db_manager or db
        self.config = config or RecommendationEngineConfig()
        self.last_trigger_time: Dict[str, float] = {}

    def _is_on_cooldown(self, rule_code: str, current_time: float) -> bool:
        """Check if a rule trigger is currently in cooldown."""
        last_time = self.last_trigger_time.get(rule_code, 0.0)
        return (current_time - last_time) < self.config.cooldown_seconds

    def evaluate_windows(
        self,
        recent_windows: List[MultimodalWindow],
        session_id: str = "default_session",
        current_time: Optional[float] = None,
    ) -> List[Suggestion]:
        """Convenience method to evaluate a collection of windows directly."""
        return self.evaluate_rules(
            session_id=session_id,
            recent_windows=recent_windows,
            current_time=current_time,
        )

    def evaluate_rules(
        self,
        session_id: str,
        recent_windows: List[MultimodalWindow],
        current_time: Optional[float] = None,
    ) -> List[Suggestion]:
        """
        Evaluate sliding window history against deterministic pedagogical rules.

        Args:
            session_id: Current session identifier.
            recent_windows: Chronologically ordered list of MultimodalWindow instances.
            current_time: Optional evaluation timestamp. Defaults to current epoch time.

        Returns:
            List of freshly triggered Suggestion objects.
        """
        now = current_time if current_time is not None else time.time()
        triggered: List[Suggestion] = []

        if not recent_windows:
            return triggered

        # Estimate window durations (assume 30s default if not detectable)
        w_dur = 30.0
        if len(recent_windows) >= 2:
            deltas = [
                recent_windows[i].window_end_sec - recent_windows[i].window_start_sec
                for i in range(len(recent_windows))
            ]
            w_dur = max(5.0, float(np.median(deltas)))

        # Convert durations to window counts (requiring at least 3 windows to prevent single-spike triggers)
        ten_min_windows = max(3, int(round((self.config.monologue_duration_min * 60.0) / w_dur)))
        three_min_windows = max(3, int(round((self.config.silence_duration_min * 60.0) / w_dur)))
        hw_warning_windows = max(2, self.config.hardware_warning_windows)
        stable_windows = max(4, self.config.stable_pacing_windows)

        # Slice recent horizons
        slice_3m = recent_windows[-three_min_windows:]
        slice_10m = recent_windows[-ten_min_windows:]
        slice_hw = recent_windows[-hw_warning_windows:]
        slice_stable = recent_windows[-stable_windows:]

        # =========================================================================
        # Rule 1: Prolonged Monologue (Introduce Q&A / Comprehension Check)
        # =========================================================================
        rule_1 = RuleCode.PROLONGED_MONOLOGUE.value
        if len(slice_10m) >= ten_min_windows and not self._is_on_cooldown(rule_1, now):
            valid_speeches = [w.speech_activity_pct for w in slice_10m if w.speech_activity_pct is not None]
            valid_motions = [w.visual_movement_index for w in slice_10m if w.visual_movement_index is not None]

            # Requires audio & video to be consistently available across the full horizon
            if len(valid_speeches) == len(slice_10m) and len(valid_motions) == len(slice_10m):
                avg_speech = float(np.mean(valid_speeches))
                avg_motion = float(np.mean(valid_motions))

                if (
                    avg_speech >= self.config.monologue_speech_pct
                    and avg_motion <= self.config.monologue_motion_max
                ):
                    avg_quality = float(np.mean([w.data_quality_score for w in slice_10m]))
                    sugg = Suggestion(
                        suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
                        session_id=session_id,
                        timestamp_sec=now,
                        trigger_rule_code=rule_1,
                        suggestion_text=(
                            f"Continuous presentation observed for {int(self.config.monologue_duration_min)} minutes. "
                            "Consider pausing for a question-and-answer activity or short peer check."
                        ),
                        title="Continuous Presentation: Introduce Q&A",
                        explanation=(
                            f"Continuous presentation detected: acoustic speech activity averaged {avg_speech:.1f}% across {len(slice_10m)} consecutive windows "
                            f"({int(self.config.monologue_duration_min)} mins) with low physical movement ({avg_motion:.1f}%)."
                        ),
                        triggering_measurements={
                            "mean_speech_pct": round(avg_speech, 1),
                            "mean_visual_movement": round(avg_motion, 1),
                            "duration_minutes": self.config.monologue_duration_min,
                            "consecutive_windows": len(slice_10m),
                        },
                        suggested_action="Introduce a question-and-answer activity, quick poll, or 2-minute think-pair-share.",
                        data_quality_limitations=(
                            "Single-channel microphone capture cannot distinguish instructor monologue from a single student presentation. "
                            "Loudness does not evaluate student attentiveness."
                        ),
                        confidence_score=avg_quality,
                    )
                    triggered.append(sugg)
                    self.last_trigger_time[rule_1] = now

        # =========================================================================
        # Rule 2: Extended Silence (Invite Voluntary Participation)
        # =========================================================================
        rule_2 = RuleCode.EXTENDED_SILENCE.value
        if len(slice_3m) >= three_min_windows and not self._is_on_cooldown(rule_2, now):
            # Must verify that microphone was ACTIVE and measured genuine silence (not missing audio!)
            has_active_audio = all(w.audio_available for w in slice_3m)
            valid_speeches = [w.speech_activity_pct for w in slice_3m if w.speech_activity_pct is not None]

            if has_active_audio and len(valid_speeches) == len(slice_3m):
                avg_speech = float(np.mean(valid_speeches))
                all_silent = all(
                    w.is_genuine_silence or (s <= self.config.silence_speech_pct)
                    for w, s in zip(slice_3m, valid_speeches)
                )

                if all_silent and avg_speech <= self.config.silence_speech_pct:
                    avg_quality = float(np.mean([w.data_quality_score for w in slice_3m]))
                    sugg = Suggestion(
                        suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
                        session_id=session_id,
                        timestamp_sec=now,
                        trigger_rule_code=rule_2,
                        suggestion_text=(
                            f"Extended classroom silence observed (>= {int(self.config.silence_duration_min)} min). "
                            "If independent exercise is underway, continue; if expecting responses, invite voluntary participation."
                        ),
                        title="Extended Silence: Invite Voluntary Participation",
                        explanation=(
                            f"Measured room vocalization was <= {self.config.silence_speech_pct}% for {len(slice_3m)} consecutive windows "
                            f"(~{int(self.config.silence_duration_min)} mins) under active microphone monitoring."
                        ),
                        triggering_measurements={
                            "mean_speech_pct": round(avg_speech, 1),
                            "duration_minutes": self.config.silence_duration_min,
                            "consecutive_windows": len(slice_3m),
                            "is_genuine_silence": True,
                        },
                        suggested_action=(
                            "If waiting for student responses, invite voluntary participation, rephrase the prompt, "
                            "or allow 60 seconds of silent writing before calling for volunteers."
                        ),
                        data_quality_limitations=(
                            "Verified genuine acoustic quietness. The system cannot determine whether students are "
                            "engaged in deep contemplation, independent reading, or experiencing difficulty."
                        ),
                        confidence_score=avg_quality,
                    )
                    triggered.append(sugg)
                    self.last_trigger_time[rule_2] = now

        # =========================================================================
        # Rule 3: Topic Transition (Conduct a Short Recap)
        # =========================================================================
        rule_3 = RuleCode.CONDUCT_RECAP.value
        if len(recent_windows) >= (ten_min_windows + three_min_windows) and not self._is_on_cooldown(rule_3, now):
            # Prior presentation block followed immediately by an activity lull
            prior_slice = recent_windows[-(ten_min_windows + three_min_windows) : -three_min_windows]
            current_slice = recent_windows[-three_min_windows:]

            prior_speeches = [w.speech_activity_pct for w in prior_slice if w.speech_activity_pct is not None]
            curr_speeches = [w.speech_activity_pct for w in current_slice if w.speech_activity_pct is not None]

            if len(prior_speeches) == len(prior_slice) and len(curr_speeches) == len(current_slice):
                prior_avg = float(np.mean(prior_speeches))
                curr_avg = float(np.mean(curr_speeches))

                # Rapid phase transition from continuous talk to quiet
                if (
                    prior_avg >= self.config.recap_prior_speech_min
                    and curr_avg <= self.config.recap_current_speech_max
                ):
                    sugg = Suggestion(
                        suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
                        session_id=session_id,
                        timestamp_sec=now,
                        trigger_rule_code=rule_3,
                        suggestion_text=(
                            "Transition detected following prolonged presentation. "
                            "Consider conducting a short recap of core concepts before moving to the next section."
                        ),
                        title="Phase Shift: Conduct a Short Recap",
                        explanation=(
                            f"Speech activity dropped from {prior_avg:.1f}% during previous presentation block "
                            f"to {curr_avg:.1f}% across recent windows, indicating a natural topic transition."
                        ),
                        triggering_measurements={
                            "prior_speech_pct": round(prior_avg, 1),
                            "current_speech_pct": round(curr_avg, 1),
                            "consecutive_windows": len(current_slice),
                        },
                        suggested_action="Conduct a 2-minute synthesis recap of the 2-3 essential takeaways before beginning the next topic.",
                        data_quality_limitations="Acoustic dynamics indicate phase transition; does not assess concept mastery.",
                        confidence_score=0.90,
                    )
                    triggered.append(sugg)
                    self.last_trigger_time[rule_3] = now

        # =========================================================================
        # Rule 4: Visual Restlessness Spike (Consider Standup Stretch / Transition)
        # =========================================================================
        rule_4 = RuleCode.RESTLESSNESS_SPIKE.value
        if len(slice_3m) >= three_min_windows and not self._is_on_cooldown(rule_4, now):
            valid_motions = [w.visual_movement_index for w in slice_3m if w.visual_movement_index is not None]
            valid_speeches = [w.speech_activity_pct for w in slice_3m if w.speech_activity_pct is not None]

            if len(valid_motions) == len(slice_3m):
                avg_motion = float(np.mean(valid_motions))
                avg_speech = float(np.mean(valid_speeches)) if valid_speeches else 0.0

                if (
                    avg_motion >= self.config.restlessness_movement_min
                    and avg_speech <= self.config.restlessness_speech_max
                ):
                    avg_quality = float(np.mean([w.data_quality_score for w in slice_3m]))
                    sugg = Suggestion(
                        suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
                        session_id=session_id,
                        timestamp_sec=now,
                        trigger_rule_code=rule_4,
                        suggestion_text=(
                            "Elevated physical movement detected during quiet time. "
                            "Students may have completed an assigned task or may benefit from a 1-minute stretch."
                        ),
                        title="Movement Shift: Consider Quick Standup Stretch",
                        explanation=(
                            f"Visual movement averaged {avg_motion:.1f}% (threshold: {self.config.restlessness_movement_min}%) "
                            f"across {len(slice_3m)} consecutive windows while speech remained quiet ({avg_speech:.1f}%)."
                        ),
                        triggering_measurements={
                            "mean_visual_movement": round(avg_motion, 1),
                            "mean_speech_pct": round(avg_speech, 1),
                            "consecutive_windows": len(slice_3m),
                        },
                        suggested_action="Check if students have finished an exercise, or offer a 60-second physical stretch or topic transition.",
                        data_quality_limitations=(
                            "Measures optical pixel displacement across the classroom. Does not identify specific "
                            "individuals or infer student mood, emotion, or restlessness cause."
                        ),
                        confidence_score=avg_quality,
                    )
                    triggered.append(sugg)
                    self.last_trigger_time[rule_4] = now

        # =========================================================================
        # Rule 5: Check Microphone Placement or Audio Quality
        # =========================================================================
        rule_5 = RuleCode.CHECK_AUDIO_QUALITY.value
        if len(slice_hw) >= hw_warning_windows and not self._is_on_cooldown(rule_5, now):
            audio_missing = all(not w.audio_available for w in slice_hw)
            audio_degraded = all(
                w.data_quality_label in (DataQualityLabel.DEGRADED_SAMPLE_RATE.value, DataQualityLabel.PARTIAL_VISUAL_ONLY.value)
                for w in slice_hw
            )

            if audio_missing or audio_degraded:
                sugg = Suggestion(
                    suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
                    session_id=session_id,
                    timestamp_sec=now,
                    trigger_rule_code=rule_5,
                    suggestion_text=(
                        "Microphone input is unavailable or experiencing persistent buffer drops. "
                        "Check microphone placement, connection, or audio permissions."
                    ),
                    title="Hardware Diagnostic: Check Microphone Placement & Quality",
                    explanation=(
                        f"Acoustic stream was unavailable or degraded across {len(slice_hw)} consecutive windows. "
                        "Analytics are currently operating in visual-only mode."
                    ),
                    triggering_measurements={
                        "audio_available": not audio_missing,
                        "consecutive_windows": len(slice_hw),
                        "data_quality_label": slice_hw[-1].data_quality_label,
                    },
                    suggested_action="Check microphone placement, verify USB connection or OS permissions, or position the microphone closer to speakers.",
                    data_quality_limitations="No acoustic features can be computed during audio sensor disconnection.",
                    confidence_score=1.0,
                )
                triggered.append(sugg)
                self.last_trigger_time[rule_5] = now

        # =========================================================================
        # Rule 6: Review Camera Angle & Placement
        # =========================================================================
        rule_6 = RuleCode.CHECK_CAMERA_ANGLE.value
        if len(slice_hw) >= hw_warning_windows and not self._is_on_cooldown(rule_6, now):
            camera_missing = all(not w.visual_available for w in slice_hw)
            zero_headcount = all(
                (w.active_person_count == 0) for w in slice_hw if w.active_person_count is not None
            )

            if camera_missing or (len(slice_hw) >= hw_warning_windows and zero_headcount):
                sugg = Suggestion(
                    suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
                    session_id=session_id,
                    timestamp_sec=now,
                    trigger_rule_code=rule_6,
                    suggestion_text=(
                        "Zero attendees detected or video feed unavailable across consecutive windows. "
                        "Review current camera angle, field of view, or lighting contrast."
                    ),
                    title="Hardware Diagnostic: Review Camera Angle & Lighting",
                    explanation=(
                        f"Optical detection recorded zero human subjects or missing video frames across {len(slice_hw)} "
                        "consecutive windows during an active session."
                    ),
                    triggering_measurements={
                        "visual_available": not camera_missing,
                        "mean_headcount": 0,
                        "consecutive_windows": len(slice_hw),
                    },
                    suggested_action="Adjust camera tilt or field of view to encompass seating area, check for strong window glare or lens obstruction.",
                    data_quality_limitations="Centroid tracking requires adequate lighting and unobstructed line of sight.",
                    confidence_score=1.0,
                )
                triggered.append(sugg)
                self.last_trigger_time[rule_6] = now

        # =========================================================================
        # Rule 7: Sudden Presence Drop
        # =========================================================================
        rule_7 = RuleCode.PRESENCE_DROP.value
        if len(recent_windows) >= 5 and not self._is_on_cooldown(rule_7, now):
            prev_counts = [w.active_person_count for w in recent_windows[-5:-1] if w.active_person_count is not None]
            curr_count = recent_windows[-1].active_person_count

            if len(prev_counts) >= 3 and curr_count is not None:
                baseline_count = float(np.mean(prev_counts))
                if baseline_count >= 4.0 and curr_count <= baseline_count * (1.0 - self.config.presence_drop_ratio):
                    drop_pct = round(((baseline_count - curr_count) / baseline_count) * 100.0, 1)
                    sugg = Suggestion(
                        suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
                        session_id=session_id,
                        timestamp_sec=now,
                        trigger_rule_code=rule_7,
                        suggestion_text=(
                            f"Significant drop in detected attendees (-{drop_pct}%). "
                            "Verify camera field of view or check if students stepped out for a break."
                        ),
                        title="Attendance Shift: Significant Presence Drop",
                        explanation=(
                            f"Detected attendees decreased from a baseline of {baseline_count:.1f} to {curr_count} "
                            f"(-{drop_pct}%) in the latest observation window."
                        ),
                        triggering_measurements={
                            "baseline_headcount": round(baseline_count, 1),
                            "current_headcount": curr_count,
                            "drop_pct": drop_pct,
                        },
                        suggested_action="Confirm camera was not bumped or obstructed; verify if class has transitioned to hallway/lab breakout.",
                        data_quality_limitations="Centroid loss can be caused by physical occlusion or moving out of webcam field of view.",
                        confidence_score=0.85,
                    )
                    triggered.append(sugg)
                    self.last_trigger_time[rule_7] = now

        # =========================================================================
        # Rule 8: Continue Current Activity (Stable Dynamics)
        # =========================================================================
        rule_8 = RuleCode.MAINTAIN_ACTIVITY.value
        # Only evaluate stability if no other pedagogical alerts fired in this cycle
        if len(triggered) == 0 and len(slice_stable) >= stable_windows and not self._is_on_cooldown(rule_8, now):
            valid_motions = [w.visual_movement_index for w in slice_stable if w.visual_movement_index is not None]
            valid_speeches = [w.speech_activity_pct for w in slice_stable if w.speech_activity_pct is not None]

            if len(valid_motions) == len(slice_stable) and len(valid_speeches) == len(slice_stable):
                motion_std = float(np.std(valid_motions))
                speech_std = float(np.std(valid_speeches))
                mean_speech = float(np.mean(valid_speeches))
                mean_motion = float(np.mean(valid_motions))

                # Dynamics remain steady within balanced interactive ranges (not stuck in extreme silence/monologue)
                if (
                    motion_std <= 12.0
                    and speech_std <= 20.0
                    and 20.0 <= mean_speech <= 70.0
                    and 10.0 <= mean_motion <= 60.0
                ):
                    sugg = Suggestion(
                        suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
                        session_id=session_id,
                        timestamp_sec=now,
                        trigger_rule_code=rule_8,
                        suggestion_text="Classroom dynamics and vocal pacing remain stable. Continue the current activity.",
                        title="Pacing Stable: Continue Current Activity",
                        explanation=(
                            f"Classroom activity indicators remain consistent across {len(slice_stable)} windows "
                            f"(Speech: {mean_speech:.1f}%, Motion: {mean_motion:.1f}%). Pacing is balanced."
                        ),
                        triggering_measurements={
                            "mean_speech_pct": round(mean_speech, 1),
                            "mean_visual_movement": round(mean_motion, 1),
                            "stable_windows": len(slice_stable),
                        },
                        suggested_action="Continue current instructional activity as planned; no pacing adjustments indicated.",
                        data_quality_limitations="Physical stability reflects macro group dynamics; does not measure individual student understanding.",
                        confidence_score=0.95,
                    )
                    triggered.append(sugg)
                    self.last_trigger_time[rule_8] = now

        # Persist triggered suggestions to SQLite if database is configured
        if self.db is not None:
            for s in triggered:
                try:
                    self.db.insert_teaching_suggestion(
                        suggestion_id=s.suggestion_id,
                        session_id=s.session_id,
                        timestamp_sec=s.timestamp_sec,
                        trigger_rule_code=s.trigger_rule_code,
                        suggestion_text=s.suggestion_text,
                    )
                except Exception as e:
                    logger.warning("Failed to persist suggestion to database: %s", e)

        return triggered
