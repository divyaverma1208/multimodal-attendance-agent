"""
Unit Tests for Recommendation Engine Deterministic Pedagogical Rules.

Verifies:
1. RULE_PROLONGED_MONOLOGUE: Continuous high speech triggers Q&A suggestion.
2. RULE_EXTENDED_SILENCE: Active audio with genuine silence triggers participation cue.
3. RULE_CONDUCT_RECAP: Rapid phase shift from presentation to lull triggers recap cue.
4. RULE_RESTLESSNESS_SPIKE: High physical movement during quiet time triggers stretch cue.
5. RULE_CHECK_AUDIO_QUALITY: Missing or degraded audio stream triggers microphone diagnostic.
6. RULE_CHECK_CAMERA_ANGLE: Missing visual or persistent zero headcount triggers camera diagnostic.
7. RULE_PRESENCE_DROP: Sudden drop in detected attendees triggers presence review.
8. RULE_MAINTAIN_ACTIVITY: Stable, balanced activity triggers continue activity cue.
9. Cooldown behavior: Verifies cooldown suppresses repetitive alert spam.
10. Immunity to single-window noise: Single spikes or short horizons do not trigger rules.
11. Structured schema verification: Verifies title, explanation, triggering measurements,
    suggested teacher action, and data-quality limitations are all present.
"""

import time
from typing import List
import pytest

from modules.database import DatabaseManager
from modules.feature_fusion import DataQualityLabel, MultimodalWindow
from modules.recommendation_engine import (
    RecommendationEngine,
    RecommendationEngineConfig,
    RuleCode,
    Suggestion,
)


@pytest.fixture
def engine() -> RecommendationEngine:
    """Fixture providing a default RecommendationEngine instance."""
    return RecommendationEngine()


def make_window(
    index: int,
    start_sec: float,
    duration: float = 30.0,
    headcount: int = 15,
    visual_movement: float = 15.0,
    speech_pct: float = 50.0,
    rms: float = 0.02,
    visual_available: bool = True,
    audio_available: bool = True,
    is_genuine_silence: bool = False,
    data_quality_score: float = 1.0,
    data_quality_label: str = DataQualityLabel.HIGH.value,
) -> MultimodalWindow:
    """Helper to generate parameterized MultimodalWindow instances."""
    return MultimodalWindow(
        window_id=f"win_{index}",
        session_id="test_sess",
        window_start_sec=start_sec,
        window_end_sec=start_sec + duration,
        active_person_count=headcount,
        visual_movement_index=visual_movement,
        head_pose_variance=1.5,
        speech_activity_pct=speech_pct,
        rms_energy_mean=rms,
        visual_available=visual_available,
        audio_available=audio_available,
        is_genuine_silence=is_genuine_silence,
        data_quality_score=data_quality_score,
        data_quality_label=data_quality_label,
    )


# --- 1. Rule 1: Prolonged Monologue ---


def test_rule_prolonged_monologue_trigger(engine: RecommendationEngine) -> None:
    """Test that prolonged high speech (>=10 mins) triggers RULE_PROLONGED_MONOLOGUE."""
    now = 1000.0
    # 21 windows of 30s = 10.5 minutes
    windows = [
        make_window(i, start_sec=now + i * 30.0, visual_movement=10.0, speech_pct=88.0)
        for i in range(21)
    ]

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 650.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    assert RuleCode.PROLONGED_MONOLOGUE.value in rule_codes

    monologue_sugg = next(s for s in suggestions if s.trigger_rule_code == RuleCode.PROLONGED_MONOLOGUE.value)
    assert monologue_sugg.title == "Continuous Presentation: Introduce Q&A"
    assert "Continuous presentation" in monologue_sugg.explanation
    assert "mean_speech_pct" in monologue_sugg.triggering_measurements
    assert monologue_sugg.triggering_measurements["mean_speech_pct"] == 88.0
    assert "question-and-answer activity" in monologue_sugg.suggested_action
    assert len(monologue_sugg.data_quality_limitations) > 0


# --- 2. Rule 2: Extended Silence ---


def test_rule_extended_silence_trigger(engine: RecommendationEngine) -> None:
    """Test that active audio with genuine silence (>=3 mins) triggers RULE_EXTENDED_SILENCE."""
    now = 1000.0
    # 7 windows of silence (3.5 mins)
    windows = [
        make_window(
            i,
            start_sec=now + i * 30.0,
            visual_movement=12.0,
            speech_pct=0.0,
            is_genuine_silence=True,
        )
        for i in range(7)
    ]

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 220.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    assert RuleCode.EXTENDED_SILENCE.value in rule_codes

    silence_sugg = next(s for s in suggestions if s.trigger_rule_code == RuleCode.EXTENDED_SILENCE.value)
    assert silence_sugg.title == "Extended Silence: Invite Voluntary Participation"
    assert "voluntary participation" in silence_sugg.suggested_action
    assert silence_sugg.triggering_measurements["is_genuine_silence"] is True


def test_rule_extended_silence_not_triggered_by_missing_audio(engine: RecommendationEngine) -> None:
    """Missing microphone stream must NOT trigger extended silence (it triggers hardware check)."""
    now = 1000.0
    # Audio unavailable (not genuine silence)
    windows = [
        make_window(
            i,
            start_sec=now + i * 30.0,
            visual_movement=12.0,
            speech_pct=0.0,
            audio_available=False,
            is_genuine_silence=False,
        )
        for i in range(7)
    ]

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 220.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    # Extended silence must NOT fire
    assert RuleCode.EXTENDED_SILENCE.value not in rule_codes
    # Instead, hardware audio check fires
    assert RuleCode.CHECK_AUDIO_QUALITY.value in rule_codes


# --- 3. Rule 3: Conduct a Short Recap ---


def test_rule_conduct_recap_trigger(engine: RecommendationEngine) -> None:
    """Test that a transition from high presentation to a lull triggers RULE_CONDUCT_RECAP."""
    now = 1000.0
    windows: List[MultimodalWindow] = []

    # 20 windows of high presentation (10 minutes)
    for i in range(20):
        windows.append(
            make_window(len(windows), start_sec=now + len(windows) * 30.0, speech_pct=85.0)
        )
    # Followed by 6 windows of low speech (3 minutes of transition)
    for i in range(6):
        windows.append(
            make_window(len(windows), start_sec=now + len(windows) * 30.0, speech_pct=10.0)
        )

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 800.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    assert RuleCode.CONDUCT_RECAP.value in rule_codes

    recap_sugg = next(s for s in suggestions if s.trigger_rule_code == RuleCode.CONDUCT_RECAP.value)
    assert recap_sugg.title == "Phase Shift: Conduct a Short Recap"
    assert "prior_speech_pct" in recap_sugg.triggering_measurements
    assert "current_speech_pct" in recap_sugg.triggering_measurements
    assert "recap of the 2-3 essential takeaways" in recap_sugg.suggested_action


# --- 4. Rule 4: Visual Restlessness Spike ---


def test_rule_restlessness_spike_trigger(engine: RecommendationEngine) -> None:
    """Test elevated physical motion (>=70) during quiet time triggers RULE_RESTLESSNESS_SPIKE."""
    now = 1000.0
    windows = [
        make_window(
            i,
            start_sec=now + i * 30.0,
            visual_movement=78.0,
            speech_pct=5.0,
        )
        for i in range(7)
    ]

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 220.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    assert RuleCode.RESTLESSNESS_SPIKE.value in rule_codes

    restless_sugg = next(s for s in suggestions if s.trigger_rule_code == RuleCode.RESTLESSNESS_SPIKE.value)
    assert restless_sugg.title == "Movement Shift: Consider Quick Standup Stretch"
    assert restless_sugg.triggering_measurements["mean_visual_movement"] == 78.0
    assert "stretch" in restless_sugg.suggested_action


# --- 5. Rule 5: Check Microphone Placement & Quality ---


def test_rule_check_audio_quality_trigger(engine: RecommendationEngine) -> None:
    """Test that missing or degraded audio triggers RULE_CHECK_AUDIO_QUALITY."""
    now = 1000.0
    windows = [
        make_window(
            i,
            start_sec=now + i * 30.0,
            audio_available=False,
            data_quality_label=DataQualityLabel.PARTIAL_VISUAL_ONLY.value,
        )
        for i in range(4)
    ]

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 130.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    assert RuleCode.CHECK_AUDIO_QUALITY.value in rule_codes

    audio_sugg = next(s for s in suggestions if s.trigger_rule_code == RuleCode.CHECK_AUDIO_QUALITY.value)
    assert audio_sugg.title == "Hardware Diagnostic: Check Microphone Placement & Quality"
    assert "microphone placement" in audio_sugg.suggested_action.lower()


# --- 6. Rule 6: Review Camera Angle & Placement ---


def test_rule_check_camera_angle_trigger(engine: RecommendationEngine) -> None:
    """Test that missing visual stream or zero detected headcount triggers RULE_CHECK_CAMERA_ANGLE."""
    now = 1000.0
    windows = [
        make_window(
            i,
            start_sec=now + i * 30.0,
            headcount=0,
            visual_movement=0.0,
            speech_pct=40.0,
        )
        for i in range(4)
    ]

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 130.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    assert RuleCode.CHECK_CAMERA_ANGLE.value in rule_codes

    cam_sugg = next(s for s in suggestions if s.trigger_rule_code == RuleCode.CHECK_CAMERA_ANGLE.value)
    assert cam_sugg.title == "Hardware Diagnostic: Review Camera Angle & Lighting"
    assert "camera tilt" in cam_sugg.suggested_action.lower()


# --- 7. Rule 7: Sudden Presence Drop ---


def test_rule_presence_drop_trigger(engine: RecommendationEngine) -> None:
    """Test that sudden headcount drop (>30%) triggers RULE_PRESENCE_DROP."""
    now = 1000.0
    # 4 windows of 20 students, followed by 1 window of 10 students (50% drop)
    windows = [
        make_window(0, now + 0.0, headcount=20),
        make_window(1, now + 30.0, headcount=20),
        make_window(2, now + 60.0, headcount=20),
        make_window(3, now + 90.0, headcount=20),
        make_window(4, now + 120.0, headcount=10),
    ]

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 130.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    assert RuleCode.PRESENCE_DROP.value in rule_codes

    drop_sugg = next(s for s in suggestions if s.trigger_rule_code == RuleCode.PRESENCE_DROP.value)
    assert drop_sugg.title == "Attendance Shift: Significant Presence Drop"
    assert drop_sugg.triggering_measurements["drop_pct"] == 50.0


# --- 8. Rule 8: Maintain Activity (Stable Dynamics) ---


def test_rule_maintain_activity_trigger(engine: RecommendationEngine) -> None:
    """Test that sustained balanced activity triggers RULE_MAINTAIN_ACTIVITY."""
    now = 1000.0
    # 12 windows of balanced activity (speech ~45%, motion ~25%)
    windows = [
        make_window(
            i,
            start_sec=now + i * 30.0,
            visual_movement=25.0 + (i % 3),
            speech_pct=45.0 + (i % 4),
        )
        for i in range(12)
    ]

    suggestions = engine.evaluate_rules("test_sess", windows, current_time=now + 380.0)

    rule_codes = [s.trigger_rule_code for s in suggestions]
    assert RuleCode.MAINTAIN_ACTIVITY.value in rule_codes

    stable_sugg = next(s for s in suggestions if s.trigger_rule_code == RuleCode.MAINTAIN_ACTIVITY.value)
    assert stable_sugg.title == "Pacing Stable: Continue Current Activity"
    assert "Continue current instructional activity" in stable_sugg.suggested_action


# --- 9. Cooldown Suppression ---


def test_rule_cooldown_prevents_spam(engine: RecommendationEngine) -> None:
    """Test that rule trigger cooldown prevents duplicate alert spamming."""
    now = 1000.0
    windows = [
        make_window(i, start_sec=now + i * 30.0, speech_pct=0.0, is_genuine_silence=True)
        for i in range(7)
    ]

    first_run = engine.evaluate_rules("sess_3", windows, current_time=now + 210.0)
    assert len(first_run) >= 1

    # Immediate second evaluation should be suppressed by cooldown
    second_run = engine.evaluate_rules("sess_3", windows, current_time=now + 220.0)
    assert len(second_run) == 0

    # Once cooldown expires (e.g. +310s), it can trigger again
    third_run = engine.evaluate_rules("sess_3", windows, current_time=now + 525.0)
    assert len(third_run) >= 1


# --- 10. Single Noisy Window Immunity ---


def test_single_noisy_window_does_not_trigger_rules(engine: RecommendationEngine) -> None:
    """A single noisy spike or short sequence must never trigger pedagogical rules."""
    now = 1000.0
    # Only 1 window of extreme motion (restlessness) or monologue
    single_window = [make_window(0, now, visual_movement=95.0, speech_pct=95.0)]

    suggestions = engine.evaluate_rules("sess_noise", single_window, current_time=now + 30.0)
    assert len(suggestions) == 0


# --- 11. Structured Schema & Dictionary Serialization ---


def test_suggestion_structured_schema_serialization() -> None:
    """Verify that Suggestion to_dict includes all 5 required fields."""
    sugg = Suggestion(
        suggestion_id="sug_123",
        session_id="sess_abc",
        timestamp_sec=120.0,
        trigger_rule_code=RuleCode.PROLONGED_MONOLOGUE.value,
        suggestion_text="Consider pausing for Q&A.",
        title="Continuous Presentation: Introduce Q&A",
        explanation="Detected high speech activity for 10 minutes.",
        triggering_measurements={"speech_pct": 85.0},
        suggested_action="Introduce Q&A poll.",
        data_quality_limitations="Single-channel audio limitation.",
        confidence_score=0.95,
    )

    d = sugg.to_dict()
    assert d["title"] == "Continuous Presentation: Introduce Q&A"
    assert d["explanation"] == "Detected high speech activity for 10 minutes."
    assert d["triggering_measurements"] == {"speech_pct": 85.0}
    assert d["suggested_action"] == "Introduce Q&A poll."
    assert d["data_quality_limitations"] == "Single-channel audio limitation."
    assert d["confidence_score"] == 0.95


# --- 12. Database Persistence Integration ---


def test_suggestions_persist_to_sqlite(tmp_path) -> None:
    """Verify triggered suggestions are written to SQLite teaching_suggestions table."""
    db_file = tmp_path / "test_sugg.db"
    db = DatabaseManager(db_path=db_file)
    db.init_db()
    db.create_session("sess_db", "CS 101", "Prof Smith", video_source_type="WEBCAM")

    engine = RecommendationEngine(db_manager=db)
    now = 1000.0
    windows = [
        make_window(i, start_sec=now + i * 30.0, speech_pct=0.0, is_genuine_silence=True)
        for i in range(7)
    ]

    suggestions = engine.evaluate_rules("sess_db", windows, current_time=now + 220.0)
    assert len(suggestions) >= 1

    persisted = db.get_teaching_suggestions("sess_db")
    assert len(persisted) >= 1
    assert persisted[0]["trigger_rule_code"] == RuleCode.EXTENDED_SILENCE.value
    assert len(persisted[0]["suggestion_text"]) > 0
