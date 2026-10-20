"""
Unit Tests for Engagement Analytics and Feature Fusion.

Verifies:
1. Sub-second sample buffering and window aggregation.
2. Normal inputs summary KPI calculation.
3. Missing modalities handling without zero-interpretation.
4. Constant signals (verifying delta=0, STABLE classification, no false trends).
5. Empty sessions handling without crashing.
6. Out-of-range value sanitization (negative and >100 clamped safely).
7. Incorrect and out-of-order timestamps sorting and normalization.
8. Configurable feature weights shifting composite activity predictably.
9. Sustained increase and decrease trend detection.
10. Return of both numeric metrics and documented calculation explanations.
"""

import time
import pytest

from modules.audio_processor import AudioFeatures
from modules.engagement_analytics import (
    ActivityWeights,
    EngagementAnalytics,
    SessionAnalyticsSummary,
    TrendEvent,
)
from modules.feature_fusion import FeatureFusionEngine, MultimodalWindow


def test_feature_fusion_window_aggregation() -> None:
    """Test sub-second sample buffering and window aggregation."""
    engine = FeatureFusionEngine(window_duration_sec=30)
    now = time.time()
    engine.start_window(now)

    # Ingest 3 samples
    engine.record_sample(
        active_persons=5,
        movement_index=20.0,
        audio_features=AudioFeatures(0.02, 0.04, 300.0, True, 60.0),
        timestamp=now + 5.0,
    )
    engine.record_sample(
        active_persons=7,
        movement_index=30.0,
        audio_features=AudioFeatures(0.04, 0.03, 350.0, True, 80.0),
        timestamp=now + 15.0,
    )
    engine.record_sample(
        active_persons=6,
        movement_index=25.0,
        audio_features=AudioFeatures(0.01, 0.05, 250.0, False, 10.0),
        timestamp=now + 25.0,
    )

    assert engine.is_window_ready(now + 31.0) is True

    window = engine.close_and_fuse_window(session_id="test_sess", current_timestamp=now + 31.0)
    assert window is not None
    assert window.active_person_count == 6  # Mean of [5, 7, 6]
    assert window.visual_movement_index == 25.0  # Mean of [20, 30, 25]
    assert window.speech_activity_pct == 50.0  # Mean of [60, 80, 10]


def test_engagement_analytics_summarization() -> None:
    """Test summary KPI calculation from multiple multimodal windows."""
    w1 = MultimodalWindow("w1", "s1", 0.0, 30.0, 10, 15.0, 2.0, 90.0, 0.03)  # monologue
    w2 = MultimodalWindow("w2", "s1", 30.0, 60.0, 12, 40.0, 5.0, 50.0, 0.02) # discussion
    w3 = MultimodalWindow("w3", "s1", 60.0, 90.0, 11, 10.0, 1.0, 0.0, 0.001) # silence

    summary = EngagementAnalytics.summarize_windows([w1, w2, w3])
    assert summary.total_windows == 3
    assert summary.avg_headcount == 11.0
    assert summary.peak_headcount == 12
    assert summary.estimated_monologue_minutes == 0.5
    assert summary.estimated_discussion_minutes == 0.5
    assert summary.silent_period_minutes == 0.5
    assert isinstance(summary.metric_explanations, dict)
    assert "visual_movement_index" in summary.metric_explanations


def test_engagement_analytics_dataframe() -> None:
    """Test DataFrame conversion, relative timeline minutes, and delta columns."""
    w1 = MultimodalWindow("w1", "s1", 100.0, 130.0, 8, 20.0, 1.0, 45.0, 0.02)
    w2 = MultimodalWindow("w2", "s1", 130.0, 160.0, 9, 30.0, 1.5, 60.0, 0.03)

    df = EngagementAnalytics.to_dataframe([w1, w2])

    assert not df.empty
    assert "timeline_min" in df.columns
    assert "composite_activity" in df.columns
    assert "activity_delta" in df.columns
    assert "trend_direction" in df.columns
    assert df["timeline_min"].iloc[0] == 0.0
    assert df["timeline_min"].iloc[1] == 0.5
    assert df["headcount"].iloc[0] == 8
    assert df["headcount"].iloc[1] == 9
    assert df["activity_delta"].iloc[0] == 0.0  # Initial delta is zero
    assert df["activity_delta"].iloc[1] > 0.0  # Increased from w1 to w2


# --- New Comprehensive Feature Tests ---


def test_missing_modalities_reweights_composite() -> None:
    """
    Missing modalities must not be interpreted as zero.
    When audio is missing, composite activity should dynamically re-weight to visual.
    When visual is missing, composite activity should dynamically re-weight to audio.
    """
    # Visual-only window: motion = 40.0, audio missing (None)
    w_vis_only = MultimodalWindow("w1", "s1", 0.0, 30.0, 5, 40.0, 0.0, None, None, visual_available=True, audio_available=False)

    score_vis = EngagementAnalytics.calculate_composite_activity(
        visual_movement=w_vis_only.visual_movement_index,
        speech_pct=w_vis_only.speech_activity_pct,
    )
    # Re-weighted visual should be 40.0, NOT halved to 20.0!
    assert score_vis == 40.0

    # Audio-only window: speech = 70.0, visual missing (None)
    w_aud_only = MultimodalWindow("w2", "s1", 30.0, 60.0, None, None, None, 70.0, 0.03, visual_available=False, audio_available=True)

    score_aud = EngagementAnalytics.calculate_composite_activity(
        visual_movement=w_aud_only.visual_movement_index,
        speech_pct=w_aud_only.speech_activity_pct,
    )
    # Re-weighted audio should be 70.0, NOT halved to 35.0!
    assert score_aud == 70.0

    # Both missing: returns None
    assert EngagementAnalytics.calculate_composite_activity(None, None) is None


def test_constant_signals_produce_zero_deltas_and_stable_trend() -> None:
    """
    Constant visual and audio signals across consecutive windows must produce
    zero deltas, STABLE trend classifications, and zero false sustained trends.
    """
    windows = [
        MultimodalWindow(f"w{i}", "s1", i * 30.0, (i + 1) * 30.0, 10, 25.0, 1.0, 40.0, 0.02)
        for i in range(6)
    ]

    df = EngagementAnalytics.to_dataframe(windows)
    assert not df.empty
    # From window index 1 onwards, delta must be 0.0
    for delta in df["activity_delta"].iloc[1:]:
        assert delta == 0.0
    for trend in df["trend_direction"].iloc[1:]:
        assert trend == "STABLE"

    # Sustained trends must be empty
    trends = EngagementAnalytics.detect_sustained_trends(windows)
    assert len(trends) == 0


def test_empty_session_handling() -> None:
    """Empty list of windows must return safe default summaries without exceptions."""
    summary = EngagementAnalytics.summarize_windows([])
    assert summary.total_windows == 0
    assert summary.avg_headcount == 0.0
    assert summary.peak_headcount == 0
    assert summary.mean_composite_activity == 0.0
    assert summary.estimated_monologue_minutes == 0.0
    assert len(summary.metric_explanations) > 0

    df = EngagementAnalytics.to_dataframe([])
    assert df.empty

    trends = EngagementAnalytics.detect_sustained_trends([])
    assert trends == []

    report = EngagementAnalytics.generate_analytical_report([])
    assert report["total_windows"] == 0
    assert report["summary"].total_windows == 0


def test_out_of_range_values_sanitization() -> None:
    """
    Negative values or values > 100 must be safely clamped without throwing exceptions.
    """
    # Negative values
    assert EngagementAnalytics.sanitize_metric(-20.0, 0.0, 100.0) == 0.0
    # Excessive values
    assert EngagementAnalytics.sanitize_metric(180.0, 0.0, 100.0) == 100.0
    # Non-finite values
    assert EngagementAnalytics.sanitize_metric(float("nan"), 0.0, 100.0) is None
    assert EngagementAnalytics.sanitize_metric(float("inf"), 0.0, 100.0) is None

    # Calculate composite with out-of-range values
    comp = EngagementAnalytics.calculate_composite_activity(
        visual_movement=150.0,  # Clamped to 100
        speech_pct=-10.0,       # Clamped to 0
    )
    # 50% of 100 + 50% of 0 = 50.0
    assert comp == 50.0


def test_incorrect_and_disordered_timestamps() -> None:
    """
    Windows supplied in random, disordered chronological order must be sorted
    automatically before computing timeline minutes and deltas.
    """
    w_early = MultimodalWindow("w1", "s1", 0.0, 30.0, 5, 10.0, 0.5, 20.0, 0.01)
    w_mid = MultimodalWindow("w2", "s1", 30.0, 60.0, 7, 20.0, 1.0, 40.0, 0.02)
    w_late = MultimodalWindow("w3", "s1", 60.0, 90.0, 10, 30.0, 1.5, 60.0, 0.03)

    # Supply out-of-order: w_late, w_early, w_mid
    df = EngagementAnalytics.to_dataframe([w_late, w_early, w_mid])

    # Must be sorted chronologically: 0.0, 30.0, 60.0
    assert list(df["start_sec"]) == [0.0, 30.0, 60.0]
    assert list(df["timeline_min"]) == [0.0, 0.5, 1.0]
    assert list(df["headcount"]) == [5, 7, 10]


def test_configurable_weights_impact() -> None:
    """
    Altering ActivityWeights must shift the composite score predictably.
    """
    movement = 80.0
    speech = 20.0

    # 1. Pure Visual Weight (visual=1.0, speech=0.0)
    w_vis = ActivityWeights(visual_movement=1.0, speech_activity=0.0)
    score_vis = EngagementAnalytics.calculate_composite_activity(movement, speech, weights=w_vis)
    assert score_vis == 80.0

    # 2. Pure Audio Weight (visual=0.0, speech=1.0)
    w_aud = ActivityWeights(visual_movement=0.0, speech_activity=1.0)
    score_aud = EngagementAnalytics.calculate_composite_activity(movement, speech, weights=w_aud)
    assert score_aud == 20.0

    # 3. 75/25 weighting: 0.75 * 80 + 0.25 * 20 = 60 + 5 = 65.0
    w_custom = ActivityWeights(visual_movement=3.0, speech_activity=1.0)
    score_custom = EngagementAnalytics.calculate_composite_activity(movement, speech, weights=w_custom)
    assert score_custom == 65.0

    # Invalid weights raise ValueError
    with pytest.raises(ValueError):
        ActivityWeights(visual_movement=-1.0, speech_activity=0.5)
    with pytest.raises(ValueError):
        ActivityWeights(visual_movement=0.0, speech_activity=0.0)


def test_sustained_increase_and_decrease_detection() -> None:
    """
    Verify that consecutive monotonic increases or decreases are identified
    as TrendEvents with human-interpretable explanations.
    """
    # 6 windows simulating ramp-up (activity 20 -> 30 -> 40 -> 50) then lull (50 -> 40 -> 25)
    windows = [
        MultimodalWindow("w0", "s1", 0.0, 30.0, 10, 20.0, 1.0, 20.0, 0.01),   # comp = 20
        MultimodalWindow("w1", "s1", 30.0, 60.0, 10, 30.0, 1.0, 30.0, 0.01),  # comp = 30
        MultimodalWindow("w2", "s1", 60.0, 90.0, 10, 40.0, 1.0, 40.0, 0.02),  # comp = 40
        MultimodalWindow("w3", "s1", 90.0, 120.0, 10, 50.0, 1.0, 50.0, 0.02), # comp = 50
        MultimodalWindow("w4", "s1", 120.0, 150.0, 10, 35.0, 1.0, 35.0, 0.02),# comp = 35
        MultimodalWindow("w5", "s1", 150.0, 180.0, 10, 20.0, 1.0, 20.0, 0.01),# comp = 20
    ]

    trends = EngagementAnalytics.detect_sustained_trends(
        windows,
        min_consecutive_windows=3,
        min_delta_threshold=15.0,
    )

    assert len(trends) >= 1
    ramp_up = next((t for t in trends if t.trend_type == "SUSTAINED_INCREASE"), None)
    assert ramp_up is not None
    assert ramp_up.delta_value >= 15.0
    assert "Sustained activity ramp-up" in ramp_up.explanation
    assert ramp_up.start_sec == 0.0

    summary = EngagementAnalytics.summarize_windows(windows)
    assert summary.sustained_increases_count >= 1


def test_analytical_report_generation() -> None:
    """
    generate_analytical_report must return both numerical summaries,
    trend detections, data quality breakdowns, and metric explanations.
    """
    w1 = MultimodalWindow("w1", "s1", 0.0, 30.0, 8, 20.0, 1.0, 40.0, 0.02)
    w2 = MultimodalWindow("w2", "s1", 30.0, 60.0, 8, 30.0, 1.0, 60.0, 0.03)

    report = EngagementAnalytics.generate_analytical_report([w1, w2])

    assert "summary" in report
    assert "trends" in report
    assert "data_quality_distribution" in report
    assert "metric_explanations" in report
    assert "visual_movement_index" in report["metric_explanations"]
    assert "composite_activity_index" in report["metric_explanations"]
    assert report["total_windows"] == 2
