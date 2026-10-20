"""
Unit tests for Feature Fusion Module.

Tests:
1. Timestamp alignment across asynchronous visual and audio streams.
2. Missing audio handling (visual-only modality).
3. Missing visual handling (audio-only modality).
4. Distinguishing genuine measured silence from unavailable audio input.
5. Handling empty inputs, degenerate windows, and empty stream fusion.
6. Aggregation across multiple consecutive time windows.
7. Configurable aggregation window durations.
8. Data quality scores and categorical quality labels.
9. Correlation-adjusted activity score calculation (avoiding collinear double-counting).
10. Structured dictionary serialization for analytics and plotting.
"""

from typing import List
import pytest

from modules.audio_processor import AudioFeatures
from modules.database import DatabaseManager
from modules.feature_fusion import (
    DataQualityLabel,
    FeatureFusionEngine,
    MultimodalWindow,
    VisualFeatures,
)


@pytest.fixture
def fusion_engine() -> FeatureFusionEngine:
    """Fixture providing a standard FeatureFusionEngine with 30s windows."""
    return FeatureFusionEngine(window_duration_sec=30)


# --- 1. Timestamp Alignment ---


def test_asynchronous_timestamp_alignment():
    """
    Test aligning video arriving at 10 Hz with audio arriving at 2 Hz into a single window.
    Both streams have distinct timestamps but overlap in the [0.0, 30.0) interval.
    """
    engine = FeatureFusionEngine(
        window_duration_sec=30,
        expected_video_fps=0.5,
        expected_audio_hz=0.2,
    )
    w_start = 100.0

    # Ingest 15 visual frames spread over 30s (0.5 fps sample rate)
    for i in range(15):
        ts = w_start + (i * 2.0)  # 100.0, 102.0, ... 128.0
        engine.record_visual_sample(
            VisualFeatures(
                timestamp=ts,
                active_person_count=10,
                visual_movement_index=20.0,
                head_pose_variance=2.5,
            )
        )

    # Ingest 6 audio chunks spread over 30s (every 5 seconds)
    for j in range(6):
        ts = w_start + (j * 5.0) + 1.0  # 101.0, 106.0, ... 126.0
        engine.record_audio_sample(
            AudioFeatures(
                rms_energy=0.03,
                zero_crossing_rate=0.08,
                spectral_centroid=1500.0,
                is_speech=True,
                speech_ratio_pct=60.0,
                timestamp=ts,
            ),
            timestamp=ts,
        )

    assert engine.is_window_ready(current_timestamp=w_start + 30.0)

    window = engine.close_and_fuse_window(session_id="sess_sync", current_timestamp=w_start + 30.0)
    assert window is not None
    assert window.session_id == "sess_sync"
    assert window.window_start_sec == 100.0
    assert window.window_end_sec == 130.0
    assert window.active_person_count == 10
    assert window.visual_movement_index == 20.0
    assert window.speech_activity_pct == 60.0
    assert window.rms_energy_mean == 0.03
    assert window.visual_sample_count == 15
    assert window.audio_sample_count == 6
    assert window.visual_available is True
    assert window.audio_available is True
    assert window.data_quality_label == DataQualityLabel.HIGH.value


# --- 2. Missing Audio (Visual Only) ---


def test_missing_audio_modality(fusion_engine: FeatureFusionEngine):
    """
    When no microphone stream is available, audio fields must be None (not 0.0)
    and audio_available must be False.
    """
    t0 = 50.0
    fusion_engine.start_window(t0)

    for i in range(5):
        fusion_engine.record_visual_sample(
            VisualFeatures(
                timestamp=t0 + i,
                active_person_count=8,
                visual_movement_index=15.0,
            )
        )

    window = fusion_engine.close_and_fuse_window(session_id="s1", current_timestamp=t0 + 30.0)
    assert window is not None
    assert window.visual_available is True
    assert window.audio_available is False
    assert window.active_person_count == 8
    assert window.visual_movement_index == 15.0
    assert window.speech_activity_pct is None
    assert window.rms_energy_mean is None
    assert window.is_genuine_silence is False
    assert window.data_quality_label == DataQualityLabel.PARTIAL_VISUAL_ONLY.value


# --- 3. Missing Visual (Audio Only) ---


def test_missing_visual_modality(fusion_engine: FeatureFusionEngine):
    """
    When no camera is available, visual fields must be None and visual_available False.
    """
    t0 = 200.0
    fusion_engine.start_window(t0)

    for i in range(4):
        fusion_engine.record_audio_sample(
            AudioFeatures(0.04, 0.05, 1200.0, True, 75.0, timestamp=t0 + i * 5),
            timestamp=t0 + i * 5,
        )

    window = fusion_engine.close_and_fuse_window(session_id="s1", current_timestamp=t0 + 30.0)
    assert window is not None
    assert window.visual_available is False
    assert window.audio_available is True
    assert window.active_person_count is None
    assert window.visual_movement_index is None
    assert window.speech_activity_pct == 75.0
    assert window.rms_energy_mean == 0.04
    assert window.data_quality_label == DataQualityLabel.PARTIAL_AUDIO_ONLY.value


# --- 4. Genuine Silence vs Missing Audio ---


def test_genuine_silence_distinguished_from_missing_audio():
    """
    Verify that an active microphone measuring 0.0% speech is flagged as is_genuine_silence=True,
    whereas a missing microphone has speech_activity_pct=None and is_genuine_silence=False.
    """
    engine = FeatureFusionEngine(window_duration_sec=30)
    t = 0.0

    # Window A: Active microphone detecting complete silence
    engine.start_window(t)
    engine.record_sample(
        active_persons=5,
        movement_index=5.0,
        audio_features=AudioFeatures(0.001, 0.01, 250.0, False, 0.0, timestamp=t),
        timestamp=t,
    )
    window_silent = engine.close_and_fuse_window(session_id="test", current_timestamp=t + 30.0)
    assert window_silent is not None
    assert window_silent.audio_available is True
    assert window_silent.speech_activity_pct == 0.0
    assert window_silent.is_genuine_silence is True

    # Window B: Missing microphone
    t += 30.0
    engine.start_window(t)
    engine.record_sample(
        active_persons=5,
        movement_index=5.0,
        audio_features=None,  # No audio stream
        timestamp=t,
    )
    window_no_audio = engine.close_and_fuse_window(session_id="test", current_timestamp=t + 30.0)
    assert window_no_audio is not None
    assert window_no_audio.audio_available is False
    assert window_no_audio.speech_activity_pct is None
    assert window_no_audio.is_genuine_silence is False


# --- 5. Empty Inputs and Degenerate Windows ---


def test_empty_window_closing_returns_none(fusion_engine: FeatureFusionEngine):
    """Closing a window with zero samples must return None without raising an exception."""
    fusion_engine.start_window(0.0)
    assert fusion_engine.close_and_fuse_window("sess", 30.0) is None


def test_fuse_streams_empty_input(fusion_engine: FeatureFusionEngine):
    """fuse_streams with empty visual and audio lists must return an empty list."""
    windows = fusion_engine.fuse_streams("sess", [], [])
    assert windows == []


# --- 6. Aggregation Across Multiple Time Windows ---


def test_aggregation_across_multiple_time_windows():
    """
    Feed a multi-minute timeseries and verify that fuse_streams partitions
    samples into consecutive non-overlapping windows.
    """
    engine = FeatureFusionEngine(window_duration_sec=20)
    session_id = "multi_win_sess"

    # 60-second timeseries -> three 20s windows: [0, 20), [20, 40), [40, 60)
    visual_samples: List[VisualFeatures] = []
    audio_samples: List[AudioFeatures] = []

    # Window 1: High movement, moderate speech
    for t in range(0, 20, 2):
        visual_samples.append(VisualFeatures(float(t), 12, 45.0, 4.0))
        audio_samples.append(AudioFeatures(0.03, 0.05, 1200.0, True, 50.0, timestamp=float(t)))

    # Window 2: Low movement, high speech (monologue)
    for t in range(20, 40, 2):
        visual_samples.append(VisualFeatures(float(t), 12, 10.0, 1.0))
        audio_samples.append(AudioFeatures(0.05, 0.08, 1800.0, True, 90.0, timestamp=float(t)))

    # Window 3: Quiet independent study (low speech, low movement)
    for t in range(40, 60, 2):
        visual_samples.append(VisualFeatures(float(t), 12, 5.0, 0.5))
        audio_samples.append(AudioFeatures(0.001, 0.01, 300.0, False, 0.0, timestamp=float(t)))

    windows = engine.fuse_streams(
        session_id=session_id,
        visual_samples=visual_samples,
        audio_samples=audio_samples,
        window_duration_sec=20,
    )

    assert len(windows) == 3

    # Check Window 1
    assert windows[0].window_start_sec == 0.0
    assert windows[0].window_end_sec == 20.0
    assert windows[0].visual_movement_index == 45.0
    assert windows[0].speech_activity_pct == 50.0

    # Check Window 2
    assert windows[1].window_start_sec == 20.0
    assert windows[1].window_end_sec == 40.0
    assert windows[1].visual_movement_index == 10.0
    assert windows[1].speech_activity_pct == 90.0

    # Check Window 3
    assert windows[2].window_start_sec == 40.0
    assert windows[2].window_end_sec == 60.0
    assert windows[2].visual_movement_index == 5.0
    assert windows[2].speech_activity_pct == 0.0
    assert windows[2].is_genuine_silence is True


# --- 7. Configurable Aggregation Window Durations ---


@pytest.mark.parametrize("duration_sec", [5, 10, 15, 60])
def test_configurable_window_duration(duration_sec: int):
    """Engine must respect configured window duration for readiness and duration calculations."""
    engine = FeatureFusionEngine(window_duration_sec=duration_sec)
    engine.start_window(0.0)
    engine.record_sample(active_persons=3, movement_index=10.0, timestamp=1.0)

    assert not engine.is_window_ready(current_timestamp=duration_sec - 1.0)
    assert engine.is_window_ready(current_timestamp=duration_sec + 0.1)

    window = engine.close_and_fuse_window(session_id="test_cfg", current_timestamp=float(duration_sec))
    assert window is not None
    assert (window.window_end_sec - window.window_start_sec) == pytest.approx(duration_sec)


# --- 8. Non-Redundant Correlation-Adjusted Activity Score ---


def test_collinear_features_not_treated_as_independent_additive_evidence():
    """
    Test that simultaneously high acoustic RMS and speech ratio do not linearly explode.
    Speech and movement combined should saturate smoothly rather than double-counting.
    """
    engine = FeatureFusionEngine()

    # Case A: Speech only (80% speech, 0.04 RMS, 0 motion)
    score_speech_only = engine.compute_composite_activity_score(
        visual_movement=0.0,
        head_pose_variance=0.0,
        speech_activity_pct=80.0,
        rms_energy=0.04,
    )
    assert 50.0 <= score_speech_only <= 75.0

    # Case B: Motion only (40 motion, 0 speech)
    score_motion_only = engine.compute_composite_activity_score(
        visual_movement=40.0,
        head_pose_variance=0.0,
        speech_activity_pct=0.0,
        rms_energy=0.0,
    )
    assert 50.0 <= score_motion_only <= 75.0

    # Case C: Joint motion AND speech
    score_joint = engine.compute_composite_activity_score(
        visual_movement=40.0,
        head_pose_variance=0.0,
        speech_activity_pct=80.0,
        rms_energy=0.04,
    )
    # Bounded combination should be strictly greater than individual scores, but bounded <= 100
    assert score_joint > score_speech_only
    assert score_joint > score_motion_only
    assert score_joint <= 100.0
    # Naive addition would have been score_speech_only + score_motion_only (> 120), which is prevented
    assert score_joint < (score_speech_only + score_motion_only)


# --- 9. Structured Dictionary & Serialization ---


def test_multimodal_window_to_dict():
    """Verify dictionary serialization contains expected keys for Plotly and export."""
    window = MultimodalWindow(
        window_id="win_test_0",
        session_id="sess_123",
        window_start_sec=0.0,
        window_end_sec=30.0,
        active_person_count=15,
        visual_movement_index=22.4,
        head_pose_variance=3.1,
        speech_activity_pct=45.0,
        rms_energy_mean=0.025,
        visual_available=True,
        audio_available=True,
        visual_sample_count=150,
        audio_sample_count=30,
        data_quality_score=0.95,
        data_quality_label=DataQualityLabel.HIGH.value,
        is_genuine_silence=False,
        correlation_adjusted_activity_score=58.2,
    )

    d = window.to_dict()
    assert d["window_id"] == "win_test_0"
    assert d["headcount"] == 15
    assert d["duration_sec"] == 30.0
    assert d["visual_available"] is True
    assert d["audio_available"] is True
    assert d["composite_activity_score"] == 58.2
    assert d["data_quality_label"] == "HIGH"


# --- 10. Database Persistence with Quality Indicators ---


def test_database_persistence_with_quality_indicators(tmp_path):
    """Verify that fused windows persist to SQLite with data quality columns."""
    db_file = tmp_path / "test_fusion.db"
    db = DatabaseManager(db_path=db_file)
    db.init_db()

    # Create session
    db.create_session("sess_db_test", "Math 101", "Dr. Alan", video_source_type="WEBCAM")

    engine = FeatureFusionEngine(window_duration_sec=30, db_manager=db)
    engine.start_window(0.0)
    engine.record_sample(active_persons=10, movement_index=25.0, timestamp=5.0)

    window = engine.close_and_fuse_window("sess_db_test", 30.0)
    assert window is not None

    # Retrieve from DB
    persisted = db.get_activity_windows("sess_db_test")
    assert len(persisted) == 1
    row = persisted[0]
    assert row["window_id"] == window.window_id
    assert row["active_person_count"] == 10
    assert row["visual_movement_index"] == 25.0
    assert row["data_quality_label"] == DataQualityLabel.PARTIAL_VISUAL_ONLY.value


def test_degraded_sample_rate_detection():
    """Verify that severe frame drops produce DEGRADED_SAMPLE_RATE label."""
    # Expect 10 fps (150 frames over 30s) but only 2 frames arrive
    engine = FeatureFusionEngine(window_duration_sec=30, expected_video_fps=10.0, expected_audio_hz=2.0)
    engine.start_window(0.0)
    engine.record_visual_sample(VisualFeatures(1.0, 5, 10.0))
    engine.record_visual_sample(VisualFeatures(2.0, 5, 12.0))
    engine.record_audio_sample(AudioFeatures(0.02, 0.05, 1000.0, True, 50.0, timestamp=1.0))

    window = engine.close_and_fuse_window("sess_drop", 30.0)
    assert window is not None
    assert window.data_quality_label == DataQualityLabel.DEGRADED_SAMPLE_RATE.value
    assert window.data_quality_score < 0.7
