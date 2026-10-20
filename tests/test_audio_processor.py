"""
Tests for Audio Processor Module.

Verifies:
1. Pure silence input handling.
2. Speech-like synthetic acoustic input classification.
3. Invalid audio files, unsupported formats, and non-existent paths.
4. Missing, empty, or degenerate audio inputs.
5. Ambient background noise calibration and dynamic thresholding.
6. Windowed speech duration and silence duration calculations.
7. Privacy guardrail prohibiting raw audio disk persistence by default.
8. Graceful degradation when microphone stream fails or permissions are absent.
9. Valid multi-channel and resampled audio file loading.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from modules.audio_processor import (
    AudioFeatures,
    AudioProcessor,
    AudioStatus,
    AudioWindowResult,
)


@pytest.fixture
def processor() -> AudioProcessor:
    """Fixture providing a standard AudioProcessor instance with audio-saving disabled."""
    return AudioProcessor(sample_rate=16000, allow_save_audio=False)


def generate_synthetic_speech(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    f0: float = 220.0,
) -> np.ndarray:
    """
    Generate synthetic speech-like acoustic signal with harmonic formants and periodic pulses.
    Produces typical human speech frequencies (F0=220Hz, F1=660Hz, F2=1320Hz)
    within standard speech ZCR and spectral centroid envelopes.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Fundamental and harmonics
    harmonics = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * (f0 * 3) * t)
        + 0.2 * np.sin(2 * np.pi * (f0 * 6) * t)
    )
    # Low-frequency amplitude envelope (syllabic rate ~ 4 Hz)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    speech = harmonics * envelope * 0.4
    return speech.astype(np.float32)


# --- 1. Silence Handling ---


def test_silence_input_handling(processor: AudioProcessor):
    """Zero-amplitude signal should register zero RMS, 0% speech, and is_speech=False."""
    silence = np.zeros(16000, dtype=np.float32)

    features = processor.extract_features(silence, timestamp=1.0)

    assert isinstance(features, AudioFeatures)
    assert features.rms_energy == 0.0
    assert features.zero_crossing_rate == 0.0
    assert features.speech_ratio_pct == 0.0
    assert features.is_speech is False
    assert features.timestamp == 1.0


def test_silence_window_duration(processor: AudioProcessor):
    """A window containing silence should allocate 100% of duration to silence."""
    silence = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds of silence

    window_res = processor.analyze_window(silence, window_start_sec=0.0, window_end_sec=3.0)

    assert isinstance(window_res, AudioWindowResult)
    assert window_res.window_duration_sec == 3.0
    assert window_res.speech_duration_sec == 0.0
    assert window_res.silence_duration_sec == 3.0
    assert window_res.speech_ratio_pct == 0.0
    assert window_res.is_speech is False


# --- 2. Speech-Like Audio Handling ---


def test_speech_like_audio_feature_extraction(processor: AudioProcessor):
    """Synthetic speech-like signal should exceed energy threshold and trigger VAD."""
    speech = generate_synthetic_speech(duration_sec=1.0, sample_rate=16000, f0=250.0)

    features = processor.extract_features(speech, timestamp=2.0)

    assert features.rms_energy > 0.02
    assert 0.01 <= features.zero_crossing_rate <= 0.40
    assert 200.0 <= features.spectral_centroid <= 4000.0
    assert features.speech_ratio_pct > 30.0
    assert features.is_speech is True


def test_speech_window_duration_calculation(processor: AudioProcessor):
    """Windowed speech duration should scale proportionally with active speech frames."""
    speech = generate_synthetic_speech(duration_sec=2.0, sample_rate=16000, f0=220.0)

    result = processor.analyze_window(speech, window_start_sec=10.0, window_end_sec=12.0)

    assert result.window_duration_sec == 2.0
    assert result.speech_duration_sec > 0.5
    assert result.silence_duration_sec < 1.5
    assert result.speech_duration_sec + result.silence_duration_sec == pytest.approx(2.0, abs=0.05)
    assert result.is_speech is True


# --- 3. Invalid Files and Unsupported Formats ---


def test_nonexistent_audio_file(processor: AudioProcessor):
    """Attempting to load a non-existent audio file must raise FileNotFoundError."""
    missing_path = Path("/tmp/non_existent_audio_test_12345.wav")
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        processor.load_audio_file(missing_path)


def test_unsupported_file_extension(processor: AudioProcessor, tmp_path: Path):
    """Files with unsupported extensions (.txt, .pdf, .csv) must raise ValueError."""
    invalid_file = tmp_path / "recording.txt"
    invalid_file.write_text("not an audio file")

    with pytest.raises(ValueError, match="Unsupported audio format"):
        processor.load_audio_file(invalid_file)


def test_corrupted_audio_file(processor: AudioProcessor, tmp_path: Path):
    """A corrupted .wav file must raise ValueError."""
    corrupted_file = tmp_path / "corrupted.wav"
    corrupted_file.write_bytes(b"RIFF\x00\x00\x00\x00CORRUPTED_HEADER_DATA")

    with pytest.raises(ValueError, match="Corrupted or invalid audio file"):
        processor.load_audio_file(corrupted_file)


# --- 4. Degenerate and Missing Inputs ---


def test_empty_audio_chunk_handling(processor: AudioProcessor):
    """Processing empty numpy array should return safe zeroed features without crashing."""
    empty_arr = np.array([], dtype=np.float32)

    features = processor.extract_features(empty_arr, timestamp=5.0)
    assert features.rms_energy == 0.0
    assert features.zero_crossing_rate == 0.0
    assert features.spectral_centroid == 0.0
    assert features.is_speech is False
    assert features.speech_ratio_pct == 0.0

    assert processor.compute_rms(empty_arr) == 0.0
    assert processor.compute_zcr(empty_arr) == 0.0
    assert processor.compute_spectral_centroid(empty_arr) == 0.0
    assert processor.is_speech_subframe(empty_arr) is False


def test_nan_and_inf_resilience(processor: AudioProcessor):
    """NaN and infinite values in raw audio arrays must be sanitized gracefully."""
    dirty_arr = np.array([0.1, np.nan, 0.2, np.inf, -np.inf, 0.05], dtype=np.float32)

    features = processor.extract_features(dirty_arr, timestamp=1.0)
    assert not np.isnan(features.rms_energy)
    assert not np.isinf(features.rms_energy)
    assert features.rms_energy >= 0.0


# --- 5. Ambient Noise Calibration ---


def test_ambient_noise_calibration(processor: AudioProcessor):
    """Calibrating on ambient room noise should update the baseline noise floor."""
    initial_floor = processor.noise_floor

    # Ambient room noise with RMS ~ 0.015
    np.random.seed(42)
    room_noise = np.random.normal(0.0, 0.015, 16000).astype(np.float32)
    calibrated_floor = processor.calibrate_noise_floor(room_noise)

    assert calibrated_floor > initial_floor
    assert processor.noise_floor == calibrated_floor
    assert 0.01 <= calibrated_floor <= 0.02


def test_dynamic_vad_threshold_with_high_noise_floor(processor: AudioProcessor):
    """Elevated noise floor prevents moderate background noise from triggering speech VAD."""
    processor.noise_floor = 0.05  # High noise environment (e.g. loud projector/AC)

    # Subframe with RMS 0.06 is above default threshold (0.015) but below dynamic threshold (0.05 * 2.2 = 0.11)
    subframe = np.full(480, 0.06, dtype=np.float32)
    assert processor.is_speech_subframe(subframe) is False


# --- 6. Privacy Guardrails ---


def test_privacy_saving_prohibited_by_default(processor: AudioProcessor, tmp_path: Path):
    """Saving raw audio when allow_save_audio=False must raise PermissionError."""
    target_path = tmp_path / "diagnostic.wav"
    audio_data = np.zeros(16000, dtype=np.float32)

    with pytest.raises(PermissionError, match="Privacy Policy Violation"):
        processor.save_audio_diagnostic(audio_data, target_path)


def test_privacy_saving_allowed_when_explicitly_configured(tmp_path: Path):
    """Saving raw audio is permitted only when allow_save_audio=True."""
    processor = AudioProcessor(allow_save_audio=True)
    target_path = tmp_path / "allowed_diagnostic.wav"
    audio_data = np.zeros(16000, dtype=np.float32)

    saved_path = processor.save_audio_diagnostic(audio_data, target_path)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


# --- 7. Microphone Stream Error & Permission Handling ---


def test_mic_stream_device_unavailable_handling(processor: AudioProcessor):
    """When no audio devices exist or permissions fail, start_mic_stream returns False gracefully."""
    with patch("sounddevice.query_devices", return_value=[]):
        success = processor.start_mic_stream()
        assert success is False
        assert processor.status == AudioStatus.DEVICE_UNAVAILABLE
        assert "No audio input devices detected" in (processor.error_message or "")


def test_mic_stream_permission_denied_exception(processor: AudioProcessor):
    """When an exception occurs initializing InputStream, status becomes DEVICE_UNAVAILABLE."""
    with patch("sounddevice.query_devices", return_value=[{"name": "Default Mic"}]):
        with patch("sounddevice.InputStream", side_effect=RuntimeError("Permission Denied: Mic Access")):
            success = processor.start_mic_stream()
            assert success is False
            assert processor.status == AudioStatus.DEVICE_UNAVAILABLE
            assert "Permission Denied" in (processor.error_message or "")


def test_mic_stream_read_and_stop(processor: AudioProcessor):
    """Queue reading and stopping microphone stream behaves deterministically."""
    assert processor.read_mic_chunk() is None  # Queue empty

    mock_stream = MagicMock()
    processor._mic_stream = mock_stream
    processor.status = AudioStatus.ACTIVE_STREAMING

    processor.stop_mic_stream()
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    assert processor._mic_stream is None
    assert processor.status == AudioStatus.STOPPED


# --- 8. File Ingestion, Stereo Downmixing, and Resampling ---


def test_load_valid_wav_stereo_downmix(processor: AudioProcessor, tmp_path: Path):
    """Stereo audio files must be downmixed to mono and resampled if needed."""
    wav_path = tmp_path / "stereo_test.wav"
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    stereo_data = np.column_stack([left, right]).astype(np.float32)

    sf.write(str(wav_path), stereo_data, sr)

    data, loaded_sr = processor.load_audio_file(wav_path)

    assert loaded_sr == processor.sample_rate  # Resampled to 16000
    assert data.ndim == 1  # Downmixed to mono
    assert len(data) == int(duration * processor.sample_rate)


def test_process_file_slice_convenience(processor: AudioProcessor, tmp_path: Path):
    """process_file_slice reads a specific time interval and returns features."""
    wav_path = tmp_path / "slice_test.wav"
    speech = generate_synthetic_speech(duration_sec=2.0, sample_rate=16000)
    sf.write(str(wav_path), speech, 16000)

    slice_feat = processor.process_file_slice(wav_path, start_sec=0.5, duration_sec=1.0)
    assert isinstance(slice_feat, AudioFeatures)
    assert slice_feat.rms_energy > 0.01
    assert slice_feat.timestamp == 1.5
