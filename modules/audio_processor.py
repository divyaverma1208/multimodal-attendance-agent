"""
Audio Processor Module.

Provides offline, privacy-first acoustic analysis for classroom intelligence.
Extracts short-time Root Mean Square (RMS) energy, Zero-Crossing Rate (ZCR),
Spectral Centroid, and Voice Activity Detection (VAD) ratios from audio files or
live microphone streams.

ETHICAL & PRIVACY GUARDRAILS:
- Operates 100% locally with zero cloud API transmission.
- Ephemeral audio buffers: Raw audio arrays are processed in RAM and discarded immediately.
- Never saves raw audio to disk unless explicitly enabled in controlled diagnostic mode.
- Does not perform automated speech recognition (ASR) or conversation transcription.
- Does not identify speakers or extract biometric voiceprints.
- Does not infer psychological attentiveness, emotion, or comprehension from acoustic loudness.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import queue
import time
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from config.settings import settings

logger = logging.getLogger(__name__)


class AudioStatus(str, Enum):
    """Operational status of the audio processing subsystem."""
    UNINITIALIZED = "UNINITIALIZED"
    ACTIVE_STREAMING = "ACTIVE_STREAMING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


@dataclass
class AudioFeatures:
    """Acoustic feature representation for a momentary slice."""
    rms_energy: float
    zero_crossing_rate: float
    spectral_centroid: float
    is_speech: bool
    speech_ratio_pct: float
    timestamp: float = 0.0

    @property
    def zcr(self) -> float:
        """Alias for zero_crossing_rate."""
        return self.zero_crossing_rate

    @property
    def speech_ratio(self) -> float:
        """Normalized speech activity proportion [0.0 to 1.0]."""
        return max(0.0, min(1.0, self.speech_ratio_pct / 100.0))

    @property
    def speech_pct(self) -> float:
        """Speech activity percentage [0.0 to 100.0]."""
        return self.speech_ratio_pct


@dataclass
class AudioWindowResult:
    """Timestamped structured acoustic metrics across an aggregation window."""
    timestamp: float
    window_start_sec: float
    window_end_sec: float
    window_duration_sec: float
    rms_energy_mean: float
    rms_energy_max: float
    zero_crossing_rate_mean: float
    spectral_centroid_mean: float
    speech_duration_sec: float
    silence_duration_sec: float
    speech_ratio_pct: float
    is_speech: bool
    noise_floor_estimate: float


class AudioProcessor:
    """
    Analyzes acoustic properties of classroom audio streams.
    Extracts loudness dynamics, frequency center of mass, and voice activity ratios.
    """

    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"}

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        energy_threshold: Optional[float] = None,
        zcr_threshold: Optional[float] = None,
        allow_save_audio: bool = False,
    ) -> None:
        """
        Initialize AudioProcessor.

        Args:
            sample_rate: Target audio sampling frequency in Hz (default: 16000).
            energy_threshold: Base RMS threshold for voice activity classification.
            zcr_threshold: Base zero-crossing rate threshold for speech detection.
            allow_save_audio: Guardrail flag. If False, raw audio persistence is prohibited.
        """
        self.sample_rate = sample_rate or settings.audio.sample_rate
        self.energy_threshold = energy_threshold or settings.audio.energy_vad_threshold
        self.zcr_threshold = zcr_threshold or settings.audio.zcr_threshold
        self.allow_save_audio = allow_save_audio

        self.noise_floor: float = 0.002
        self.status = AudioStatus.UNINITIALIZED
        self.error_message: Optional[str] = None

        # Microphone streaming state
        self._mic_stream = None
        self._mic_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=10)

    # --- Core Acoustic Signal Descriptors ---

    def compute_rms(self, signal: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy / loudness of signal."""
        if signal.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(signal**2)))

    def compute_zcr(self, signal: np.ndarray) -> float:
        """Calculate Zero-Crossing Rate (ZCR) of audio signal."""
        if signal.size < 2:
            return 0.0
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / 2.0
        return float(zero_crossings / len(signal))

    def compute_spectral_centroid(self, signal: np.ndarray) -> float:
        """
        Compute spectral centroid (center of spectral mass / acoustic brightness).
        Uses fast Discrete Fourier Transform (FFT) with Hanning window for real-time efficiency.
        """
        if signal.size < 64:
            return 0.0

        try:
            window = np.hanning(len(signal))
            spectrum = np.abs(np.fft.rfft(signal * window))
            freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)
            sum_spec = np.sum(spectrum)
            return float(np.sum(freqs * spectrum) / sum_spec) if sum_spec > 1e-12 else 0.0
        except Exception:
            try:
                # Librosa fallback if FFT computation encounters an unexpected error
                n_fft = min(2048, 1 << (len(signal).bit_length() - 1))
                n_fft = max(64, n_fft)
                hop_length = max(16, n_fft // 4)
                centroids = librosa.feature.spectral_centroid(
                    y=signal, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length
                )
                return float(np.mean(centroids))
            except Exception:
                return 0.0

    def calibrate_noise_floor(self, noise_sample: np.ndarray) -> float:
        """
        Calibrate ambient background noise floor from an initial room audio slice.

        Args:
            noise_sample: Representative audio slice of ambient background room sound.

        Returns:
            Updated noise floor estimate.
        """
        if noise_sample.size > 0:
            sample_rms = self.compute_rms(noise_sample)
            # Smooth noise floor estimate
            self.noise_floor = max(0.0005, round(sample_rms, 5))
            logger.info("Calibrated ambient noise floor to RMS: %s", self.noise_floor)
        return self.noise_floor

    # --- Voice Activity Detection (VAD) Baseline Engine ---

    def is_speech_subframe(self, subframe: np.ndarray) -> bool:
        r"""
        Multi-feature baseline Voice Activity Detector (VAD).
        Evaluates:
        1. RMS energy above adaptive ambient noise floor threshold ($RMS > \max(T_{default}, 2.2 \times \text{noise\_floor})$)
        2. Zero-Crossing Rate within human speech boundaries ($0.004 \le ZCR \le 0.48$)
        3. Spectral Centroid within characteristic human vocal frequency range ($150\text{ Hz} \le SC \le 4800\text{ Hz}$)

        Args:
            subframe: Short audio slice (~25-30ms).

        Returns:
            True if acoustic characteristics indicate vocal speech activity.
        """
        if subframe.size == 0:
            return False

        rms = self.compute_rms(subframe)
        dynamic_threshold = max(self.energy_threshold, self.noise_floor * 2.0)

        # 1. Energy threshold check
        if rms < dynamic_threshold:
            return False

        # 2. ZCR check (speech has structured periodicity vs broadband white noise)
        zcr = self.compute_zcr(subframe)
        if zcr < 0.003 or zcr > 0.50:
            return False

        # 3. Spectral Centroid check (vocal resonance center of mass)
        centroid = self.compute_spectral_centroid(subframe)
        if centroid < 120.0 or centroid > 5200.0:
            return False

        return True

    def extract_features(
        self,
        audio_chunk: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> AudioFeatures:
        """
        Extract acoustic features and voice activity ratio from an audio buffer.

        Args:
            audio_chunk: 1D floating-point audio array (-1.0 to 1.0).
            timestamp: Observation timestamp.

        Returns:
            AudioFeatures dataclass instance.
        """
        now = timestamp if timestamp is not None else time.time()
        if audio_chunk.size == 0:
            return AudioFeatures(0.0, 0.0, 0.0, False, 0.0, timestamp=now)

        # Ensure single-channel mono and finite values
        clean_chunk = np.nan_to_num(audio_chunk, nan=0.0, posinf=1.0, neginf=-1.0)
        if clean_chunk.ndim > 1:
            clean_chunk = np.mean(clean_chunk, axis=1)

        rms = self.compute_rms(clean_chunk)
        zcr = self.compute_zcr(clean_chunk)
        centroid = self.compute_spectral_centroid(clean_chunk)

        # Analyze sub-frames (30ms chunks) for speech ratio
        frame_len = max(64, int(self.sample_rate * 0.03))  # 30ms sub-frame
        total_subframes = max(1, len(clean_chunk) // frame_len)
        speech_subframes = 0

        for i in range(total_subframes):
            sub = clean_chunk[i * frame_len : (i + 1) * frame_len]
            if self.is_speech_subframe(sub):
                speech_subframes += 1

        speech_ratio = (speech_subframes / total_subframes) * 100.0
        is_speech = speech_ratio >= 8.0  # Overall chunk classified as speech if >=8% active

        # Adaptively track ambient noise floor during non-speech intervals
        dynamic_threshold = max(self.energy_threshold, self.noise_floor * 2.0)
        if rms < dynamic_threshold and not is_speech:
            self.noise_floor = max(0.0005, min(0.02, 0.95 * self.noise_floor + 0.05 * rms))

        return AudioFeatures(
            rms_energy=round(rms, 4),
            zero_crossing_rate=round(zcr, 4),
            spectral_centroid=round(centroid, 1),
            is_speech=is_speech,
            speech_ratio_pct=round(min(100.0, speech_ratio), 1),
            timestamp=now,
        )

    def analyze_window(
        self,
        audio_chunk: np.ndarray,
        window_start_sec: float,
        window_end_sec: float,
    ) -> AudioWindowResult:
        """
        Perform comprehensive windowed analysis calculating speech and silence durations.

        Args:
            audio_chunk: Audio waveform array for the time window.
            window_start_sec: Offset start of window.
            window_end_sec: Offset end of window.

        Returns:
            Structured AudioWindowResult object.
        """
        duration = max(0.001, window_end_sec - window_start_sec)
        feat = self.extract_features(audio_chunk, timestamp=window_end_sec)

        speech_duration = round((feat.speech_ratio_pct / 100.0) * duration, 2)
        silence_duration = round(max(0.0, duration - speech_duration), 2)

        # Max sub-frame energy in window
        frame_len = max(64, int(self.sample_rate * 0.03))
        max_rms = 0.0
        if audio_chunk.size >= frame_len:
            sub_energies = [
                self.compute_rms(audio_chunk[i : i + frame_len])
                for i in range(0, len(audio_chunk) - frame_len, frame_len)
            ]
            max_rms = float(max(sub_energies)) if sub_energies else feat.rms_energy

        return AudioWindowResult(
            timestamp=window_end_sec,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            window_duration_sec=round(duration, 2),
            rms_energy_mean=feat.rms_energy,
            rms_energy_max=round(max_rms, 4),
            zero_crossing_rate_mean=feat.zero_crossing_rate,
            spectral_centroid_mean=feat.spectral_centroid,
            speech_duration_sec=speech_duration,
            silence_duration_sec=silence_duration,
            speech_ratio_pct=feat.speech_ratio_pct,
            is_speech=feat.is_speech,
            noise_floor_estimate=round(self.noise_floor, 5),
        )

    # --- File Ingestion & Format Validation ---

    def load_audio_file(
        self,
        audio_path: Union[str, Path],
        start_sec: float = 0.0,
        duration_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio from file with validation of formats and existence.

        Args:
            audio_path: Path to recorded audio file.
            start_sec: Offset in seconds to seek into file.
            duration_sec: Duration to read in seconds (reads to EOF if None).

        Returns:
            Tuple of (audio_waveform: np.ndarray, sample_rate: int).

        Raises:
            FileNotFoundError: If audio file does not exist.
            ValueError: If file format is unsupported or file is corrupted.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{path.suffix}'. Supported formats: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        try:
            with sf.SoundFile(str(path)) as f:
                sr = f.samplerate
                start_frame = int(start_sec * sr)
                if start_frame >= len(f):
                    return np.array([], dtype=np.float32), sr

                f.seek(start_frame)
                frames_to_read = int(duration_sec * sr) if duration_sec is not None else -1
                data = f.read(frames=frames_to_read, dtype="float32")

                # Convert multi-channel audio to mono
                if data.ndim > 1:
                    data = np.mean(data, axis=1)

                # Resample to target sample_rate if different
                if sr != self.sample_rate and len(data) > 0:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate

                return data, sr

        except Exception as e:
            logger.error("Failed to decode audio file %s: %s", path, e)
            raise ValueError(f"Corrupted or invalid audio file: {path}. Details: {str(e)}") from e

    def process_file_slice(
        self,
        audio_path: Union[str, Path],
        start_sec: float,
        duration_sec: float,
    ) -> AudioFeatures:
        """
        Convenience method to load and process an audio file interval.
        """
        try:
            chunk, _ = self.load_audio_file(audio_path, start_sec=start_sec, duration_sec=duration_sec)
            return self.extract_features(chunk, timestamp=start_sec + duration_sec)
        except Exception as e:
            logger.warning("Error processing audio slice: %s. Returning silent features.", e)
            return AudioFeatures(0.0, 0.0, 0.0, False, 0.0, timestamp=start_sec + duration_sec)

    # --- Live Microphone Capture with Safe Degradation ---

    @staticmethod
    def get_available_microphones() -> List[dict]:
        """List available audio input devices detected on host system."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            inputs = []
            default_in = sd.default.device[0] if hasattr(sd.default, "device") else None
            for idx, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    inputs.append({
                        "index": idx,
                        "name": d.get("name", f"Microphone {idx}"),
                        "is_default": (idx == default_in),
                    })
            return inputs
        except Exception as e:
            logger.debug("Failed to query sound devices: %s", e)
            return []

    def test_microphone(self, device_index: Optional[int] = None, duration_sec: float = 1.0) -> Tuple[bool, str, float, float]:
        """
        Record a short sample from the microphone to test connectivity and speech levels.

        Returns:
            Tuple of (success: bool, message: str, rms: float, speech_ratio: float).
        """
        try:
            import sounddevice as sd
            frames_to_read = int(self.sample_rate * duration_sec)
            recording = sd.rec(
                frames=frames_to_read,
                samplerate=self.sample_rate,
                channels=1,
                device=device_index,
                dtype="float32",
            )
            sd.wait()
            flat_audio = recording.flatten()
            features = self.extract_features(flat_audio)
            msg = f"Microphone operational. RMS: {features.rms_energy:.4f}, Speech Ratio: {features.speech_ratio_pct:.1f}%"
            return True, msg, features.rms_energy, features.speech_ratio_pct
        except Exception as e:
            return False, f"Microphone test failed: {e}", 0.0, 0.0

    def start_mic_stream(self, device_index: Optional[int] = None) -> bool:
        """
        Initialize live microphone stream using sounddevice with graceful error handling and automatic device fallback.

        Returns:
            True if microphone stream opened successfully, False otherwise.
        """
        self.stop_mic_stream()
        self.active_device_index = device_index
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            if not devices:
                self.status = AudioStatus.DEVICE_UNAVAILABLE
                self.error_message = "No audio input devices detected on host system."
                logger.warning(self.error_message)
                return False

            # Candidate device list to attempt in order:
            # 1. User-requested device_index (if provided)
            # 2. System default input (device=None)
            # 3. Any device reporting max_input_channels > 0
            candidates: List[Optional[int]] = []
            if device_index is not None:
                candidates.append(device_index)
            candidates.append(None)
            for idx, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0 and idx not in candidates:
                    candidates.append(idx)

            def _audio_callback(indata, frames, time_info, status):
                if status:
                    logger.debug("Audio stream callback status: %s", status)
                # Downmix to mono if stereo
                data = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
                if not self._mic_queue.full():
                    self._mic_queue.put_nowait(data.copy())

            last_err = None
            for candidate in candidates:
                try:
                    sr_to_try = [self.sample_rate]
                    if candidate is not None and candidate < len(devices):
                        dev_sr = int(devices[candidate].get("default_samplerate", 44100))
                        if dev_sr not in sr_to_try:
                            sr_to_try.append(dev_sr)

                    for sr in sr_to_try:
                        stream = None
                        try:
                            stream = sd.InputStream(
                                samplerate=sr,
                                channels=1,
                                device=candidate,
                                blocksize=max(256, int(sr * 0.5)),  # 0.5-second chunks
                                callback=_audio_callback,
                            )
                            stream.start()
                            self._mic_stream = stream
                            self.active_device_index = candidate
                            self.status = AudioStatus.ACTIVE_STREAMING
                            self.error_message = None
                            logger.info(
                                "Microphone stream started successfully at %s Hz (device=%s).",
                                sr, candidate
                            )
                            return True
                        except Exception as e:
                            last_err = e
                            if stream is not None:
                                try:
                                    stream.close()
                                except Exception:
                                    pass
                except Exception as e:
                    last_err = e

            self.status = AudioStatus.DEVICE_UNAVAILABLE
            self.error_message = (
                f"Failed to access microphone. Ensure microphone permissions are granted. Details: {last_err}"
            )
            logger.warning(self.error_message)
            self.stop_mic_stream()
            return False

        except Exception as e:
            self.status = AudioStatus.DEVICE_UNAVAILABLE
            self.error_message = f"Audio subsystem error: {e}"
            logger.warning(self.error_message)
            self.stop_mic_stream()
            return False

    def read_mic_chunk(self) -> Optional[np.ndarray]:
        """Retrieve next recorded chunk from live microphone queue."""
        try:
            return self._mic_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_mic_stream(self) -> None:
        """Safely stop and close microphone stream."""
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception as e:
                logger.warning("Error stopping mic stream: %s", e)
            finally:
                self._mic_stream = None

        if self.status == AudioStatus.ACTIVE_STREAMING:
            self.status = AudioStatus.STOPPED

    start_microphone_stream = start_mic_stream
    stop_microphone_stream = stop_mic_stream
    read_stream_chunk = read_mic_chunk
    process_audio_chunk = extract_features

    # --- Privacy Diagnostic Frame Guardrail ---

    def save_audio_diagnostic(self, audio_data: np.ndarray, target_path: Union[str, Path]) -> Path:
        """
        Diagnostic-only audio saver. Protected by strict privacy guardrail.

        Raises:
            PermissionError: If allow_save_audio is False.
        """
        if not self.allow_save_audio:
            raise PermissionError(
                "Privacy Policy Violation: Saving raw classroom audio to disk is strictly prohibited. "
                "Set allow_save_audio=True explicitly only in controlled diagnostic tests."
            )
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio_data, self.sample_rate)
        return path
