"""
Centralized Configuration and Settings.

Manages application paths, perception hyperparameters, threshold settings,
privacy toggles, and database configurations.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Tuple


# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatabaseSettings:
    """Configuration for local SQLite database."""
    db_name: str = os.getenv("APP_DB_NAME", "attendance_agent.db")
    db_dir: Path = BASE_DIR / "data"
    
    @property
    def db_path(self) -> Path:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        return self.db_dir / self.db_name


@dataclass(frozen=True)
class VideoSettings:
    """Hyperparameters for camera and video stream processing."""
    default_camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    target_fps: int = int(os.getenv("TARGET_FPS", "8"))
    frame_size: Tuple[int, int] = (640, 480)
    person_detection_confidence: float = float(os.getenv("DETECTION_CONFIDENCE", "0.65"))
    max_lost_frames: int = int(os.getenv("MAX_LOST_FRAMES", "30"))
    max_centroid_distance_px: float = float(os.getenv("MAX_CENTROID_DISTANCE", "75.0"))
    use_mediapipe_if_available: bool = os.getenv("USE_MEDIAPIPE", "true").lower() == "true"


@dataclass(frozen=True)
class AudioSettings:
    """Hyperparameters for acoustic processing and speech detection."""
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    frame_duration_ms: int = 30
    chunk_size: int = 1024
    energy_vad_threshold: float = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.005"))
    zcr_threshold: float = float(os.getenv("VAD_ZCR_THRESHOLD", "0.05"))
    enable_audio_by_default: bool = os.getenv("ENABLE_AUDIO", "true").lower() == "true"


@dataclass(frozen=True)
class AttendanceSettings:
    """Parameters for sustained presence calculation."""
    default_sustained_threshold_pct: float = float(os.getenv("SUSTAINED_THRESHOLD_PCT", "60.0"))
    min_continuous_presence_sec: float = float(os.getenv("MIN_PRESENCE_SEC", "300.0"))  # 5 minutes
    transient_filter_seconds: float = float(os.getenv("TRANSIENT_FILTER_SEC", "30.0"))


@dataclass(frozen=True)
class AnalyticsSettings:
    """Aggregation window sizes and suggestion engine thresholds."""
    window_duration_seconds: int = int(os.getenv("WINDOW_DURATION_SEC", "30"))
    monologue_alert_threshold_sec: int = int(os.getenv("MONOLOGUE_ALERT_SEC", "600"))  # 10 minutes
    silence_alert_threshold_sec: int = int(os.getenv("SILENCE_ALERT_SEC", "180"))     # 3 minutes
    restlessness_movement_threshold: float = float(os.getenv("RESTLESSNESS_THRESHOLD", "75.0"))


@dataclass(frozen=True)
class PrivacySettings:
    """Data governance, retention, and privacy enforcement."""
    store_raw_media: bool = False  # NEVER store raw frames/audio by default
    require_explicit_consent: bool = True
    auto_purge_days: int = int(os.getenv("DATA_RETENTION_DAYS", "30"))
    exports_dir: Path = BASE_DIR / "data" / "exports"

    def ensure_export_dir(self) -> Path:
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        return self.exports_dir


@dataclass
class AppConfig:
    """Master application configuration container."""
    db: DatabaseSettings = field(default_factory=DatabaseSettings)
    video: VideoSettings = field(default_factory=VideoSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    attendance: AttendanceSettings = field(default_factory=AttendanceSettings)
    analytics: AnalyticsSettings = field(default_factory=AnalyticsSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)


# Global singleton instance
settings = AppConfig()
