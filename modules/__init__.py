"""
Multimodal Attendance & Engagement Agent - Core Modules.

This package contains modular engines for video ingestion, person tracking,
sustained presence calculation, acoustic analysis, feature fusion,
time-window analytics, deterministic pedagogical suggestions, and local storage.
"""

from .database import DatabaseError, DatabaseManager
from .session_manager import SessionManager, SessionState
from .person_tracker import CentroidTracker, HeadOrientation, PersonTracker, Track, TrackingFrameResult
from .attendance import AttendanceEngine, AttendanceEvent, AttendanceEventType, AttendanceStatus
from .video_processor import VideoProcessor
from .audio_processor import AudioFeatures, AudioProcessor, AudioStatus, AudioWindowResult
from .feature_fusion import DataQualityLabel, FeatureFusionEngine, MultimodalWindow, VisualFeatures
from .engagement_analytics import ActivityWeights, EngagementAnalytics, SessionAnalyticsSummary, TrendEvent
from .recommendation_engine import (
    RecommendationEngine,
    RecommendationEngineConfig,
    RuleCode,
    Suggestion,
)
from .report_generator import ReportGenerator
from .pipeline import ClassroomPipeline, PipelineStepResult

__all__ = [
    "ClassroomPipeline",
    "PipelineStepResult",
    "DatabaseError",
    "DatabaseManager",
    "SessionManager",
    "SessionState",
    "CentroidTracker",
    "Track",
    "PersonTracker",
    "TrackingFrameResult",
    "HeadOrientation",
    "AttendanceEngine",
    "AttendanceStatus",
    "AttendanceEvent",
    "AttendanceEventType",
    "VideoProcessor",
    "AudioProcessor",
    "AudioFeatures",
    "AudioStatus",
    "AudioWindowResult",
    "FeatureFusionEngine",
    "MultimodalWindow",
    "VisualFeatures",
    "DataQualityLabel",
    "EngagementAnalytics",
    "ActivityWeights",
    "SessionAnalyticsSummary",
    "TrendEvent",
    "RecommendationEngine",
    "RecommendationEngineConfig",
    "RuleCode",
    "Suggestion",
    "ReportGenerator",
]
