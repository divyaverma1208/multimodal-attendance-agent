"""
Session Manager Module.

Coordinates classroom observation session lifecycles (READY -> ACTIVE <-> PAUSED -> CONCLUDED).
Tracks active elapsed duration, timing slices, and ensures atomic state synchronization
with the local database.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
from typing import Optional
import uuid

from modules.database import DatabaseManager


class SessionState(str, Enum):
    """Classroom session lifecycle states."""
    READY = "READY"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CONCLUDED = "CONCLUDED"


@dataclass
class SessionMetadata:
    """In-memory representation of session metadata."""
    session_id: str
    course_code: str
    session_title: str
    status: SessionState
    video_source_type: str
    audio_enabled: bool
    sustained_threshold_pct: float
    started_at: Optional[float] = None
    concluded_at: Optional[float] = None
    accumulated_active_seconds: float = 0.0


class SessionManager:
    """Manages session lifecycle state transitions and elapsed active duration."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        db: Optional[DatabaseManager] = None,
    ) -> None:
        """
        Initialize session manager with a database instance.

        Args:
            db_manager: Persistent SQLite storage layer.
            db: Alias for db_manager.
        """
        self.db = db_manager or db or DatabaseManager()
        self.current_session: Optional[SessionMetadata] = None
        self._last_resume_timestamp: Optional[float] = None

    def create_session(
        self,
        course_code: str,
        session_title: str,
        video_source_type: str = "webcam",
        audio_enabled: bool = True,
        sustained_threshold_pct: float = 60.0,
    ) -> SessionMetadata:
        """
        Initialize a new classroom session in READY state.

        Args:
            course_code: Course identifier (e.g., 'CS101').
            session_title: Human-readable topic or title.
            video_source_type: 'webcam' or 'file'.
            audio_enabled: Whether acoustic analysis is enabled.
            sustained_threshold_pct: % of active session required for sustained attendance.

        Returns:
            Configured SessionMetadata instance.
        """
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.db.create_session(
            session_id=session_id,
            course_code=course_code,
            session_title=session_title,
            video_source_type=video_source_type,
            audio_enabled=audio_enabled,
            sustained_threshold_pct=sustained_threshold_pct,
        )

        metadata = SessionMetadata(
            session_id=session_id,
            course_code=course_code,
            session_title=session_title,
            status=SessionState.READY,
            video_source_type=video_source_type,
            audio_enabled=audio_enabled,
            sustained_threshold_pct=sustained_threshold_pct,
        )
        self.current_session = metadata
        self._last_resume_timestamp = None
        return metadata

    def start_session(self) -> None:
        """Transition session from READY or PAUSED to ACTIVE."""
        if not self.current_session:
            raise RuntimeError("No session currently initialized.")
        if self.current_session.status not in (SessionState.READY, SessionState.PAUSED):
            raise ValueError(f"Cannot start session from state '{self.current_session.status}'.")

        now = time.time()
        now_iso = datetime.now().isoformat()
        if self.current_session.started_at is None:
            self.current_session.started_at = now
            self.db.update_session_status(
                session_id=self.current_session.session_id,
                status=SessionState.ACTIVE.value,
                started_at=now_iso,
            )
        else:
            self.db.update_session_status(
                session_id=self.current_session.session_id,
                status=SessionState.ACTIVE.value,
            )

        self._last_resume_timestamp = now
        self.current_session.status = SessionState.ACTIVE

    def resume_session(self) -> None:
        """Alias for start_session() to resume from PAUSED state."""
        self.start_session()

    def pause_session(self) -> None:
        """Pause active session and accumulate elapsed active time slice."""
        if not self.current_session or self.current_session.status != SessionState.ACTIVE:
            return

        now = time.time()
        if self._last_resume_timestamp is not None:
            slice_duration = now - self._last_resume_timestamp
            self.current_session.accumulated_active_seconds += max(0.0, slice_duration)
            self._last_resume_timestamp = None

        self.current_session.status = SessionState.PAUSED
        self.db.update_session_status(
            session_id=self.current_session.session_id,
            status=SessionState.PAUSED.value,
            total_active_seconds=self.current_session.accumulated_active_seconds,
        )

    def conclude_session(self) -> float:
        """
        Conclude session, freeze elapsed duration, and persist final status.

        Returns:
            Total active seconds recorded for the session.
        """
        if not self.current_session or self.current_session.status == SessionState.CONCLUDED:
            return 0.0

        if self.current_session.status == SessionState.ACTIVE:
            self.pause_session()

        now = time.time()
        now_iso = datetime.now().isoformat()
        self.current_session.concluded_at = now
        self.current_session.status = SessionState.CONCLUDED

        total_sec = self.current_session.accumulated_active_seconds
        self.db.update_session_status(
            session_id=self.current_session.session_id,
            status=SessionState.CONCLUDED.value,
            total_active_seconds=total_sec,
            concluded_at=now_iso,
        )
        return total_sec

    def get_active_elapsed_seconds(self) -> float:
        """
        Calculate total live elapsed active seconds, including current running slice.
        """
        if not self.current_session:
            return 0.0

        accumulated = self.current_session.accumulated_active_seconds
        if self.current_session.status == SessionState.ACTIVE and self._last_resume_timestamp is not None:
            live_slice = time.time() - self._last_resume_timestamp
            return accumulated + max(0.0, live_slice)
        return accumulated
