"""
Attendance Engine Module.

Implements sustained-presence attendance verification from timestamped tracking observations.
Tracks rolling observation windows, presence/absence intervals, brief detection gap bridging,
provisional vs verified sustained attendance, and timestamped attendance lifecycle events.

ETHICAL & SECURITY GUARANTEES:
- Do not use a single detection to confirm attendance.
- Never equate tracking IDs to student identities autonomously.
- Associate student names strictly through explicit teacher claim or authenticated token check-in.
- Prevent double-counting students across active tracks.
- Support manual teacher overrides with full audit logging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple
import uuid

import pandas as pd

from config.settings import settings
from modules.database import DatabaseManager
from modules.person_tracker import Track

logger = logging.getLogger(__name__)


class AttendanceStatus(str, Enum):
    """Attendance verification statuses."""
    SUSTAINED_PRESENT = "SUSTAINED_PRESENT"  # Verified sustained presence meets or exceeds threshold
    PROVISIONAL = "PROVISIONAL"              # Active participant meeting early provisional threshold
    PARTIAL = "PARTIAL"                      # Present beyond transient filter but below sustained threshold
    TRANSIENT = "TRANSIENT"                  # Passer-by or brief walk-in (< transient threshold)
    ABSENT = "ABSENT"                        # Enrolled roster member with no verified presence


class AttendanceEventType(str, Enum):
    """Types of timestamped attendance lifecycle events."""
    FIRST_SEEN = "FIRST_SEEN"
    BRIEF_OCCLUSION_BRIDGED = "BRIEF_OCCLUSION_BRIDGED"
    PROVISIONAL_ACHIEVED = "PROVISIONAL_ACHIEVED"
    SUSTAINED_ACHIEVED = "SUSTAINED_ACHIEVED"
    LEFT_VIEW = "LEFT_VIEW"
    RE_ENTERED = "RE_ENTERED"
    MANUAL_CORRECTION = "MANUAL_CORRECTION"
    DUPLICATE_SUPPRESSED = "DUPLICATE_SUPPRESSED"


@dataclass
class AttendanceEvent:
    """Timestamped attendance lifecycle event."""
    event_id: str
    session_id: str
    track_id: int
    timestamp: float
    event_type: AttendanceEventType
    description: str


@dataclass
class ParticipantObservationState:
    """Detailed temporal presence state for a tracked participant."""
    track_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float
    observation_timestamps: List[float] = field(default_factory=list)
    presence_intervals: List[Tuple[float, float]] = field(default_factory=list)
    absence_intervals: List[Tuple[float, float]] = field(default_factory=list)
    provisional_achieved: bool = False
    sustained_achieved: bool = False
    assigned_student_id: Optional[str] = None
    student_name: Optional[str] = None
    verification_method: str = "ANONYMOUS"
    manual_override: bool = False
    override_status: Optional[AttendanceStatus] = None

    @property
    def total_observation_count(self) -> int:
        """Total number of discrete frame observations."""
        return len(self.observation_timestamps)

    @property
    def total_detected_seconds(self) -> float:
        """Calculate cumulative detected presence time across presence intervals."""
        if not self.presence_intervals:
            return max(0.0, self.last_seen_timestamp - self.first_seen_timestamp)
        return sum(max(0.0, end - start) for start, end in self.presence_intervals)

    @property
    def total_absence_seconds(self) -> float:
        """Calculate cumulative absence time between confirmed intervals."""
        return sum(max(0.0, end - start) for start, end in self.absence_intervals)


@dataclass
class AttendanceEvaluation:
    """Comprehensive evaluated attendance record for reporting and database storage."""
    track_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float
    total_detected_seconds: float
    presence_percentage: float
    observation_ratio: float
    status: AttendanceStatus
    assigned_student_id: Optional[str] = None
    student_name: Optional[str] = None
    verification_method: str = "ANONYMOUS"
    presence_intervals_count: int = 0
    absence_intervals_count: int = 0


def mask_identifier(identifier: Optional[str]) -> str:
    """Mask student or participant identifiers for privacy-compliant logging."""
    if not identifier:
        return "[ANONYMOUS]"
    clean = str(identifier).strip()
    if len(clean) <= 3:
        return "***"
    return f"{clean[:2]}***{clean[-1:]}"


class AttendanceEngine:

    """
    Evaluates participant presence from timestamped observations,
    manages rolling observation windows, and generates attendance events.
    """

    def __init__(
        self,
        default_threshold_pct: Optional[float] = None,
        provisional_threshold_pct: float = 20.0,
        transient_filter_sec: Optional[float] = None,
        gap_tolerance_sec: float = 10.0,
        rolling_window_sec: float = 300.0,
        min_detections_for_presence: int = 2,
    ) -> None:
        """
        Initialize AttendanceEngine.

        Args:
            default_threshold_pct: % of active session required for SUSTAINED_PRESENT (default 60%).
            provisional_threshold_pct: % of session required for PROVISIONAL status (default 20%).
            transient_filter_sec: Maximum presence duration classified as TRANSIENT passer-by (default 30s).
            gap_tolerance_sec: Inter-detection gap tolerance in seconds to bridge brief occlusions.
            rolling_window_sec: Window size in seconds for rolling observation calculations.
            min_detections_for_presence: Minimum discrete observations needed to verify presence.
        """
        self.default_threshold_pct = (
            default_threshold_pct or settings.attendance.default_sustained_threshold_pct
        )
        self.provisional_threshold_pct = provisional_threshold_pct
        self.transient_filter_sec = (
            transient_filter_sec or settings.attendance.transient_filter_seconds
        )
        self.gap_tolerance_sec = gap_tolerance_sec
        self.rolling_window_sec = rolling_window_sec
        self.min_detections = max(2, min_detections_for_presence)

        # In-memory session tracking states: {track_id: ParticipantObservationState}
        self.participants: Dict[int, ParticipantObservationState] = {}
        self.events: List[AttendanceEvent] = []
        self._processed_event_keys: Set[Tuple[int, float, str]] = set()

    def record_observation(
        self,
        session_id: str,
        track_id: int,
        timestamp: float,
        is_visible: bool = True,
    ) -> List[AttendanceEvent]:
        """
        Ingest a single timestamped tracking observation for a participant.

        Args:
            session_id: Active session identifier.
            track_id: Anonymous tracking integer.
            timestamp: Observation timestamp.
            is_visible: True if visible in current frame; False if missed/occluded.

        Returns:
            List of freshly triggered AttendanceEvents.
        """
        generated_events: List[AttendanceEvent] = []

        if track_id not in self.participants:
            # Case 1: First detection of this participant
            state = ParticipantObservationState(
                track_id=track_id,
                first_seen_timestamp=timestamp,
                last_seen_timestamp=timestamp,
                observation_timestamps=[timestamp],
                presence_intervals=[(timestamp, timestamp)],
            )
            self.participants[track_id] = state

            event = self._create_event(
                session_id=session_id,
                track_id=track_id,
                timestamp=timestamp,
                event_type=AttendanceEventType.FIRST_SEEN,
                description=f"Anonymous participant Track-{track_id} first observed at {timestamp:.2f}s",
            )
            if event:
                generated_events.append(event)
            return generated_events

        state = self.participants[track_id]
        prev_timestamp = state.last_seen_timestamp
        time_delta = timestamp - prev_timestamp

        if not is_visible:
            # Subject missed in this frame
            return generated_events

        state.observation_timestamps.append(timestamp)
        state.last_seen_timestamp = timestamp

        # Manage presence and absence intervals with gap tolerance
        if time_delta <= self.gap_tolerance_sec:
            # Continuous presence or brief gap successfully bridged
            if state.presence_intervals:
                start_t, _ = state.presence_intervals[-1]
                state.presence_intervals[-1] = (start_t, timestamp)
            else:
                state.presence_intervals.append((timestamp, timestamp))

            if time_delta > 2.0:
                event = self._create_event(
                    session_id=session_id,
                    track_id=track_id,
                    timestamp=timestamp,
                    event_type=AttendanceEventType.BRIEF_OCCLUSION_BRIDGED,
                    description=f"Track-{track_id} brief {time_delta:.1f}s gap bridged as continuous presence",
                )
                if event:
                    generated_events.append(event)
        else:
            # Subject re-entered view after an extended absence period
            state.absence_intervals.append((prev_timestamp, timestamp))
            state.presence_intervals.append((timestamp, timestamp))

            event = self._create_event(
                session_id=session_id,
                track_id=track_id,
                timestamp=timestamp,
                event_type=AttendanceEventType.RE_ENTERED,
                description=f"Track-{track_id} re-entered view after {time_delta:.1f}s absence",
            )
            if event:
                generated_events.append(event)

        return generated_events

    def record_frame_observations(
        self,
        session_id: str,
        timestamp: float,
        active_tracks: Dict[int, Track],
    ) -> List[AttendanceEvent]:
        """
        Batch ingest all active and occluded tracks for a given frame.

        Args:
            session_id: Active session identifier.
            timestamp: Frame timestamp.
            active_tracks: Dictionary of {track_id: Track}.

        Returns:
            List of generated attendance events.
        """
        all_events: List[AttendanceEvent] = []
        for track_id, track in active_tracks.items():
            events = self.record_observation(
                session_id=session_id,
                track_id=track_id,
                timestamp=timestamp,
                is_visible=track.is_visible,
            )
            all_events.extend(events)
        return all_events

    def _create_event(
        self,
        session_id: str,
        track_id: int,
        timestamp: float,
        event_type: AttendanceEventType,
        description: str,
    ) -> Optional[AttendanceEvent]:
        """Create and record an attendance event, suppressing exact duplicates."""
        event_key = (track_id, round(timestamp, 2), event_type.value)
        if event_key in self._processed_event_keys:
            return None

        self._processed_event_keys.add(event_key)
        event = AttendanceEvent(
            event_id=f"evt_{uuid.uuid4().hex[:10]}",
            session_id=session_id,
            track_id=track_id,
            timestamp=timestamp,
            event_type=event_type,
            description=description,
        )
        self.events.append(event)
        return event

    def calculate_observation_ratio(
        self,
        track_id: int,
        expected_duration_sec: float,
    ) -> float:
        """
        Calculate the proportion of expected observations in which participant was detected.

        Args:
            track_id: Participant tracking ID.
            expected_duration_sec: Expected total duration of observation.

        Returns:
            Proportion in [0.0, 1.0].
        """
        if track_id not in self.participants or expected_duration_sec <= 0:
            return 0.0

        state = self.participants[track_id]
        detected = state.total_detected_seconds
        ratio = min(1.0, max(0.0, detected / expected_duration_sec))
        return round(ratio, 4)

    def evaluate_track_status(
        self,
        track_id: int,
        total_session_active_seconds: float,
        threshold_pct: Optional[float] = None,
    ) -> AttendanceStatus:
        """
        Determine formal attendance status for a participant state.

        Args:
            track_id: Tracking ID.
            total_session_active_seconds: Cumulative active session time in seconds.
            threshold_pct: Configured sustained threshold percentage.

        Returns:
            AttendanceStatus enum value.
        """
        if track_id not in self.participants:
            return AttendanceStatus.ABSENT

        state = self.participants[track_id]

        # Respect authorized manual teacher override
        if state.manual_override and state.override_status is not None:
            return state.override_status

        detected_sec = state.total_detected_seconds

        # Guardrail: Never use a single detection to confirm attendance
        if state.total_observation_count < self.min_detections:
            return AttendanceStatus.TRANSIENT

        # Transient passer-by filter (e.g. < 30 seconds)
        if detected_sec < self.transient_filter_sec:
            return AttendanceStatus.TRANSIENT

        if total_session_active_seconds <= 0:
            return AttendanceStatus.PROVISIONAL

        target_pct = threshold_pct if threshold_pct is not None else self.default_threshold_pct
        presence_ratio = (detected_sec / total_session_active_seconds) * 100.0

        # Verified sustained attendance
        if presence_ratio >= target_pct:
            return AttendanceStatus.SUSTAINED_PRESENT

        # Provisional attendance
        if presence_ratio >= self.provisional_threshold_pct:
            return AttendanceStatus.PROVISIONAL

        return AttendanceStatus.PARTIAL

    def evaluate_session_presence(
        self,
        tracks: Optional[List[Track]] = None,
        total_session_active_seconds: float = 0.0,
        threshold_pct: Optional[float] = None,
        existing_roster_mapping: Optional[Dict[int, Dict[str, str]]] = None,
    ) -> List[AttendanceEvaluation]:
        """
        Evaluate all participant observations and return structured evaluations.

        Args:
            tracks: Optional list of Track objects to synchronize into participant states.
            total_session_active_seconds: Cumulative active session duration in seconds.
            threshold_pct: Required sustained presence threshold percentage.
            existing_roster_mapping: Dict of {track_id: {'assigned_student_id': ..., 'student_name': ...}}.

        Returns:
            List of AttendanceEvaluation objects.
        """
        # Synchronize any tracks passed directly
        if tracks:
            for t in tracks:
                if t.track_id not in self.participants:
                    self.participants[t.track_id] = ParticipantObservationState(
                        track_id=t.track_id,
                        first_seen_timestamp=t.first_seen_timestamp,
                        last_seen_timestamp=t.last_seen_timestamp,
                        observation_timestamps=[t.first_seen_timestamp, t.last_seen_timestamp],
                        presence_intervals=[(t.first_seen_timestamp, t.last_seen_timestamp)],
                    )

        roster_map = existing_roster_mapping or {}
        evaluations: List[AttendanceEvaluation] = []

        for track_id, state in sorted(self.participants.items(), key=lambda kv: kv[0]):
            status = self.evaluate_track_status(
                track_id=track_id,
                total_session_active_seconds=total_session_active_seconds,
                threshold_pct=threshold_pct,
            )
            detected_sec = state.total_detected_seconds
            presence_pct = (
                (detected_sec / total_session_active_seconds * 100.0)
                if total_session_active_seconds > 0
                else 0.0
            )
            obs_ratio = self.calculate_observation_ratio(
                track_id=track_id,
                expected_duration_sec=total_session_active_seconds,
            )

            # Roster metadata (strict explicit binding)
            student_meta = roster_map.get(track_id, {})
            assigned_id = state.assigned_student_id or student_meta.get("assigned_student_id")
            student_name = state.student_name or student_meta.get("student_name")
            method = state.verification_method

            evaluations.append(
                AttendanceEvaluation(
                    track_id=track_id,
                    first_seen_timestamp=state.first_seen_timestamp,
                    last_seen_timestamp=state.last_seen_timestamp,
                    total_detected_seconds=round(detected_sec, 2),
                    presence_percentage=round(min(100.0, presence_pct), 1),
                    observation_ratio=obs_ratio,
                    status=status,
                    assigned_student_id=assigned_id,
                    student_name=student_name,
                    verification_method=method,
                    presence_intervals_count=len(state.presence_intervals),
                    absence_intervals_count=len(state.absence_intervals),
                )
            )

        return evaluations

    def claim_track_for_student(
        self,
        db: DatabaseManager,
        session_id: str,
        track_id: int,
        student_id: Optional[str],
        method: str = "TEACHER_CONFIRMED",
    ) -> bool:
        """
        Manually link an anonymous tracking ID to an enrolled student record.
        STRICT GUARDRAIL: Prevents double-counting by ensuring a student is not
        simultaneously assigned to multiple tracks in the same session.

        Args:
            db: DatabaseManager instance.
            session_id: Session identifier.
            track_id: Anonymous tracking ID.
            student_id: Enrolled student ID or None to unassign.
            method: Verification method.

        Returns:
            True if assignment succeeded, False if rejected due to double-counting.
        """
        if student_id:
            # 1. Check in-memory participant states to prevent double-counting
            for tid, p in self.participants.items():
                if p.assigned_student_id == student_id and tid != track_id:
                    logger.warning(
                        "Double-counting prevented: Student [%s] is already assigned to Track-%s",
                        mask_identifier(student_id),
                        tid,
                    )
                    return False

            # 2. Check persistent database records
            existing_records = db.get_session_attendance(session_id)
            for r in existing_records:
                if r["assigned_student_id"] == student_id and r["track_id"] != track_id:
                    logger.warning(
                        "Double-counting prevented: Student [%s] is already assigned to Track-%s",
                        mask_identifier(student_id),
                        r["track_id"],
                    )
                    return False

        # Apply assignment to in-memory state
        now = time.time()
        if track_id in self.participants:
            self.participants[track_id].assigned_student_id = student_id
            self.participants[track_id].verification_method = method
        else:
            self.participants[track_id] = ParticipantObservationState(
                track_id=track_id,
                first_seen_timestamp=now,
                last_seen_timestamp=now,
                assigned_student_id=student_id,
                verification_method=method,
            )

        # Persist to database
        db.assign_student_to_track(
            session_id=session_id,
            track_id=track_id,
            student_id=student_id,
            verification_method=method,
        )

        event = self._create_event(
            session_id=session_id,
            track_id=track_id,
            timestamp=now,
            event_type=AttendanceEventType.MANUAL_CORRECTION,
            description=f"Teacher confirmed assignment of Student {student_id} to Track-{track_id}",
        )
        if event:
            db.insert_attendance_event(
                event_id=event.event_id,
                session_id=event.session_id,
                track_id=event.track_id,
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                description=event.description,
            )
        return True

    def manual_override_status(
        self,
        db: DatabaseManager,
        session_id: str,
        track_id: int,
        new_status: AttendanceStatus,
        teacher_note: str = "",
    ) -> None:
        """
        Authorized teacher manual override of verified attendance status.

        Args:
            db: DatabaseManager instance.
            session_id: Session identifier.
            track_id: Tracking ID.
            new_status: Desired AttendanceStatus override.
            teacher_note: Contextual note explaining manual correction.
        """
        now = time.time()
        if track_id in self.participants:
            state = self.participants[track_id]
            state.manual_override = True
            state.override_status = new_status
            state.verification_method = "TEACHER_OVERRIDE"

        db.update_attendance_status(
            session_id=session_id,
            track_id=track_id,
            status=new_status.value,
            verification_method="TEACHER_OVERRIDE",
        )

        event = self._create_event(
            session_id=session_id,
            track_id=track_id,
            timestamp=now,
            event_type=AttendanceEventType.MANUAL_CORRECTION,
            description=f"Teacher manually updated Track-{track_id} status to {new_status.value}. Note: {teacher_note}",
        )
        if event:
            db.insert_attendance_event(
                event_id=event.event_id,
                session_id=event.session_id,
                track_id=event.track_id,
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                description=event.description,
            )

    def sync_to_database(
        self,
        db: DatabaseManager,
        session_id: str,
        evaluations: List[AttendanceEvaluation],
    ) -> None:
        """
        Persist attendance records and generated events to local SQLite storage.
        """
        for record in evaluations:
            record_id = f"att_{session_id}_{record.track_id}"
            db.upsert_attendance_record(
                record_id=record_id,
                session_id=session_id,
                track_id=record.track_id,
                first_seen_timestamp=record.first_seen_timestamp,
                last_seen_timestamp=record.last_seen_timestamp,
                total_detected_seconds=record.total_detected_seconds,
                sustained_presence_status=record.status.value,
            )
            if record.assigned_student_id:
                db.assign_student_to_track(
                    session_id=session_id,
                    track_id=record.track_id,
                    student_id=record.assigned_student_id,
                    verification_method=record.verification_method,
                )

        # Sync attendance events
        for evt in self.events:
            db.insert_attendance_event(
                event_id=evt.event_id,
                session_id=evt.session_id,
                track_id=evt.track_id,
                timestamp=evt.timestamp,
                event_type=evt.event_type.value,
                description=evt.description,
            )

    def export_attendance_csv(
        self,
        db: DatabaseManager,
        session_id: str,
        export_dir: Optional[Path] = None,
    ) -> Path:
        """
        Export verified session attendance to a local CSV file.
        """
        from modules.report_generator import ReportGenerator
        records = db.get_session_attendance(session_id)
        df = ReportGenerator.generate_attendance_dataframe(records=records)
        return ReportGenerator.export_attendance_csv(
            df=df,
            session_id=session_id,
            export_dir=export_dir,
        )

    def evaluate_track(
        self,
        track: Track,
        total_session_active_seconds: float,
        threshold_pct: Optional[float] = None,
    ) -> AttendanceStatus:
        """
        Backward-compatible single-track presence evaluation method.
        """
        if track.track_id not in self.participants:
            self.participants[track.track_id] = ParticipantObservationState(
                track_id=track.track_id,
                first_seen_timestamp=track.first_seen_timestamp,
                last_seen_timestamp=track.last_seen_timestamp,
                observation_timestamps=[track.first_seen_timestamp, track.last_seen_timestamp],
                presence_intervals=[(track.first_seen_timestamp, track.last_seen_timestamp)],
            )
        return self.evaluate_track_status(
            track_id=track.track_id,
            total_session_active_seconds=total_session_active_seconds,
            threshold_pct=threshold_pct,
        )
