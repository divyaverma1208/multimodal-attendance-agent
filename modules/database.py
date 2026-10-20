"""
Database Manager Module.

Provides local SQLite persistence for sessions, course rosters, anonymous attendance
ledgers, time-windowed activity metrics, and teaching suggestion history.
Enforces foreign key integrity, WAL mode, transaction safety, parameterized queries,
and strict privacy preservation (zero storage of raw video or audio).
"""

from contextlib import contextmanager
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception raised for database operational and integrity errors."""
    pass


class DatabaseManager:
    """Manages SQLite connections, schema initialization, and transactional queries."""

    # Allowed columns for dynamic updates to prevent SQL injection
    ALLOWED_SESSION_COLUMNS = {
        "course_code", "session_title", "status", "started_at",
        "concluded_at", "total_active_seconds", "sustained_threshold_pct",
        "video_source_type", "audio_enabled",
    }
    ALLOWED_STUDENT_COLUMNS = {"course_code", "full_name", "email"}

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize database manager with a configurable local path.

        Args:
            db_path: Path to the SQLite database file or ':memory:'.
                     Defaults to settings.db.db_path.
        """
        if db_path is None:
            self.db_path: Union[str, Path] = settings.db.db_path
        elif isinstance(db_path, str) and db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)

        # Ensure target directory exists for local file-based databases
        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys=ON;")

        self.init_db()

    def close(self) -> None:
        """Close persistent in-memory database connection if open."""
        if self._mem_conn is not None:
            try:
                self._mem_conn.close()
            except Exception:
                pass
            self._mem_conn = None

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Provide a transactional database connection context with WAL mode and foreign keys.

        Yields:
            Configured sqlite3.Connection instance with sqlite3.Row row factory.

        Raises:
            DatabaseError: If connection or transaction execution fails.
        """
        if self._mem_conn is not None:
            try:
                self._mem_conn.execute("PRAGMA foreign_keys=ON;")
                yield self._mem_conn
                self._mem_conn.commit()
            except sqlite3.Error as e:
                self._mem_conn.rollback()
                logger.error("Database transaction rolled back due to error: %s", e)
                raise DatabaseError(f"Database transaction error: {e}") from e
            except Exception as e:
                self._mem_conn.rollback()
                logger.error("Unexpected error in database transaction: %s", e)
                raise DatabaseError(f"Unexpected database transaction error: {e}") from e
            return

        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,
            )
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            logger.error("Failed to connect to SQLite database at %s: %s", self.db_path, e)
            raise DatabaseError(f"Database connection error: {e}") from e

        try:
            # Enable WAL mode for concurrency and enforce foreign key constraints
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            logger.error("Database transaction rolled back due to error: %s", e)
            raise DatabaseError(f"Database transaction error: {e}") from e
        except Exception as e:
            conn.rollback()
            logger.error("Unexpected error in database transaction: %s", e)
            raise DatabaseError(f"Unexpected database transaction error: {e}") from e
        finally:
            conn.close()

    def init_db(self) -> None:
        """
        Create tables and indexes automatically if they do not already exist.
        Guarantees that no raw video frames or raw audio samples are stored.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    course_code TEXT NOT NULL,
                    session_title TEXT NOT NULL,
                    status TEXT CHECK(status IN ('READY', 'ACTIVE', 'PAUSED', 'CONCLUDED')) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    concluded_at TIMESTAMP,
                    total_active_seconds REAL DEFAULT 0.0,
                    sustained_threshold_pct REAL DEFAULT 60.0,
                    video_source_type TEXT NOT NULL,
                    audio_enabled INTEGER DEFAULT 1
                );
                """
            )

            # 2. Students (Roster) table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    course_code TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # 3. Attendance records table (Anonymous tracks mapped to student roster)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance_records (
                    record_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    track_id INTEGER NOT NULL,
                    first_seen_timestamp REAL NOT NULL,
                    last_seen_timestamp REAL NOT NULL,
                    total_detected_seconds REAL NOT NULL,
                    sustained_presence_status TEXT CHECK(
                        sustained_presence_status IN ('SUSTAINED_PRESENT', 'PROVISIONAL', 'PARTIAL', 'TRANSIENT', 'ABSENT')
                    ) NOT NULL,
                    assigned_student_id TEXT REFERENCES students(student_id) ON DELETE SET NULL,
                    verification_method TEXT CHECK(
                        verification_method IN ('ANONYMOUS', 'TEACHER_CONFIRMED', 'OPT_IN_TOKEN', 'TEACHER_OVERRIDE', 'TEACHER_MANUAL_ENTRY')
                    ) DEFAULT 'ANONYMOUS',
                    verified_at TIMESTAMP,
                    correction_notes TEXT,
                    UNIQUE(session_id, track_id)
                );
                """
            )

            # 4. Attendance Events Log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    track_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # 5. Activity windows table (Aggregated time-window metrics, nullable for missing modalities)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_windows (
                    window_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    window_start_sec REAL NOT NULL,
                    window_end_sec REAL NOT NULL,
                    active_person_count INTEGER,
                    visual_movement_index REAL,
                    head_pose_variance REAL,
                    speech_activity_pct REAL,
                    rms_energy_mean REAL,
                    data_quality_score REAL DEFAULT 1.0,
                    data_quality_label TEXT DEFAULT 'HIGH',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # 6. Teaching suggestions log
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS teaching_suggestions (
                    suggestion_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                    timestamp_sec REAL NOT NULL,
                    trigger_rule_code TEXT NOT NULL,
                    suggestion_text TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Indexes for fast relational querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance_records(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON attendance_events(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_windows_session ON activity_windows(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_session ON teaching_suggestions(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_course ON students(course_code);")

    # =========================================================================
    # Session CRUD Operations
    # =========================================================================

    def create_session(
        self,
        session_id: str,
        course_code: str,
        session_title: str,
        video_source_type: str,
        audio_enabled: bool = True,
        sustained_threshold_pct: float = 60.0,
    ) -> None:
        """
        Create a new classroom session in READY state using parameterized SQL.

        Args:
            session_id: Unique session identifier.
            course_code: Course identifier (e.g. 'CS101').
            session_title: Human-readable topic or session title.
            video_source_type: 'webcam' or 'file'.
            audio_enabled: Whether audio stream processing is active.
            sustained_threshold_pct: Threshold percentage for sustained attendance.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, course_code, session_title, status,
                    video_source_type, audio_enabled, sustained_threshold_pct
                ) VALUES (?, ?, ?, 'READY', ?, ?, ?);
                """,
                (
                    session_id,
                    course_code,
                    session_title,
                    video_source_type,
                    1 if audio_enabled else 0,
                    sustained_threshold_pct,
                ),
            )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single session by its unique ID.

        Args:
            session_id: Unique session identifier.

        Returns:
            Dictionary representation of the session, or None if not found.
        """
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?;", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        course_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List sessions ordered by creation date descending with optional pagination and filtering.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            course_code: Optional course filter.

        Returns:
            List of session record dictionaries.
        """
        with self.get_connection() as conn:
            if course_code:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions 
                    WHERE course_code = ? 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?;
                    """,
                    (course_code, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM sessions 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?;
                    """,
                    (limit, offset),
                ).fetchall()
            return [dict(r) for r in rows]

    def update_session_status(
        self,
        session_id: str,
        status: str,
        total_active_seconds: Optional[float] = None,
        started_at: Optional[str] = None,
        concluded_at: Optional[str] = None,
    ) -> None:
        """
        Update lifecycle state and durations for an existing session.

        Args:
            session_id: Target session identifier.
            status: New status ('READY', 'ACTIVE', 'PAUSED', 'CONCLUDED').
            total_active_seconds: Cumulative active duration.
            started_at: ISO timestamp of start.
            concluded_at: ISO timestamp of conclusion.
        """
        updates: List[str] = ["status = ?"]
        params: List[Any] = [status]

        if total_active_seconds is not None:
            updates.append("total_active_seconds = ?")
            params.append(total_active_seconds)
        if started_at is not None:
            updates.append("started_at = ?")
            params.append(started_at)
        if concluded_at is not None:
            updates.append("concluded_at = ?")
            params.append(concluded_at)

        params.append(session_id)
        query = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?;"

        with self.get_connection() as conn:
            conn.execute(query, tuple(params))

    def update_session(self, session_id: str, **kwargs: Any) -> bool:
        """
        Update arbitrary session fields safely using column validation and parameterization.

        Args:
            session_id: Target session ID.
            **kwargs: Field names and values to update.

        Returns:
            True if a record was updated, False otherwise.
        """
        if not kwargs:
            return False

        invalid_cols = set(kwargs.keys()) - self.ALLOWED_SESSION_COLUMNS
        if invalid_cols:
            raise ValueError(f"Invalid column names for session update: {invalid_cols}")

        updates = [f"{col} = ?" for col in kwargs.keys()]
        params = list(kwargs.values()) + [session_id]
        query = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?;"

        with self.get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and cascade deletion to all associated attendance records,
        events, activity windows, and teaching suggestions.

        Args:
            session_id: Unique session identifier to delete.

        Returns:
            True if the session existed and was deleted, False otherwise.
        """
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
            return cursor.rowcount > 0

    # =========================================================================
    # Participant Roster Operations
    # =========================================================================

    def insert_student(
        self,
        student_id: str,
        course_code: str,
        full_name: str,
        email: Optional[str] = None,
    ) -> None:
        """
        Add or update an enrolled student in the course roster.

        Args:
            student_id: Unique student identifier.
            course_code: Course identifier (e.g. 'CS101').
            full_name: Student full name.
            email: Optional contact email.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO students (student_id, course_code, full_name, email)
                VALUES (?, ?, ?, ?);
                """,
                (student_id, course_code, full_name, email),
            )

    def bulk_insert_students(self, students: List[Dict[str, Any]]) -> int:
        """
        Insert multiple students into the roster atomically within a transaction.

        Args:
            students: List of student dictionaries with 'student_id', 'course_code',
                      'full_name', and optional 'email'.

        Returns:
            Number of student records inserted or updated.
        """
        if not students:
            return 0

        params = [
            (s["student_id"], s["course_code"], s["full_name"], s.get("email"))
            for s in students
        ]

        with self.get_connection() as conn:
            cursor = conn.executemany(
                """
                INSERT OR REPLACE INTO students (student_id, course_code, full_name, email)
                VALUES (?, ?, ?, ?);
                """,
                params,
            )
            return cursor.rowcount

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single student record by student ID.

        Args:
            student_id: Unique student identifier.

        Returns:
            Dictionary representing student record, or None if not found.
        """
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM students WHERE student_id = ?;", (student_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_students(self, course_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve student roster, optionally filtered by course.

        Args:
            course_code: Optional course identifier filter.

        Returns:
            List of student dictionaries ordered by full name.
        """
        with self.get_connection() as conn:
            if course_code:
                rows = conn.execute(
                    "SELECT * FROM students WHERE course_code = ? ORDER BY full_name ASC;",
                    (course_code,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM students ORDER BY full_name ASC;").fetchall()
            return [dict(r) for r in rows]

    def list_students(self, course_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Alias for get_students."""
        return self.get_students(course_code=course_code)

    def update_student(
        self,
        student_id: str,
        full_name: Optional[str] = None,
        course_code: Optional[str] = None,
        email: Optional[str] = None,
    ) -> bool:
        """
        Update student attributes safely.

        Args:
            student_id: Student ID to update.
            full_name: Updated student name.
            course_code: Updated course code.
            email: Updated contact email.

        Returns:
            True if student was updated, False if student was not found.
        """
        updates: List[str] = []
        params: List[Any] = []

        if full_name is not None:
            updates.append("full_name = ?")
            params.append(full_name)
        if course_code is not None:
            updates.append("course_code = ?")
            params.append(course_code)
        if email is not None:
            updates.append("email = ?")
            params.append(email)

        if not updates:
            return False

        params.append(student_id)
        query = f"UPDATE students SET {', '.join(updates)} WHERE student_id = ?;"

        with self.get_connection() as conn:
            cursor = conn.execute(query, tuple(params))
            return cursor.rowcount > 0

    def delete_student(self, student_id: str) -> bool:
        """
        Remove a student from the roster. Associated attendance records will have
        assigned_student_id set to NULL to preserve anonymous attendance metrics.

        Args:
            student_id: Unique student ID to remove.

        Returns:
            True if deleted, False otherwise.
        """
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM students WHERE student_id = ?;", (student_id,))
            return cursor.rowcount > 0

    # =========================================================================
    # Attendance Records & Manual Teacher Corrections
    # =========================================================================

    def upsert_attendance_record(
        self,
        record_id: str,
        session_id: str,
        track_id: int,
        first_seen_timestamp: float,
        last_seen_timestamp: float,
        total_detected_seconds: float,
        sustained_presence_status: str,
        assigned_student_id: Optional[str] = None,
        verification_method: str = "ANONYMOUS",
        correction_notes: Optional[str] = None,
    ) -> None:
        """
        Upsert a tracked subject's presence log using parameterized SQL.

        Args:
            record_id: Unique record ID.
            session_id: Session identifier.
            track_id: Anonymous tracking identifier.
            first_seen_timestamp: First observed timestamp.
            last_seen_timestamp: Last observed timestamp.
            total_detected_seconds: Total detected presence duration.
            sustained_presence_status: Status classification.
            assigned_student_id: Optional linked student ID.
            verification_method: Verification method.
            correction_notes: Optional teacher notes.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO attendance_records (
                    record_id, session_id, track_id, first_seen_timestamp,
                    last_seen_timestamp, total_detected_seconds, sustained_presence_status,
                    assigned_student_id, verification_method, correction_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, track_id) DO UPDATE SET
                    last_seen_timestamp = excluded.last_seen_timestamp,
                    total_detected_seconds = excluded.total_detected_seconds,
                    sustained_presence_status = excluded.sustained_presence_status,
                    assigned_student_id = COALESCE(excluded.assigned_student_id, attendance_records.assigned_student_id),
                    verification_method = COALESCE(excluded.verification_method, attendance_records.verification_method),
                    correction_notes = COALESCE(excluded.correction_notes, attendance_records.correction_notes);
                """,
                (
                    record_id,
                    session_id,
                    track_id,
                    first_seen_timestamp,
                    last_seen_timestamp,
                    total_detected_seconds,
                    sustained_presence_status,
                    assigned_student_id,
                    verification_method,
                    correction_notes,
                ),
            )

    insert_attendance_record = upsert_attendance_record

    def get_attendance_record(self, session_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single attendance record by session ID and track ID.

        Args:
            session_id: Session identifier.
            track_id: Tracking identifier.

        Returns:
            Dictionary of record fields, or None if not found.
        """
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT a.*, s.full_name AS student_name, s.course_code
                FROM attendance_records a
                LEFT JOIN students s ON a.assigned_student_id = s.student_id
                WHERE a.session_id = ? AND a.track_id = ?;
                """,
                (session_id, track_id),
            ).fetchone()
            return dict(row) if row else None

    def get_session_attendance(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve attendance ledger with linked student names if assigned.

        Args:
            session_id: Target session identifier.

        Returns:
            List of attendance record dictionaries sorted by track ID.
        """
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT 
                    a.record_id,
                    a.session_id,
                    a.track_id,
                    a.first_seen_timestamp,
                    a.last_seen_timestamp,
                    a.total_detected_seconds,
                    a.sustained_presence_status,
                    a.assigned_student_id,
                    a.verification_method,
                    a.verified_at,
                    a.correction_notes,
                    s.full_name AS student_name,
                    s.course_code
                FROM attendance_records a
                LEFT JOIN students s ON a.assigned_student_id = s.student_id
                WHERE a.session_id = ?
                ORDER BY a.track_id ASC;
                """,
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def assign_student_to_track(
        self,
        session_id: str,
        track_id: int,
        student_id: Optional[str],
        verification_method: str = "TEACHER_CONFIRMED",
    ) -> None:
        """
        Manually link an anonymous tracking ID to an enrolled student record.
        Strictly requires teacher confirmation or explicit token.

        Args:
            session_id: Session identifier.
            track_id: Anonymous tracking identifier.
            student_id: Enrolled student identifier, or None to unassign.
            verification_method: Verification method label.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE attendance_records
                SET assigned_student_id = ?,
                    verification_method = ?,
                    verified_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND track_id = ?;
                """,
                (student_id, verification_method, session_id, track_id),
            )

    def update_attendance_status(
        self,
        session_id: str,
        track_id: int,
        status: str,
        verification_method: str = "TEACHER_OVERRIDE",
        notes: Optional[str] = None,
    ) -> None:
        """
        Support authorized teacher correction/override of attendance status.

        Args:
            session_id: Target session ID.
            track_id: Target track ID.
            status: New attendance status ('SUSTAINED_PRESENT', 'PROVISIONAL', etc.).
            verification_method: Verification method.
            notes: Optional explanatory notes.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE attendance_records
                SET sustained_presence_status = ?,
                    verification_method = ?,
                    verified_at = CURRENT_TIMESTAMP,
                    correction_notes = COALESCE(?, correction_notes)
                WHERE session_id = ? AND track_id = ?;
                """,
                (status, verification_method, notes, session_id, track_id),
            )

    def override_attendance_record(
        self,
        session_id: str,
        track_id: int,
        status: str,
        student_id: Optional[str] = None,
        verification_method: str = "TEACHER_OVERRIDE",
        reason: Optional[str] = None,
    ) -> bool:
        """
        Comprehensive manual attendance correction method.
        Updates status, optional student assignment, and appends an audit event.

        Args:
            session_id: Session identifier.
            track_id: Track identifier.
            status: New sustained presence status.
            student_id: Optional student ID assignment.
            verification_method: Override method descriptor.
            reason: Explanation for manual override.

        Returns:
            True if record was found and updated, False otherwise.
        """
        with self.get_connection() as conn:
            # Check existing record
            existing = conn.execute(
                "SELECT * FROM attendance_records WHERE session_id = ? AND track_id = ?;",
                (session_id, track_id),
            ).fetchone()

            if not existing:
                return False

            if student_id is not None:
                conn.execute(
                    """
                    UPDATE attendance_records
                    SET sustained_presence_status = ?,
                        assigned_student_id = ?,
                        verification_method = ?,
                        verified_at = CURRENT_TIMESTAMP,
                        correction_notes = ?
                    WHERE session_id = ? AND track_id = ?;
                    """,
                    (status, student_id, verification_method, reason, session_id, track_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE attendance_records
                    SET sustained_presence_status = ?,
                        verification_method = ?,
                        verified_at = CURRENT_TIMESTAMP,
                        correction_notes = ?
                    WHERE session_id = ? AND track_id = ?;
                    """,
                    (status, verification_method, reason, session_id, track_id),
                )

            # Log audit event
            import time
            event_id = f"evt_override_{session_id}_{track_id}_{int(time.time() * 1000)}"
            desc = f"Teacher override on Track-{track_id}: status={status}. Reason: {reason or 'None provided'}"
            conn.execute(
                """
                INSERT INTO attendance_events (event_id, session_id, track_id, timestamp, event_type, description)
                VALUES (?, ?, ?, ?, 'MANUAL_CORRECTION', ?);
                """,
                (event_id, session_id, track_id, time.time(), desc),
            )
            return True

    manual_attendance_override = override_attendance_record

    def create_manual_attendance_record(
        self,
        record_id: str,
        session_id: str,
        track_id: int,
        status: str = "SUSTAINED_PRESENT",
        student_id: Optional[str] = None,
        verification_method: str = "TEACHER_MANUAL_ENTRY",
        reason: Optional[str] = "Manual entry by teacher",
    ) -> None:
        """
        Insert a manual attendance entry for a student who was not automatically tracked.

        Args:
            record_id: Unique record ID.
            session_id: Session identifier.
            track_id: Track ID (typically negative or distinct for manual entries).
            status: Attendance status.
            student_id: Enrolled student ID.
            verification_method: Verification method tag.
            reason: Explanatory note.
        """
        import time
        now = time.time()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO attendance_records (
                    record_id, session_id, track_id, first_seen_timestamp,
                    last_seen_timestamp, total_detected_seconds, sustained_presence_status,
                    assigned_student_id, verification_method, verified_at, correction_notes
                ) VALUES (?, ?, ?, ?, ?, 0.0, ?, ?, ?, CURRENT_TIMESTAMP, ?);
                """,
                (
                    record_id,
                    session_id,
                    track_id,
                    now,
                    now,
                    status,
                    student_id,
                    verification_method,
                    reason,
                ),
            )

    # =========================================================================
    # Timestamped Attendance Events
    # =========================================================================

    def insert_attendance_event(
        self,
        event_id: str,
        session_id: str,
        track_id: int,
        timestamp: float,
        event_type: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Log a timestamped attendance event using parameterized SQL.

        Args:
            event_id: Unique event identifier.
            session_id: Session identifier.
            track_id: Anonymous track ID.
            timestamp: Event epoch timestamp.
            event_type: Event classification ('ENTRY', 'EXIT', 'SUSTAINED_PRESENCE_MET', etc.).
            description: Human-readable context.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO attendance_events (event_id, session_id, track_id, timestamp, event_type, description)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (event_id, session_id, track_id, timestamp, event_type, description),
            )

    def log_attendance_event(
        self,
        event_id: str,
        session_id: str,
        track_id: int,
        timestamp: float,
        event_type: str,
        description: Optional[str] = None,
    ) -> None:
        """Convenience alias for insert_attendance_event."""
        self.insert_attendance_event(
            event_id=event_id,
            session_id=session_id,
            track_id=track_id,
            timestamp=timestamp,
            event_type=event_type,
            description=description,
        )

    def get_attendance_events(
        self,
        session_id: str,
        track_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve timestamped attendance events for a session, optionally filtered by track.

        Args:
            session_id: Session identifier.
            track_id: Optional track ID filter.

        Returns:
            List of event record dictionaries ordered by timestamp ascending.
        """
        with self.get_connection() as conn:
            if track_id is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM attendance_events 
                    WHERE session_id = ? AND track_id = ? 
                    ORDER BY timestamp ASC;
                    """,
                    (session_id, track_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM attendance_events 
                    WHERE session_id = ? 
                    ORDER BY timestamp ASC;
                    """,
                    (session_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    # =========================================================================
    # Aggregated Activity Measurements (Time-Window Metrics)
    # =========================================================================

    def insert_activity_window(
        self,
        window_id: str,
        session_id: str,
        window_start_sec: float,
        window_end_sec: float,
        active_person_count: Optional[int] = None,
        visual_movement_index: Optional[float] = None,
        head_pose_variance: Optional[float] = None,
        speech_activity_pct: Optional[float] = None,
        rms_energy_mean: Optional[float] = None,
        data_quality_score: float = 1.0,
        data_quality_label: str = "HIGH",
    ) -> None:
        """
        Record aggregated multimodal metrics for a time window using parameterized SQL.
        Explicitly supports NULL values for missing modalities to prevent conflating
        missing hardware signals with true zero measurements.

        Args:
            window_id: Unique window identifier.
            session_id: Session identifier.
            window_start_sec: Window start relative offset (seconds).
            window_end_sec: Window end relative offset (seconds).
            active_person_count: Detected person count (None if vision unavailable).
            visual_movement_index: Aggregated movement index (None if vision unavailable).
            head_pose_variance: Head pose variance (None if vision unavailable).
            speech_activity_pct: Speech activity proportion (None if audio unavailable).
            rms_energy_mean: Mean acoustic RMS energy (None if audio unavailable).
            data_quality_score: Quality index in [0.0, 1.0].
            data_quality_label: Quality rating ('HIGH', 'DEGRADED', 'UNAVAILABLE').
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO activity_windows (
                    window_id, session_id, window_start_sec, window_end_sec,
                    active_person_count, visual_movement_index, head_pose_variance,
                    speech_activity_pct, rms_energy_mean, data_quality_score, data_quality_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    window_id,
                    session_id,
                    window_start_sec,
                    window_end_sec,
                    active_person_count,
                    visual_movement_index,
                    head_pose_variance,
                    speech_activity_pct,
                    rms_energy_mean,
                    data_quality_score,
                    data_quality_label,
                ),
            )

    def get_activity_windows(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve chronologically ordered activity windows for a session.

        Args:
            session_id: Target session identifier.

        Returns:
            List of window dictionaries sorted by window_start_sec ascending.
        """
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM activity_windows
                WHERE session_id = ?
                ORDER BY window_start_sec ASC;
                """,
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_latest_activity_window(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recently recorded activity window for a session.

        Args:
            session_id: Target session identifier.

        Returns:
            Dictionary of the latest window, or None if none exist.
        """
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM activity_windows
                WHERE session_id = ?
                ORDER BY window_start_sec DESC
                LIMIT 1;
                """,
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    # =========================================================================
    # Teaching Suggestions (Recommendation History)
    # =========================================================================

    def insert_teaching_suggestion(
        self,
        suggestion_id: str,
        session_id: str,
        timestamp_sec: float,
        trigger_rule_code: str,
        suggestion_text: str,
    ) -> None:
        """
        Log a triggered pedagogical cue using parameterized SQL.

        Args:
            suggestion_id: Unique suggestion ID.
            session_id: Associated session ID.
            timestamp_sec: Timestamp in seconds relative to session start.
            trigger_rule_code: Standardized RuleCode string.
            suggestion_text: Human-readable actionable guidance.
        """
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO teaching_suggestions (
                    suggestion_id, session_id, timestamp_sec, trigger_rule_code, suggestion_text
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (suggestion_id, session_id, timestamp_sec, trigger_rule_code, suggestion_text),
            )

    def log_teaching_suggestion(
        self,
        suggestion_id: str,
        session_id: str,
        timestamp_sec: float,
        trigger_rule_code: str,
        suggestion_text: str,
    ) -> None:
        """Convenience alias for insert_teaching_suggestion."""
        self.insert_teaching_suggestion(
            suggestion_id=suggestion_id,
            session_id=session_id,
            timestamp_sec=timestamp_sec,
            trigger_rule_code=trigger_rule_code,
            suggestion_text=suggestion_text,
        )

    def get_teaching_suggestions(
        self,
        session_id: str,
        unacknowledged_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve pedagogical suggestions triggered during a session.

        Args:
            session_id: Target session identifier.
            unacknowledged_only: If True, returns only unacknowledged items.

        Returns:
            List of suggestion dictionaries ordered by timestamp descending.
        """
        with self.get_connection() as conn:
            if unacknowledged_only:
                rows = conn.execute(
                    """
                    SELECT * FROM teaching_suggestions
                    WHERE session_id = ? AND acknowledged = 0
                    ORDER BY timestamp_sec DESC;
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM teaching_suggestions
                    WHERE session_id = ?
                    ORDER BY timestamp_sec DESC;
                    """,
                    (session_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def acknowledge_suggestion(self, suggestion_id: str) -> bool:
        """
        Mark a pedagogical suggestion as acknowledged by the teacher.

        Args:
            suggestion_id: Unique suggestion identifier.

        Returns:
            True if the suggestion was found and updated, False otherwise.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE teaching_suggestions SET acknowledged = 1 WHERE suggestion_id = ?;",
                (suggestion_id,),
            )
            return cursor.rowcount > 0

    # =========================================================================
    # Configurable Record Retention & Data Minimization
    # =========================================================================

    def purge_old_records(self, days_threshold: Optional[int] = None) -> int:
        """
        Purge sessions and all cascading telemetry created older than N days.
        Default days_threshold is loaded from settings.privacy.auto_purge_days.

        Args:
            days_threshold: Number of days to retain records.

        Returns:
            Number of purged sessions.
        """
        if days_threshold is None:
            days_threshold = settings.privacy.auto_purge_days

        days_threshold = max(0, int(days_threshold))

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM sessions
                WHERE created_at < datetime('now', '-' || ? || ' days');
                """,
                (str(days_threshold),),
            )
            return cursor.rowcount

    def purge_session_records(self, session_id: str) -> bool:
        """
        Explicitly purge all records, events, telemetry, and suggestions for a specific session.

        Args:
            session_id: Session ID to purge.

        Returns:
            True if session existed and was purged, False otherwise.
        """
        return self.delete_session(session_id)

    def delete_telemetry_for_session(self, session_id: str) -> Tuple[int, int]:
        """
        Delete aggregated activity windows and suggestion logs for a session
        while retaining verified attendance records for audit and compliance.

        Args:
            session_id: Session identifier.

        Returns:
            Tuple of (deleted_windows_count, deleted_suggestions_count).
        """
        with self.get_connection() as conn:
            cur1 = conn.execute(
                "DELETE FROM activity_windows WHERE session_id = ?;", (session_id,)
            )
            cur2 = conn.execute(
                "DELETE FROM teaching_suggestions WHERE session_id = ?;", (session_id,)
            )
            return cur1.rowcount, cur2.rowcount

    # =========================================================================
    # Privacy Architecture Verification
    # =========================================================================

    def verify_no_raw_media_stored(self) -> bool:
        """
        Verify the architectural privacy constraint: assert that NO table in SQLite
        stores raw video frames or raw audio waveforms (BLOB types).

        Returns:
            True if all tables comply with zero raw media storage.
        """
        with self.get_connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()

            for table_row in tables:
                table_name = table_row["name"]
                columns = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
                for col in columns:
                    col_name = col["name"].lower()
                    col_type = col["type"].upper()
                    if "BLOB" in col_type or "IMAGE" in col_type or "FRAME" in col_type:
                        logger.error(
                            "Privacy violation detected: Column '%s' in table '%s' has disallowed type '%s'",
                            col_name,
                            table_name,
                            col_type,
                        )
                        return False
                    if col_name in ("raw_frame", "raw_audio", "video_buffer", "audio_samples"):
                        logger.error(
                            "Privacy violation detected: Disallowed media column '%s' in table '%s'",
                            col_name,
                            table_name,
                        )
                        return False

        return True
