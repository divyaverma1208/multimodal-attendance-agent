"""
Report Generator Module.

Compiles verified attendance records and multimodal activity summaries into
standardized Pandas DataFrames and downloadable CSV files.
"""

from datetime import datetime
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Formats and exports attendance data and classroom analytics."""

    @staticmethod
    def generate_attendance_dataframe(
        records: List[Dict[str, Any]],
        session_title: str = "",
        course_code: str = "",
    ) -> pd.DataFrame:
        """
        Convert raw attendance records into an educator-friendly Pandas DataFrame.

        Args:
            records: List of attendance record dictionaries from DatabaseManager.
            session_title: Human-readable title of the session.
            course_code: Course code.

        Returns:
            Formatted Pandas DataFrame.
        """
        if not records:
            return pd.DataFrame(
                columns=[
                    "Track ID",
                    "Assigned Student ID",
                    "Student Name",
                    "Attendance Status",
                    "First Seen",
                    "Last Seen",
                    "Duration (Min)",
                    "Verification Method",
                ]
            )

        rows = []
        for r in records:
            # Format timestamps into human-readable local time strings
            first_seen = (
                datetime.fromtimestamp(r["first_seen_timestamp"]).strftime("%H:%M:%S")
                if r.get("first_seen_timestamp")
                else "N/A"
            )
            last_seen = (
                datetime.fromtimestamp(r["last_seen_timestamp"]).strftime("%H:%M:%S")
                if r.get("last_seen_timestamp")
                else "N/A"
            )
            duration_min = round(float(r.get("total_detected_seconds", 0.0)) / 60.0, 1)

            rows.append({
                "Track ID": f"Track-{r.get('track_id')}",
                "Assigned Student ID": r.get("assigned_student_id") or "Unassigned",
                "Student Name": r.get("student_name") or "Anonymous Track",
                "Attendance Status": r.get("sustained_presence_status", "UNKNOWN"),
                "First Seen": first_seen,
                "Last Seen": last_seen,
                "Duration (Min)": duration_min,
                "Verification Method": r.get("verification_method", "ANONYMOUS"),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def export_attendance_csv(
        df: pd.DataFrame,
        session_id: str,
        export_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save the attendance DataFrame to a CSV file in the exports directory.

        Args:
            df: Formatted attendance DataFrame.
            session_id: Session identifier used for filename.
            export_dir: Target directory (defaults to settings.privacy.ensure_export_dir()).

        Returns:
            Path to the saved CSV file.
        """
        import re
        target_dir = export_dir or settings.privacy.ensure_export_dir()
        safe_session_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(session_id))
        filename = f"attendance_{safe_session_id}.csv"
        file_path = (target_dir / filename).resolve()
        if not str(file_path).startswith(str(target_dir.resolve())):
            raise ValueError(f"Path traversal detected: {filename} attempts to escape export directory.")
        df.to_csv(file_path, index=False)
        logger.info("Exported attendance report to: %s", file_path)
        return file_path

    @staticmethod
    def to_csv_bytes(df: pd.DataFrame) -> bytes:
        """Convert DataFrame to UTF-8 encoded CSV bytes for Streamlit downloads."""
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")
