"""
Read-only tool registry for the Phase 6 AI Assistant.

This is the ONLY component the LLM can influence that touches data. Every tool
is a thin wrapper around an existing Phase 4/5 service method, so the assistant
inherits the application's business logic instead of duplicating it.

Security properties:
- Dispatch is by exact name from a frozen allowlist. There is no dynamic
  attribute access, no `getattr` on a service, and no SQL anywhere in this file.
- Every tool is read-only. Write methods on the underlying services are never
  referenced.
- Arguments arrive pre-validated from `src.ai.intent_schema`.
"""

from typing import Any, Callable, Dict, List, Optional

import config
from src.ai import intent_schema as schema
from src.ai.prompts import SYSTEM_FACTS
from src.services.student_service import StudentService
from src.services.session_service import SessionService
from src.services.statistics_service import StatisticsService
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Errors
# ============================================================================

class ToolError(Exception):
    """Base exception for tool execution failures."""
    pass


class UnknownToolError(ToolError):
    """Raised when a tool name is not in the allowlist."""
    pass


class InvalidToolArgumentsError(ToolError):
    """Raised when arguments are missing or unusable."""
    pass


# ============================================================================
# Notes injected into every tool payload so the model cannot infer absence
# ============================================================================

_DATA_MODEL_NOTE = (
    "Absence counts and attendance percentages are NOT available: this database "
    "stores only recorded present/late events and has no expected-student roster. "
    "Do not state absence or attendance percentage as fact."
)


def _student_view(student: Dict[str, Any]) -> Dict[str, Any]:
    """Project a student row down to the fields safe to send to the model."""
    return {
        "student_db_id": student.get("id"),
        "student_id": student.get("student_id"),
        "name": student.get("name"),
        "department": student.get("department") or "—",
        "is_active": bool(student.get("is_active")),
    }


def _student_stats_view(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Project student statistics, replacing unsupported figures with None."""
    return {
        "student_id": stats.get("student_id"),
        "name": stats.get("name"),
        "department": stats.get("department"),
        "is_active": stats.get("is_active"),
        "recorded_attendance_events": stats.get("total_marked", 0),
        "present_count": stats.get("present_count", 0),
        "late_count": stats.get("late_count", 0),
        "punctuality_percentage": stats.get("punctuality_percentage", 0.0),
        "attendance_percentage": None,
        "absent_count": None,
        "note": _DATA_MODEL_NOTE,
    }


def _session_stats_view(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Project session statistics, replacing unsupported figures with None."""
    return {
        "session_db_id": stats.get("session_id"),
        "session_name": stats.get("session_name"),
        "course_code": stats.get("course_code"),
        "start_time": stats.get("start_time"),
        "end_time": stats.get("end_time"),
        "status": stats.get("status"),
        "late_threshold_minutes": stats.get("late_threshold_minutes"),
        "recorded_attendance_events": stats.get("total_marked", 0),
        "present_count": stats.get("present_count", 0),
        "late_count": stats.get("late_count", 0),
        "punctuality_percentage": stats.get("punctuality_percentage", 0.0),
        "expected_students": None,
        "attendance_percentage": None,
        "absent_count": None,
        "note": _DATA_MODEL_NOTE,
    }


def _history_row_view(row: Dict[str, Any]) -> Dict[str, Any]:
    """Project one attendance record row down to safe fields."""
    return {
        "student_id": row.get("student_id"),
        "student_name": row.get("student_name"),
        "session_name": row.get("session_name"),
        "course_code": row.get("course_code"),
        "session_start_time": row.get("session_start_time"),
        "status": row.get("status"),
        "marked_at": row.get("marked_at"),
    }


# ============================================================================
# Registry
# ============================================================================

class AttendanceToolRegistry:
    """
    Executes validated intents against the existing application services.

    All data access for the AI Assistant flows through this class.
    """

    def __init__(self, database_path=None):
        """
        Initialize the registry and its underlying services.

        Args:
            database_path: Optional database path override (used by tests).
        """
        self.database_path = database_path
        self.student_service = StudentService(database_path)
        self.session_service = SessionService(database_path)
        self.statistics_service = StatisticsService(database_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_tool_names(self) -> List[str]:
        """Return the names of every callable tool (for documentation/tests)."""
        return sorted(name for name in dir(self) if name.startswith("tool_"))

    def dispatch(self, tool_call: schema.ToolCall) -> Dict[str, Any]:
        """
        Execute a validated intent and return a structured result.

        The tool name is derived from the intent through an explicit mapping -
        it is never taken from model output as a method name.

        Args:
            tool_call: A validated `ToolCall`.

        Returns:
            Structured result dict, always containing an `ok` key.

        Raises:
            UnknownToolError: If the intent has no registered tool.
        """
        handler = _INTENT_TO_TOOL.get(tool_call.intent)
        if handler is None:
            raise UnknownToolError(f"No tool registered for intent '{tool_call.intent}'.")

        method = getattr(self, handler, None)
        if method is None or handler not in self.list_tool_names():
            # Defence in depth: never dispatch to an unregistered method.
            raise UnknownToolError(f"No tool registered for intent '{tool_call.intent}'.")

        # Only pass arguments this specific tool declares. An argument name can
        # be individually valid (e.g. "limit") yet not belong to this tool, so
        # dispatching it blindly would raise TypeError.
        accepted = self._accepted_arguments(handler)
        filtered = {
            key: value for key, value in tool_call.arguments.items()
            if key in accepted
        }

        return method(**filtered)

    @staticmethod
    def _accepted_arguments(handler: str) -> set:
        """
        Return the argument names a tool method actually accepts.

        Read from the method signature rather than a hand-maintained list so the
        allowlist cannot drift out of sync with the code.
        """
        import inspect

        signature = inspect.signature(getattr(AttendanceToolRegistry, handler))
        return {
            name
            for name, param in signature.parameters.items()
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def resolve_student(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve a student reference to exactly one student.

        Handles an explicit `student_db_id`, an exact student ID, or a
        case-insensitive name match.

        Returns:
            Dict with one of:
              {"status": "resolved", "student": {...}}
              {"status": "ambiguous", "candidates": [...]}
              {"status": "not_found", "query": "..."}
        """
        db_id = arguments.get("student_db_id")
        if isinstance(db_id, int) and db_id > 0:
            student = self.student_service.get_student_by_id(db_id)
            if not student:
                return {"status": "not_found", "query": str(db_id)}
            return {"status": "resolved", "student": _student_view(student)}

        query = arguments.get("student_query")
        if not query:
            return {"status": "not_found", "query": ""}

        query = str(query).strip()
        if not query:
            return {"status": "not_found", "query": ""}

        # Exact student ID match wins outright.
        try:
            exact = self.student_service.get_student_by_student_id(query)
        except Exception:
            exact = None
        if exact:
            return {"status": "resolved", "student": _student_view(exact)}

        all_students = self.student_service.get_all_students()
        needle = query.lower()
        matches = [
            s for s in all_students
            if needle in str(s.get("name", "")).lower()
            or needle == str(s.get("student_id", "")).lower()
        ]

        if not matches:
            return {"status": "not_found", "query": query}
        if len(matches) == 1:
            return {"status": "resolved", "student": _student_view(matches[0])}

        return {
            "status": "ambiguous",
            "query": query,
            "candidates": [
                {
                    "student_id": s.get("student_id"),
                    "name": s.get("name"),
                    "department": s.get("department") or "—",
                }
                for s in matches[:8]
            ],
        }

    def resolve_session(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve a session reference to exactly one session.

        Returns:
            {"status": "resolved", "session": {...}} | {"status": "ambiguous", ...} | {"status": "not_found", ...}
        """
        db_id = arguments.get("session_db_id")
        if isinstance(db_id, int) and db_id > 0:
            try:
                session = self.session_service.get_session(db_id)
            except Exception:
                session = None
            if not session:
                return {"status": "not_found", "query": str(db_id)}
            return {"status": "resolved", "session": self._session_brief(session)}

        query = arguments.get("session_query")
        if not query:
            return {"status": "not_found", "query": ""}

        query = str(query).strip()
        if not query:
            return {"status": "not_found", "query": ""}

        if query.isdigit():
            try:
                session = self.session_service.get_session(int(query))
            except Exception:
                session = None
            if session:
                return {"status": "resolved", "session": self._session_brief(session)}

        sessions = self.session_service.get_all_sessions()
        needle = query.lower()
        matches = [
            s for s in sessions
            if needle in str(s.get("session_name", "")).lower()
            or needle == str(s.get("course_code") or "").lower()
        ]

        if not matches:
            return {"status": "not_found", "query": query}
        if len(matches) == 1:
            return {"status": "resolved", "session": self._session_brief(matches[0])}

        # Prefer the most recent session when several match.
        matches.sort(key=lambda s: str(s.get("start_time") or ""), reverse=True)
        return {
            "status": "ambiguous",
            "query": query,
            "candidates": [
                {
                    "session_db_id": s.get("id"),
                    "session_name": s.get("session_name"),
                    "course_code": s.get("course_code"),
                    "start_time": s.get("start_time"),
                }
                for s in matches[:8]
            ],
        }

    @staticmethod
    def _session_brief(session: Dict[str, Any]) -> Dict[str, Any]:
        """Project a session row down to safe fields."""
        return {
            "session_db_id": session.get("id"),
            "session_name": session.get("session_name"),
            "course_code": session.get("course_code"),
            "start_time": session.get("start_time"),
            "end_time": session.get("end_time"),
            "status": session.get("status"),
            "late_threshold_minutes": session.get("late_threshold_minutes"),
        }

    # ------------------------------------------------------------------
    # Tools (one per intent)
    # ------------------------------------------------------------------

    def tool_overall_stats(self) -> Dict[str, Any]:
        """System-wide totals: students, sessions, attendance records, punctuality."""
        stats = self.statistics_service.get_overall_dashboard_stats()
        return {
            "ok": True,
            "tool": "overall_stats",
            "data": {
                "total_students": stats.get("total_students", 0),
                "active_students": stats.get("active_students", 0),
                "total_sessions": stats.get("total_sessions", 0),
                "active_sessions": stats.get("active_sessions", 0),
                "closed_sessions": stats.get("closed_sessions", 0),
                "total_attendance_records": stats.get("total_records", 0),
                "total_present": stats.get("total_present", 0),
                "total_late": stats.get("total_late", 0),
                "punctuality_rate": stats.get("punctuality_rate", 0.0),
                "attendance_rate": None,
                "absent_count": None,
                "note": _DATA_MODEL_NOTE,
            },
        }

    def tool_student_lookup(self, student_query: Optional[str] = None) -> Dict[str, Any]:
        """Find a student by name or student ID."""
        if not student_query:
            raise InvalidToolArgumentsError("A student name or ID is required.")

        resolved = self.resolve_student({"student_query": student_query})
        if resolved["status"] == "resolved":
            return {
                "ok": True,
                "tool": "student_lookup",
                "data": {"matched_students": [resolved["student"]]},
            }
        if resolved["status"] == "ambiguous":
            return {
                "ok": True,
                "tool": "student_lookup",
                "needs_clarification": True,
                "data": {"matched_students": resolved["candidates"]},
            }
        return {
            "ok": True,
            "tool": "student_lookup",
            "not_found": True,
            "data": {"query": resolved.get("query", student_query)},
        }

    def tool_student_stats(self, student_query: Optional[str] = None,
                           student_db_id: Optional[int] = None) -> Dict[str, Any]:
        """Recorded present/late counts and punctuality for one student."""
        resolved = self.resolve_student(
            {"student_query": student_query, "student_db_id": student_db_id}
        )
        if resolved["status"] == "ambiguous":
            return {
                "ok": True,
                "tool": "student_stats",
                "needs_clarification": True,
                "data": {"candidates": resolved["candidates"]},
            }
        if resolved["status"] == "not_found":
            return {
                "ok": True,
                "tool": "student_stats",
                "not_found": True,
                "data": {"query": resolved.get("query", "")},
            }

        student = resolved["student"]
        stats = self.statistics_service.get_student_statistics(student["student_db_id"])
        if not stats:
            return {
                "ok": True,
                "tool": "student_stats",
                "not_found": True,
                "data": {"query": student.get("name")},
            }
        return {"ok": True, "tool": "student_stats", "data": _student_stats_view(stats)}

    def tool_student_history(self, student_query: Optional[str] = None,
                             student_db_id: Optional[int] = None,
                             limit: Optional[int] = None,
                             status_filter: Optional[str] = None) -> Dict[str, Any]:
        """Per-session attendance rows for one student."""
        resolved = self.resolve_student(
            {"student_query": student_query, "student_db_id": student_db_id}
        )
        if resolved["status"] == "ambiguous":
            return {
                "ok": True,
                "tool": "student_history",
                "needs_clarification": True,
                "data": {"candidates": resolved["candidates"]},
            }
        if resolved["status"] == "not_found":
            return {
                "ok": True,
                "tool": "student_history",
                "not_found": True,
                "data": {"query": resolved.get("query", "")},
            }

        student = resolved["student"]
        rows = self.statistics_service.get_filtered_history(
            student_id_str=student.get("student_id"),
            status=status_filter,
        )
        capped = rows[: limit or 20]
        return {
            "ok": True,
            "tool": "student_history",
            "data": {
                "student": _student_view(student),
                "total_records": len(rows),
                "returned_records": len(capped),
                "records": [_history_row_view(r) for r in capped],
                "note": _DATA_MODEL_NOTE,
            },
        }

    def tool_student_ranking(self, order: str = "asc", limit: Optional[int] = None,
                             threshold: Optional[float] = None) -> Dict[str, Any]:
        """Rank students by punctuality (lowest first when order='asc')."""
        all_stats = self.statistics_service.get_all_students_statistics()

        rows = []
        for s in all_stats:
            marked = s.get("total_marked") or 0
            if marked == 0:
                # No recorded events: punctuality is undefined, not zero.
                continue
            rows.append({
                "student_id": s.get("student_id"),
                "name": s.get("name"),
                "department": s.get("department"),
                "recorded_attendance_events": marked,
                "present_count": s.get("present_count", 0),
                "late_count": s.get("late_count", 0),
                "punctuality_percentage": s.get("punctuality_percentage", 0.0),
            })

        reverse = str(order).lower() == "desc"
        rows.sort(key=lambda r: r["punctuality_percentage"], reverse=reverse)

        if threshold is not None:
            rows = [r for r in rows if r["punctuality_percentage"] < threshold]

        capped = rows[: limit or 5]
        return {
            "ok": True,
            "tool": "student_ranking",
            "data": {
                "order": "desc (highest punctuality first)" if reverse else "asc (lowest punctuality first)",
                "students_ranked": len(rows),
                "students_without_records_excluded": len(all_stats) - len(rows) if not threshold else None,
                "ranking": capped,
                "note": (
                    "Ranked by PUNCTUALITY (present / (present + late)) of RECORDED events "
                    "only. Attendance percentage and absence are not available."
                ),
            },
        }

    def tool_session_list(self, course_code: Optional[str] = None,
                          limit: Optional[int] = None) -> Dict[str, Any]:
        """List sessions, optionally filtered by course."""
        if course_code:
            try:
                sessions = self.session_service.get_sessions_by_course(course_code)
            except Exception:
                sessions = []
        else:
            sessions = self.session_service.get_all_sessions()

        brief = [self._session_brief(s) for s in sessions]
        brief.sort(key=lambda s: str(s.get("start_time") or ""), reverse=True)
        capped = brief[: limit or 10]
        return {
            "ok": True,
            "tool": "session_list",
            "data": {
                "total_sessions": len(brief),
                "returned_sessions": len(capped),
                "course_filter": course_code,
                "sessions": capped,
            },
        }

    def tool_session_stats(self, session_query: Optional[str] = None,
                           session_db_id: Optional[int] = None) -> Dict[str, Any]:
        """Counts and metadata for one session."""
        resolved = self.resolve_session(
            {"session_query": session_query, "session_db_id": session_db_id}
        )
        if resolved["status"] == "ambiguous":
            return {
                "ok": True,
                "tool": "session_stats",
                "needs_clarification": True,
                "data": {"candidates": resolved["candidates"]},
            }
        if resolved["status"] == "not_found":
            return {
                "ok": True,
                "tool": "session_stats",
                "not_found": True,
                "data": {"query": resolved.get("query", "")},
            }

        session = resolved["session"]
        stats = self.statistics_service.get_session_statistics(session["session_db_id"])
        if not stats:
            return {
                "ok": True,
                "tool": "session_stats",
                "not_found": True,
                "data": {"query": session.get("session_name")},
            }
        return {"ok": True, "tool": "session_stats", "data": _session_stats_view(stats)}

    def tool_session_attendance(self, session_query: Optional[str] = None,
                                session_db_id: Optional[int] = None,
                                status_filter: Optional[str] = None) -> Dict[str, Any]:
        """Which students were recorded present/late in a session."""
        resolved = self.resolve_session(
            {"session_query": session_query, "session_db_id": session_db_id}
        )
        if resolved["status"] == "ambiguous":
            return {
                "ok": True,
                "tool": "session_attendance",
                "needs_clarification": True,
                "data": {"candidates": resolved["candidates"]},
            }
        if resolved["status"] == "not_found":
            return {
                "ok": True,
                "tool": "session_attendance",
                "not_found": True,
                "data": {"query": resolved.get("query", "")},
            }

        session = resolved["session"]
        rows = self.statistics_service.get_filtered_history(
            session_id=session["session_db_id"],
            status=status_filter,
        )
        capped = rows[:100]
        return {
            "ok": True,
            "tool": "session_attendance",
            "data": {
                "session": session,
                "status_filter": status_filter,
                "total_records": len(rows),
                "records": [_history_row_view(r) for r in capped],
                "note": (
                    "These are students RECORDED in this session. Students with no record "
                    "cannot be reported as absent because no expected-student roster exists."
                ),
            },
        }

    def tool_course_stats(self, course_code: Optional[str] = None) -> Dict[str, Any]:
        """Per-course aggregated attendance counts."""
        stats = self.statistics_service.get_course_statistics()
        courses = []
        for row in stats:
            code = row.get("course_code")
            if course_code and str(code).lower() != str(course_code).strip().lower():
                continue
            courses.append({
                "course_code": code,
                "total_sessions": row.get("total_sessions", 0),
                "total_attendance_records": row.get("total_marked", 0),
                "present_count": row.get("present_count", 0),
                "late_count": row.get("late_count", 0),
                "attendance_percentage": None,
                "absent_count": None,
            })

        return {
            "ok": True,
            "tool": "course_stats",
            "data": {
                "course_filter": course_code,
                "total_courses": len(courses),
                "courses": courses,
                "note": _DATA_MODEL_NOTE,
            },
        }

    def tool_attendance_trend(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Day-by-day present/late counts."""
        trend = self.statistics_service.get_attendance_trend()
        capped = trend[: limit or 30] if limit else trend
        return {
            "ok": True,
            "tool": "attendance_trend",
            "data": {
                "total_days": len(trend),
                "returned_days": len(capped),
                "days": [
                    {
                        "date": row.get("attendance_date"),
                        "total_records": row.get("total_marked", 0),
                        "present_count": row.get("present_count", 0),
                        "late_count": row.get("late_count", 0),
                    }
                    for row in capped
                ],
                "note": _DATA_MODEL_NOTE,
            },
        }

    def tool_system_info(self) -> Dict[str, Any]:
        """Static system capabilities and data-model limits."""
        return {
            "ok": True,
            "tool": "system_info",
            "data": dict(SYSTEM_FACTS),
        }


# Explicit intent -> method-name mapping. Model output selects an intent from a
# fixed set; the mapping to a bound method is never taken from model output.
_INTENT_TO_TOOL: Dict[str, str] = {
    schema.INTENT_OVERALL_STATS: "tool_overall_stats",
    schema.INTENT_STUDENT_LOOKUP: "tool_student_lookup",
    schema.INTENT_STUDENT_STATS: "tool_student_stats",
    schema.INTENT_STUDENT_HISTORY: "tool_student_history",
    schema.INTENT_STUDENT_RANKING: "tool_student_ranking",
    schema.INTENT_SESSION_LIST: "tool_session_list",
    schema.INTENT_SESSION_STATS: "tool_session_stats",
    schema.INTENT_SESSION_ATTENDANCE: "tool_session_attendance",
    schema.INTENT_COURSE_STATS: "tool_course_stats",
    schema.INTENT_ATTENDANCE_TREND: "tool_attendance_trend",
    schema.INTENT_SYSTEM_INFO: "tool_system_info",
}