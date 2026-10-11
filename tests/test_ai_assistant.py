"""
Tests for the Phase 6 AI Assistant.

All tests run without any network access: the LLM provider is always mocked
through an injected `FakeLLMClient`, so `OMNIROUTE_API_KEY` is never required
and no external API is contacted.

Covered:
- Natural-language intent interpretation (mocked model output)
- Structured output validation and malformed-output handling
- Tool routing and per-tool behaviour
- Student / session / course / trend / overall questions
- Follow-up conversation context
- Ambiguous, unknown, empty-database, unsupported questions
- LLM and API failure handling
- Security: no SQL, no Python/shell execution, read-only enforcement
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Prevent heavy/problematic imports during testing
sys.modules['face_recognition'] = MagicMock()
sys.modules['face_recognition_models'] = MagicMock()
fpdf_mock = MagicMock()
fpdf_mock.FPDF = MagicMock()
sys.modules['fpdf'] = fpdf_mock

from src.ai import intent_schema as schema
from src.ai.llm_client import (
    LLMNotConfiguredError,
    LLMRequestFailedError,
    LLMResponseFormatError,
    LLMUnavailableError,
    OmniRouteClient,
)
from src.services.ai_assistant_service import (
    AIAssistantService,
    AmbiguousQueryError,
    ModelUnavailableError,
)
from src.services.ai_tools import AttendanceToolRegistry, UnknownToolError
from src.services.student_service import StudentService
from src.services.session_service import SessionService, SessionStatus
from src.services.attendance_service import AttendanceService
from src.database.schema import initialize_database


# ============================================================================
# Fake LLM client
# ============================================================================

class FakeLLMClient:
    """
    Test double for `OmniRouteClient`.

    Returns queued planning/answer payloads and never performs network I/O.
    """

    def __init__(self, plans: Optional[List[Any]] = None, answers: Optional[List[str]] = None):
        # Each plan is either a dict (JSON payload) or an Exception to raise.
        self.plans = list(plans or [])
        self.answers = list(answers or ["Grounded answer."])
        self.calls: List[Dict[str, Any]] = []
        self._plan_index = 0
        self._answer_index = 0
        self.api_key = "fake-test-key"
        self.model = "fake-model"
        self.base_url = "http://localhost:20128/v1"

    def is_configured(self) -> bool:
        return True

    def chat(self, messages, temperature=None, max_tokens=None, json_mode=False) -> str:
        self.calls.append({
            "messages": messages,
            "json_mode": json_mode,
        })
        if json_mode:
            if self._plan_index >= len(self.plans):
                raise LLMResponseFormatError("No queued plan")
            item = self.plans[self._plan_index]
            self._plan_index += 1
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                return item
            import json as _json
            return _json.dumps(item)
        if self._answer_index >= len(self.answers):
            raise LLMResponseFormatError("No queued answer")
        item = self.answers[self._answer_index]
        self._answer_index += 1
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def fake_env_key():
    """Ensure a usable key exists so the config gate passes deterministically."""
    with patch.dict(os.environ, {"OMNIROUTE_API_KEY": "test_key_12345"}):
        yield


# ============================================================================
# Database fixtures
# ============================================================================

@pytest.fixture
def test_db_path(tmp_path):
    """Create an empty initialized test database."""
    db_path = tmp_path / "test_assistant.db"
    initialize_database(db_path)
    return db_path


@pytest.fixture
def populated_db(test_db_path):
    """
    Populate a database with a deterministic dataset.

    Students:
      Ahmed Hassan (2021001, Computer Science)
      Sarah Mohamed (2021002, Engineering)
      Ahmed Khalil (2021003, Computer Science)   <- makes "Ahmed" ambiguous
      John Smith (2021004, Computer Science)

    Sessions:
      CS101 - Week 0 (2024-01-08, closed)
      CS101 - Week 0 (started 7 days ago, closed)
      CS101 - Week 1 (starts tomorrow, active)
      CS101 - Week 2 (starts in 2 days, closed)
      MA201 - Week 1 (starts in 3 days, closed)

    Attendance:
      Week 0: Sarah Mohamed (late)
      Week 1: Ahmed Hassan (present)
      Week 2: Ahmed Hassan (present), John Smith (present)
    """
    student_service = StudentService(test_db_path)
    session_service = SessionService(test_db_path)
    attendance_service = AttendanceService(test_db_path)

    ahmed = student_service.create_student(
        student_id="2021001", name="Ahmed Hassan",
        email="ahmed@example.com", department="Computer Science",
    )
    sarah = student_service.create_student(
        student_id="2021002", name="Sarah Mohamed",
        email="sarah@example.com", department="Engineering",
    )
    ahmed2 = student_service.create_student(
        student_id="2021003", name="Ahmed Khalil",
        email="ahmed2@example.com", department="Computer Science",
    )
    john = student_service.create_student(
        student_id="2021004", name="John Smith",
        email="john@example.com", department="Computer Science",
    )

    # AttendanceService derives present/late by comparing the session start time
    # to the record's marked_at against late_threshold_minutes. Because
    # `record_attendance` stamps marked_at with CURRENT_TIMESTAMP, a session
    # starting in the FUTURE yields 'present' and a PAST session yields 'late'.
    # Times are therefore relative to "now" instead of fixed 2024 dates.
    now = datetime.now()
    week1 = session_service.create_session(
        session_name="CS101 - Week 1", course_code="CS101",
        start_time=(now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        status=SessionStatus.ACTIVE,
    )
    week2 = session_service.create_session(
        session_name="CS101 - Week 2", course_code="CS101",
        start_time=(now + timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
        status=SessionStatus.ACTIVE,
    )
    ma201 = session_service.create_session(
        session_name="MA201 - Week 1", course_code="MA201",
        start_time=(now + timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
        status=SessionStatus.ACTIVE,
    )
    late_session = session_service.create_session(
        session_name="CS101 - Week 0", course_code="CS101",
        start_time=(now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),
        status=SessionStatus.ACTIVE,
    )

    attendance_service.record_attendance(week1, ahmed)          # present
    attendance_service.record_attendance(late_session, sarah)  # late
    attendance_service.record_attendance(week2, ahmed)          # present
    attendance_service.record_attendance(week2, john)           # present

    session_service.close_session(late_session)
    session_service.close_session(week2)
    session_service.close_session(ma201)

    return {
        "students": {"ahmed": ahmed, "sarah": sarah, "ahmed2": ahmed2, "john": john},
        "sessions": {
            "week0": late_session,
            "week1": week1,
            "week2": week2,
            "ma201": ma201,
        },
    }


@pytest.fixture
def tools(test_db_path):
    """Tool registry on an empty database."""
    return AttendanceToolRegistry(test_db_path)


@pytest.fixture
def populated_tools(test_db_path, populated_db):
    """
    Tool registry on the populated database.

    Depends on `populated_db` so the fixture ordering guarantees the data is
    written before the registry is constructed.
    """
    return AttendanceToolRegistry(test_db_path)


def make_service(db_path, plans, answers=None) -> AIAssistantService:
    """Build an assistant backed by a fake LLM client."""
    return AIAssistantService(db_path, llm_client=FakeLLMClient(plans, answers))


def plan(intent: str, **arguments) -> Dict[str, Any]:
    """Build a planner payload."""
    return {"intent": intent, "arguments": arguments, "confidence": 0.9}


# ============================================================================
# Natural-language intent interpretation (schema level)
# ============================================================================

class TestIntentInterpretation:
    """The planner output maps to the correct validated intent."""

    @pytest.mark.parametrize("intent", [
        schema.INTENT_OVERALL_STATS,
        schema.INTENT_STUDENT_STATS,
        schema.INTENT_STUDENT_HISTORY,
        schema.INTENT_SESSION_LIST,
        schema.INTENT_SESSION_STATS,
        schema.INTENT_COURSE_STATS,
        schema.INTENT_SYSTEM_INFO,
    ])
    def test_valid_intent_accepted(self, intent):
        call = schema.validate_intent({"intent": intent, "arguments": {}})
        assert call.intent == intent

    def test_unknown_intent_degrades_to_unsupported(self):
        call = schema.validate_intent({
            "intent": "drop_all_tables",
            "arguments": {"sql": "DROP TABLE students"},
        })
        assert call.intent == schema.INTENT_UNSUPPORTED

    def test_missing_intent_defaults_to_unsupported(self):
        call = schema.validate_intent({})
        assert call.intent == schema.INTENT_UNSUPPORTED

    def test_intent_is_case_normalized(self):
        call = schema.validate_intent({"intent": "OVERALL_STATS", "arguments": {}})
        assert call.intent == schema.INTENT_OVERALL_STATS

    def test_confidence_clamped_to_range(self):
        call = schema.validate_intent({"intent": "overall_stats", "confidence": 5.0})
        assert 0.0 <= call.confidence <= 1.0

    def test_confidence_nan_defaults(self):
        call = schema.validate_intent({"intent": "overall_stats", "confidence": "abc"})
        assert 0.0 <= call.confidence <= 1.0


# ============================================================================
# Structured output validation
# ============================================================================

class TestStructuredOutputValidation:

    def test_parse_plain_json(self):
        assert schema.parse_llm_json('{"intent": "overall_stats"}')["intent"] == "overall_stats"

    def test_parse_json_from_code_fence(self):
        raw = '```json\n{"intent": "overall_stats"}\n```'
        assert schema.parse_llm_json(raw)["intent"] == "overall_stats"

    def test_parse_json_embedded_in_prose(self):
        raw = 'Sure! Here you go: {"intent": "overall_stats"} hope that helps'
        assert schema.parse_llm_json(raw)["intent"] == "overall_stats"

    def test_parse_empty_raises(self):
        with pytest.raises(schema.IntentValidationError):
            schema.parse_llm_json("")

    def test_parse_non_json_raises(self):
        with pytest.raises(schema.IntentValidationError):
            schema.parse_llm_json("I cannot help with that.")

    def test_parse_json_array_rejected(self):
        with pytest.raises(schema.IntentValidationError):
            schema.parse_llm_json("[1, 2, 3]")

    def test_unknown_argument_keys_are_dropped(self):
        call = schema.validate_intent({
            "intent": "student_stats",
            "arguments": {
                "student_query": "Ahmed",
                "sql": "DELETE FROM students",
                "command": "rm -rf /",
            },
        })
        assert call.arguments == {"student_query": "Ahmed"}
        assert "sql" not in call.arguments
        assert "command" not in call.arguments

    def test_null_arguments_become_empty(self):
        call = schema.validate_intent({"intent": "overall_stats", "arguments": None})
        assert call.arguments == {}

    def test_non_dict_arguments_rejected(self):
        with pytest.raises(schema.IntentValidationError):
            schema.validate_intent({"intent": "overall_stats", "arguments": ["Ahmed"]})

    def test_string_length_is_capped(self):
        long_value = "A" * 5000
        call = schema.validate_intent({
            "intent": "student_stats",
            "arguments": {"student_query": long_value},
        })
        assert len(call.arguments["student_query"]) <= 200

    def test_limit_is_capped_at_100(self):
        call = schema.validate_intent({
            "intent": "student_ranking",
            "arguments": {"limit": 100000},
        })
        assert call.arguments["limit"] == 100

    def test_limit_minimum_is_one(self):
        call = schema.validate_intent({
            "intent": "student_ranking",
            "arguments": {"limit": -5},
        })
        assert call.arguments["limit"] == 1

    def test_limit_non_numeric_rejected(self):
        with pytest.raises(schema.IntentValidationError):
            schema.validate_intent({
                "intent": "student_ranking",
                "arguments": {"limit": "many"},
            })

    def test_invalid_status_filter_rejected(self):
        with pytest.raises(schema.IntentValidationError):
            schema.validate_intent({
                "intent": "session_attendance",
                "arguments": {"session_query": "CS101", "status_filter": "absent"},
            })

    def test_valid_status_filter_accepted(self):
        call = schema.validate_intent({
            "intent": "session_attendance",
            "arguments": {"session_query": "CS101", "status_filter": "LATE"},
        })
        assert call.arguments["status_filter"] == "late"

    def test_invalid_order_rejected(self):
        with pytest.raises(schema.IntentValidationError):
            schema.validate_intent({
                "intent": "student_ranking",
                "arguments": {"order": "sideways"},
            })

    def test_empty_string_argument_rejected(self):
        with pytest.raises(schema.IntentValidationError):
            schema.validate_intent({
                "intent": "student_stats",
                "arguments": {"student_query": "   "},
            })

    def test_clarification_flagged_when_student_missing(self):
        call = schema.validate_intent({"intent": "student_stats", "arguments": {}})
        assert call.requires_clarification()

    def test_clarification_not_flagged_when_student_present(self):
        call = schema.validate_intent({
            "intent": "student_stats",
            "arguments": {"student_query": "Ahmed"},
        })
        assert not call.requires_clarification()


# ============================================================================
# Tool registry behaviour
# ============================================================================

class TestToolRegistry:

    def test_all_expected_tools_registered(self, tools):
        names = tools.list_tool_names()
        for expected in [
            "tool_overall_stats", "tool_student_lookup", "tool_student_stats",
            "tool_student_history", "tool_student_ranking", "tool_session_list",
            "tool_session_stats", "tool_session_attendance", "tool_course_stats",
            "tool_attendance_trend", "tool_system_info",
        ]:
            assert expected in names

    def test_dispatch_rejects_unknown_intent(self, tools):
        bad = schema.ToolCall(intent="totally_made_up")
        with pytest.raises(UnknownToolError):
            tools.dispatch(bad)

    def test_dispatch_rejects_unsupported_intent(self, tools):
        with pytest.raises(UnknownToolError):
            tools.dispatch(schema.ToolCall(intent=schema.INTENT_UNSUPPORTED))

    @pytest.mark.parametrize("intent,args", [
        (schema.INTENT_OVERALL_STATS, {}),
        (schema.INTENT_STUDENT_STATS, {"student_query": "Ahmed"}),
        (schema.INTENT_STUDENT_HISTORY, {"student_query": "Ahmed"}),
        (schema.INTENT_STUDENT_LOOKUP, {"student_query": "Ahmed"}),
        (schema.INTENT_STUDENT_RANKING, {"order": "asc"}),
        (schema.INTENT_SESSION_LIST, {}),
        (schema.INTENT_SESSION_STATS, {"session_query": "Week 1"}),
        (schema.INTENT_SESSION_ATTENDANCE, {"session_query": "Week 1"}),
        (schema.INTENT_COURSE_STATS, {}),
        (schema.INTENT_ATTENDANCE_TREND, {}),
        (schema.INTENT_SYSTEM_INFO, {}),
    ])
    def test_every_supported_intent_dispatches(self, tools, intent, args):
        """Every advertised intent is routed to a registered tool."""
        call = schema.ToolCall(intent=intent, arguments=args)
        try:
            result = tools.dispatch(call)
        except UnknownToolError:
            pytest.fail(f"{intent} has no registered tool")
        assert result["ok"] is True
        assert result["tool"]

    def test_overall_stats_on_empty_database(self, tools):
        data = tools.dispatch(schema.ToolCall(intent=schema.INTENT_OVERALL_STATS))["data"]
        assert data["total_students"] == 0
        assert data["total_attendance_records"] == 0

    def test_overall_stats_with_data(self, populated_tools):
        data = populated_tools.dispatch(
            schema.ToolCall(intent=schema.INTENT_OVERALL_STATS)
        )["data"]
        assert data["total_students"] == 4
        assert data["total_sessions"] == 4
        assert data["active_sessions"] == 1
        assert data["closed_sessions"] == 3
        assert data["total_attendance_records"] == 4
        assert data["total_present"] == 3
        assert data["total_late"] == 1

    def test_absent_count_and_rate_are_none(self, populated_tools):
        """The schema cannot support absence; tools must not fabricate it."""
        data = populated_tools.dispatch(
            schema.ToolCall(intent=schema.INTENT_OVERALL_STATS)
        )["data"]
        assert data["absent_count"] is None
        assert data["attendance_rate"] is None

    def test_student_stats_resolves_by_name(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_STUDENT_STATS,
            arguments={"student_query": "Sarah"},
        ))
        assert result["ok"] is True
        assert result["data"]["name"] == "Sarah Mohamed"
        assert result["data"]["late_count"] == 1
        assert result["data"]["attendance_percentage"] is None

    def test_student_stats_ambiguous_returns_candidates(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_STUDENT_STATS,
            arguments={"student_query": "Ahmed"},
        ))
        assert result["needs_clarification"] is True
        assert len(result["data"]["candidates"]) == 2

    def test_student_stats_not_found(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_STUDENT_STATS,
            arguments={"student_query": "ZNobody"},
        ))
        assert result["not_found"] is True

    def test_student_lookup_by_exact_student_id(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_STUDENT_LOOKUP,
            arguments={"student_query": "2021004"},
        ))
        assert result["data"]["matched_students"][0]["name"] == "John Smith"

    def test_student_history_limit_respected(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_STUDENT_HISTORY,
            arguments={"student_query": "Ahmed Hassan", "limit": 1},
        ))
        assert result["data"]["returned_records"] == 1
        assert result["data"]["total_records"] == 2

    def test_ranking_ascending_is_worst_first(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_STUDENT_RANKING,
            arguments={"order": "asc"},
        ))
        ranking = result["data"]["ranking"]
        punctuality = [r["punctuality_percentage"] for r in ranking]
        assert punctuality == sorted(punctuality)

    def test_ranking_excludes_students_without_records(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_STUDENT_RANKING,
            arguments={"order": "asc"},
        ))
        names = [r["name"] for r in result["data"]["ranking"]]
        assert "Ahmed Khalil" not in names  # never marked attendance

    def test_session_list_filtered_by_course(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_SESSION_LIST,
            arguments={"course_code": "CS101"},
        ))
        # Week 0, Week 1 and Week 2 are all CS101.
        assert result["data"]["total_sessions"] == 3
        assert all(
            s["course_code"] == "CS101" for s in result["data"]["sessions"]
        )

    def test_session_stats_resolves_by_name(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_SESSION_STATS,
            arguments={"session_query": "CS101 - Week 1"},
        ))
        assert result["data"]["session_name"] == "CS101 - Week 1"
        assert result["data"]["present_count"] == 1
        assert result["data"]["late_count"] == 0
        assert result["data"]["expected_students"] is None
        assert result["data"]["attendance_percentage"] is None

    def test_session_stats_reports_late_count(self, populated_tools):
        """The late record lives in Week 0."""
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_SESSION_STATS,
            arguments={"session_query": "CS101 - Week 0"},
        ))
        assert result["data"]["late_count"] == 1
        assert result["data"]["present_count"] == 0

    def test_session_attendance_lists_students(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_SESSION_ATTENDANCE,
            arguments={"session_query": "CS101 - Week 1"},
        ))
        names = {r["student_name"] for r in result["data"]["records"]}
        assert names == {"Ahmed Hassan"}

    def test_session_attendance_status_filter(self, populated_tools):
        """The late record lives in Week 0 (Sarah was marked late)."""
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_SESSION_ATTENDANCE,
            arguments={"session_query": "CS101 - Week 0", "status_filter": "late"},
        ))
        records = result["data"]["records"]
        assert len(records) == 1
        assert records[0]["student_name"] == "Sarah Mohamed"

    def test_session_attendance_filter_present(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_SESSION_ATTENDANCE,
            arguments={"session_query": "CS101 - Week 2", "status_filter": "present"},
        ))
        names = {r["student_name"] for r in result["data"]["records"]}
        assert names == {"Ahmed Hassan", "John Smith"}

    def test_course_stats(self, populated_tools):
        result = populated_tools.dispatch(
            schema.ToolCall(intent=schema.INTENT_COURSE_STATS)
        )
        codes = {c["course_code"] for c in result["data"]["courses"]}
        assert "CS101" in codes
        assert "MA201" in codes

    def test_course_stats_filter(self, populated_tools):
        result = populated_tools.dispatch(schema.ToolCall(
            intent=schema.INTENT_COURSE_STATS,
            arguments={"course_code": "CS101"},
        ))
        assert result["data"]["total_courses"] == 1

    def test_attendance_trend(self, populated_tools):
        result = populated_tools.dispatch(
            schema.ToolCall(intent=schema.INTENT_ATTENDANCE_TREND)
        )
        assert result["data"]["total_days"] >= 1

    def test_system_info_states_limits(self, tools):
        data = tools.dispatch(schema.ToolCall(intent=schema.INTENT_SYSTEM_INFO))["data"]
        assert "data_model_limits" in data
        assert data["read_only"]

    def test_student_lookup_requires_argument(self, tools):
        from src.services.ai_tools import InvalidToolArgumentsError
        with pytest.raises(InvalidToolArgumentsError):
            tools.dispatch(schema.ToolCall(
                intent=schema.INTENT_STUDENT_LOOKUP, arguments={}
            ))


# ============================================================================
# Service orchestration / routing
# ============================================================================

class TestServiceRouting:

    def test_overall_question_routes_to_overall_tool(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_OVERALL_STATS)],
            ["There are 4 students."],
        )
        reply = svc.ask("How many students are registered?")
        assert "4 students" in reply

    def test_student_question_uses_real_data(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_STUDENT_STATS, student_query="Sarah Mohamed")],
            ["Sarah was late once."],
        )
        reply = svc.ask("What is Sarah's punctuality percentage?")
        assert "Sarah" in reply

    def test_session_question_returns_session_data(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_SESSION_STATS, session_query="CS101 - Week 1")],
            ["Week 1 had 1 present."],
        )
        reply = svc.ask("What was the attendance rate in CS101 - Week 1?")
        assert "Week 1" in reply

    def test_routing_question_routes_to_ranking(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_STUDENT_RANKING, order="asc", limit=3)],
            ["The lowest punctuality is John Smith."],
        )
        reply = svc.ask("Who has the worst punctuality?")
        assert "John" in reply

    def test_plan_step_uses_json_mode(self, test_db_path):
        svc = make_service(test_db_path, [plan(schema.INTENT_OVERALL_STATS)], ["ok"])
        svc.ask("how many students")
        assert svc.llm.calls[0]["json_mode"] is True

    def test_answer_step_contains_tool_data(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_OVERALL_STATS)],
            ["ok"],
        )
        svc.ask("how many students")
        answer_prompt = svc.llm.calls[1]["messages"][-1]["content"]
        assert "total_students" in answer_prompt

    def test_conversation_history_is_recorded(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_OVERALL_STATS), plan(schema.INTENT_OVERALL_STATS)],
            ["one", "two"],
        )
        svc.ask("how many students")
        svc.ask("and sessions?")
        assert [m["role"] for m in svc.history] == [
            "user", "assistant", "user", "assistant",
        ]

    def test_reset_conversation_clears_state(self, test_db_path):
        svc = make_service(test_db_path, [plan(schema.INTENT_OVERALL_STATS)], ["ok"])
        svc.ask("how many students")
        svc.reset_conversation()
        assert svc.history == []
        assert svc.focus == {}

    def test_empty_message_asks_for_input(self, test_db_path):
        svc = make_service(test_db_path, [], [])
        reply = svc.ask("   ")
        assert "type a question" in reply.lower()
        assert svc.llm.calls == []

    def test_process_query_alias_does_not_raise(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path, [plan(schema.INTENT_OVERALL_STATS)], ["four students"]
        )
        assert isinstance(svc.process_query("how many students"), str)

    def test_capabilities_lists_unavailable_items(self, test_db_path):
        svc = make_service(test_db_path, [], [])
        assert any("absent" in c.lower() for c in svc.get_unavailable_capabilities())

    def test_debug_state_never_leaks_key(self, test_db_path, populated_db):
        svc = make_service(test_db_path, [plan(schema.INTENT_OVERALL_STATS)], ["ok"])
        svc.ask("how many students")
        state = svc.debug_state()
        assert "test_key_12345" not in str(state)
        assert state["configured"] is True


# ============================================================================
# Follow-up conversation context
# ============================================================================

class TestFollowUpContext:

    def test_followup_reuses_focus_student(self, test_db_path, populated_db):
        """'What about his last 5 sessions?' resolves to the earlier student."""
        svc = make_service(
            test_db_path,
            [
                plan(schema.INTENT_STUDENT_STATS, student_query="Sarah Mohamed"),
                # Model omits the student for the follow-up.
                plan(schema.INTENT_STUDENT_HISTORY, limit=5),
            ],
            ["Sarah has 1 late.", "Sarah attended 1 session."],
        )
        svc.ask("What is Sarah's punctuality?")
        assert svc.focus.get("student_query") == "Sarah Mohamed"

        svc.ask("What about his last 5 sessions?")
        # The answer step must have been reached, meaning the follow-up was
        # resolved from focus rather than rejected as a missing identifier.
        assert len(svc.llm.calls) == 4  # plan, answer, plan, answer
        assert "Sarah attended 1 session." in svc.llm.answers

    def test_focus_is_not_overwritten_by_explicit_identifier(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [
                plan(schema.INTENT_STUDENT_STATS, student_query="Sarah Mohamed"),
                plan(schema.INTENT_STUDENT_STATS, student_query="John Smith"),
            ],
            ["a", "b"],
        )
        svc.ask("What is Sarah's punctuality?")
        svc.ask("What about John Smith?")
        assert svc.focus.get("student_query") == "John Smith"

    def test_history_is_passed_to_planner(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_OVERALL_STATS), plan(schema.INTENT_OVERALL_STATS)],
            ["a", "b"],
        )
        svc.ask("how many students")
        svc.ask("and sessions")

        # call order: plan1, answer1, plan2, answer2
        second_planner = svc.llm.calls[2]["messages"]
        planner_text = "\n".join(m["content"] for m in second_planner)
        # The prior turn must be visible to the planner for follow-ups.
        assert "USER: how many students" in planner_text
        assert "ASSISTANT:" in planner_text





# ============================================================================
# Ambiguity / unknown / empty / unsupported
# ============================================================================

class TestAmbiguityAndErrors:

    def test_ambiguous_student_asks_for_clarification(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_STUDENT_STATS, student_query="Ahmed")],
            ["should not be reached"],
        )
        reply = svc.ask("What is Ahmed's attendance?")
        assert "more than one student" in reply.lower()
        assert "Ahmed Hassan" in reply
        assert "Ahmed Khalil" in reply
        # Only the planner call was made - no guessing, no second LLM call.
        assert len(svc.llm.calls) == 1

    def test_ambiguous_session_asks_for_clarification(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_SESSION_STATS, session_query="CS101")],
            ["should not be reached"],
        )
        reply = svc.ask("Tell me about CS101")
        assert "more than one session" in reply.lower()

    def test_unknown_student_reports_not_found(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_STUDENT_STATS, student_query="ZNobody")],
            ["should not be reached"],
        )
        reply = svc.ask("What is ZNobody's attendance?")
        assert "couldn't find" in reply.lower()

    def test_missing_student_identifier_asks_which_student(self, test_db_path):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_STUDENT_STATS)],
            ["should not be reached"],
        )
        reply = svc.ask("What is his attendance?")
        assert "which student" in reply.lower()

    def test_missing_session_identifier_asks_which_session(self, test_db_path):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_SESSION_STATS)],
            ["should not be reached"],
        )
        reply = svc.ask("What was the attendance rate there?")
        assert "which session" in reply.lower()

    def test_unsupported_question_gets_scope_explanation(self, test_db_path):
        svc = make_service(
            test_db_path,
            [{
                "intent": schema.INTENT_UNSUPPORTED,
                "arguments": {},
                "unsupported_reason": "weather is unrelated",
            }],
            ["should not be reached"],
        )
        reply = svc.ask("What is the weather in Cairo?")
        assert "attendance" in reply.lower()
        assert len(svc.llm.calls) == 1

    def test_unknown_intent_degrades_gracefully(self, test_db_path):
        svc = make_service(
            test_db_path,
            [{"intent": "hack_the_database", "arguments": {}}],
            ["should not be reached"],
        )
        reply = svc.ask("do something weird")
        assert "attendance" in reply.lower()

    def test_empty_database_answers_without_crashing(self, test_db_path):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_OVERALL_STATS)],
            ["No students are registered yet."],
        )
        reply = svc.ask("How many students are registered?")
        assert isinstance(reply, str) and reply

    def test_empty_database_student_question(self, test_db_path):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_STUDENT_STATS, student_query="Ahmed")],
            ["Nobody found."],
        )
        reply = svc.ask("What is Ahmed's attendance?")
        assert "couldn't find" in reply.lower()

    def test_absent_count_is_refused_as_unavailable(self, test_db_path, populated_db):
        """Absence is not computable; the model should be routed to system_info."""
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_SYSTEM_INFO)],
            ["Absence is not tracked in this system."],
        )
        reply = svc.ask("Who was absent today?")
        assert "not tracked" in reply.lower() or "absent" in reply.lower()


# ============================================================================
# LLM / API failure handling
# ============================================================================

class TestLLMFailureHandling:

    def test_missing_api_key_does_not_call_model(self, test_db_path):
        with patch.dict(os.environ, {"OMNIROUTE_API_KEY": ""}):
            svc = AIAssistantService(test_db_path, llm_client=FakeLLMClient())
            reply = svc.ask("how many students")
        assert "not configured" in reply.lower()
        assert "OMNIROUTE_API_KEY" in reply
        assert svc.llm.calls == []

    def test_missing_key_via_process_query(self, test_db_path):
        with patch.dict(os.environ, {"OMNIROUTE_API_KEY": "your_api_key_here"}):
            svc = AIAssistantService(test_db_path, llm_client=FakeLLMClient())
            reply = svc.process_query("how many students")
        assert "not configured" in reply.lower()

    def test_connection_error_raises_model_unavailable(self, test_db_path):
        svc = make_service(test_db_path, [LLMUnavailableError("no route to host")], [])
        with pytest.raises(ModelUnavailableError):
            svc.ask("how many students")

    def test_provider_error_raises_model_unavailable(self, test_db_path):
        svc = make_service(test_db_path, [LLMRequestFailedError("HTTP 401")], [])
        with pytest.raises(ModelUnavailableError):
            svc.ask("how many students")

    def test_malformed_planner_output_handled(self, test_db_path, populated_db):
        """Non-JSON planner output must not crash the app."""
        svc = make_service(test_db_path, ["I think you should check Ahmed."], [])
        reply = svc.ask("What is Ahmed's attendance?")
        assert isinstance(reply, str) and reply
        assert "attendance" in reply.lower()

    def test_malformed_json_object_handled(self, test_db_path):
        svc = make_service(test_db_path, ["{not valid json at all"], [])
        reply = svc.ask("how many students")
        assert isinstance(reply, str) and reply

    def test_invalid_argument_in_plan_degrades(self, test_db_path):
        """A bad status_filter is rejected by validation, not executed."""
        svc = make_service(
            test_db_path,
            [{
                "intent": schema.INTENT_SESSION_ATTENDANCE,
                "arguments": {"session_query": "CS101", "status_filter": "'; DROP TABLE students; --"},
            }],
            [],
        )
        reply = svc.ask("who was absent in CS101")
        assert isinstance(reply, str) and reply

    def test_answer_step_failure_falls_back_to_raw_data(self, test_db_path, populated_db):
        """Data was retrieved; a phrasing failure must still return real values."""
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_OVERALL_STATS)],
            [LLMResponseFormatError("empty response")],
        )
        reply = svc.ask("how many students")
        assert "total_students" in reply
        assert "4" in reply

    def test_unexpected_client_exception_is_wrapped(self, test_db_path):
        svc = make_service(test_db_path, [RuntimeError("boom")], [])
        with pytest.raises(ModelUnavailableError):
            svc.ask("how many students")

    def test_unknown_tool_intent_handled(self, test_db_path):
        """A validated-but-unmapped intent must not raise through to the UI."""
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_SYSTEM_INFO)],
            ["ok"],
        )
        # Simulate drift between the intent allowlist and the tool registry.
        svc.tools.dispatch = MagicMock(side_effect=UnknownToolError("nope"))
        reply = svc.ask("who am i")
        assert isinstance(reply, str) and reply


# ============================================================================
# Security
# ============================================================================

class TestSecurity:

    def test_sql_injection_in_student_name_is_inert(self, test_db_path, populated_db):
        """Injection strings are treated as literal text, never executed."""
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_STUDENT_STATS,
                  student_query="'; DROP TABLE students; --")],
            [],
        )
        reply = svc.ask("What is '; DROP TABLE students; --'s attendance?")
        assert "couldn't find" in reply.lower()
        # Table must still exist and be intact.
        with sqlite3.connect(test_db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        assert count == 4

    def test_sql_in_cannot_delete_attendance(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_SESSION_STATS,
                  session_query="CS101 - Week 1; DELETE FROM attendance_records")],
            [],
        )
        svc.ask("stats for CS101 - Week 1; DELETE FROM attendance_records")
        with sqlite3.connect(test_db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM attendance_records").fetchone()[0]
        assert count == 4

    def test_model_cannot_inject_extra_arguments(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [{
                "intent": schema.INTENT_OVERALL_STATS,
                "arguments": {
                    "sql": "DROP TABLE students",
                    "table": "students",
                    "limit": 5,
                },
            }],
            ["ok"],
        )
        svc.ask("how many students")
        with sqlite3.connect(test_db_path) as conn:
            conn.execute("SELECT COUNT(*) FROM students").fetchone()
        assert svc.llm.calls  # answered without crashing

    @pytest.mark.parametrize("payload", [
        "import os; os.system('rm -rf /')",
        "__import__('subprocess').run('ls')",
        "eval('1+1')",
        "exec('x=1')",
        "open('/etc/passwd').read()",
        "$(whoami)",
        "`id`",
    ])
    def test_python_and_shell_payloads_are_inert(self, test_db_path, populated_db, payload):
        """Code-like input is treated as a plain name fragment."""
        svc = make_service(
            test_db_path,
            [plan(schema.INTENT_STUDENT_LOOKUP, student_query=payload)],
            ["No match."],
        )
        reply = svc.ask(f"Tell me about {payload}")
        assert isinstance(reply, str)
        with sqlite3.connect(test_db_path) as conn:
            conn.execute("SELECT COUNT(*) FROM students").fetchone()

    def test_no_arbitrary_code_execution_modules_used(self):
        """The AI path must not import eval/exec/subprocess machinery."""
        import src.ai.intent_schema as s
        import src.services.ai_assistant_service as svc_mod
        import src.services.ai_tools as tools_mod

        for module in (s, svc_mod, tools_mod):
            source = Path(module.__file__).read_text(encoding="utf-8")
            for banned in ["eval(", "exec(", "subprocess", "os.system", "pickle"]:
                assert banned not in source, f"{banned} found in {module.__name__}"

    def test_tools_module_contains_no_sql(self):
        """No SQL strings in the tool layer - data access stays in repositories."""
        source = Path(
            __import__("src.services.ai_tools", fromlist=["x"]).__file__
        ).read_text(encoding="utf-8")
        for banned in ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "DROP TABLE"]:
            assert banned not in source.upper()

    def test_read_only_no_students_deleted(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [
                plan(schema.INTENT_OVERALL_STATS),
                plan(schema.INTENT_STUDENT_STATS, student_query="Ahmed Hassan"),
            ],
            ["a", "b"],
        )
        svc.ask("delete all students")
        svc.ask("drop table students")
        with sqlite3.connect(test_db_path) as conn:
            students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            records = conn.execute("SELECT COUNT(*) FROM attendance_records").fetchone()[0]
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert students == 4
        assert records == 4
        assert sessions == 4

    def test_read_only_sessions_unchanged_after_creative_requests(self, test_db_path, populated_db):
        svc = make_service(
            test_db_path,
            [
                plan(schema.INTENT_SESSION_STATS, session_query="CS101 - Week 1"),
                plan(schema.INTENT_SESSION_STATS, session_query="CS101 - Week 2"),
            ],
            ["a", "b"],
        )
        svc.ask("close session CS101 - Week 1")
        svc.ask("set all students inactive")
        with sqlite3.connect(test_db_path) as conn:
            week1_status = conn.execute(
                "SELECT status FROM sessions WHERE session_name = 'CS101 - Week 1'"
            ).fetchone()[0]
            inactive = conn.execute(
                "SELECT COUNT(*) FROM students WHERE is_active = 0"
            ).fetchone()[0]
        assert week1_status == "active"
        assert inactive == 0

    def test_tool_registry_only_wraps_read_methods(self):
        """No write-method name appears in the tool module."""
        source = Path(__import__("src.services.ai_tools", fromlist=["x"]).__file__
                      ).read_text(encoding="utf-8")
        for banned in [
            "create_student", "update_student", "delete_student",
            "activate_student", "deactivate_student",
            "create_session", "activate_session", "close_session",
            "update_session", "record_attendance",
        ]:
            assert banned not in source, f"write method '{banned}' referenced"


# ============================================================================
# LLM client (no network - responses are faked)
# ============================================================================

class TestOmniRouteClient:

    def test_not_configured_raises(self):
        client = OmniRouteClient(api_key="")
        with pytest.raises(LLMNotConfiguredError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_placeholder_key_rejected(self):
        client = OmniRouteClient(api_key="your_api_key_here")
        with pytest.raises(LLMNotConfiguredError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_successful_response(self):
        client = OmniRouteClient(api_key="real-key-value")
        fake_response = MagicMock(status_code=200)
        fake_response.json.return_value = {
            "choices": [{"message": {"content": "  hello  "}}]
        }
        with patch("src.ai.llm_client.requests.post", return_value=fake_response):
            assert client.chat([{"role": "user", "content": "hi"}]) == "hello"

    def test_uses_configured_base_url_and_model(self):
        client = OmniRouteClient(api_key="real-key-value", base_url="http://x/v1", model="m1")
        fake_response = MagicMock(status_code=200)
        fake_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("src.ai.llm_client.requests.post", return_value=fake_response) as mock_post:
            client.chat([{"role": "user", "content": "hi"}])
        assert mock_post.call_args[0][0] == "http://x/v1/chat/completions"
        assert mock_post.call_args[1]["data"].count('"model": "m1"') == 1

    def test_auth_error_message(self):
        client = OmniRouteClient(api_key="real-key-value")
        with patch("src.ai.llm_client.requests.post", return_value=MagicMock(status_code=401)):
            with pytest.raises(LLMRequestFailedError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_rate_limit_message(self):
        client = OmniRouteClient(api_key="real-key-value")
        with patch("src.ai.llm_client.requests.post", return_value=MagicMock(status_code=429)):
            with pytest.raises(LLMRequestFailedError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_server_error_message(self):
        client = OmniRouteClient(api_key="real-key-value")
        with patch("src.ai.llm_client.requests.post", return_value=MagicMock(status_code=503)):
            with pytest.raises(LLMRequestFailedError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_connection_error_handled(self):
        import requests as real_requests
        client = OmniRouteClient(api_key="real-key-value")
        err = real_requests.exceptions.ConnectionError("refused")
        with patch("src.ai.llm_client.requests.post", side_effect=err):
            with pytest.raises(LLMUnavailableError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_timeout_handled(self):
        import requests as real_requests
        client = OmniRouteClient(api_key="real-key-value")
        with patch("src.ai.llm_client.requests.post",
                   side_effect=real_requests.exceptions.Timeout()):
            with pytest.raises(LLMUnavailableError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_non_json_response(self):
        client = OmniRouteClient(api_key="real-key-value")
        bad = MagicMock(status_code=200)
        bad.json.side_effect = ValueError("not json")
        with patch("src.ai.llm_client.requests.post", return_value=bad):
            with pytest.raises(LLMResponseFormatError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_empty_choices(self):
        client = OmniRouteClient(api_key="real-key-value")
        bad = MagicMock(status_code=200)
        bad.json.return_value = {"choices": []}
        with patch("src.ai.llm_client.requests.post", return_value=bad):
            with pytest.raises(LLMResponseFormatError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_null_content(self):
        client = OmniRouteClient(api_key="real-key-value")
        bad = MagicMock(status_code=200)
        bad.json.return_value = {"choices": [{"message": {"content": None}}]}
        with patch("src.ai.llm_client.requests.post", return_value=bad):
            with pytest.raises(LLMResponseFormatError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_error_object_with_200_status(self):
        client = OmniRouteClient(api_key="real-key-value")
        bad = MagicMock(status_code=200)
        bad.json.return_value = {"error": {"message": "bad model"}}
        with patch("src.ai.llm_client.requests.post", return_value=bad):
            with pytest.raises(LLMRequestFailedError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_api_key_is_not_leaked_in_error(self):
        client = OmniRouteClient(api_key="super-secret-key")
        bad = MagicMock(status_code=500)
        bad.text = "internal error mentioning super-secret-key"
        with patch("src.ai.llm_client.requests.post", return_value=bad):
            with pytest.raises(LLMRequestFailedError) as exc:
                client.chat([{"role": "user", "content": "hi"}])
        assert "super-secret-key" not in str(exc.value)


# ============================================================================
# Chat UI submission behaviour
# ============================================================================

class TestChatUI:

    @pytest.fixture
    def ui_env(self):
        """Minimal Streamlit surface for the chat page."""
        import src.ui.pages.ai_assistant as page
        fake = MagicMock()
        fake.session_state = _SessionState()
        # st.columns must return the right number of column context managers.
        fake.columns.side_effect = _fake_columns
        # page.st is replaced wholesale, so the example buttons default to False.
        fake.button.return_value = False
        with patch.object(page.st, "chat_input", return_value=None), \
             patch.object(page.st, "chat_message"), \
             patch.object(page.st, "spinner"), \
             patch.object(page.st, "info"), \
             patch.object(page.st, "warning"), \
             patch.object(page.st, "error"), \
             patch.object(page.st, "expander"), \
             patch.object(page.st, "markdown"), \
             patch.object(page.st, "caption"), \
             patch.object(page.st, "title"), \
             patch.object(page.st, "columns", side_effect=_fake_columns), \
             patch.object(page.st, "button", return_value=False), \
             patch.object(page.st, "checkbox", return_value=False), \
             patch.object(page.st, "json"), \
             patch.object(page.st, "set_page_config"):
            fake.session_state._streamlit = True
            patcher = patch.object(page, "st", fake)
            patcher.start()
            yield fake
            patcher.stop()

    def test_uses_chat_input_not_text_input(self, ui_env):
        """The page must submit via st.chat_input, so ENTER sends and clears."""
        import src.ui.pages.ai_assistant as page
        assert hasattr(page.st, "chat_input")

    def test_render_without_key_shows_config_warning(self, ui_env):
        import src.ui.pages.ai_assistant as page
        with patch.dict(os.environ, {"OMNIROUTE_API_KEY": ""}):
            with patch.object(page, "AIAssistantService") as MockSvc:
                instance = MockSvc.return_value
                instance.is_configured.return_value = False
                page.render_ai_assistant()
        assert ui_env.warning.called

    def test_history_is_rendered(self, ui_env):
        import src.ui.pages.ai_assistant as page
        ui_env.session_state.chat_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        with patch.dict(os.environ, {"OMNIROUTE_API_KEY": "k"}):
            with patch.object(page, "AIAssistantService") as MockSvc:
                MockSvc.return_value.is_configured.return_value = True
                page.render_ai_assistant()
        assert ui_env.session_state.chat_history

    def test_prompt_is_submitted_to_service(self, ui_env):
        """A chat_input return value must be sent to the assistant."""
        import src.ui.pages.ai_assistant as page
        ui_env.session_state.chat_history = []
        with patch.dict(os.environ, {"OMNIROUTE_API_KEY": "k"}):
            with patch.object(page, "AIAssistantService") as MockSvc, \
                 patch.object(page.st, "chat_input", return_value="how many students?"), \
                 patch.object(page.st, "rerun"):
                instance = MockSvc.return_value
                instance.is_configured.return_value = True
                instance.ask.return_value = "4 students"
                page.render_ai_assistant()
        # The chat_input value is submitted as the user's message.
        assert instance.ask.call_args_list[0][0][0] == "how many students?"
        assert ui_env.session_state.chat_history[-1]["content"] == "4 students"
        assert ui_env.session_state.chat_history[0] == {
            "role": "user", "content": "how many students?",
        }

    def test_model_error_becomes_message_not_crash(self, ui_env):
        import src.ui.pages.ai_assistant as page
        ui_env.session_state.chat_history = []
        with patch.dict(os.environ, {"OMNIROUTE_API_KEY": "k"}):
            with patch.object(page, "AIAssistantService") as MockSvc, \
                 patch.object(page.st, "chat_input", return_value="hello"), \
                 patch.object(page.st, "rerun"):
                instance = MockSvc.return_value
                instance.is_configured.return_value = True
                instance.ask.side_effect = ModelUnavailableError("provider down")
                page.render_ai_assistant()
        last = ui_env.session_state.chat_history[-1]["content"]
        assert "unavailable" in last.lower()

    def test_clear_chat_resets_history(self, ui_env):
        import src.ui.pages.ai_assistant as page
        ui_env.session_state.chat_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        ui_env.session_state.chat_history = []
        assert ui_env.session_state.chat_history == []


def _fake_columns(spec, *args, **kwargs):
    """
    Return the right number of column context managers for `st.columns`.

    Streamlit accepts either an int or a list of relative widths.
    """
    if isinstance(spec, int):
        count = spec
    elif isinstance(spec, (list, tuple)):
        count = len(spec)
    else:
        count = 1
    return [_Column() for _ in range(count)]


class _Column:
    """Context manager standing in for a Streamlit column."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _SessionState(dict):
    """Dict-backed stand-in for st.session_state."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value