"""
AI Assistant Service for natural-language attendance questions (Phase 6).

Architecture:

    Streamlit UI
        -> AIAssistantService (this module)
            -> LLM: plan a structured intent  (natural-language understanding)
            -> validate + allowlist the intent (src.ai.intent_schema)
            -> AttendanceToolRegistry (src.services.ai_tools) -> existing services
            -> LLM: phrase a grounded answer
        -> Streamlit UI

Design rules:
- The LLM performs reasoning and phrasing ONLY. Every attendance fact comes
  from the existing Phase 4/5 services via the read-only tool registry.
- The LLM never produces executable SQL, Python, or shell commands. Intent is
  expressed as a validated JSON object drawn from a fixed allowlist.
- This assistant is strictly read-only.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config
from src.ai import intent_schema as schema
from src.ai.llm_client import (
    LLMError,
    LLMNotConfiguredError,
    LLMRequestFailedError,
    LLMResponseFormatError,
    LLMUnavailableError,
    OmniRouteClient,
)
from src.ai.prompts import (
    ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYSTEM_FACTS,
    build_answer_prompt,
)
from src.services.ai_tools import (
    AttendanceToolRegistry,
    InvalidToolArgumentsError,
    ToolError,
    UnknownToolError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Service-level Exceptions
# ============================================================================

class AIAssistantError(Exception):
    """Base exception for AI assistant errors."""
    pass


class ModelUnavailableError(AIAssistantError):
    """Raised when the AI model/API is unavailable."""
    pass


class AmbiguousQueryError(AIAssistantError):
    """Raised when a query is ambiguous and requires clarification."""
    pass


# ---------------------------------------------------------------------------
# Pronouns / ellipsis that mean "reuse the subject we just discussed".
# ---------------------------------------------------------------------------
_FOLLOWUP_PATTERNS = [
    r"\bhis\b", r"\bher\b", r"\btheir\b", r"\bthey\b", r"\bhim\b",
    r"\bthat student\b", r"\bthis student\b", r"\bsame student\b",
    r"\bshe\b", r"\bhe\b",
]
_ELLIPSIS_RE = re.compile(r"^\s*(and|so|what about|how about)\b", re.IGNORECASE)


class AIAssistantService:
    """
    Conversational, read-only assistant over the attendance system.

    Uses the project's configured OmniRoute model for natural-language
    understanding and answer generation, while all data access goes through
    the existing application services.
    """

    def __init__(
        self,
        database_path=None,
        llm_client: Optional[OmniRouteClient] = None,
    ):
        """
        Initialize the AI Assistant.

        Args:
            database_path: Optional database path override (used by tests).
            llm_client: Optional injected LLM client (used by tests to mock the
                        provider without any network access).
        """
        self.database_path = database_path
        self.tools = AttendanceToolRegistry(database_path)
        self.llm = llm_client if llm_client is not None else OmniRouteClient()

        # Conversation memory for the current session.
        self.history: List[Dict[str, str]] = []
        # Entities most recently resolved, used to resolve follow-up questions.
        self.focus: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Backwards-compatible accessors (Phase 6 UI/tests)
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> str:
        """The resolved OmniRoute API key (never logged or returned to the UI)."""
        return self.llm.api_key

    def is_configured(self) -> bool:
        """
        True when a usable OmniRoute API key is available.

        Both the application configuration and the injected client are checked,
        so an unconfigured environment is reported even if a client reports
        itself as ready.
        """
        return config.is_ai_assistant_configured() and self.llm.is_configured()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, user_message: str) -> str:
        """
        Handle one user turn and return the assistant's reply.

        Args:
            user_message: Raw natural-language message from the user.

        Returns:
            The assistant's natural-language reply.

        Raises:
            ModelUnavailableError: The provider could not be reached or failed.
        """
        message = (user_message or "").strip()
        if not message:
            return self._clarify(
                "Please type a question about students, sessions, or attendance."
            )

        if not self.is_configured():
            # Graceful, actionable message - no API call attempted.
            self.history.append({"role": "user", "content": message})
            reply = (
                f"⚠️ AI Assistant is not configured. Please set OMNIROUTE_API_KEY "
                f"in your `.env` file (`{config.ENV_FILE}`) and restart the app."
            )
            self.history.append({"role": "assistant", "content": reply})
            return reply

        try:
            tool_call = self._plan(message)
        except ModelUnavailableError:
            raise
        except LLMError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except schema.IntentValidationError as exc:
            logger.warning(f"Rejected malformed model intent: {exc}")
            tool_call = schema.ToolCall(
                intent=schema.INTENT_UNSUPPORTED,
                unsupported_reason="The assistant could not interpret that request.",
            )

        try:
            reply = self._answer(message, tool_call)
        except ModelUnavailableError as exc:
            # The intent was understood and the data was already retrieved, so a
            # failure in the phrasing step must not lose the answer. Fall back to
            # a deterministic rendering of the real tool results.
            logger.warning(f"Falling back to raw results: {exc}")
            reply = self._format_raw_fallback(message, tool_call)

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def process_query(self, user_question: str) -> str:
        """
        Backwards-compatible alias for `ask`.

        Kept so existing Phase 6 callers and tests continue to work.
        """
        try:
            return self.ask(user_question)
        except ModelUnavailableError as exc:
            return f"⚠️ Model unavailable: {exc}"

    def reset_conversation(self) -> None:
        """Clear chat history and entity focus."""
        self.history.clear()
        self.focus.clear()

    def get_available_capabilities(self) -> List[str]:
        """Human-readable list of what the assistant can do."""
        return [
            "System-wide totals (students, sessions, attendance records, punctuality)",
            "Student lookup by name or student ID",
            "Per-student present/late counts and punctuality",
            "Per-student attendance history",
            "Student punctuality ranking (best / worst)",
            "Session listing and per-session counts",
            "Per-session attendance records (who was present or late)",
            "Per-course aggregated statistics",
            "Day-by-day attendance trend",
            "System capabilities and data-availability questions",
        ]

    def get_unavailable_capabilities(self) -> List[str]:
        """Things the assistant must not claim to know."""
        return [
            "Who was absent from a session (no expected-student roster exists)",
            "Attendance percentage / attendance rate (denominator is unknown)",
        ]

    # ------------------------------------------------------------------
    # Step 1: planning
    # ------------------------------------------------------------------

    def _plan(self, message: str) -> schema.ToolCall:
        """
        Ask the LLM for a structured intent and validate it.

        Raises:
            ModelUnavailableError: Provider failure.
            IntentValidationError: Model output could not be validated.
        """
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        ]
        recent = self._recent_history()
        if recent:
            messages.append({"role": "user", "content": recent})

        messages.append({
            "role": "user",
            "content": f"TODAY: {datetime.now().strftime('%Y-%m-%d')}\n\nREQUEST: {message}",
        })

        raw = self._call_llm(messages, json_mode=True)
        payload = schema.parse_llm_json(raw)
        tool_call = schema.validate_intent(payload)

        self._resolve_followups(tool_call, message)
        logger.debug(f"Planned intent: {tool_call.intent} args={tool_call.arguments}")
        return tool_call

    def _resolve_followups(self, tool_call: schema.ToolCall, message: str) -> None:
        """
        Fill in missing identifiers from conversation focus for follow-ups.

        This makes "what about his last 5 sessions?" work after an earlier
        question about a specific student, without the model having to guess.
        """
        if not self.focus:
            return

        is_followup = bool(_ELLIPSIS_RE.search(message)) or any(
            re.search(pattern, message, re.IGNORECASE) for pattern in _FOLLOWUP_PATTERNS
        )
        if not is_followup:
            return

        # Only fill a gap; never overwrite an explicit identifier.
        if tool_call.intent in schema.INTENTS_REQUIRING_STUDENT:
            if not tool_call.arguments.get("student_query") and not tool_call.arguments.get("student_db_id"):
                if self.focus.get("student_query"):
                    tool_call.arguments["student_query"] = self.focus["student_query"]
                elif self.focus.get("student_db_id"):
                    tool_call.arguments["student_db_id"] = self.focus["student_db_id"]

        if tool_call.intent in schema.INTENTS_REQUIRING_SESSION:
            if not tool_call.arguments.get("session_query") and not tool_call.arguments.get("session_db_id"):
                if self.focus.get("session_query"):
                    tool_call.arguments["session_query"] = self.focus["session_query"]

    # ------------------------------------------------------------------
    # Step 2: tool execution + answer
    # ------------------------------------------------------------------

    def _answer(self, message: str, tool_call: schema.ToolCall) -> str:
        """
        Execute the validated intent and produce the final reply.

        Raises:
            ModelUnavailableError: Provider failure.
        """
        if tool_call.intent == schema.INTENT_UNSUPPORTED:
            return self._unsupported_reply(message, tool_call)

        if tool_call.requires_clarification():
            return self._clarify(
                self._missing_identifier_text(tool_call.intent)
            )

        try:
            result = self.tools.dispatch(tool_call)
        except (UnknownToolError, InvalidToolArgumentsError) as exc:
            logger.warning(f"Rejected tool dispatch: {exc}")
            return self._unsupported_reply(message, tool_call)
        except ToolError as exc:
            logger.error(f"Tool execution failed: {exc}")
            return self._clarify("I hit a problem reading that data. Please try rephrasing.")
        except (OSError, ValueError) as exc:
            # Database/driver level failure while reading data.
            logger.error(f"Data access failed while running {tool_call.intent}: {exc}")
            return self._clarify("I hit a problem reading that data. Please try again.")

        # Ambiguity and not-found are handled without another LLM call, so the
        # assistant can never guess an identifier.
        if result.get("needs_clarification"):
            return self._clarify(self._ambiguity_text(tool_call, result))
        if result.get("not_found"):
            return self._not_found_reply(tool_call, result)

        self._update_focus(tool_call, result)

        if config.AI_RAW_MODE:
            return self._format_raw_fallback(message, tool_call)

        return self._phrase_answer(message, tool_call, result)

    def _phrase_answer(
        self,
        message: str,
        tool_call: schema.ToolCall,
        result: Dict[str, Any],
    ) -> str:
        """Ask the LLM to phrase a grounded answer from tool results only."""
        system_prompt = build_answer_prompt(
            today=datetime.now().strftime("%Y-%m-%d"),
            data_note="(the tool result is in the user message below)",
        )
        user_prompt = (
            f"DATA (JSON, trustworthy - use only these values):\n"
            f"```json\n{json.dumps(result.get('data', {}), default=str, ensure_ascii=False)}\n```\n\n"
            f"USER QUESTION: {message}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_llm(messages)

    def _call_llm(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """
        Single LLM call wrapper that normalizes every provider failure.

        Raises:
            ModelUnavailableError: For all provider-level failures.
        """
        try:
            return self.llm.chat(messages, json_mode=json_mode)
        except LLMNotConfiguredError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except LLMUnavailableError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except LLMRequestFailedError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except LLMResponseFormatError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except LLMError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except Exception as exc:  # defensive: never let a provider bug crash the UI
            logger.error(f"Unexpected LLM failure: {type(exc).__name__}")
            raise ModelUnavailableError(
                "The AI provider could not be reached."
            ) from exc

    # ------------------------------------------------------------------
    # Conversation state
    # ------------------------------------------------------------------

    def _recent_history(self) -> str:
        """Render recent turns as plain text for the planner prompt."""
        turns = self.history[-config.AI_MAX_HISTORY_MESSAGES:]
        if not turns:
            return ""
        return "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)

    def _update_focus(self, tool_call: schema.ToolCall, result: Dict[str, Any]) -> None:
        """Remember the last resolved student/session for follow-up questions."""
        data = result.get("data", {})
        intent = tool_call.intent

        if intent in (schema.INTENT_STUDENT_STATS, schema.INTENT_STUDENT_HISTORY):
            # student_stats returns a flat view; student_history nests under
            # "student". Both carry the display name and the public ID.
            student = data.get("student") or data
            name = student.get("name")
            public_id = student.get("student_id")
            if name or public_id:
                self.focus = {
                    "student_query": name or public_id,
                    "student_db_id": student.get("student_db_id"),
                }
        elif intent == schema.INTENT_STUDENT_LOOKUP:
            matches = data.get("matched_students") or []
            if len(matches) == 1:
                self.focus = {
                    "student_query": matches[0].get("name") or matches[0].get("student_id"),
                    "student_db_id": matches[0].get("student_db_id"),
                }
        elif intent in (schema.INTENT_SESSION_STATS, schema.INTENT_SESSION_ATTENDANCE):
            session = data.get("session")
            if not session and result.get("data", {}).get("session_name"):
                session = {
                    "session_name": result["data"].get("session_name"),
                    "session_db_id": result["data"].get("session_db_id"),
                }
            if session:
                self.focus = {
                    "session_query": session.get("session_name"),
                    "session_db_id": session.get("session_db_id"),
                }

    # ------------------------------------------------------------------
    # Message helpers (deterministic, never guessed)
    # ------------------------------------------------------------------

    def _ambiguity_text(self, tool_call: schema.ToolCall, result: Dict[str, Any]) -> str:
        """Build a clarification request listing the candidates."""
        data = result.get("data", {})
        if tool_call.intent in schema.INTENTS_REQUIRING_STUDENT:
            candidates = data.get("candidates") or []
            if not candidates:
                candidates = data.get("matched_students") or []
            options = ", ".join(
                f"{c.get('name')} (ID: {c.get('student_id')})" for c in candidates[:5]
            )
            return (
                "I found more than one student matching that. Which one did you mean?\n"
                f"{options}"
            )

        candidates = data.get("candidates") or []
        options = ", ".join(
            f"{c.get('session_name')} (ID: {c.get('session_db_id')})" for c in candidates[:5]
        )
        return f"I found more than one session matching that. Which one did you mean?\n{options}"

    def _not_found_reply(self, tool_call: schema.ToolCall, result: Dict[str, Any]) -> str:
        """Explain that the referenced record does not exist."""
        query = (result.get("data") or {}).get("query", "")
        if tool_call.intent in schema.INTENTS_REQUIRING_STUDENT:
            return (
                f"I couldn't find any student matching `{query}` in the registry. "
                "Check the Student Registry page for the exact name or ID."
            )
        if tool_call.intent in schema.INTENTS_REQUIRING_SESSION:
            return f"I couldn't find any session matching `{query}`."
        return f"I couldn't find anything matching `{query}`."

    def _missing_identifier_text(self, intent: str) -> str:
        """Ask which entity the user meant."""
        if intent in schema.INTENTS_REQUIRING_STUDENT:
            return "Which student do you mean? Give me a name or student ID."
        return "Which session do you mean? Give me a session name or ID."

    def _unsupported_reply(self, message: str, tool_call: schema.ToolCall) -> str:
        """Polite scope explanation for out-of-domain or unanswerable requests."""
        reason = (tool_call.unsupported_reason or "").strip()
        base = (
            "I'm built for attendance analytics in this system - students, sessions, "
            "and recorded attendance. "
        )
        if reason:
            base += f"({reason}) "
        base += (
            "I can also explain what data the system does and doesn't record. "
            "What would you like to know?"
        )
        return base

    def _clarify(self, message: str) -> str:
        """Return a clarification message and record it in history."""
        self.history.append({"role": "assistant", "content": message})
        return message

    def _format_raw_fallback(
        self, message: str, tool_call: schema.ToolCall, result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deterministic rendering used when the phrasing step is unavailable."""
        if result is None:
            try:
                result = self.tools.dispatch(tool_call)
            except (ToolError, Exception) as exc:
                logger.error(f"Raw fallback failed: {exc}")
                return "I couldn't retrieve that information."

        return (
            f"**{tool_call.intent.replace('_', ' ').title()}**\n\n"
            f"```json\n{json.dumps(result.get('data', {}), default=str, indent=2)}\n```"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def debug_state(self) -> Dict[str, Any]:
        """Safe debug snapshot. Never includes the API key value."""
        return {
            "configured": self.is_configured(),
            "model": getattr(self.llm, "model", None),
            "base_url": getattr(self.llm, "base_url", None),
            "history_length": len(self.history),
            "focus": dict(self.focus),
            "available_tools": self.tools.list_tool_names(),
        }