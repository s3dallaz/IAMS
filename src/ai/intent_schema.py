"""
Structured intent schema for the Phase 6 AI Assistant.

This module is the SECURITY BOUNDARY between the LLM and the application.
The LLM may only ever express intent through a validated `ToolCall` built from
a fixed allowlist. There is no code path that lets model output become SQL,
Python, or a shell command.

Validation rules:
- The tool name must exist in the allowlist (checked again at dispatch time).
- Only declared argument names are accepted; unknown keys are dropped.
- Values are coerced to a narrow set of primitive types with length caps.
- Rejected input never reaches a repository.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import config


# ============================================================================
# Intent taxonomy
# ============================================================================

# The complete set of intents the assistant understands.
# Anything outside this set is treated as unsupported.
INTENT_OVERALL_STATS = "overall_stats"
INTENT_STUDENT_STATS = "student_stats"
INTENT_STUDENT_HISTORY = "student_history"
INTENT_STUDENT_LOOKUP = "student_lookup"
INTENT_STUDENT_RANKING = "student_ranking"
INTENT_SESSION_LIST = "session_list"
INTENT_SESSION_STATS = "session_stats"
INTENT_SESSION_ATTENDANCE = "session_attendance"
INTENT_COURSE_STATS = "course_stats"
INTENT_ATTENDANCE_TREND = "attendance_trend"
INTENT_SYSTEM_INFO = "system_info"
INTENT_UNSUPPORTED = "unsupported"

VALID_INTENTS: Set[str] = {
    INTENT_OVERALL_STATS,
    INTENT_STUDENT_STATS,
    INTENT_STUDENT_HISTORY,
    INTENT_STUDENT_LOOKUP,
    INTENT_STUDENT_RANKING,
    INTENT_SESSION_LIST,
    INTENT_SESSION_STATS,
    INTENT_SESSION_ATTENDANCE,
    INTENT_COURSE_STATS,
    INTENT_ATTENDANCE_TREND,
    INTENT_SYSTEM_INFO,
    INTENT_UNSUPPORTED,
}

# Intents that require a student identifier to be present and unambiguous.
INTENTS_REQUIRING_STUDENT: Set[str] = {
    INTENT_STUDENT_STATS,
    INTENT_STUDENT_HISTORY,
}

# Intents that require a session identifier.
INTENTS_REQUIRING_SESSION: Set[str] = {
    INTENT_SESSION_STATS,
    INTENT_SESSION_ATTENDANCE,
}


# ============================================================================
# Errors
# ============================================================================

class IntentValidationError(ValueError):
    """Raised when model-provided intent data cannot be safely used."""
    pass


# ============================================================================
# Argument specification
# ============================================================================

# Argument name -> allowed primitive types. This is the complete set of
# argument names any intent may carry.
ARG_TYPES: Dict[str, Set[type]] = {
    "student_query": {str},
    "student_db_id": {int},
    "student_id": {str},
    "session_db_id": {int},
    "session_query": {str},
    "course_code": {str},
    "status_filter": {str},
    "limit": {int},
    "threshold": {float, int},
    "order": {str},
}

# Enumerated argument values. Anything outside these sets is rejected so the
# model cannot smuggle arbitrary strings into query filters.
ARG_ENUMS: Dict[str, Set[str]] = {
    "status_filter": {"present", "late"},
    "order": {"asc", "desc"},
    "limit": set(),
}


@dataclass
class ToolCall:
    """
    A validated, application-level intent produced from LLM output.

    Attributes:
        intent: One of `VALID_INTENTS`.
        arguments: Sanitized primitive arguments for the tool.
        confidence: Model-reported confidence (0.0 - 1.0).
        unsupported_reason: Why the request was classified as unsupported.
    """

    intent: str = INTENT_UNSUPPORTED
    arguments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    unsupported_reason: str = ""

    def requires_clarification(self) -> bool:
        """True when required identifiers are missing."""
        if self.intent in INTENTS_REQUIRING_STUDENT:
            has_student = bool(
                self.arguments.get("student_query") or self.arguments.get("student_db_id")
            )
            return not has_student
        if self.intent in INTENTS_REQUIRING_SESSION:
            has_session = bool(
                self.arguments.get("session_query") or self.arguments.get("session_db_id")
            )
            return not has_session
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation."""
        return {
            "intent": self.intent,
            "arguments": dict(self.arguments),
            "confidence": self.confidence,
        }


# ============================================================================
# Parsing
# ============================================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Extract a JSON object from raw LLM text.

    Tolerates markdown code fences and surrounding prose.

    Raises:
        IntentValidationError: If no JSON object could be recovered.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise IntentValidationError("Empty response from the AI model.")

    text = raw.strip()

    # Prefer content inside a fenced block when present.
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fall back to the first balanced-looking object in the text.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise IntentValidationError(
                "The AI model did not return a valid structured response."
            ) from None
        try:
            parsed = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            raise IntentValidationError(
                "The AI model did not return a valid structured response."
            ) from None

    if not isinstance(parsed, dict):
        raise IntentValidationError(
            "The AI model returned an unexpected structured response type."
        )
    return parsed


def _coerce_argument(name: str, value: Any) -> Any:
    """
    Coerce a single argument to its declared type.

    Raises:
        IntentValidationError: If the value cannot be safely coerced.
    """
    allowed = ARG_TYPES[name]
    max_len = config.AI_MAX_ARG_LENGTH

    if name == "limit":
        # Numeric with a hard ceiling, regardless of whether JSON sent int/float/str.
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise IntentValidationError(f"Argument 'limit' must be an integer.") from None
        return max(1, min(number, 100))

    if name == "threshold":
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise IntentValidationError(f"Argument 'threshold' must be numeric.") from None
        return max(0.0, min(number, 100.0))

    if int in allowed:
        if isinstance(value, bool):
            raise IntentValidationError(f"Argument '{name}' must be an integer.")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value.strip())
        raise IntentValidationError(f"Argument '{name}' must be an integer.")

    # String-only arguments
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        raise IntentValidationError(f"Argument '{name}' must be a string.")

    cleaned = value.strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    if not cleaned:
        raise IntentValidationError(f"Argument '{name}' cannot be empty.")

    enum = ARG_ENUMS.get(name)
    if enum:
        lowered = cleaned.lower()
        if lowered not in enum:
            raise IntentValidationError(
                f"Argument '{name}' must be one of: {', '.join(sorted(enum))}."
            )
        return lowered
    return cleaned


def validate_intent(payload: Dict[str, Any]) -> ToolCall:
    """
    Validate raw model output into a safe `ToolCall`.

    Unknown intents degrade to `unsupported` rather than raising, so a confused
    model still produces a graceful answer. Malformed arguments raise.

    Args:
        payload: Decoded JSON object from the model.

    Returns:
        A validated `ToolCall`.

    Raises:
        IntentValidationError: If the payload is structurally invalid or an
            argument cannot be safely coerced.
    """
    if not isinstance(payload, dict):
        raise IntentValidationError("Structured response must be a JSON object.")

    raw_intent = payload.get("intent", INTENT_UNSUPPORTED)
    intent = str(raw_intent).strip().lower() if raw_intent is not None else INTENT_UNSUPPORTED

    if intent not in VALID_INTENTS:
        # Unknown intent -> unsupported, keeping any reason text.
        reason = str(payload.get("unsupported_reason", "") or "")[
            : config.AI_MAX_ARG_LENGTH
        ]
        return ToolCall(
            intent=INTENT_UNSUPPORTED,
            arguments={},
            confidence=_safe_float(payload.get("confidence"), 0.0),
            unsupported_reason=reason,
        )

    raw_args = payload.get("arguments", {})
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        raise IntentValidationError("'arguments' must be a JSON object.")

    arguments: Dict[str, Any] = {}
    for name, value in raw_args.items():
        if name not in ARG_TYPES:
            # Silently drop unknown keys: they can never reach a tool.
            continue
        if value is None:
            continue
        arguments[name] = _coerce_argument(name, value)

    return ToolCall(
        intent=intent,
        arguments=arguments,
        confidence=_safe_float(payload.get("confidence"), 0.5),
        unsupported_reason=str(payload.get("unsupported_reason", "") or "")[
            : config.AI_MAX_ARG_LENGTH
        ],
    )


def _safe_float(value: Any, default: float) -> float:
    """Best-effort float conversion."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(0.0, min(number, 1.0))


def is_unsupported(tool_call: Optional[ToolCall]) -> bool:
    """Convenience helper: True when the model reported an unsupported request."""
    return tool_call is None or tool_call.intent == INTENT_UNSUPPORTED