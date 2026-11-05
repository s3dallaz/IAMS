"""
The activity log: who did what, and when.

Services write an entry after each change they make (accounts, sign-ins,
students, courses and class lists, sessions, attendance set by hand,
check-out / spot checks, the camera choice). It is append-only: nobody can
edit or delete entries from the app. The camera's own marks are not logged
here (each attendance record already says the camera made it).

Who: the signed-in user of the page that made the change. The app tells
the log how to find them (set_actor_provider). Changes made in the
background - the scheduler starting / closing sessions or checks - are
"System (automatic)". A service can also name the actor itself (e.g. a
sign-in, before anyone is signed in).

Writing the log never breaks the change itself: a failure is only logged
to the app log.
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.data.repositories.activity_repository import ActivityRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


class Category:
    SIGN_IN = 'sign_in'
    ACCOUNTS = 'accounts'
    STUDENTS = 'students'
    COURSES = 'courses'
    SESSIONS = 'sessions'
    ATTENDANCE = 'attendance'
    CHECKS = 'checks'
    SETTINGS = 'settings'


CATEGORY_LABELS = {
    Category.SIGN_IN: "Sign-in",
    Category.ACCOUNTS: "Accounts",
    Category.STUDENTS: "Students & faces",
    Category.COURSES: "Courses & class lists",
    Category.SESSIONS: "Sessions",
    Category.ATTENDANCE: "Attendance (by hand)",
    Category.CHECKS: "Check-out & spot checks",
    Category.SETTINGS: "Settings",
}

SYSTEM_ACTOR: Dict[str, Any] = {'id': None, 'username': None, 'full_name': "System (automatic)", 'role': None}

_actor_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None
_actor_override: ContextVar = ContextVar('activity_actor', default=None)


def set_actor_provider(provider: Optional[Callable[[], Optional[Dict[str, Any]]]]) -> None:
    """How to find the signed-in user of the current page (the app sets this once)."""
    global _actor_provider
    _actor_provider = provider


def acting_user() -> Optional[Dict[str, Any]]:
    """The user making the current change, or None (background work)."""
    override = _actor_override.get()
    if override is not None:
        return override
    if _actor_provider is None:
        return None
    try:
        return _actor_provider()
    except Exception:
        return None


@contextmanager
def acting_as(actor: Dict[str, Any]):
    """Changes made inside this block are logged as `actor` (e.g. SYSTEM_ACTOR in the scheduler)."""
    token = _actor_override.set(actor)
    try:
        yield
    finally:
        _actor_override.reset(token)


def changes(before: Dict[str, Any], after: Dict[str, Any], fields: Iterable[str]) -> Dict[str, List[Any]]:
    """{field: [old, new]} for the fields whose value changed."""
    return {field: [before.get(field), after.get(field)] for field in fields
            if before.get(field) != after.get(field)}


def describe_changes(changed: Dict[str, List[Any]], labels: Optional[Dict[str, str]] = None) -> str:
    """'name: "A" -> "B"; email: "" -> "x@y"' for a summary line."""
    labels = labels or {}

    def show(value):
        return "—" if value in (None, "") else f'"{value}"'

    return "; ".join(f"{labels.get(field, field.replace('_', ' '))}: {show(old)} → {show(new)}"
                     for field, (old, new) in changed.items())


class ActivityLog:
    """Writes and reads the activity log."""

    def __init__(self, database_path: Optional[Path] = None):
        self.repository = ActivityRepository(database_path)

    def record(self, category: str, action: str, summary: str, *, target_type: Optional[str] = None,
               target_id: Optional[int] = None, session_id: Optional[int] = None,
               details: Optional[Dict[str, Any]] = None, actor: Optional[Dict[str, Any]] = None,
               now: Optional[datetime] = None) -> Optional[int]:
        """
        Add an entry. `actor` defaults to the signed-in user (else System).
        Never raises: returns None when the entry could not be written.
        """
        try:
            who = actor or acting_user() or SYSTEM_ACTOR
            entry = {
                'at': (now or datetime.now()).strftime(TIME_FORMAT),
                'user_id': who.get('id'),
                'username': who.get('username'),
                'full_name': who.get('full_name') or who.get('username'),
                'role': who.get('role'),
                'category': category,
                'action': action,
                'summary': summary,
                'target_type': target_type,
                'target_id': target_id,
                'session_id': session_id,
                'details': json.dumps(details, default=str, ensure_ascii=False) if details else None,
            }
            return self.repository.add(entry)
        except Exception as e:
            logger.warning(f"Could not write the activity log ({action}: {summary}): {e}")
            return None

    def search(self, limit: int = 500, **filters) -> List[Dict[str, Any]]:
        """
        Newest first. Filters: date_from / date_to ('YYYY-MM-DD HH:MM:SS'),
        username, category, session_id, target=(type, id), search (text).
        """
        return self.repository.search(limit=limit, **filters)

    def count(self, **filters) -> int:
        return self.repository.count(**filters)

    def people(self) -> List[Dict[str, Any]]:
        return self.repository.people()
