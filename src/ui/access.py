"""
What each signed-in user may see.

Pages: the lowest role that may open each sidebar page. Instructors run their
classes; data management (students, courses, accounts) and the AI Assistant,
which reads all data, are for Admin / IT.

Sessions: Admin / IT see every session. An instructor sees the sessions they
own - those of the sections linked to their account, and the ones they
created themselves.
"""

from typing import Any, Dict, Iterable, List, Optional

from src.services.auth_service import Role, has_role

MY_CLASSES = "My Classes"

PAGE_ROLES = {
    "Home": Role.INSTRUCTOR,
    MY_CLASSES: Role.INSTRUCTOR,
    "Real-Time Attendance": Role.INSTRUCTOR,
    "Session Management": Role.INSTRUCTOR,
    "Student Registry": Role.ADMIN,
    "Courses & Sections": Role.ADMIN,
    "Statistics & Reports": Role.INSTRUCTOR,
    "AI Assistant": Role.ADMIN,
    "Users": Role.ADMIN,
}


def can_open(user: Optional[Dict[str, Any]], page: str) -> bool:
    return page in PAGE_ROLES and has_role(user, PAGE_ROLES[page])


def pages_for(user: Optional[Dict[str, Any]], teaches: bool = False) -> List[str]:
    """
    The sidebar pages `user` may open, in sidebar order. Admin / IT get
    "My Classes" only when sections are linked to them (`teaches`).
    """
    pages = [page for page in PAGE_ROLES if can_open(user, page)]
    if MY_CLASSES in pages and sees_everything(user) and not teaches:
        pages.remove(MY_CLASSES)
    return pages


def sees_everything(user: Optional[Dict[str, Any]]) -> bool:
    """Admin / IT: every session, course and report."""
    return has_role(user, Role.ADMIN)


def owns_session(user: Optional[Dict[str, Any]], session: Optional[Dict[str, Any]]) -> bool:
    """
    True if `user` may see and run `session`. Needs the session's
    section_instructor_id (SessionRepository adds it) and created_by_user_id.
    """
    if not user or not session:
        return False
    if sees_everything(user):
        return True
    return user['id'] in (session.get('section_instructor_id'), session.get('created_by_user_id'))


def visible_sessions(user: Optional[Dict[str, Any]], sessions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The sessions `user` may see, in the given order."""
    return [s for s in sessions if owns_session(user, s)]
