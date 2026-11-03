"""
Which sidebar pages each role can open (the lowest role that may open it).

Instructors run their classes; data management (students, courses,
accounts) and the AI Assistant, which reads all data, are for Admin / IT.
"""

from typing import Any, Dict, List, Optional

from src.services.auth_service import Role, has_role

PAGE_ROLES = {
    "Home": Role.INSTRUCTOR,
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


def pages_for(user: Optional[Dict[str, Any]]) -> List[str]:
    """The sidebar pages `user` may open, in sidebar order."""
    return [page for page in PAGE_ROLES if can_open(user, page)]
