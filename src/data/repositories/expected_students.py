"""
The one rule for which students a session expects, as an SQL condition
shared by Session Details, the absence counts and the reports.

A student is expected at a session when they are active and:
- the session has a section: they were on its class list before the session
  started (section_students.added_at <= start)
- the session has no section: they were registered before the session
  started (students.enrolled_at <= start) - every student, as before sections

Anyone with an attendance record in a session counts too (marked by hand
or by the camera); the queries add that themselves. Session start times are
local; datetime(..., 'utc') converts them to UTC like the database-default
timestamps enrolled_at and added_at.
"""


def expected_condition(student: str = "s", session: str = "sess") -> str:
    """
    SQL that is true when the student row `student` is expected at the
    session row `session` (table aliases of the surrounding query).
    """
    return f"""({student}.is_active = 1 AND CASE
        WHEN {session}.section_id IS NULL
            THEN {student}.enrolled_at <= datetime({session}.start_time, 'utc')
        ELSE EXISTS (SELECT 1 FROM section_students listed
                     WHERE listed.section_id = {session}.section_id
                       AND listed.student_id = {student}.id
                       AND listed.added_at <= datetime({session}.start_time, 'utc'))
    END)"""
