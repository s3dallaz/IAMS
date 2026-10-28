"""
Tests for the Home page helpers: today's sessions, the next session, the
14-day chart data, the getting-started steps and the system check rows.
"""

from datetime import datetime

from src.ui.pages.home import (
    PAGE_ATTENDANCE, PAGE_SESSIONS, PAGE_STUDENTS,
    guide_steps, next_session, sessions_in_last_days, system_checks,
    todays_sessions, trend_last_days,
)

NOW = datetime(2026, 10, 28, 12, 0, 0)


def _session(id_, start, status='scheduled'):
    return {'id': id_, 'session_name': f"S{id_}", 'start_time': start, 'status': status}


def test_todays_sessions_keeps_today_and_running_ones_in_time_order():
    sessions = [
        _session(1, '2026-10-28 14:00:00'),
        _session(2, '2026-10-28 08:00:00', 'closed'),
        _session(3, '2026-10-27 09:00:00', 'closed'),   # yesterday, closed -> out
        _session(4, '2026-10-27 23:00:00', 'active'),   # started yesterday, still running -> in
        _session(5, '2026-10-29 09:00:00'),             # tomorrow -> out
    ]
    assert [s['id'] for s in todays_sessions(sessions, NOW)] == [4, 2, 1]


def test_next_session_is_the_earliest_future_scheduled_one():
    sessions = [
        _session(1, '2026-10-30 09:00:00'),
        _session(2, '2026-10-28 15:00:00'),
        _session(3, '2026-10-28 11:00:00'),             # already due -> skipped
        _session(4, '2026-10-28 13:00:00', 'active'),   # not scheduled -> skipped
    ]
    assert next_session(sessions, NOW)['id'] == 2
    assert next_session([], NOW) is None


def test_sessions_in_last_days_counts_the_past_week_only():
    sessions = [
        _session(1, '2026-10-28 09:00:00'),
        _session(2, '2026-10-22 09:00:00'),
        _session(3, '2026-10-20 09:00:00'),   # more than 7 days ago
        _session(4, '2026-10-29 09:00:00'),   # in the future
        _session(5, None),
    ]
    assert sessions_in_last_days(sessions, NOW) == 2


def test_trend_fills_missing_days_with_zero():
    trend = [
        {'attendance_date': '2026-10-28', 'present_count': 5, 'late_count': 2},
        {'attendance_date': '2026-10-26', 'present_count': 3, 'late_count': None},
        {'attendance_date': '2026-09-01', 'present_count': 9, 'late_count': 9},  # too old
    ]
    df = trend_last_days(trend, NOW, days=3)
    assert df['Day'].tolist() == ['26 Oct', '27 Oct', '28 Oct']
    assert df['Present'].tolist() == [3, 0, 5]
    assert df['Late'].tolist() == [0, 0, 2]


def test_guide_steps_follow_the_data():
    steps = guide_steps(students=0, students_with_faces=0, sessions=0, records=0)
    assert [s['done'] for s in steps] == [False] * 4
    assert [s['page'] for s in steps] == [PAGE_STUDENTS, PAGE_STUDENTS, PAGE_SESSIONS, PAGE_ATTENDANCE]

    steps = guide_steps(students=10, students_with_faces=3, sessions=1, records=0)
    assert [s['done'] for s in steps] == [True, True, True, False]


def test_system_check_all_good():
    rows = system_checks(students_without_faces=0, pending_samples=0, scheduler_running=True,
                         camera_session=None, ai_ready=True, missing_models=[])
    assert all(r[''] in ("✅", "ℹ️") for r in rows)
    camera = next(r for r in rows if r['Item'] == "Camera")
    assert camera[''] == "ℹ️" and camera['Result'].startswith("Off")


def test_system_check_reports_problems():
    rows = system_checks(students_without_faces=4, pending_samples=2, scheduler_running=False,
                         camera_session=7, ai_ready=False, missing_models=["MiniFASNetV2.onnx"])
    by_item = {r['Item']: r for r in rows}
    for item in ("Face models", "Session timer", "Recognition data", "Face data", "AI Assistant"):
        assert by_item[item][''] == "⚠️", item
    assert "MiniFASNetV2.onnx" in by_item["Face models"]['Result']
    assert "4 active student(s)" in by_item["Face data"]['Result']
    assert by_item["Camera"]['Result'] == "On - session #7"
