"""
Tests for Phase 5 Statistics, Analytics & Report Services.

Tests:
1. AnalyticsRepository aggregation queries
2. StatisticsService metric calculations, rates, percentages, student & session summaries
3. Filtered history querying across multi-parameter criteria
4. Edge cases (zero sessions, zero attendance, all late, empty database)
5. ReportService CSV generation (history, student, session)
6. ReportService PDF generation (student report, session report)
7. UI Page rendering in isolated environment
"""

import pytest
import tempfile
import sqlite3
import csv
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.database.schema import initialize_database
from src.services.student_service import StudentService
from src.services.session_service import SessionService, SessionStatus
from src.services.attendance_service import AttendanceService, AttendanceStatus
from src.services.statistics_service import StatisticsService
from src.services.report_service import ReportService
from src.data.repositories.analytics_repository import AnalyticsRepository
from src.ui.pages.statistics_reports import (
    render_statistics_reports_page,
    render_overview_dashboard,
    render_student_statistics_tab,
    render_session_course_tab,
    render_filterable_history_tab
)


@pytest.fixture
def temp_db():
    """Create a temporary database with initialized schema for analytics testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def populated_db(temp_db):
    """Seed database with students, sessions, and attendance records for testing."""
    student_svc = StudentService(temp_db)
    session_svc = SessionService(temp_db)
    att_svc = AttendanceService(temp_db)
    
    # Create 3 students
    s1 = student_svc.create_student("STU001", "Alice Smith", "alice@test.com", "Computer Science")
    s2 = student_svc.create_student("STU002", "Bob Jones", "bob@test.com", "Computer Science")
    s3 = student_svc.create_student("STU003", "Charlie Brown", "charlie@test.com", "Mathematics")
    
    # Create sessions as ACTIVE first so attendance can be recorded, then close if needed
    sess1 = session_svc.create_session(
        session_name="CS101 Intro Lecture",
        course_code="CS101",
        start_time="2026-10-01 09:00:00",
        late_threshold_minutes=15,
        status=SessionStatus.ACTIVE
    )
    sess2 = session_svc.create_session(
        session_name="CS101 Lab 1",
        course_code="CS101",
        start_time="2026-10-03 14:00:00",
        late_threshold_minutes=10,
        status=SessionStatus.ACTIVE
    )
    sess3 = session_svc.create_session(
        session_name="MATH201 Calculus",
        course_code="MATH201",
        start_time="2026-10-05 10:00:00",
        late_threshold_minutes=15,
        status=SessionStatus.SCHEDULED
    )
    
    # Record attendance
    # Session 1: s1 Present, s2 Late (s3 absent)
    att_svc.record_attendance(sess1, s1, marked_at="2026-10-01 09:05:00", detection_confidence=0.95, liveness_score=0.9)
    att_svc.record_attendance(sess1, s2, marked_at="2026-10-01 09:20:00", detection_confidence=0.88, liveness_score=0.85)
    
    # Now close session 1
    session_svc.close_session(sess1)
    
    # Session 2: s1 Present, s3 Present (s2 absent)
    att_svc.record_attendance(sess2, s1, marked_at="2026-10-03 14:02:00", detection_confidence=0.92, liveness_score=0.91)
    att_svc.record_attendance(sess2, s3, marked_at="2026-10-03 14:05:00", detection_confidence=0.96, liveness_score=0.89)
    
    return {
        'db_path': temp_db,
        'student_ids': [s1, s2, s3],
        'session_ids': [sess1, sess2, sess3]
    }


# ============================================================================
# 1. Analytics Repository Tests
# ============================================================================

def test_analytics_repo_dashboard_summary(populated_db):
    repo = AnalyticsRepository(populated_db['db_path'])
    summary = repo.get_dashboard_summary()
    
    assert summary['total_students_count'] == 3
    assert summary['active_students_count'] == 3
    assert summary['total_sessions_count'] == 3
    assert summary['active_sessions_count'] == 1
    assert summary['total_records_count'] == 4
    assert summary['total_present_count'] == 3
    assert summary['total_late_count'] == 1


def test_analytics_repo_student_summary(populated_db):
    repo = AnalyticsRepository(populated_db['db_path'])
    s1_db_id = populated_db['student_ids'][0]
    
    summary = repo.get_student_attendance_summary(s1_db_id)
    assert summary['total_marked'] == 2
    assert summary['present_count'] == 2
    assert summary['late_count'] == 0
    assert summary['total_sessions'] == 2  # active + closed


def test_analytics_repo_filtered_history_by_student(populated_db):
    repo = AnalyticsRepository(populated_db['db_path'])
    
    records = repo.get_filtered_attendance_history(student_id_str="STU001")
    assert len(records) == 2
    assert all(r['student_id'] == "STU001" for r in records)


def test_analytics_repo_filtered_history_by_course(populated_db):
    repo = AnalyticsRepository(populated_db['db_path'])
    
    records = repo.get_filtered_attendance_history(course_code="CS101")
    assert len(records) == 4


def test_analytics_repo_filtered_history_by_status(populated_db):
    repo = AnalyticsRepository(populated_db['db_path'])
    
    late_records = repo.get_filtered_attendance_history(status="late")
    assert len(late_records) == 1
    assert late_records[0]['student_id'] == "STU002"


def test_analytics_repo_filtered_history_by_date_range(populated_db):
    repo = AnalyticsRepository(populated_db['db_path'])
    
    records = repo.get_filtered_attendance_history(
        start_date="2026-10-01",
        end_date="2026-10-02"
    )
    assert len(records) == 2


def test_analytics_repo_attendance_trend(populated_db):
    repo = AnalyticsRepository(populated_db['db_path'])

    trend = repo.get_attendance_trend()

    assert trend == [
        {'attendance_date': '2026-10-01', 'total_marked': 2, 'present_count': 1, 'late_count': 1},
        {'attendance_date': '2026-10-03', 'total_marked': 2, 'present_count': 2, 'late_count': 0},
    ]


# ============================================================================
# 2. Statistics Service Tests
# ============================================================================

def test_stats_service_dashboard_metrics(populated_db):
    svc = StatisticsService(populated_db['db_path'])
    stats = svc.get_overall_dashboard_stats()
    
    assert stats['total_students'] == 3
    assert stats['total_sessions'] == 3
    assert stats['total_records'] == 4
    assert stats['total_present'] == 3
    assert stats['total_late'] == 1
    assert stats['punctuality_rate'] == 75.0  # 3 present out of 4 marked
    assert stats['attendance_rate'] is None
    assert stats['total_absent'] is None
    assert stats['absence_supported'] is False


def test_stats_service_empty_database_edge_case(temp_db):
    svc = StatisticsService(temp_db)
    stats = svc.get_overall_dashboard_stats()
    
    assert stats['total_students'] == 0
    assert stats['total_sessions'] == 0
    assert stats['total_records'] == 0
    assert stats['attendance_rate'] is None
    assert stats['punctuality_rate'] == 0.0


def test_stats_service_student_statistics(populated_db):
    svc = StatisticsService(populated_db['db_path'])
    s1_id = populated_db['student_ids'][0]
    
    stats = svc.get_student_statistics(s1_id)
    assert stats['student_id'] == "STU001"
    assert stats['name'] == "Alice Smith"
    assert stats['total_marked'] == 2
    assert stats['present_count'] == 2
    assert stats['late_count'] == 0
    assert stats['absent_count'] is None
    assert stats['attendance_percentage'] is None
    assert len(stats['history']) == 2


def test_stats_service_student_zero_attendance(populated_db, temp_db):
    student_svc = StudentService(temp_db)
    s_new = student_svc.create_student("ZERO_01", "Zero Attendance")
    
    svc = StatisticsService(temp_db)
    stats = svc.get_student_statistics(s_new)
    
    assert stats['total_marked'] == 0
    assert stats['present_count'] == 0
    assert stats['absent_count'] is None
    assert stats['attendance_percentage'] is None
    assert len(stats['history']) == 0


def test_stats_service_session_statistics(populated_db):
    svc = StatisticsService(populated_db['db_path'])
    sess1_id = populated_db['session_ids'][0]
    
    stats = svc.get_session_statistics(sess1_id)
    assert stats['session_name'] == "CS101 Intro Lecture"
    assert stats['expected_students'] is None
    assert stats['total_marked'] == 2
    assert stats['present_count'] == 1
    assert stats['late_count'] == 1
    assert stats['absent_count'] is None
    assert stats['attendance_percentage'] is None


def test_stats_service_all_students_statistics(populated_db):
    svc = StatisticsService(populated_db['db_path'])
    all_stats = svc.get_all_students_statistics()
    
    assert len(all_stats) == 3
    ids = [s['student_id'] for s in all_stats]
    assert "STU001" in ids
    assert "STU002" in ids
    assert "STU003" in ids
    assert all(student['attendance_percentage'] is None for student in all_stats)


def test_stats_service_course_statistics(populated_db):
    svc = StatisticsService(populated_db['db_path'])
    courses = svc.get_course_statistics()
    
    assert len(courses) == 2  # CS101, MATH201
    cs_course = next(c for c in courses if c['course_code'] == 'CS101')
    assert cs_course['total_sessions'] == 2
    assert cs_course['total_marked'] == 4


def test_stats_service_attendance_trend(populated_db):
    svc = StatisticsService(populated_db['db_path'])

    assert len(svc.get_attendance_trend()) == 2


# ============================================================================
# 3. Report Service Tests (CSV & PDF Export)
# ============================================================================

def test_report_service_export_history_csv(populated_db):
    stats_svc = StatisticsService(populated_db['db_path'])
    report_svc = ReportService(statistics_service=stats_svc)
    
    records = stats_svc.get_filtered_history()
    csv_str = report_svc.export_attendance_history_csv(records)
    
    reader = list(csv.reader(io.StringIO(csv_str)))
    assert len(reader) == 5  # 1 header + 4 rows
    assert reader[0][1] == "Student ID"
    assert reader[0][8] == "Status"


def test_report_service_export_student_summary_csv(populated_db):
    stats_svc = StatisticsService(populated_db['db_path'])
    report_svc = ReportService(statistics_service=stats_svc)
    s1_id = populated_db['student_ids'][0]
    
    stats = stats_svc.get_student_statistics(s1_id)
    csv_str = report_svc.export_student_summary_csv(stats)
    
    assert "STUDENT ATTENDANCE REPORT" in csv_str
    assert "STU001" in csv_str
    assert "Alice Smith" in csv_str


def test_report_service_export_session_summary_csv(populated_db):
    stats_svc = StatisticsService(populated_db['db_path'])
    report_svc = ReportService(statistics_service=stats_svc)
    sess1_id = populated_db['session_ids'][0]
    
    stats = stats_svc.get_session_statistics(sess1_id)
    csv_str = report_svc.export_session_summary_csv(stats)
    
    assert "SESSION ATTENDANCE REPORT" in csv_str
    assert "CS101 Intro Lecture" in csv_str


def test_report_service_generate_student_report_pdf(populated_db):
    stats_svc = StatisticsService(populated_db['db_path'])
    report_svc = ReportService(statistics_service=stats_svc)
    s1_id = populated_db['student_ids'][0]
    
    stats = stats_svc.get_student_statistics(s1_id)
    pdf_bytes = report_svc.generate_student_report_pdf(stats)
    
    assert pdf_bytes is not None
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


def test_report_service_generate_session_report_pdf(populated_db):
    stats_svc = StatisticsService(populated_db['db_path'])
    report_svc = ReportService(statistics_service=stats_svc)
    sess1_id = populated_db['session_ids'][0]
    
    stats = stats_svc.get_session_statistics(sess1_id)
    pdf_bytes = report_svc.generate_session_report_pdf(stats)
    
    assert pdf_bytes is not None
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


# ============================================================================
# 4. UI Page Render Tests (Mocked UI)
# ============================================================================

@patch("streamlit.tabs")
def test_render_statistics_reports_page(mock_tabs):
    mock_tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    with patch("src.ui.pages.statistics_reports.render_overview_dashboard"), \
         patch("src.ui.pages.statistics_reports.render_student_statistics_tab"), \
         patch("src.ui.pages.statistics_reports.render_session_course_tab"), \
         patch("src.ui.pages.statistics_reports.render_filterable_history_tab"):
        render_statistics_reports_page()
        mock_tabs.assert_called_once()
