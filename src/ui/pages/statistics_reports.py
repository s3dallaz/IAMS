"""
Statistics, Analytics & Reports UI Page for Intelligent Attendance Monitoring System.

Provides an analytical dashboard for:
1. System-wide Key Attendance Metrics (Present, Late, Absent, Attendance %, Punctuality %)
2. Visual Analytics & Charts (Status distribution, course comparisons, trends, most absences)
3. Student Attendance Analysis with history breakdown, CSV export, and PDF Report generation
4. Session / Course Statistics with attendance breakdown, CSV export, and PDF Report generation
5. Filterable Attendance History with multi-parameter search, date filtering, and CSV export

Absences follow the rule of Session Details (StatisticsService). Only the
open tab runs, and report files are built when their download button is
clicked, so the page stays quick with tens of thousands of students.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from src.services.attendance_service import marked_by_label
from src.services.statistics_service import StatisticsService
from src.services.report_service import ReportService
from src.services.student_service import StudentService
from src.services.session_service import SessionService
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_ABSENT_STUDENTS = 20      # rows of the "most absences" table
STUDENT_SEARCH_RESULTS = 50   # students offered in the picker for a search
ABSENCE_NOTE = ("Absent = expected but not marked. Expected students of a session are the active "
                "students registered before it started, plus anyone marked in it. Totals count "
                "closed sessions only.")
STATUS_LABELS = {'present': "✅ Present", 'late': "⏰ Late", 'absent': "❌ Absent"}


def render_statistics_reports_page():
    """Main entry point to render the Statistics, Analytics & Reports page."""
    st.title("📊 Statistics, Analytics & Reports")
    st.caption("Attendance insights, history, and export")
    st.markdown("---")
    
    # Initialize Services without loading heavy CV models or camera
    stats_service = StatisticsService()
    report_service = ReportService(statistics_service=stats_service)
    student_service = StudentService()
    session_service = SessionService()
    
    # Top Level Navigation: only the open tab runs
    tab_dash, tab_student, tab_session, tab_history = st.tabs([
        "📈 Overview Dashboard",
        "👤 Student Statistics",
        "📅 Session & Course Analytics",
        "🔍 Filterable History & Export"
    ], key="stats_tab", on_change="rerun")
    
    if tab_dash.open:
        with tab_dash:
            render_overview_dashboard(stats_service)
    
    if tab_student.open:
        with tab_student:
            render_student_statistics_tab(stats_service, report_service, student_service)
    
    if tab_session.open:
        with tab_session:
            render_session_course_tab(stats_service, report_service, session_service)
    
    if tab_history.open:
        with tab_history:
            render_filterable_history_tab(stats_service, report_service, session_service)


# ============================================================================
# Tab 1: Overview Dashboard
# ============================================================================

def render_overview_dashboard(stats_service: StatisticsService):
    """Render high-level attendance summary cards and visual analytics."""
    st.subheader("📈 System-Wide Attendance Overview")
    
    stats = stats_service.get_overall_dashboard_stats()
    
    # 1. Metric Cards
    cols = st.columns(6)
    cols[0].metric("Total Students", f"{stats['total_students']:,}")
    cols[1].metric("Total Sessions", stats['total_sessions'],
                   help=f"{stats['active_sessions']} active, {stats['closed_sessions']} closed")
    cols[2].metric("Attendance Rate", f"{stats['attendance_rate']}%",
                   help="Attended / expected, over closed sessions")
    cols[3].metric("Present", f"{stats['total_present']:,}", delta=f"{stats['punctuality_rate']}% on time")
    cols[4].metric("Late", f"{stats['total_late']:,}", delta_color="inverse")
    cols[5].metric("Absent", f"{stats['total_absent']:,}", help="In closed sessions")
    st.caption(ABSENCE_NOTE)
    
    st.markdown("---")
    
    # 2. Visual Analytics & Charts
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        st.markdown("##### 🥧 Present vs Late")
        present = stats['total_present']
        late = stats['total_late']
        if (present + late) > 0:
            status_df = pd.DataFrame({
                'Status': ['Present', 'Late'],
                'Count': [present, late]
            })
            st.bar_chart(status_df.set_index('Status'), color="#3b82f6")
        else:
            st.info("No attendance records to chart yet.")
    
    course_stats = stats_service.get_course_statistics()
    with col_ch2:
        st.markdown("##### 📚 Course Attendance Comparison")
        if course_stats:
            c_df = pd.DataFrame(course_stats)
            c_df_plot = c_df[['course_code', 'present_count', 'late_count']].rename(
                columns={'course_code': 'Course', 'present_count': 'Present', 'late_count': 'Late'}
            ).set_index('Course')
            st.bar_chart(c_df_plot)
        else:
            st.info("No course attendance data available.")
    
    st.markdown("---")
    
    trend = stats_service.get_attendance_trend()
    session_stats = stats_service.get_all_sessions_statistics()
    col_trend, col_sessions = st.columns(2)
    with col_trend:
        st.markdown("##### Attendance trend")
        if trend:
            trend_df = pd.DataFrame(trend).rename(
                columns={
                    'attendance_date': 'Date',
                    'present_count': 'Present',
                    'late_count': 'Late',
                }
            )
            st.line_chart(trend_df, x='Date', y=['Present', 'Late'])
        else:
            st.info("No recorded attendance trend yet.")
    with col_sessions:
        st.markdown("##### Session comparison")
        if session_stats:
            session_df = pd.DataFrame(session_stats).rename(
                columns={'session_name': 'Session', 'total_marked': 'Recorded attendance'}
            )
            st.bar_chart(session_df, x='Session', y='Recorded attendance')
        else:
            st.info("No sessions available for comparison.")
    
    # 3. The students who miss the most (a short list, not every student)
    st.markdown("##### 🚩 Students With the Most Absences")
    top_absent = stats_service.get_students_with_most_absences(TOP_ABSENT_STUDENTS)
    if top_absent:
        st.dataframe([{
            'Student ID': s['student_id'],
            'Name': s['name'],
            'Department': s['department'],
            'Absent': s['absent_count'],
            'Expected Sessions': s['expected_sessions'],
            'Attendance': f"{s['attendance_percentage']}%",
            'Present': s['present_count'],
            'Late': s['late_count'],
        } for s in top_absent], hide_index=True, width="stretch")
        st.caption(f"Top {len(top_absent)} by absences in closed sessions. "
                   f"Look up any student in the **👤 Student Statistics** tab.")
    else:
        st.info("No absences in closed sessions yet.")


# ============================================================================
# Tab 2: Student Statistics
# ============================================================================

def render_student_statistics_tab(
    stats_service: StatisticsService,
    report_service: ReportService,
    student_service: StudentService
):
    """Render individual student attendance statistics with history and report export."""
    st.subheader("👤 Individual Student Attendance Analysis")
    
    # Search first: a picker with every registered student would be too heavy
    query = st.text_input("Find a student", placeholder="Name or student ID", key="stats_student_search")
    if not query.strip():
        st.info("Type a name or student ID to find a student.")
        return
    
    students = student_service.search_students(query, limit=STUDENT_SEARCH_RESULTS)
    if not students:
        st.info("No student matches.")
        return
    
    student_options = {s['id']: f"{s['name']} (ID: {s['student_id']})" for s in students}
    
    selected_id = st.selectbox(
        "Select Student to Analyze",
        options=list(student_options.keys()),
        format_func=lambda x: student_options[x],
        key="stats_student_selector"
    )
    if len(students) == STUDENT_SEARCH_RESULTS:
        st.caption(f"Showing the first {STUDENT_SEARCH_RESULTS} matches — type more to narrow it down.")
    
    if not selected_id:
        return
    
    stats = stats_service.get_student_statistics(selected_id)
    if not stats:
        st.warning("Could not calculate statistics for selected student.")
        return
    
    # Student Summary Header
    st.markdown(f"### **{stats['name']}** (`{stats['student_id']}`)")
    st.caption(f"Department: {stats['department']} | Status: {'Active' if stats['is_active'] else 'Inactive'}")
    
    # Metrics Row
    m = st.columns(6)
    m[0].metric("Expected Sessions", stats['expected_sessions'], help="Closed sessions held after registering")
    m[1].metric("Present (On Time)", stats['present_count'])
    m[2].metric("Late", stats['late_count'])
    m[3].metric("Absent", stats['absent_count'], help="Closed sessions missed")
    m[4].metric("Attendance", f"{stats['attendance_percentage']}%")
    m[5].metric("On-time Rate", f"{stats['punctuality_percentage']}%")
    
    st.markdown("---")
    
    # Export Buttons: the files are built when clicked
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="📥 Download Student CSV Report",
            data=lambda: report_service.export_student_summary_csv(stats),
            file_name=f"attendance_student_{stats['student_id']}.csv",
            mime="text/csv",
            on_click="ignore",
            key=f"btn_dl_csv_stu_{selected_id}"
        )
    with col_exp2:
        st.download_button(
            label="📄 Download Student PDF Report",
            data=lambda: report_service.generate_student_report_pdf(stats),
            file_name=f"attendance_student_{stats['student_id']}.pdf",
            mime="application/pdf",
            on_click="ignore",
            key=f"btn_dl_pdf_stu_{selected_id}"
        )
    
    st.markdown("---")
    
    # History Table for this student: attended sessions, then the ones missed
    st.markdown("#### 📋 Attendance History")
    table_data = [{
        'Session Name': r.get('session_name', '—'),
        'Course': r.get('course_code') or '—',
        'Status': STATUS_LABELS.get(r.get('status'), r.get('status')),
        'Marked Time': format_datetime(r.get('marked_at')),
        'Confidence': f"{r.get('detection_confidence', 0):.2f}" if r.get('detection_confidence') is not None else '—',
        'Marked By': marked_by_label(r.get('marked_by'), r.get('verified_by')) or '—'
    } for r in stats.get('history', [])]
    table_data += [{
        'Session Name': r.get('session_name', '—'),
        'Course': r.get('course_code') or '—',
        'Status': STATUS_LABELS['absent'],
        'Marked Time': f"(session {format_datetime(r.get('session_start_time'))})",
        'Confidence': '—',
        'Marked By': '—'
    } for r in stats.get('missed_sessions', [])]
    if table_data:
        st.dataframe(table_data, hide_index=True, width="stretch")
    else:
        st.info("No attendance recorded for this student yet.")


# ============================================================================
# Tab 3: Session & Course Analytics
# ============================================================================

def render_session_course_tab(
    stats_service: StatisticsService,
    report_service: ReportService,
    session_service: SessionService
):
    """Render session and course attendance statistics and document export."""
    st.subheader("📅 Session & Course Attendance Analytics")
    
    sessions = session_service.get_all_sessions()
    if not sessions:
        st.info("No sessions created in system.")
        return
    
    session_options = {s['id']: f"#{s['id']} - {s['session_name']} ({s.get('course_code') or 'General'})" for s in sessions}
    
    selected_sess_id = st.selectbox(
        "Select Session to Inspect",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x],
        key="stats_session_selector"
    )
    
    if not selected_sess_id:
        return
    
    sess_stats = stats_service.get_session_statistics(selected_sess_id)
    if not sess_stats:
        st.warning("Could not calculate statistics for selected session.")
        return
    
    # Session Header
    st.markdown(f"### **{sess_stats['session_name']}** (`{sess_stats['course_code']}`)")
    st.caption(f"Start Time: {format_datetime(sess_stats['start_time'])} | Status: {sess_stats['status'].upper()} | Late Cutoff: {sess_stats['late_threshold_minutes']} min")
    
    # Metrics: the same numbers as Session Management -> Session Details
    started = sess_stats['absence_supported']
    c = st.columns(6)
    c[0].metric("Expected", f"{sess_stats['expected_students']:,}" if started else "—")
    c[1].metric("Present", sess_stats['present_count'])
    c[2].metric("Late", sess_stats['late_count'])
    c[3].metric("Absent", f"{sess_stats['absent_count']:,}" if started else "—")
    c[4].metric("Attendance", f"{sess_stats['attendance_percentage']}%" if started else "—")
    c[5].metric("On-time Rate", f"{sess_stats['punctuality_percentage']}%")
    if not started:
        st.caption("This session has not started yet.")
    elif sess_stats['status'] == 'active':
        st.caption("The session is still running: Absent means *not marked yet*.")
    
    st.markdown("---")
    
    # Export Buttons: the files are built when clicked
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            label="📥 Download Session CSV Report",
            data=lambda: report_service.export_session_summary_csv(sess_stats),
            file_name=f"attendance_session_{selected_sess_id}.csv",
            mime="text/csv",
            on_click="ignore",
            help="Every expected student: present, late and absent",
            key=f"btn_dl_csv_sess_{selected_sess_id}"
        )
    with col_e2:
        st.download_button(
            label="📄 Download Session PDF Report",
            data=lambda: report_service.generate_session_report_pdf(sess_stats),
            file_name=f"attendance_session_{selected_sess_id}.pdf",
            mime="application/pdf",
            on_click="ignore",
            key=f"btn_dl_pdf_sess_{selected_sess_id}"
        )
    
    st.markdown("---")
    
    # Records Table
    st.markdown("#### 👥 Marked Students in this Session")
    records = sess_stats.get('records', [])
    if records:
        table_rows = []
        for r in records:
            table_rows.append({
                'Student ID': r.get('student_id', '—'),
                'Name': r.get('student_name', '—'),
                'Department': r.get('student_department', '—'),
                'Status': STATUS_LABELS.get(r.get('status'), r.get('status')),
                'Marked Time': format_datetime(r.get('marked_at')),
                'Confidence': f"{r.get('detection_confidence', 0):.2f}" if r.get('detection_confidence') is not None else '—',
                'Liveness': f"{r.get('liveness_score', 0):.2f}" if r.get('liveness_score') is not None else '—',
                'Marked By': marked_by_label(r.get('marked_by'), r.get('verified_by')) or '—'
            })
        st.dataframe(table_rows, hide_index=True, width="stretch")
    else:
        st.info("No attendance records logged for this session yet.")
    if started:
        st.caption("Absent students are in the CSV report and in **Session Management → Session Details**.")
    
    st.markdown("#### Course summary")
    course_stats = stats_service.get_course_statistics()
    if course_stats:
        course_rows = pd.DataFrame(course_stats).rename(
            columns={
                'course_code': 'Course',
                'total_sessions': 'Sessions',
                'total_marked': 'Recorded attendance',
                'present_count': 'Present',
                'late_count': 'Late',
                'absent_count': 'Absent',
                'attendance_rate': 'Attendance %',
            }
        )[['Course', 'Sessions', 'Recorded attendance', 'Present', 'Late', 'Absent', 'Attendance %']]
        st.dataframe(course_rows, hide_index=True)
        st.caption(ABSENCE_NOTE)
    else:
        st.info("No course data available.")


# ============================================================================
# Tab 4: Filterable Attendance History
# ============================================================================

def render_filterable_history_tab(
    stats_service: StatisticsService,
    report_service: ReportService,
    session_service: SessionService
):
    """Render search and multi-parameter filterable history with CSV export."""
    st.subheader("🔍 Searchable Attendance History")
    st.markdown("Filter attendance records across students, courses, sessions, statuses, and date ranges.")
    
    # Filter Controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        f_student_id = st.text_input("Filter Student ID", placeholder="e.g., 2021001", key="hist_f_stu_id")
        f_student_name = st.text_input("Filter Student Name", placeholder="e.g., John", key="hist_f_stu_name")
    with col_f2:
        f_course = st.text_input("Filter Course Code", placeholder="e.g., CS101", key="hist_f_course")
        f_status = st.selectbox("Status", ["All", "Present", "Late"], key="hist_f_status")
    with col_f3:
        all_sessions = session_service.get_all_sessions()
        sess_map = {s['id']: s['session_name'] for s in all_sessions}
        sess_options = ["All"] + list(sess_map.keys())
        f_session = st.selectbox(
            "Session",
            options=sess_options,
            format_func=lambda x: "All Sessions" if x == "All" else f"#{x} - {sess_map[x]}",
            key="hist_f_session"
        )
        
        # Date Range Filter
        f_date_enable = st.checkbox("Filter by Date Range", value=False, key="hist_f_date_enable")
        
    start_date_val = None
    end_date_val = None
    if f_date_enable:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            d_start = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30), key="hist_d_start")
            start_date_val = d_start.strftime('%Y-%m-%d')
        with col_d2:
            d_end = st.date_input("End Date", value=datetime.now().date(), key="hist_d_end")
            end_date_val = d_end.strftime('%Y-%m-%d')
    
    # Query filtered data through service
    session_id_param = None if f_session == "All" else int(f_session)
    status_param = None if f_status == "All" else f_status.lower()
    
    records = stats_service.get_filtered_history(
        student_id_str=f_student_id if f_student_id else None,
        student_name=f_student_name if f_student_name else None,
        course_code=f_course if f_course else None,
        session_id=session_id_param,
        status=status_param,
        start_date=start_date_val,
        end_date=end_date_val
    )
    
    st.markdown("---")
    
    # Results Count & CSV Export
    col_res_hdr, col_res_btn = st.columns([3, 1])
    with col_res_hdr:
        st.markdown(f"#### Found **{len(records)}** Record(s)")
    with col_res_btn:
        if records:
            st.download_button(
                label="📥 Export to CSV",
                data=lambda: report_service.export_attendance_history_csv(records),
                on_click="ignore",
                file_name=f"attendance_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="btn_export_history_csv"
            )
    
    if not records:
        st.info("No attendance records match the selected filters.")
        return
    
    # Table Display
    display_rows = []
    for r in records:
        display_rows.append({
            'Record ID': r['record_id'],
            'Student ID': r['student_id'],
            'Student Name': r['student_name'],
            'Course': r.get('course_code') or '—',
            'Session': r['session_name'],
            'Status': "✅ Present" if r['status'] == 'present' else "⏰ Late",
            'Marked At': format_datetime(r['marked_at']),
            'Confidence': f"{r.get('detection_confidence', 0):.2f}" if r.get('detection_confidence') is not None else '—',
            'Liveness': f"{r.get('liveness_score', 0):.2f}" if r.get('liveness_score') is not None else '—',
            'Marked By': marked_by_label(r.get('marked_by'), r.get('verified_by')) or '—'
        })
        
    st.dataframe(display_rows, hide_index=True)


def format_datetime(dt_string: Optional[str]) -> str:
    """Format a datetime string for clean display."""
    if not dt_string:
        return '—'
    try:
        dt = datetime.fromisoformat(dt_string.replace(' ', 'T'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return dt_string
