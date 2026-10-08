"""
Report Service for CSV export and PDF document generation.

Provides formatted CSV export and PDF reports for:
1. Filtered attendance history
2. Individual student attendance reports
3. Session and course attendance reports
"""

import io
import csv
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from fpdf import FPDF

from src.services.statistics_service import StatisticsService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttendancePDF(FPDF):
    """Custom FPDF subclass with consistent header, footer, and styling."""
    
    def __init__(self, title_text: str = "Attendance Report"):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.title_text = title_text
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 41, 59)  # Dark slate
        self.cell(0, 10, self.title_text, border=False, new_x='LMARGIN', new_y='NEXT', align='L')
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Intelligent Attendance Monitoring System", border=False, new_x='LMARGIN', new_y='NEXT', align='L')
        self.set_draw_color(226, 232, 240)
        self.line(self.l_margin, self.get_y() + 2, self.epw + self.l_margin, self.get_y() + 2)
        self.ln(6)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align='C')


class ReportService:
    """
    Service layer for generating CSV and PDF reports.
    """
    
    def __init__(self, database_path: Optional[Path] = None, statistics_service: Optional[StatisticsService] = None):
        """
        Initialize ReportService.
        
        Args:
            database_path: Optional database path
            statistics_service: Injected StatisticsService instance (optional)
        """
        self.statistics_service = statistics_service or StatisticsService(database_path)

    # ========================================================================
    # CSV Generation
    # ========================================================================

    def export_attendance_history_csv(self, records: List[Dict[str, Any]]) -> str:
        """
        Export a list of attendance history records to CSV formatted string.
        
        Args:
            records: List of attendance history dictionaries
            
        Returns:
            CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        headers = [
            "Record ID",
            "Student ID",
            "Student Name",
            "Department",
            "Course Code",
            "Session Name",
            "Session Start Time",
            "Marked Time",
            "Status",
            "Detection Confidence",
            "Liveness Score",
            "Marked By",
            "Notes"
        ]
        writer.writerow(headers)
        
        for r in records:
            writer.writerow([
                r.get('record_id', ''),
                r.get('student_id', ''),
                r.get('student_name', ''),
                r.get('student_department', '') or '',
                r.get('course_code', '') or '',
                r.get('session_name', ''),
                r.get('session_start_time', ''),
                r.get('marked_at', ''),
                r.get('status', '').upper(),
                f"{r.get('detection_confidence', 0):.2f}" if r.get('detection_confidence') is not None else '',
                f"{r.get('liveness_score', 0):.2f}" if r.get('liveness_score') is not None else '',
                r.get('marked_by', ''),
                r.get('notes', '') or ''
            ])
            
        return output.getvalue()

    def export_student_summary_csv(self, student_stats: Dict[str, Any]) -> str:
        """
        Export a student's attendance summary and history to CSV string.
        
        Args:
            student_stats: Dictionary from StatisticsService.get_student_statistics
            
        Returns:
            CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Summary Header
        writer.writerow(["STUDENT ATTENDANCE REPORT"])
        writer.writerow(["Student ID", student_stats.get('student_id', '')])
        writer.writerow(["Student Name", student_stats.get('name', '')])
        writer.writerow(["Department", student_stats.get('department', '')])
        writer.writerow(["Sessions With Attendance Records", student_stats.get('total_sessions', 0)])
        writer.writerow(["Present Count", student_stats.get('present_count', 0)])
        writer.writerow(["Late Count", student_stats.get('late_count', 0)])
        writer.writerow(["Absences", "Not available: no session roster is recorded"])
        writer.writerow(["Attendance Rate", "Not available: no session roster is recorded"])
        writer.writerow([])
        
        # History Table
        writer.writerow(["Session Name", "Course", "Session Time", "Marked Time", "Status", "Confidence", "Marked By"])
        for r in student_stats.get('history', []):
            writer.writerow([
                r.get('session_name', ''),
                r.get('course_code', '') or '',
                r.get('session_start_time', ''),
                r.get('marked_at', ''),
                r.get('status', '').upper(),
                f"{r.get('detection_confidence', 0):.2f}" if r.get('detection_confidence') is not None else '',
                r.get('marked_by', '')
            ])
            
        return output.getvalue()

    def export_session_summary_csv(self, session_stats: Dict[str, Any]) -> str:
        """
        Export a session's attendance statistics and student attendance records to CSV string.
        
        Args:
            session_stats: Dictionary from StatisticsService.get_session_statistics
            
        Returns:
            CSV content as string
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(["SESSION ATTENDANCE REPORT"])
        writer.writerow(["Session Name", session_stats.get('session_name', '')])
        writer.writerow(["Course Code", session_stats.get('course_code', '')])
        writer.writerow(["Start Time", session_stats.get('start_time', '')])
        writer.writerow(["Status", session_stats.get('status', '').upper()])
        writer.writerow(["Expected Students", "Not available: no session roster is recorded"])
        writer.writerow(["Total Marked", session_stats.get('total_marked', 0)])
        writer.writerow(["Present Count", session_stats.get('present_count', 0)])
        writer.writerow(["Late Count", session_stats.get('late_count', 0)])
        writer.writerow(["Absences", "Not available: no session roster is recorded"])
        writer.writerow(["Attendance Rate", "Not available: no session roster is recorded"])
        writer.writerow([])
        
        writer.writerow(["Student ID", "Student Name", "Department", "Marked Time", "Status", "Confidence", "Liveness", "Marked By"])
        for r in session_stats.get('records', []):
            writer.writerow([
                r.get('student_id', ''),
                r.get('student_name', ''),
                r.get('student_department', '') or '',
                r.get('marked_at', ''),
                r.get('status', '').upper(),
                f"{r.get('detection_confidence', 0):.2f}" if r.get('detection_confidence') is not None else '',
                f"{r.get('liveness_score', 0):.2f}" if r.get('liveness_score') is not None else '',
                r.get('marked_by', '')
            ])
            
        return output.getvalue()

    # ========================================================================
    # PDF Generation
    # ========================================================================

    def generate_student_report_pdf(self, student_stats: Dict[str, Any]) -> bytes:
        """
        Generate a PDF document for an individual student attendance report.
        
        Args:
            student_stats: Dictionary from StatisticsService.get_student_statistics
            
        Returns:
            PDF file bytes
        """
        pdf = AttendancePDF(title_text="Student Attendance Report")
        pdf.add_page()
        
        # Student Info Box
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(pdf.l_margin, pdf.get_y(), pdf.epw, 24, 'FD')
        
        pdf.set_xy(pdf.l_margin + 4, pdf.get_y() + 3)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(40, 6, "Student Name:", new_x='RIGHT', new_y='TOP')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(55, 6, str(student_stats.get('name', '—')), new_x='RIGHT', new_y='TOP')
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(35, 6, "Student ID:", new_x='RIGHT', new_y='TOP')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, str(student_stats.get('student_id', '—')), new_x='LMARGIN', new_y='NEXT')
        
        pdf.set_x(pdf.l_margin + 4)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(40, 6, "Department:", new_x='RIGHT', new_y='TOP')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(55, 6, str(student_stats.get('department', '—')), new_x='RIGHT', new_y='TOP')
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(35, 6, "Profile Status:", new_x='RIGHT', new_y='TOP')
        pdf.set_font('Helvetica', '', 10)
        status_text = "Active" if student_stats.get('is_active') else "Inactive"
        pdf.cell(0, 6, status_text, new_x='LMARGIN', new_y='NEXT')
        
        pdf.ln(10)
        
        # Key Metrics Grid
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 7, "Attendance Performance Summary", new_x='LMARGIN', new_y='NEXT')
        
        col_w = pdf.epw / 4
        metrics = [
            ("Recorded Sessions", str(student_stats.get('total_sessions', 0))),
            ("Present", str(student_stats.get('present_count', 0))),
            ("Late", str(student_stats.get('late_count', 0))),
            ("On-time Rate", f"{student_stats.get('punctuality_percentage', 0)}%")
        ]
        
        pdf.set_fill_color(241, 245, 249)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(71, 85, 105)
        for label, _ in metrics:
            pdf.cell(col_w, 6, label, border=1, align='C', fill=True)
        pdf.ln()
        
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(15, 23, 42)
        for _, val in metrics:
            pdf.cell(col_w, 8, val, border=1, align='C')
        pdf.ln(12)
        
        # History Table
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 7, "Attendance Records History", new_x='LMARGIN', new_y='NEXT')
        
        table_headers = [
            ("Session Name", 65),
            ("Course", 30),
            ("Marked Time", 45),
            ("Status", 25),
            ("Marked By", 25)
        ]
        
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 8)
        for title, width in table_headers:
            pdf.cell(width, 7, title, border=1, align='C', fill=True)
        pdf.ln()
        
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(15, 23, 42)
        
        history = student_stats.get('history', [])
        if not history:
            pdf.cell(pdf.epw, 8, "No attendance records recorded for this student.", border=1, align='C')
            pdf.ln()
        else:
            for i, r in enumerate(history):
                fill_row = (i % 2 == 1)
                pdf.set_fill_color(248, 250, 252) if fill_row else pdf.set_fill_color(255, 255, 255)
                
                pdf.cell(65, 6, str(r.get('session_name', '—'))[:35], border=1, fill=fill_row)
                pdf.cell(30, 6, str(r.get('course_code', '—'))[:15], border=1, fill=fill_row)
                pdf.cell(45, 6, str(r.get('marked_at', '—'))[:19], border=1, align='C', fill=fill_row)
                
                status_val = str(r.get('status', '—')).upper()
                pdf.cell(25, 6, status_val, border=1, align='C', fill=fill_row)
                pdf.cell(25, 6, str(r.get('marked_by', 'system')), border=1, align='C', fill=fill_row)
                pdf.ln()
                
        return bytes(pdf.output())

    def generate_session_report_pdf(self, session_stats: Dict[str, Any]) -> bytes:
        """
        Generate a PDF document for a session attendance report.
        
        Args:
            session_stats: Dictionary from StatisticsService.get_session_statistics
            
        Returns:
            PDF file bytes
        """
        pdf = AttendancePDF(title_text="Session Attendance Report")
        pdf.add_page()
        
        # Session Metadata Box
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(pdf.l_margin, pdf.get_y(), pdf.epw, 24, 'FD')
        
        pdf.set_xy(pdf.l_margin + 4, pdf.get_y() + 3)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(35, 6, "Session Name:", new_x='RIGHT', new_y='TOP')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(60, 6, str(session_stats.get('session_name', '—')), new_x='RIGHT', new_y='TOP')
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(30, 6, "Course Code:", new_x='RIGHT', new_y='TOP')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, str(session_stats.get('course_code', '—')), new_x='LMARGIN', new_y='NEXT')
        
        pdf.set_x(pdf.l_margin + 4)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(35, 6, "Start Time:", new_x='RIGHT', new_y='TOP')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(60, 6, str(session_stats.get('start_time', '—')), new_x='RIGHT', new_y='TOP')
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(30, 6, "Status:", new_x='RIGHT', new_y='TOP')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, str(session_stats.get('status', '—')).upper(), new_x='LMARGIN', new_y='NEXT')
        
        pdf.ln(10)
        
        # Performance Summary Metrics
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 7, "Session Attendance Metrics", new_x='LMARGIN', new_y='NEXT')
        
        col_w = pdf.epw / 4
        metrics = [
            ("Total Marked", str(session_stats.get('total_marked', 0))),
            ("Present", str(session_stats.get('present_count', 0))),
            ("Late", str(session_stats.get('late_count', 0))),
            ("On-time Rate", f"{session_stats.get('punctuality_percentage', 0)}%")
        ]
        
        pdf.set_fill_color(241, 245, 249)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(71, 85, 105)
        for label, _ in metrics:
            pdf.cell(col_w, 6, label, border=1, align='C', fill=True)
        pdf.ln()
        
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(15, 23, 42)
        for _, val in metrics:
            pdf.cell(col_w, 8, val, border=1, align='C')
        pdf.ln(12)
        
        # Records Table
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 7, "Recorded Student Attendance", new_x='LMARGIN', new_y='NEXT')
        
        table_headers = [
            ("Student ID", 35),
            ("Student Name", 60),
            ("Department", 35),
            ("Status", 25),
            ("Marked At", 35)
        ]
        
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 8)
        for title, width in table_headers:
            pdf.cell(width, 7, title, border=1, align='C', fill=True)
        pdf.ln()
        
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(15, 23, 42)
        
        records = session_stats.get('records', [])
        if not records:
            pdf.cell(pdf.epw, 8, "No student attendance records logged for this session.", border=1, align='C')
            pdf.ln()
        else:
            for i, r in enumerate(records):
                fill_row = (i % 2 == 1)
                pdf.set_fill_color(248, 250, 252) if fill_row else pdf.set_fill_color(255, 255, 255)
                
                pdf.cell(35, 6, str(r.get('student_id', '—'))[:15], border=1, fill=fill_row)
                pdf.cell(60, 6, str(r.get('student_name', '—'))[:30], border=1, fill=fill_row)
                pdf.cell(35, 6, str(r.get('student_department', '—'))[:18], border=1, fill=fill_row)
                
                status_val = str(r.get('status', '—')).upper()
                pdf.cell(25, 6, status_val, border=1, align='C', fill=fill_row)
                pdf.cell(35, 6, str(r.get('marked_at', '—'))[:19], border=1, align='C', fill=fill_row)
                pdf.ln()
                
        return bytes(pdf.output())
