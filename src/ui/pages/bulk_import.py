"""
Bulk Import view of the Student Registry.

Tab 1 imports students from an Excel / CSV file (checked and previewed first,
saved in one transaction). Tab 2 enrolls face photos from a ZIP file named by
student ID, with a progress bar and a downloadable report.
"""

import time

import pandas as pd
import streamlit as st

from src.services.bulk_import_service import (
    BulkImportError,
    StudentImportPlan,
    apply_student_import,
    import_face_zip,
    plan_student_import,
    read_student_table,
    student_errors_csv,
    student_template_csv,
)
from src.services.enrollment_service import EnrollmentService
from src.services.student_service import StudentService
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREVIEW_ROWS = 500   # rows shown per preview table (the import itself has no such limit)


def render_bulk_import_view(student_service: StudentService, enrollment_service: EnrollmentService):
    """Bulk import page: students from Excel/CSV, then their face photos from a ZIP."""
    st.subheader("📥 Bulk Import")
    if st.button("⬅️ Back to Student Directory", key="btn_back_bulk"):
        st.session_state['current_view'] = 'directory'
        st.rerun()
    st.caption("Step 1: import the student list. Step 2: import their face photos. "
               "Both are safe to repeat - nothing is duplicated.")

    tab_students, tab_faces = st.tabs(["1️⃣ Students (Excel / CSV)", "2️⃣ Face photos (ZIP)"])
    with tab_students:
        _render_student_import(student_service)
    with tab_faces:
        _render_face_import(student_service, enrollment_service)


# ============================================================================
# Tab 1: students from Excel / CSV
# ============================================================================

def _render_student_import(student_service: StudentService):
    st.markdown(
        "Upload an **Excel (.xlsx)** or **CSV** file with one student per row. "
        "Required columns: `student_id`, `name`. Optional: `email`, `department`, `notes`."
    )
    st.download_button("⬇️ Download template (CSV)", student_template_csv(),
                       file_name="students_template.csv", mime="text/csv", key="btn_student_template")

    if 'bulk_students_upload_key' not in st.session_state:
        st.session_state['bulk_students_upload_key'] = 0
    uploaded = st.file_uploader("Student list", type=["xlsx", "csv"],
                                key=f"bulk_students_file_{st.session_state['bulk_students_upload_key']}")

    result = st.session_state.get('bulk_students_result')
    if result:
        st.success(f"✅ Import finished: **{result['created']}** new student(s) added, "
                   f"**{result['updated']}** updated, {result['unchanged']} unchanged"
                   + (f", {result['errors']} row(s) with errors skipped." if result['errors'] else "."))
    if uploaded is None:
        return
    st.session_state['bulk_students_result'] = None

    data = uploaded.getvalue()
    try:
        table, ignored = read_student_table(data, uploaded.name)
        plan = plan_student_import(table, student_service, ignored)
    except BulkImportError as e:
        st.error(f"❌ {e}")
        return
    if not plan.rows:
        st.warning("The file has headers but no student rows.")
        return

    counts = plan.counts
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🆕 New", counts["create"])
    c2.metric("✏️ Updated", counts["update"])
    c3.metric("➖ Unchanged", counts["unchanged"])
    c4.metric("❌ Errors", counts["error"])
    if plan.ignored_columns:
        st.info(f"These columns are not used and will be ignored: {', '.join(plan.ignored_columns)}")

    if counts["error"]:
        st.markdown("#### ❌ Rows with errors (not imported)")
        st.dataframe(_rows_table(plan, "error", with_errors=True), hide_index=True, width="stretch")
        st.download_button("⬇️ Download the rows with errors", student_errors_csv(plan),
                           file_name="students_errors.csv", mime="text/csv", key="btn_student_errors")
        st.caption("Fix them in the file and upload it again; the rows already imported stay as they are.")

    if counts["create"]:
        with st.expander(f"🆕 New students ({counts['create']})", expanded=counts["create"] <= 20):
            st.dataframe(_rows_table(plan, "create"), hide_index=True, width="stretch")
    update_existing = True
    if counts["update"]:
        with st.expander(f"✏️ Students whose details change ({counts['update']})"):
            st.dataframe(_rows_table(plan, "update", with_changes=True), hide_index=True, width="stretch")
        update_existing = st.checkbox("Update the details of existing students", value=True,
                                      key="chk_bulk_update_existing",
                                      help="Off: only new students are added; existing ones are left as they are. "
                                           "Empty cells never erase existing details.")

    to_save = counts["create"] + (counts["update"] if update_existing else 0)
    if st.button(f"✅ Import {to_save} student(s)", type="primary", disabled=to_save == 0,
                 key="btn_bulk_import_students"):
        try:
            outcome = apply_student_import(plan, student_service, update_existing=update_existing)
        except BulkImportError as e:
            st.error(f"❌ {e}")
            return
        st.session_state['bulk_students_result'] = outcome
        st.session_state['bulk_students_upload_key'] += 1   # clear the uploader
        st.rerun()


def _rows_table(plan: StudentImportPlan, action: str, with_errors: bool = False,
                with_changes: bool = False) -> pd.DataFrame:
    rows = []
    for row in plan.with_action(action)[:PREVIEW_ROWS]:
        item = {"Row": row.row_number, "Student ID": row.student_id, "Name": row.name,
                "Email": row.email or "", "Department": row.department or ""}
        if with_changes:
            item["Changes"] = ", ".join(row.changes)
        if with_errors:
            item["Problem"] = "; ".join(row.errors)
        rows.append(item)
    return pd.DataFrame(rows)


# ============================================================================
# Tab 2: face photos from a ZIP
# ============================================================================

def _render_face_import(student_service: StudentService, enrollment_service: EnrollmentService):
    st.markdown(
        "Upload a **ZIP file** of face photos named after the student IDs "
        "(import the student list first):"
    )
    st.code("photos.zip\n"
            "├── 2021001.jpg          ← one photo per student\n"
            "├── 2021002.jpg\n"
            "├── 2021003_2.jpg        ← extra photos: _2, -2 or \" (2)\"\n"
            "└── 2021004/             ← or one folder per student\n"
            "    ├── front.jpg\n"
            "    └── side.jpg", language=None)
    st.caption("Use clear, front-facing photos with one face each. Large photos are fine; they are "
               "shrunk automatically. Keep this page open while the import runs.")

    if 'bulk_faces_upload_key' not in st.session_state:
        st.session_state['bulk_faces_upload_key'] = 0
    uploaded = st.file_uploader("Face photos (ZIP)", type=["zip"],
                                key=f"bulk_faces_file_{st.session_state['bulk_faces_upload_key']}")
    skip_existing = st.checkbox(
        "Skip students who already have face data", value=True, key="chk_bulk_skip_faces",
        help="On: only students without face data are enrolled, so running the same ZIP again "
             "continues where an interrupted import stopped. Off: the photos are added as extra samples.")

    if uploaded is not None and st.button("🚀 Start face import", type="primary", key="btn_bulk_import_faces"):
        progress_bar = st.progress(0.0, text="Starting...")
        started = time.time()

        def show_progress(done: int, total: int, current: str):
            if done == total or done % 5 == 0:
                text = f"{done} / {total} photos" + (f" - {current}" if current else "")
                progress_bar.progress(done / total if total else 1.0, text=text)

        try:
            report = import_face_zip(uploaded.getvalue(), enrollment_service, student_service,
                                     skip_students_with_faces=skip_existing, progress=show_progress)
        except BulkImportError as e:
            st.error(f"❌ {e}")
            return
        except Exception as e:
            logger.exception("Face ZIP import failed")
            st.error(f"❌ The import stopped: {e}. Photos enrolled so far are kept; run it again to continue.")
            return
        st.session_state['bulk_faces_report'] = (report, time.time() - started)
        st.session_state['bulk_faces_upload_key'] += 1
        st.rerun()

    stored = st.session_state.get('bulk_faces_report')
    if stored:
        _render_face_report(*stored)


def _render_face_report(report, seconds: float):
    counts = report.counts
    st.success(f"✅ Face import finished in {seconds:.0f} s. The live attendance camera picks up "
               f"new faces within about 30 seconds.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Enrolled", counts["enrolled"])
    c2.metric("⚠️ Enrolled with warning", counts["warnings"])
    c3.metric("⏭️ Skipped", counts["skipped"])
    c4.metric("❌ Failed", counts["failed"])

    problems = [r for r in report.results if r.status == "failed" or (r.status == "enrolled" and r.message)]
    if problems:
        st.markdown("#### Photos to check")
        st.dataframe(pd.DataFrame([{"File": r.path, "Student ID": r.student_id,
                                    "Result": "❌ failed" if r.status == "failed" else "⚠️ enrolled",
                                    "Reason": r.message} for r in problems[:PREVIEW_ROWS]]),
                     hide_index=True, width="stretch")
    if report.ignored_files:
        st.caption(f"{len(report.ignored_files)} file(s) in the ZIP are not photos and were ignored "
                   f"(e.g. {report.ignored_files[0]}).")

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button("⬇️ Download the full report (CSV)", report.report_csv(),
                           file_name="face_import_report.csv", mime="text/csv", key="btn_face_report")
    with col_b:
        if report.students_without_faces:
            st.download_button(f"⬇️ {len(report.students_without_faces)} active student(s) still without "
                               f"face data (CSV)", report.missing_faces_csv(),
                               file_name="students_without_faces.csv", mime="text/csv",
                               key="btn_missing_faces")
        else:
            st.markdown("🎉 Every active student has face data.")
