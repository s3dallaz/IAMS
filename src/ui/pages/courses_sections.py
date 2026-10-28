"""
Courses & Sections page.

Tab 1 lists the courses (add, edit, deactivate, delete). Tab 2 opens one
course's sections and a section's class list (add students by search or by
pasting IDs, remove them). Tab 3 imports class lists from Excel / CSV
(checked and previewed first, saved in one transaction).
"""

import re
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.services.bulk_import_service import BulkImportError, _csv_bytes
from src.services.class_list_import_service import (
    ClassListImportPlan, apply_class_list_import, class_list_errors_csv,
    class_list_template_csv, plan_class_list_import, read_class_list_table,
)
from src.services.course_service import CourseService, CourseServiceError
from src.services.student_service import StudentService
from src.utils.logger import get_logger

logger = get_logger(__name__)

TAB_COURSES = "📚 Courses"
TAB_SECTIONS = "👥 Sections & class lists"
TAB_IMPORT = "📥 Import class lists"
TABS_KEY = "courses_tab"
COURSE_SELECTOR_KEY = "courses_section_course"
SECTION_SELECTOR_KEY = "courses_section_section"
FLASH_KEY = "courses_flash"
TABLE_VERSION_KEY = "courses_table_version"

CLASS_LIST_ROWS = 500     # rows of a class list shown at once (search finds the rest)
SEARCH_RESULTS = 50       # students offered when searching to add
PREVIEW_ROWS = 500        # rows per import preview table


def render_courses_sections():
    """Main function to render the Courses & Sections page."""
    st.title("📚 Courses & Sections")
    st.markdown("A **course** (e.g. TM103) has one or more **sections** (e.g. A, B, Lab 1). "
                "Each section has an instructor and its own **class list** of students.")
    st.markdown("---")

    course_service = CourseService()
    student_service = StudentService()
    _show_flash()

    tab_courses, tab_sections, tab_import = st.tabs(
        [TAB_COURSES, TAB_SECTIONS, TAB_IMPORT], key=TABS_KEY, on_change="rerun"
    )
    if tab_courses.open:
        with tab_courses:
            _render_courses_tab(course_service)
    if tab_sections.open:
        with tab_sections:
            _render_sections_tab(course_service, student_service)
    if tab_import.open:
        with tab_import:
            _render_import_tab(course_service, student_service)


# ============================================================================
# Tab 1: courses
# ============================================================================

def _render_courses_tab(course_service: CourseService):
    with st.expander("➕ Add a course", expanded=False):
        with st.form(f"add_course_form_{_table_version()}"):
            c1, c2 = st.columns([1, 2])
            code = c1.text_input("Course code *", placeholder="e.g. TM103")
            name = c2.text_input("Course name *", placeholder="e.g. Computer Organization")
            department = st.text_input("Department", placeholder="optional")
            notes = st.text_input("Notes", placeholder="optional")
            if st.form_submit_button("Add course", type="primary"):
                try:
                    course_service.create_course(code, name, department, notes)
                except CourseServiceError as e:
                    st.error(f"❌ {e}")
                else:
                    _flash(f"✅ Course **{code.strip()}** added. Add its sections in **{TAB_SECTIONS}**.")
                    _bump_table_version()
                    st.rerun()

    courses = course_service.list_courses()
    if not courses:
        st.info("No courses yet. Add one above, or import class lists in the **📥 Import class lists** tab - "
                "courses and sections in the file are created for you.")
        return

    search = st.text_input("Search", placeholder="Course code or name", key="courses_search").strip().lower()
    if search:
        courses = [c for c in courses if search in c['course_code'].lower() or search in c['name'].lower()]
    if not courses:
        st.info("No courses match.")
        return

    st.caption(f"{len(courses)} course(s). **Select a row** to edit, deactivate or delete it, or to open its sections.")
    event = st.dataframe(
        [{
            'Code': c['course_code'],
            'Name': c['name'],
            'Department': c['department'] or '—',
            'Sections': c['section_count'],
            'Students': c['student_count'],
            'Status': "✅ Active" if c['is_active'] else "⏸️ Inactive",
        } for c in courses],
        width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row",
        key=f"courses_table_{search}_{_table_version()}",
    )
    rows = event.selection.rows if event else []
    if rows and rows[0] < len(courses):
        _render_course_actions(course_service, courses[rows[0]])


def _render_course_actions(course_service: CourseService, course: Dict[str, Any]):
    st.markdown(f"#### ⚡ {course['course_code']} - {course['name']}")
    b1, b2, b3, b4 = st.columns(4)
    b1.button("👥 Open sections", key=f"open_course_{course['id']}", type="primary",
              on_click=_open_course_sections, args=(course['id'],), width="stretch")
    edit = b2.toggle("✏️ Edit", key=f"edit_course_{course['id']}")
    if course['is_active']:
        if b3.button("⏸️ Deactivate", key=f"deact_course_{course['id']}", width="stretch",
                     help="Hidden when creating new sessions. Nothing is deleted."):
            _run(lambda: course_service.set_course_active(course['id'], False),
                 f"⏸️ Course **{course['course_code']}** deactivated.")
    elif b3.button("▶️ Activate", key=f"act_course_{course['id']}", width="stretch"):
        _run(lambda: course_service.set_course_active(course['id'], True),
             f"▶️ Course **{course['course_code']}** activated.")
    confirm = b4.toggle("🗑️ Delete", key=f"del_course_{course['id']}")

    if edit:
        with st.form(f"edit_course_form_{course['id']}"):
            c1, c2 = st.columns([1, 2])
            code = c1.text_input("Course code *", value=course['course_code'])
            name = c2.text_input("Course name *", value=course['name'])
            department = st.text_input("Department", value=course['department'] or "")
            notes = st.text_input("Notes", value=course['notes'] or "")
            if st.form_submit_button("Save changes", type="primary"):
                _run(lambda: course_service.update_course(course['id'], code, name, department, notes),
                     f"✅ Course **{code.strip()}** saved.")
    if confirm:
        st.warning(f"Delete **{course['course_code']}** with its {course['section_count']} section(s) and their "
                   f"class lists? The students themselves are not deleted. This cannot be undone.")
        if st.button("Delete course", key=f"confirm_del_course_{course['id']}", type="primary"):
            _run(lambda: course_service.delete_course(course['id']),
                 f"🗑️ Course **{course['course_code']}** deleted.")


def _open_course_sections(course_id: int):
    """'Open sections' callback: runs before the rerun, so the sections tab opens on this course."""
    st.session_state[COURSE_SELECTOR_KEY] = course_id
    st.session_state.pop(SECTION_SELECTOR_KEY, None)
    st.session_state[TABS_KEY] = TAB_SECTIONS


# ============================================================================
# Tab 2: sections and class lists
# ============================================================================

def _render_sections_tab(course_service: CourseService, student_service: StudentService):
    courses = course_service.list_courses()
    if not courses:
        st.info(f"Add a course first in **{TAB_COURSES}**.")
        return
    course_labels = {c['id']: f"{c['course_code']} - {c['name']}" + ("" if c['is_active'] else " (inactive)")
                     for c in courses}
    if st.session_state.get(COURSE_SELECTOR_KEY) not in course_labels:
        st.session_state[COURSE_SELECTOR_KEY] = courses[0]['id']
    course_id = st.selectbox("Course", options=list(course_labels), format_func=course_labels.get,
                             key=COURSE_SELECTOR_KEY)

    with st.expander("➕ Add a section to this course", expanded=False):
        with st.form(f"add_section_form_{course_id}_{_table_version()}"):
            c1, c2 = st.columns([1, 2])
            name = c1.text_input("Section name *", placeholder="e.g. A, B, Lab 1")
            instructor = c2.text_input("Instructor", placeholder="e.g. Dr. Sara Ali")
            notes = st.text_input("Notes", placeholder="optional, e.g. Sunday 10:00, Hall 3")
            if st.form_submit_button("Add section", type="primary"):
                try:
                    section_id = course_service.create_section(course_id, name, instructor, notes)
                except CourseServiceError as e:
                    st.error(f"❌ {e}")
                else:
                    st.session_state[SECTION_SELECTOR_KEY] = section_id
                    _flash(f"✅ Section **{name.strip()}** added. Now add its students below.")
                    _bump_table_version()
                    st.rerun()

    sections = course_service.list_sections(course_id=course_id)
    if not sections:
        st.info("This course has no sections yet. Add one above.")
        return
    section_labels = {s['id']: f"{s['name']} · {s['instructor_name'] or 'no instructor'} · "
                               f"{s['student_count']:,} student(s)" + ("" if s['is_active'] else " (inactive)")
                      for s in sections}
    if st.session_state.get(SECTION_SELECTOR_KEY) not in section_labels:
        st.session_state[SECTION_SELECTOR_KEY] = sections[0]['id']
    section_id = st.selectbox("Section", options=list(section_labels), format_func=section_labels.get,
                              key=SECTION_SELECTOR_KEY)
    section = next(s for s in sections if s['id'] == section_id)

    _render_section_settings(course_service, section)
    st.markdown("---")
    _render_class_list(course_service, student_service, section)


def _render_section_settings(course_service: CourseService, section: Dict[str, Any]):
    with st.expander(f"⚙️ Section {section['name']} settings", expanded=False):
        with st.form(f"edit_section_form_{section['id']}"):
            c1, c2 = st.columns([1, 2])
            name = c1.text_input("Section name *", value=section['name'])
            instructor = c2.text_input("Instructor", value=section['instructor_name'] or "")
            notes = st.text_input("Notes", value=section['notes'] or "")
            if st.form_submit_button("Save changes", type="primary"):
                _run(lambda: course_service.update_section(section['id'], name, instructor, notes),
                     f"✅ Section **{name.strip()}** saved.")
        b1, b2 = st.columns(2)
        if section['is_active']:
            if b1.button("⏸️ Deactivate section", key=f"deact_section_{section['id']}", width="stretch",
                         help="Hidden when creating new sessions. Its class list is kept."):
                _run(lambda: course_service.set_section_active(section['id'], False),
                     f"⏸️ Section **{section['name']}** deactivated.")
        elif b1.button("▶️ Activate section", key=f"act_section_{section['id']}", width="stretch"):
            _run(lambda: course_service.set_section_active(section['id'], True),
                 f"▶️ Section **{section['name']}** activated.")
        if b2.toggle("🗑️ Delete section", key=f"del_section_{section['id']}"):
            st.warning(f"Delete section **{section['name']}** and its class list of "
                       f"{section['student_count']:,} student(s)? The students themselves are not deleted.")
            if st.button("Delete section", key=f"confirm_del_section_{section['id']}", type="primary"):
                st.session_state.pop(SECTION_SELECTOR_KEY, None)
                _run(lambda: course_service.delete_section(section['id']),
                     f"🗑️ Section **{section['name']}** deleted.")


def _render_class_list(course_service: CourseService, student_service: StudentService, section: Dict[str, Any]):
    total = section['student_count']
    st.markdown(f"#### 📋 Class list: {section['course_code']} / {section['name']} ({total:,} student(s))")

    # A key keeps the expander open while searching (each keystroke reruns the page)
    with st.expander("➕ Add students", expanded=total == 0, key=f"add_students_open_{section['id']}"):
        _render_add_students(course_service, student_service, section)

    if not total:
        st.info("No students on this class list yet.")
        return

    search = st.text_input("Search the class list", placeholder="Name or student ID",
                           key=f"class_list_search_{section['id']}").strip()
    students = course_service.get_class_list(section['id'], search=search, limit=CLASS_LIST_ROWS)
    if not students:
        st.info("No student on this class list matches.")
        return
    shown = f"Showing the first {len(students):,}; search to find others. " if len(students) == CLASS_LIST_ROWS else ""
    st.caption(f"{shown}**Select rows** to remove students from this class list.")
    event = st.dataframe(
        [{
            'Student ID': s['student_id'],
            'Name': s['name'],
            'Department': s['department'] or '—',
            'Status': "✅ Active" if s['is_active'] else "⏸️ Inactive",
        } for s in students],
        width="stretch", hide_index=True, on_select="rerun", selection_mode="multi-row",
        key=f"class_list_table_{section['id']}_{search}_{_table_version()}",
    )
    selected = [students[i] for i in (event.selection.rows if event else []) if i < len(students)]

    c1, c2 = st.columns(2)
    if c1.button(f"➖ Remove {len(selected)} selected", key=f"remove_students_{section['id']}",
                 disabled=not selected, width="stretch"):
        _run(lambda: course_service.remove_students(section['id'], [s['student_db_id'] for s in selected]),
             f"➖ {len(selected)} student(s) removed from section **{section['name']}**.")
    c2.download_button(
        "⬇️ Download class list (CSV)",
        _class_list_csv(course_service, section),
        file_name=f"class_list_{section['course_code']}_{section['name']}.csv".replace(" ", "_"),
        mime="text/csv", key=f"download_class_list_{section['id']}", width="stretch",
    )


def _render_add_students(course_service: CourseService, student_service: StudentService, section: Dict[str, Any]):
    on_list = course_service.get_class_list_student_ids(section['id'])

    st.markdown("**Search and pick**")
    query = st.text_input("Find students", placeholder="Name or student ID",
                          key=f"add_search_{section['id']}").strip()
    if query:
        found = [s for s in student_service.search_students(query, limit=SEARCH_RESULTS) if s['id'] not in on_list]
        if not found:
            st.caption("No students found who aren't already on this class list.")
        else:
            labels = {s['id']: f"{s['student_id']} - {s['name']}" for s in found}
            picked = st.multiselect("Students to add", options=list(labels), format_func=labels.get,
                                    key=f"add_pick_{section['id']}_{query}")
            if st.button(f"➕ Add {len(picked)} student(s)", key=f"add_picked_{section['id']}",
                         type="primary", disabled=not picked):
                _run(lambda: course_service.add_students(section['id'], picked),
                     f"➕ {len(picked)} student(s) added to section **{section['name']}**.")

    st.markdown("**Or paste student IDs**")
    pasted = st.text_area("Student IDs", placeholder="2021001, 2021002 ... (commas, spaces or one per line)",
                          key=f"add_paste_{section['id']}_{_table_version()}", height=100)
    ids = list(dict.fromkeys(part for part in re.split(r"[\s,;]+", pasted) if part))
    if st.button(f"➕ Add {len(ids)} pasted ID(s)", key=f"add_pasted_{section['id']}", disabled=not ids):
        _add_pasted_ids(course_service, student_service, section, ids)


def _add_pasted_ids(course_service: CourseService, student_service: StudentService,
                    section: Dict[str, Any], ids: List[str]):
    db_ids, missing = [], []
    for student_id in ids:
        student = student_service.get_student_by_student_id(student_id)
        if student:
            db_ids.append(student['id'])
        else:
            missing.append(student_id)
    try:
        added = course_service.add_students(section['id'], db_ids)
    except CourseServiceError as e:
        st.error(f"❌ {e}")
        return
    message = f"➕ {added} student(s) added to section **{section['name']}**"
    if len(db_ids) > added:
        message += f"; {len(db_ids) - added} were already on it"
    if missing:
        message += f". ⚠️ Not in the Student Registry: {', '.join(missing[:20])}" + (" ..." if len(missing) > 20 else "")
    _flash(message + ".", "warning" if missing else "success")
    _bump_table_version()
    st.rerun()


def _class_list_csv(course_service: CourseService, section: Dict[str, Any]) -> bytes:
    rows = [{'course_code': section['course_code'], 'section': section['name'], 'student_id': s['student_id'],
             'name': s['name'], 'department': s['department']}
            for s in course_service.get_class_list(section['id'])]
    return _csv_bytes(['course_code', 'section', 'student_id', 'name', 'department'], rows)


# ============================================================================
# Tab 3: import class lists
# ============================================================================

def _render_import_tab(course_service: CourseService, student_service: StudentService):
    st.markdown(
        "Upload an **Excel (.xlsx)** or **CSV** file with one row per student and section. "
        "Required columns: `course_code`, `section`, `student_id`. Optional: `course_name`, `instructor` "
        "(used when a course or section is new).")
    st.caption("Courses and sections that don't exist yet are created. Students must already be in the "
               "Student Registry. Importing the same file again is safe, and nobody is removed by an import.")
    st.download_button("⬇️ Download template (CSV)", class_list_template_csv(),
                       file_name="class_lists_template.csv", mime="text/csv", key="btn_class_list_template")

    if 'class_list_upload_key' not in st.session_state:
        st.session_state['class_list_upload_key'] = 0
    uploaded = st.file_uploader("Class lists", type=["xlsx", "csv"],
                                key=f"class_list_file_{st.session_state['class_list_upload_key']}")

    result = st.session_state.get('class_list_result')
    if result:
        st.success(f"✅ Import finished: **{result['added']:,}** student(s) added to class lists, "
                   f"{result['courses']} new course(s), {result['sections']} new section(s)"
                   + (f", {result['skipped']:,} skipped" if result['skipped'] else "")
                   + (f", {result['errors']:,} row(s) with errors not imported." if result['errors'] else "."))
    if uploaded is None:
        return
    st.session_state['class_list_result'] = None

    try:
        table, ignored = read_class_list_table(uploaded.getvalue(), uploaded.name)
        plan = plan_class_list_import(table, course_service, student_service, ignored)
    except BulkImportError as e:
        st.error(f"❌ {e}")
        return
    if not plan.rows:
        st.warning("The file has headers but no rows.")
        return

    counts = plan.counts
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("➕ To add", f"{counts['add']:,}")
    c2.metric("📚 New courses", len(plan.new_courses))
    c3.metric("👥 New sections", len(plan.new_sections))
    c4.metric("➖ Skipped", f"{counts['skip']:,}", help="Already on the class list, or repeated in the file")
    c5.metric("❌ Errors", f"{counts['error']:,}")
    if plan.ignored_columns:
        st.info(f"These columns are not used and will be ignored: {', '.join(plan.ignored_columns)}")

    if counts['error']:
        st.markdown("#### ❌ Rows with errors (not imported)")
        st.dataframe(_plan_table(plan, "error"), hide_index=True, width="stretch")
        st.download_button("⬇️ Download the rows with errors", class_list_errors_csv(plan),
                           file_name="class_lists_errors.csv", mime="text/csv", key="btn_class_list_errors")
    if plan.new_courses or plan.new_sections:
        with st.expander(f"📚 New courses and sections ({len(plan.new_courses)} / {len(plan.new_sections)})"):
            if plan.new_courses:
                st.dataframe([{'Course code': c['course_code'], 'Name': c['name']} for c in plan.new_courses],
                             hide_index=True, width="stretch")
            if plan.new_sections:
                st.dataframe([{'Course': s['course_code'], 'Section': s['name'],
                               'Instructor': s['instructor_name'] or '—'} for s in plan.new_sections],
                             hide_index=True, width="stretch")
    if counts['add']:
        with st.expander(f"➕ Students to add ({counts['add']:,})", expanded=counts['add'] <= 20):
            st.dataframe(_plan_table(plan, "add"), hide_index=True, width="stretch")
    if counts['skip']:
        with st.expander(f"➖ Skipped rows ({counts['skip']:,})"):
            st.dataframe(_plan_table(plan, "skip"), hide_index=True, width="stretch")

    if st.button(f"✅ Import {counts['add']:,} student(s)", type="primary",
                 disabled=not (counts['add'] or plan.new_sections), key="btn_class_list_import"):
        try:
            outcome = apply_class_list_import(plan, course_service)
        except BulkImportError as e:
            st.error(f"❌ {e}")
            return
        st.session_state['class_list_result'] = outcome
        st.session_state['class_list_upload_key'] += 1   # clear the uploader
        _bump_table_version()
        st.rerun()


def _plan_table(plan: ClassListImportPlan, action: str) -> pd.DataFrame:
    rows = []
    for row in plan.with_action(action)[:PREVIEW_ROWS]:
        item = {'Row': row.row_number, 'Course': row.course_code, 'Section': row.section,
                'Student ID': row.student_id, 'Name': row.student_name}
        if action == "error":
            item['Error'] = "; ".join(row.errors)
        elif action == "skip":
            item['Why'] = row.note
        rows.append(item)
    return pd.DataFrame(rows)


# ============================================================================
# Helpers
# ============================================================================

def _run(action, success: str):
    """Run a change, then show `success` after the rerun (or the error now)."""
    try:
        action()
    except CourseServiceError as e:
        st.error(f"❌ {e}")
        return
    except Exception as e:
        logger.error(f"Courses & Sections action failed: {e}")
        st.error(f"❌ Something went wrong: {e}")
        return
    _flash(success)
    _bump_table_version()
    st.rerun()


def _flash(message: str, kind: str = "success"):
    """Show a message on the next run (st.rerun() would discard one shown now)."""
    st.session_state[FLASH_KEY] = (kind, message)


def _show_flash():
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        kind, message = flash
        getattr(st, kind)(message)


def _table_version() -> int:
    return st.session_state.get(TABLE_VERSION_KEY, 0)


def _bump_table_version():
    """New table / form keys after a change: no selection points at a row that moved."""
    st.session_state[TABLE_VERSION_KEY] = _table_version() + 1
