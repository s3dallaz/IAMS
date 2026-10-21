"""
Tests for the bulk import of students (Excel / CSV) and face photos (ZIP).

Face detection and embedding are replaced by fakes driven by the photo's
colour, so each test controls which photos show the same person.
"""

import io
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pandas as pd
import pytest

import config
from src.computer_vision.face_detector import DetectedFace
from src.computer_vision.face_embedder import FaceEmbedding
from src.database.schema import initialize_database
from src.services.bulk_import_service import (
    MAX_PHOTO_SIDE,

    BulkImportError,
    apply_student_import,
    import_face_zip,
    plan_student_import,
    read_student_table,
    student_errors_csv,
    student_template_csv,
)
from src.services.enrollment_service import EnrollmentService
from src.services.student_service import StudentService


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Path(f.name)
    initialize_database(db)
    yield db
    db.unlink(missing_ok=True)


@pytest.fixture
def students(temp_db):
    return StudentService(temp_db)


def _csv(text: str, bom: bool = False) -> bytes:
    data = text.strip().replace("\n", "\r\n").encode("utf-8")
    return (b"\xef\xbb\xbf" + data) if bom else data


def _plan(data: bytes, students, filename="students.csv"):
    table, ignored = read_student_table(data, filename)
    return plan_student_import(table, students, ignored)


# ============================================================================
# Students from CSV / Excel
# ============================================================================

def test_headers_are_matched_loosely_and_ids_keep_their_leading_zeros(students):
    plan = _plan(_csv("""
Student ID,Full Name,E-mail,Dept,Phone
0021001,Alice Ahmed,alice@uni.edu,CS,0100
2021002.0,  Omar Hassan ,,CS,
""", bom=True), students)

    assert plan.counts == {"create": 2, "update": 0, "unchanged": 0, "error": 0}
    alice, omar = plan.rows
    assert (alice.student_id, alice.name, alice.email, alice.department) == ("0021001", "Alice Ahmed", "alice@uni.edu", "CS")
    assert (omar.student_id, omar.name, omar.email) == ("2021002", "Omar Hassan", None)   # Excel float, trimmed
    assert plan.ignored_columns == ["Phone"]


def test_semicolon_separated_csv_is_read(students):
    plan = _plan(_csv("student_id;name\n2021001;Alice"), students)
    assert plan.rows[0].student_id == "2021001" and plan.rows[0].name == "Alice"


def test_arabic_windows_csv_is_read(students):
    data = "student_id,name\n2021001,أحمد علي\n".encode("cp1256")
    assert _plan(data, students).rows[0].name == "أحمد علي"


def test_missing_required_column_rejects_the_file(students):
    with pytest.raises(BulkImportError, match="Missing required column"):
        _plan(_csv("student_id,email\n2021001,a@b.com"), students)


@pytest.mark.parametrize("filename, message", [("students.xls", "not supported"), ("students.pdf", "Upload a .xlsx")])
def test_unsupported_files_are_rejected(students, filename, message):
    with pytest.raises(BulkImportError, match=message):
        read_student_table(b"x", filename)


def test_bad_rows_are_reported_and_not_imported(students):
    plan = _plan(_csv("""
student_id,name,email
2021001,Alice,alice@uni.edu
,No ID,
2021003,,
2021004,Bad Mail,not-an-email
../evil,Path,
2.02108E+11,Scientific,
2021007,Twin One,
2021007,Twin Two,
"""), students)

    problems = {row.row_number: " ".join(row.errors) for row in plan.with_action("error")}
    assert plan.counts["create"] == 1
    assert "student_id is empty" in problems[3]
    assert "name is empty" in problems[4]
    assert "not a valid address" in problems[5]
    assert "may only contain" in problems[6]
    assert "scientific format" in problems[7]
    assert "also on row(s) 9" in problems[8] and "also on row(s) 8" in problems[9]

    report = student_errors_csv(plan).decode("utf-8-sig")
    assert report.splitlines()[0] == "row,student_id,name,email,department,notes,error"
    assert len(report.splitlines()) == 1 + 7


def test_import_saves_new_students_in_one_go(students):
    plan = _plan(_csv("student_id,name,department\n2021001,Alice,CS\n2021002,Omar,IT\n,Broken,"), students)
    outcome = apply_student_import(plan, students)

    assert outcome == {"created": 2, "updated": 0, "unchanged": 0, "errors": 1}
    saved = {s["student_id"]: s for s in students.get_all_students()}
    assert saved["2021001"]["name"] == "Alice" and saved["2021002"]["department"] == "IT"


def test_reimport_updates_changes_and_empty_cells_keep_existing_details(students):
    students.create_student("2021001", "Alice", email="alice@uni.edu", department="CS")
    students.create_student("2021002", "Omar", department="IT")

    plan = _plan(_csv("student_id,name,email,department\n2021001,Alice Ahmed,,\n2021002,Omar,,IT"), students)
    assert plan.counts == {"create": 0, "update": 1, "unchanged": 1, "error": 0}
    assert plan.with_action("update")[0].changes == ["name"]

    apply_student_import(plan, students)
    alice = students.get_student_by_student_id("2021001")
    assert alice["name"] == "Alice Ahmed"
    assert alice["email"] == "alice@uni.edu" and alice["department"] == "CS"   # blanks kept them
    assert _plan(_csv("student_id,name\n2021001,Alice Ahmed"), students).counts["unchanged"] == 1


def test_existing_students_can_be_left_untouched(students):
    students.create_student("2021001", "Alice")
    plan = _plan(_csv("student_id,name\n2021001,Changed\n2021002,Omar"), students)
    outcome = apply_student_import(plan, students, update_existing=False)
    assert outcome["created"] == 1 and outcome["updated"] == 0
    assert students.get_student_by_student_id("2021001")["name"] == "Alice"


def test_bulk_save_is_all_or_nothing(students):
    students.create_student("2021001", "Alice")
    with pytest.raises(sqlite3.IntegrityError):
        students.repository.bulk_save(
            [{"student_id": "2021002", "name": "Omar"}, {"student_id": "2021001", "name": "Duplicate"}], [])
    assert students.get_student_by_student_id("2021002") is None


def test_template_has_the_expected_columns(students):
    template = student_template_csv()
    assert template.startswith(b"\xef\xbb\xbf")
    plan = _plan(template, students)
    assert plan.counts["create"] == 2 and plan.counts["error"] == 0


def test_excel_files_are_read(students):
    pytest.importorskip("openpyxl")
    buffer = io.BytesIO()
    pd.DataFrame({"Student ID": ["0021001", 2021002], "Name": ["Alice", "Omar"]}).to_excel(buffer, index=False)
    plan = _plan(buffer.getvalue(), students, "students.xlsx")
    assert [row.student_id for row in plan.rows] == ["0021001", "2021002"]


# ============================================================================
# Face photos from a ZIP
# ============================================================================

# Photo colour -> the face it shows (unit vectors: different people are 1.41 apart)
PERSON_A, PERSON_B, PERSON_C, NO_FACE, TINY_FACE = 10, 20, 30, 250, 240


def _embedding_for(value: int) -> np.ndarray:
    vector = np.zeros(128)
    vector[value % 128] = 1.0
    return vector


def _fake_detector(seen_sizes=None):
    detector = MagicMock()

    def detect(image):
        if seen_sizes is not None:
            seen_sizes.append(max(image.shape[:2]))
        value = int(image[0, 0, 0])
        if value == NO_FACE:
            return []
        if value == TINY_FACE:
            return [DetectedFace((5, 5, 30, 30), 0.9)]
        return [DetectedFace((10, 10, 100, 100), 0.95)]

    detector.detect_faces.side_effect = detect
    return detector


def _fake_embedder():
    embedder = MagicMock()
    embedder.backend = "dlib"
    embedder.generate_embedding.side_effect = lambda crop: FaceEmbedding(_embedding_for(int(crop[0, 0, 0])))
    return embedder


def _photo(value: int, size: int = 200) -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((size, size, 3), value, dtype=np.uint8))
    return encoded.tobytes()


def _zip(files: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture
def faces(temp_db, students):
    enc_dir, img_dir = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    seen_sizes = []
    service = EnrollmentService(database_path=temp_db, encodings_dir=enc_dir, images_dir=img_dir,
                                face_detector=_fake_detector(seen_sizes), face_embedder=_fake_embedder())
    service.seen_sizes = seen_sizes
    for sid, name in [("2021001", "Alice"), ("2021002", "Omar"), ("2021003", "Sara")]:
        students.create_student(sid, name)
    yield service
    shutil.rmtree(enc_dir, ignore_errors=True)
    shutil.rmtree(img_dir, ignore_errors=True)


def _results(report):
    return {r.path: (r.status, r.message) for r in report.results}


def _db_id(students, sid):
    return students.get_student_by_student_id(sid)["id"]


def test_photos_are_enrolled_in_every_naming_layout(faces, students):
    report = import_face_zip(_zip({
        "photos/2021001.jpg": _photo(PERSON_A),
        "photos/2021001_2.jpg": _photo(PERSON_A),
        "2021002 (2).png": _photo(PERSON_B),
        "2021003/front.jpg": _photo(PERSON_C),
        "readme.txt": b"hello",
        "__MACOSX/._2021001.jpg": b"junk",
    }), faces, students)

    assert report.counts == {"enrolled": 4, "skipped": 0, "failed": 0, "warnings": 0}
    assert report.ignored_files == ["readme.txt"]
    counts = faces.get_all_samples_counts()
    assert counts[_db_id(students, "2021001")] == 2 and counts[_db_id(students, "2021003")] == 1
    assert report.students_without_faces == []
    assert len(faces.load_active_gallery()) == 4


def test_photo_of_an_unknown_student_fails(faces, students):
    report = import_face_zip(_zip({"9999999.jpg": _photo(PERSON_A)}), faces, students)
    status, message = _results(report)["9999999.jpg"]
    assert status == "failed" and "No student with ID '9999999'" in message


def test_students_with_face_data_are_skipped_unless_asked(faces, students):
    first = import_face_zip(_zip({"2021001.jpg": _photo(PERSON_A)}), faces, students)
    assert first.counts["enrolled"] == 1

    again = import_face_zip(_zip({"2021001.jpg": _photo(PERSON_A), "2021002.jpg": _photo(PERSON_B)}), faces, students)
    assert _results(again)["2021001.jpg"][0] == "skipped"
    assert _results(again)["2021002.jpg"][0] == "enrolled"

    extra = import_face_zip(_zip({"2021001_3.jpg": _photo(PERSON_A)}), faces, students,
                            skip_students_with_faces=False)
    assert extra.counts["enrolled"] == 1


def test_the_same_face_under_two_ids_is_not_enrolled_twice(faces, students):
    report = import_face_zip(_zip({"2021001.jpg": _photo(PERSON_A), "2021002.jpg": _photo(PERSON_A)}),
                             faces, students)
    status, message = _results(report)["2021002.jpg"]
    assert status == "failed" and "Same face as student 2021001" in message
    assert _db_id(students, "2021002") not in faces.get_all_samples_counts()


def test_a_similar_face_is_enrolled_with_a_warning(faces, students, monkeypatch):
    # B is 0.4 from A: below the 0.45 similar limit (0.75 x 0.6), above the 0.3 duplicate limit
    similar = _embedding_for(PERSON_A) * np.cos(0.4027) + _embedding_for(PERSON_B) * np.sin(0.4027)
    faces.face_embedder.generate_embedding.side_effect = lambda crop: FaceEmbedding(
        similar if int(crop[0, 0, 0]) == PERSON_B else _embedding_for(int(crop[0, 0, 0])))
    report = import_face_zip(_zip({"2021001.jpg": _photo(PERSON_A), "2021002.jpg": _photo(PERSON_B)}),
                             faces, students)
    status, message = _results(report)["2021002.jpg"]
    assert status == "enrolled" and "Looks similar to student 2021001" in message
    assert report.counts["warnings"] == 1


def test_a_photo_that_does_not_match_the_students_other_photos_fails(faces, students):
    report = import_face_zip(_zip({"2021001/a.jpg": _photo(PERSON_A), "2021001/b.jpg": _photo(PERSON_C)}),
                             faces, students)
    status, message = _results(report)["2021001/b.jpg"]
    assert status == "failed" and "Does not match this student's other face photos" in message


@pytest.mark.parametrize("data, reason", [
    (_photo(NO_FACE), "No face detected"),
    (_photo(TINY_FACE), "Face too small"),
    (b"not really a jpeg", "Not a readable image"),
])
def test_unusable_photos_are_reported_and_the_import_continues(faces, students, data, reason):
    report = import_face_zip(_zip({"2021001.jpg": data, "2021002.jpg": _photo(PERSON_B)}), faces, students)
    status, message = _results(report)["2021001.jpg"]
    assert status == "failed" and reason in message
    assert _results(report)["2021002.jpg"][0] == "enrolled"
    assert [s["student_id"] for s in report.students_without_faces] == ["2021001", "2021003"]


def test_large_photos_are_shrunk_before_detection(faces, students):
    import_face_zip(_zip({"2021001.jpg": _photo(PERSON_A, size=2000)}), faces, students)
    assert faces.seen_sizes == [MAX_PHOTO_SIDE]


def test_progress_is_reported(faces, students):
    calls = []
    import_face_zip(_zip({"2021001.jpg": _photo(PERSON_A), "2021002.jpg": _photo(PERSON_B)}), faces, students,
                    progress=lambda done, total, current: calls.append((done, total)))
    assert calls == [(0, 2), (1, 2), (2, 2)]


@pytest.mark.parametrize("data, message", [
    (b"not a zip", "not a valid ZIP"),
    (_zip({"notes.txt": b"x"}), "No photos"),
])
def test_unusable_zip_files_are_rejected(faces, students, data, message):
    with pytest.raises(BulkImportError, match=message):
        import_face_zip(data, faces, students)


def test_report_csv_lists_every_photo(faces, students):
    report = import_face_zip(_zip({"2021001.jpg": _photo(PERSON_A), "9.jpg": _photo(PERSON_B)}), faces, students)
    lines = report.report_csv().decode("utf-8-sig").splitlines()
    assert lines[0] == "file,student_id,status,message" and len(lines) == 3


def test_imported_faces_are_added_to_the_recognition_data(faces, students):
    """The attendance engine must not read thousands of new sample files at its next start."""
    gallery_calls = []
    import_face_zip(_zip({"2021001.jpg": _photo(PERSON_A), "2021002.jpg": _photo(PERSON_B)}), faces, students,
                    gallery_progress=lambda done, total: gallery_calls.append((done, total)))
    assert faces.gallery_cache_status() == {"samples": 2, "pending": 0}
    assert gallery_calls[-1] == (2, 2)                   # the two new samples, read once at the end


def test_a_file_object_is_imported_without_copying_it(faces, students):
    """The page passes the uploaded file itself, not getvalue() (a second copy of a large ZIP)."""
    upload = io.BytesIO(_zip({"2021001.jpg": _photo(PERSON_A)}))
    upload.seek(0, io.SEEK_END)                          # wherever the uploader left it
    report = import_face_zip(upload, faces, students)
    assert report.counts["enrolled"] == 1

