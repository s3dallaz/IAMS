"""
Bulk import for the Student Registry.

1. Students from an Excel (.xlsx) or CSV file: one row per student with the
   columns student_id and name (required) plus email, department and notes.
   The file is checked first (plan_student_import) and nothing is saved until
   the plan is applied, in one transaction. Re-importing the same file is
   safe: existing students are updated, never duplicated.

2. Face photos from a ZIP file, one photo per student named after the student
   ID (2021001.jpg), or a folder per student (2021001/1.jpg, 2021001/2.jpg).
   Each photo goes through the same face detection + embedding as a manual
   upload. A face that looks like a DIFFERENT student's (same photo under two
   IDs, a mislabelled file) is reported and not enrolled.
"""

import csv
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from src.computer_vision.face_recognizer import FaceRecognizer
from src.services.enrollment_service import EnrollmentService, InvalidFaceImageError
from src.services.student_service import StudentService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BulkImportError(Exception):
    """The file cannot be imported at all (unreadable, missing columns, too large)."""


# ============================================================================
# 1. Students from Excel / CSV
# ============================================================================

STUDENT_COLUMNS = ("student_id", "name", "email", "department", "notes")
REQUIRED_COLUMNS = ("student_id", "name")
OPTIONAL_COLUMNS = ("email", "department", "notes")
MAX_STUDENT_ROWS = 50000   # checking 50,000 rows takes under a second

# Header spellings accepted for each column (compared lower-case, with _ and - as spaces)
COLUMN_ALIASES = {
    "student id": "student_id", "studentid": "student_id", "id": "student_id",
    "student no": "student_id", "student number": "student_id", "code": "student_id",
    "name": "name", "full name": "name", "student name": "name",
    "email": "email", "e mail": "email", "mail": "email", "email address": "email",
    "department": "department", "dept": "department", "major": "department",
    "notes": "notes", "note": "notes", "comments": "notes",
}

# Student IDs become folder names (data/face_encodings/<id>/): letters, digits, . _ -
STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,49}$")
MAX_NAME_LENGTH = 100


@dataclass
class StudentRow:
    """One row of the uploaded file and what the import will do with it."""
    row_number: int                    # row in the spreadsheet (the header is row 1)
    student_id: str
    name: str
    email: Optional[str] = None
    department: Optional[str] = None
    notes: Optional[str] = None
    action: str = "create"             # create | update | unchanged | error
    errors: List[str] = field(default_factory=list)
    changes: List[str] = field(default_factory=list)   # fields an update changes

    def values(self) -> Dict[str, Any]:
        return {column: getattr(self, column) for column in STUDENT_COLUMNS}


@dataclass
class StudentImportPlan:
    """The checked file: what each row will do. Nothing is saved yet."""
    rows: List[StudentRow]
    ignored_columns: List[str] = field(default_factory=list)

    def with_action(self, action: str) -> List[StudentRow]:
        return [row for row in self.rows if row.action == action]

    @property
    def counts(self) -> Dict[str, int]:
        return {action: len(self.with_action(action)) for action in ("create", "update", "unchanged", "error")}


def normalize_column(header: Any) -> str:
    """'Student ID', 'student-id', 'STUDENT_ID' -> 'student_id'; unknown headers stay recognisable."""
    key = re.sub(r"\s+", " ", str(header).strip().lower().replace("_", " ").replace("-", " "))
    return COLUMN_ALIASES.get(key, key.replace(" ", "_"))


def read_student_table(data: bytes, filename: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read an uploaded .csv or .xlsx file into a DataFrame with the standard
    column names. Every cell is text (IDs keep their leading zeros).

    Returns:
        (table, ignored_columns)

    Raises:
        BulkImportError: unreadable file, missing required columns, too many rows
    """
    suffix = PurePosixPath(filename.lower()).suffix
    if suffix in (".csv", ".txt"):
        table = _read_csv(data)
    elif suffix in (".xlsx", ".xlsm"):
        try:
            table = pd.read_excel(io.BytesIO(data), dtype=str, keep_default_na=False, engine="openpyxl")
        except ImportError:
            raise BulkImportError("Reading Excel files needs the 'openpyxl' package (run setup.bat), "
                                  "or save the sheet as 'CSV UTF-8' and upload that.")
        except Exception as e:
            raise BulkImportError(f"Could not read the Excel file: {e}")
    elif suffix == ".xls":
        raise BulkImportError("Old .xls files are not supported. In Excel use Save As -> "
                              "'Excel Workbook (.xlsx)' or 'CSV UTF-8'.")
    else:
        raise BulkImportError("Upload a .xlsx or .csv file.")

    renamed = [normalize_column(column) for column in table.columns]
    duplicated = sorted({column for column in renamed if renamed.count(column) > 1 and column in STUDENT_COLUMNS})
    if duplicated:
        raise BulkImportError(f"These columns appear more than once: {', '.join(duplicated)}")
    original = dict(zip(renamed, table.columns))
    table.columns = renamed

    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise BulkImportError(
            f"Missing required column(s): {', '.join(missing)}. The first row must be the headers "
            f"student_id, name (and optionally email, department, notes). Download the template to see the format.")

    ignored = [str(original[column]) for column in table.columns if column not in STUDENT_COLUMNS]
    table = table[[column for column in STUDENT_COLUMNS if column in table.columns]]
    # Drop rows that are completely empty (blank lines at the end of a sheet)
    if len(table):
        blank = table.apply(lambda row: all(not str(value).strip() for value in row), axis=1)
        table = table[~blank]
    if len(table) > MAX_STUDENT_ROWS:
        raise BulkImportError(f"The file has {len(table)} rows; the limit is {MAX_STUDENT_ROWS} per import.")
    return table, ignored


def _read_csv(data: bytes) -> pd.DataFrame:
    # Excel's "CSV UTF-8" has a BOM (utf-8-sig); plain "CSV" on Arabic Windows is cp1256
    for encoding in ("utf-8-sig", "cp1256"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding=encoding,
                               sep=None, engine="python")   # sep=None: commas or semicolons
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise BulkImportError(f"Could not read the CSV file: {e}")
    raise BulkImportError("Could not read the CSV file's text. In Excel use Save As -> 'CSV UTF-8'.")


def _clean_cell(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    # Excel stores numbers as floats: 2021001.0 -> 2021001
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    return text


def plan_student_import(table: pd.DataFrame, student_service: StudentService,
                        ignored_columns: Optional[List[str]] = None) -> StudentImportPlan:
    """
    Check every row and decide what importing it would do. A blank optional
    cell keeps the value the student already has. Nothing is saved.
    """
    existing = {s["student_id"]: s for s in student_service.get_all_students(active_only=False)}
    rows: List[StudentRow] = []
    for index, record in enumerate(table.to_dict("records")):
        cells = {column: _clean_cell(record.get(column, "")) for column in STUDENT_COLUMNS}
        row = StudentRow(row_number=index + 2, student_id=cells["student_id"], name=cells["name"],
                         email=cells["email"] or None, department=cells["department"] or None,
                         notes=cells["notes"] or None)
        row.errors = _validate_row(row, student_service)
        rows.append(row)

    # The same ID twice in the file: neither row is imported until it is fixed
    positions: Dict[str, List[int]] = {}
    for row in rows:
        if row.student_id:
            positions.setdefault(row.student_id, []).append(row.row_number)
    for row in rows:
        others = [n for n in positions.get(row.student_id, []) if n != row.row_number]
        if others:
            row.errors.append(f"student_id {row.student_id} is also on row(s) {', '.join(map(str, others))}")

    for row in rows:
        if row.errors:
            row.action = "error"
            continue
        current = existing.get(row.student_id)
        if current is None:
            row.action = "create"
            continue
        # Blank optional cells keep the stored value
        for column in OPTIONAL_COLUMNS:
            if getattr(row, column) is None:
                setattr(row, column, current.get(column))
        row.changes = [column for column in ("name",) + OPTIONAL_COLUMNS
                       if (getattr(row, column) or None) != (current.get(column) or None)]
        row.action = "update" if row.changes else "unchanged"

    return StudentImportPlan(rows=rows, ignored_columns=list(ignored_columns or []))


def _validate_row(row: StudentRow, student_service: StudentService) -> List[str]:
    errors = []
    if not row.student_id:
        errors.append("student_id is empty")
    elif re.fullmatch(r"\d+(\.\d+)?[eE]\+\d+", row.student_id):
        errors.append(f"student_id '{row.student_id}' is in Excel's scientific format; "
                      f"format the column as Text and type the IDs again")
    elif not STUDENT_ID_PATTERN.match(row.student_id):
        errors.append(f"student_id '{row.student_id}' may only contain letters, digits, '.', '_' and '-' "
                      f"(max 50 characters)")
    if not row.name:
        errors.append("name is empty")
    elif len(row.name) > MAX_NAME_LENGTH:
        errors.append(f"name is longer than {MAX_NAME_LENGTH} characters")
    if row.email and not student_service.EMAIL_PATTERN.match(row.email):
        errors.append(f"email '{row.email}' is not a valid address")
    return errors


def apply_student_import(plan: StudentImportPlan, student_service: StudentService,
                         update_existing: bool = True) -> Dict[str, int]:
    """
    Save the planned new students (and updates) in ONE transaction: either all
    of them are saved or none. Rows with errors and unchanged rows are skipped.
    """
    creates = [row.values() for row in plan.with_action("create")]
    updates = [row.values() for row in plan.with_action("update")] if update_existing else []
    try:
        student_service.repository.bulk_save(creates, updates)
    except Exception as e:
        raise BulkImportError(f"Nothing was saved - the database refused the import: {e}")
    logger.info(f"Student import: {len(creates)} created, {len(updates)} updated, "
                f"{plan.counts['error']} rows with errors skipped")
    return {"created": len(creates), "updated": len(updates),
            "unchanged": plan.counts["unchanged"], "errors": plan.counts["error"]}


def student_template_csv() -> bytes:
    """An example file with the expected columns (UTF-8 with BOM so Excel shows Arabic names)."""
    rows = [
        {"student_id": "2021001", "name": "Alice Ahmed", "email": "alice@example.edu",
         "department": "Computer Science", "notes": ""},
        {"student_id": "2021002", "name": "Omar Hassan", "email": "", "department": "Computer Science",
         "notes": "optional columns can be empty"},
    ]
    return _csv_bytes(list(STUDENT_COLUMNS), rows)


def student_errors_csv(plan: StudentImportPlan) -> bytes:
    """The rows with errors plus an 'error' column, to fix and upload again."""
    rows = [{"row": row.row_number, **row.values(), "error": "; ".join(row.errors)}
            for row in plan.with_action("error")]
    return _csv_bytes(["row", *STUDENT_COLUMNS, "error"], rows)


def _csv_bytes(columns: List[str], rows: List[Dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: "" if row.get(column) is None else row.get(column) for column in columns})
    return buffer.getvalue().encode("utf-8-sig")


# ============================================================================
# 2. Face photos from a ZIP file
# ============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_ZIP_PHOTOS = 50000
MAX_PHOTO_BYTES = 25 * 1024 * 1024            # one photo
MAX_ZIP_TOTAL_BYTES = 10 * 1024 * 1024 * 1024  # all photos, uncompressed (guards against zip bombs)
MAX_PHOTO_SIDE = 1280      # larger photos are shrunk first: same face, much faster detection
MIN_FACE_PIXELS = 60       # smaller faces give unreliable embeddings
# A face closer than this fraction of the recognition threshold to ANOTHER
# student's face is treated as the same person (e.g. one photo under two IDs)
# and not enrolled; up to the full threshold it is enrolled with a warning.
DUPLICATE_BLOCK_FRACTION = 0.5


@dataclass
class ZipPhoto:
    """A photo found in the ZIP and the student it belongs to."""
    path: str                 # path inside the ZIP
    student_id: str           # resolved student ID (or the file name if no student matched)
    known: bool               # a student with this ID exists


@dataclass
class PhotoResult:
    path: str
    student_id: str
    status: str               # enrolled | skipped | failed
    message: str = ""


@dataclass
class FaceImportReport:
    results: List[PhotoResult] = field(default_factory=list)
    ignored_files: List[str] = field(default_factory=list)   # not images (e.g. .txt, .pdf)
    students_without_faces: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        counts = {"enrolled": 0, "skipped": 0, "failed": 0, "warnings": 0}
        for result in self.results:
            counts[result.status] += 1
            if result.status == "enrolled" and result.message:
                counts["warnings"] += 1
        return counts

    def report_csv(self) -> bytes:
        rows = [{"file": r.path, "student_id": r.student_id, "status": r.status, "message": r.message}
                for r in self.results]
        return _csv_bytes(["file", "student_id", "status", "message"], rows)

    def missing_faces_csv(self) -> bytes:
        return _csv_bytes(["student_id", "name", "department"], self.students_without_faces)


def list_zip_photos(zip_file: zipfile.ZipFile, known_ids: Set[str]) -> Tuple[List[ZipPhoto], List[str]]:
    """
    The photos in the ZIP with their student IDs, and the ignored non-image
    files. Accepted layouts (at any folder depth):
        2021001.jpg               one photo
        2021001_2.jpg / 2021001-2.jpg / "2021001 (2).jpg"   more photos
        2021001/front.jpg         a folder per student
    """
    entries = [info for info in zip_file.infolist() if not info.is_dir()]
    photos, ignored = [], []
    total = 0
    for info in entries:
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if any(part.startswith(".") or part == "__MACOSX" for part in path.parts) or path.name.lower() == "thumbs.db":
            continue   # macOS / Windows housekeeping files
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            ignored.append(info.filename)
            continue
        if info.flag_bits & 0x1:
            raise BulkImportError("The ZIP file is password-protected; create it without a password.")
        if info.file_size > MAX_PHOTO_BYTES:
            raise BulkImportError(f"{info.filename} is larger than {MAX_PHOTO_BYTES // (1024 * 1024)} MB.")
        total += info.file_size
        if len(photos) >= MAX_ZIP_PHOTOS:
            raise BulkImportError(f"The ZIP file has more than {MAX_ZIP_PHOTOS:,} photos; "
                                  f"split it into smaller ZIP files.")
        if total > MAX_ZIP_TOTAL_BYTES:
            raise BulkImportError(f"The photos in the ZIP file add up to more than "
                                  f"{MAX_ZIP_TOTAL_BYTES // (1024 ** 3)} GB unpacked; split it into smaller ZIP files.")
        student_id = _student_id_for(path, known_ids)
        photos.append(ZipPhoto(path=info.filename, student_id=student_id or path.stem, known=student_id is not None))
    photos.sort(key=lambda photo: photo.path)
    return photos, ignored


def _student_id_for(path: PurePosixPath, known_ids: Set[str]) -> Optional[str]:
    """The student a photo belongs to: its folder name or its file name (with an optional _2 / (2) suffix)."""
    if len(path.parts) > 1 and path.parent.name in known_ids:
        return path.parent.name
    stem = path.stem.strip()
    if stem in known_ids:
        return stem
    match = re.fullmatch(r"(.+?)(?:[ _-]+\d+|\s*\(\d+\))", stem)
    if match and match.group(1) in known_ids:
        return match.group(1)
    return None


def import_face_zip(
    zip_source: Union[bytes, BinaryIO],
    enrollment_service: EnrollmentService,
    student_service: StudentService,
    skip_students_with_faces: bool = True,
    progress: Optional[Callable[[int, int, str], None]] = None,
    gallery_progress: Optional[Callable[[int, int], None]] = None,
) -> FaceImportReport:
    """
    Enroll the face photos in a ZIP file. Each photo is checked on its own; a
    bad photo is reported and the import continues with the next one. At the
    end the new samples are added to the recognition gallery cache, so the
    attendance engine does not have to read them at its next start.

    Args:
        zip_source: the ZIP's bytes, or a seekable file object (e.g. the
            uploaded file itself, so a large ZIP is not copied in memory)
        skip_students_with_faces: leave students who already have face data
            alone (makes a re-run continue where an interrupted one stopped)
        progress: called as progress(done, total, current_file)
        gallery_progress: called as gallery_progress(done, total) while face
            samples are read into the gallery cache (before and after the photos)
    """
    if isinstance(zip_source, (bytes, bytearray)):
        zip_source = io.BytesIO(zip_source)
    try:
        zip_source.seek(0)
        zip_file = zipfile.ZipFile(zip_source)
    except zipfile.BadZipFile:
        raise BulkImportError("This is not a valid ZIP file.")

    students = {s["student_id"]: s for s in student_service.get_all_students(active_only=False)}
    photos, ignored = list_zip_photos(zip_file, set(students))
    report = FaceImportReport(ignored_files=ignored)
    if not photos:
        raise BulkImportError("No photos (.jpg, .jpeg, .png, .bmp, .webp) were found in the ZIP file.")

    sample_counts = enrollment_service.get_all_samples_counts()
    gallery = _Gallery(enrollment_service, gallery_progress)
    threshold = FaceRecognizer(backend=gallery.backend).threshold
    enrolled_now: Set[str] = set()

    for done, photo in enumerate(photos):
        if progress:
            progress(done, len(photos), photo.path)
        result = _import_one_photo(zip_file, photo, students, sample_counts, enrolled_now, gallery,
                                   threshold, enrollment_service, skip_students_with_faces)
        report.results.append(result)
        if result.status == "enrolled":
            enrolled_now.add(photo.student_id)
    if progress:
        progress(len(photos), len(photos), "")

    try:
        enrollment_service.load_active_gallery(progress=gallery_progress)  # reads only the new samples
    except Exception as e:
        logger.warning(f"Could not add the imported faces to the gallery cache: {e}")

    counts_after = enrollment_service.get_all_samples_counts()
    report.students_without_faces = [
        {"student_id": s["student_id"], "name": s["name"], "department": s.get("department") or ""}
        for s in students.values() if s["is_active"] and not counts_after.get(s["id"])
    ]
    c = report.counts
    logger.info(f"Face ZIP import: {c['enrolled']} enrolled ({c['warnings']} with warnings), "
                f"{c['skipped']} skipped, {c['failed']} failed")
    return report


class _Gallery:
    """
    Embeddings of every enrolled face, growing as photos are enrolled, for the
    duplicate checks. The arrays grow in place (doubling, like a list), so
    adding a face does not copy the whole gallery and a check stays one
    matrix-vector product even with tens of thousands of faces.
    """

    def __init__(self, enrollment_service: EnrollmentService,
                 progress: Optional[Callable[[int, int], None]] = None):
        self.backend = enrollment_service._gallery_backend()
        entries = enrollment_service.load_active_gallery(progress=progress)
        self.size = 0
        self.matrix = np.zeros((0, 0))
        self.sq_norms = np.zeros(0)
        self.labels = np.zeros(0, dtype=np.int64)    # row -> index into self.student_ids
        self.student_ids: List[str] = []
        self._label_of: Dict[str, int] = {}
        for student_id, embedding in entries:
            self.add(student_id, embedding.encoding)

    def add(self, student_id: str, encoding: np.ndarray) -> None:
        encoding = np.asarray(encoding, dtype=np.float64).ravel()
        if self.size == len(self.matrix):
            capacity = max(1024, 2 * len(self.matrix))
            matrix = np.zeros((capacity, len(encoding)))
            if self.size:
                matrix[:self.size] = self.matrix[:self.size]
            self.matrix = matrix
            self.sq_norms = np.resize(self.sq_norms, capacity)
            self.labels = np.resize(self.labels, capacity)
        if student_id not in self._label_of:
            self._label_of[student_id] = len(self.student_ids)
            self.student_ids.append(student_id)
        self.matrix[self.size] = encoding
        self.sq_norms[self.size] = encoding @ encoding
        self.labels[self.size] = self._label_of[student_id]
        self.size += 1

    def compare(self, encoding: np.ndarray, student_id: str) -> Tuple[Optional[float], Optional[str], float]:
        """
        Returns:
            (closest distance to this student's own faces or None,
             the closest OTHER student or None, the distance to that student)
        """
        if not self.size:
            return None, None, float("inf")
        encoding = np.asarray(encoding, dtype=np.float64).ravel()
        # |a - b|^2 = |a|^2 - 2 a.b + |b|^2: no (faces x dims) temporary array
        sq = self.sq_norms[:self.size] - 2.0 * (self.matrix[:self.size] @ encoding) + encoding @ encoding
        distances = np.sqrt(np.maximum(sq, 0.0))
        label = self._label_of.get(student_id, -1)
        own_mask = self.labels[:self.size] == label
        own = float(distances[own_mask].min()) if own_mask.any() else None
        others = np.where(own_mask, np.inf, distances)
        closest = int(np.argmin(others))
        if not np.isfinite(others[closest]):
            return own, None, float("inf")
        return own, self.student_ids[self.labels[closest]], float(others[closest])


def _import_one_photo(zip_file, photo: ZipPhoto, students, sample_counts, enrolled_now, gallery: _Gallery,
                      threshold: float, enrollment_service: EnrollmentService,
                      skip_students_with_faces: bool) -> PhotoResult:
    def failed(message):
        return PhotoResult(photo.path, photo.student_id, "failed", message)

    if not photo.known:
        return failed(f"No student with ID '{photo.student_id}' - import the student list first, "
                      f"or name the photo <student_id>.jpg")
    student = students[photo.student_id]
    if (skip_students_with_faces and sample_counts.get(student["id"])
            and photo.student_id not in enrolled_now):
        return PhotoResult(photo.path, photo.student_id, "skipped", "Student already has face data")

    image = cv2.imdecode(np.frombuffer(zip_file.read(photo.path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return failed("Not a readable image file")
    longest = max(image.shape[:2])
    if longest > MAX_PHOTO_SIDE:
        scale = MAX_PHOTO_SIDE / longest
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    try:
        face, face_crop, embedding = enrollment_service.prepare_face_sample(image)
    except InvalidFaceImageError as e:
        return failed(str(e))
    _, _, w, h = face.bbox
    if min(w, h) < MIN_FACE_PIXELS:
        return failed(f"Face too small ({w}x{h} px, at least {MIN_FACE_PIXELS} px needed) - use a closer photo")

    # Duplicate checks against every enrolled face (including this import's)
    own, other_id, other_distance = gallery.compare(embedding.encoding, photo.student_id)
    warning = ""
    if other_id is not None:
        if other_distance < DUPLICATE_BLOCK_FRACTION * threshold:
            return failed(f"Same face as student {other_id} (distance {other_distance:.2f}) - "
                          f"check the photo is named correctly; not enrolled")
        if other_distance < threshold:
            warning = (f"Looks similar to student {other_id} (distance {other_distance:.2f}); "
                       f"enrolled, but check it is the right person")
    if own is not None and own >= threshold:
        return failed(f"Does not match this student's other face photos (distance {own:.2f}) - "
                      f"check the photo is named correctly; not enrolled")

    enrollment_service.save_face_sample(student, face_crop, embedding, float(face.confidence),
                                        enrollment_method="bulk_upload", view_angle="frontal")
    gallery.add(photo.student_id, embedding.encoding)
    return PhotoResult(photo.path, photo.student_id, "enrolled", warning)
