"""
Enrollment Service Tests

Tests face dataset management: adding face samples, validation, multiple
samples per student, gallery loading, and dataset clearing.
Uses mocked CV components to avoid physical camera/model dependencies.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

import src.services.enrollment_service as enrollment_module
from src.services.enrollment_service import (
    DuplicateFaceError,
    EnrollmentService,
    EnrollmentServiceError,
    FaceNeedsConfirmationError,
    InvalidFaceImageError
)
from src.services.student_service import StudentService
from src.computer_vision.face_detector import DetectedFace
from src.computer_vision.face_embedder import FaceEmbedding, FaceEmbeddingError
from src.database.schema import initialize_database


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    initialize_database(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def temp_dirs():
    enc_dir = Path(tempfile.mkdtemp())
    img_dir = Path(tempfile.mkdtemp())
    yield enc_dir, img_dir
    shutil.rmtree(enc_dir, ignore_errors=True)
    shutil.rmtree(img_dir, ignore_errors=True)


@pytest.fixture
def mock_detector():
    detector = MagicMock()
    detected_face = DetectedFace(bbox=(10, 10, 100, 100), confidence=0.95)
    detector.detect_faces.return_value = [detected_face]
    return detector


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedding = FaceEmbedding(np.random.randn(128).astype(np.float64))
    embedder.generate_embedding.return_value = embedding
    return embedder


@pytest.fixture
def student_service(temp_db):
    return StudentService(temp_db)


@pytest.fixture
def sample_student(student_service):
    return student_service.create_student(
        student_id="2021001",
        name="John Doe",
        department="Computer Science"
    )


@pytest.fixture
def enrollment_service(temp_db, temp_dirs, mock_detector, mock_embedder):
    enc_dir, img_dir = temp_dirs
    service = EnrollmentService(
        database_path=temp_db,
        encodings_dir=enc_dir,
        images_dir=img_dir,
        face_detector=mock_detector,
        face_embedder=mock_embedder
    )
    return service


def _make_face_image():
    return np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)


def _other_face():
    """A different person: a random 128-D vector is far from any other."""
    return FaceEmbedding(np.random.randn(128))


def _near(embedding, distance):
    """A face `distance` away from `embedding`."""
    direction = np.random.randn(128)
    return FaceEmbedding(embedding.encoding + distance * direction / np.linalg.norm(direction))


def test_existing_student_can_open_manage_face(enrollment_service, sample_student):
    count = enrollment_service.get_student_samples_count(sample_student)
    assert count == 0


def test_add_valid_face_sample(enrollment_service, sample_student):
    img = _make_face_image()
    enrollment_id = enrollment_service.add_face_sample(
        student_db_id=sample_student,
        face_image=img,
        enrollment_method="upload"
    )
    assert enrollment_id is not None
    assert isinstance(enrollment_id, int)
    assert enrollment_id > 0
    assert enrollment_service.get_student_samples_count(sample_student) == 1


def test_add_multiple_face_samples(enrollment_service, sample_student):
    for i in range(4):
        enrollment_service.add_face_sample(
            student_db_id=sample_student,
            face_image=_make_face_image(),
            enrollment_method="upload"
        )
    assert enrollment_service.get_student_samples_count(sample_student) == 4


def test_invalid_image_rejected(enrollment_service, sample_student):
    with pytest.raises(InvalidFaceImageError, match="empty"):
        enrollment_service.add_face_sample(
            student_db_id=sample_student,
            face_image=np.array([]),
            enrollment_method="upload"
        )


def test_image_without_detectable_face_rejected(enrollment_service, sample_student):
    enrollment_service._get_detector().detect_faces.return_value = []
    with pytest.raises(InvalidFaceImageError, match="No face detected"):
        enrollment_service.add_face_sample(
            student_db_id=sample_student,
            face_image=_make_face_image(),
            enrollment_method="upload"
        )


def test_student_identity_unchanged_after_enrollment(
    enrollment_service, sample_student, student_service
):
    enrollment_service.add_face_sample(
        student_db_id=sample_student,
        face_image=_make_face_image(),
        enrollment_method="upload"
    )
    student = student_service.get_student_by_id(sample_student)
    assert student['student_id'] == "2021001"
    assert student['name'] == "John Doe"


def test_existing_samples_preserved_on_new_enrollment(enrollment_service, sample_student):
    enrollment_service.add_face_sample(
        student_db_id=sample_student,
        face_image=_make_face_image(),
        enrollment_method="upload"
    )
    assert enrollment_service.get_student_samples_count(sample_student) == 1

    enrollment_service.add_face_sample(
        student_db_id=sample_student,
        face_image=_make_face_image(),
        enrollment_method="camera"
    )
    assert enrollment_service.get_student_samples_count(sample_student) == 2


def test_embedding_saved_correctly(enrollment_service, sample_student, temp_dirs):
    enc_dir, _ = temp_dirs
    enrollment_service.add_face_sample(
        student_db_id=sample_student,
        face_image=_make_face_image(),
        enrollment_method="upload"
    )
    student_enc_dir = enc_dir / "2021001"
    npy_files = list(student_enc_dir.glob("*.npy"))
    assert len(npy_files) == 1
    loaded = np.load(str(npy_files[0]))
    assert loaded.shape == (128,)


def test_gallery_loads_enrolled_data(enrollment_service, sample_student):
    enrollment_service.add_face_sample(
        student_db_id=sample_student,
        face_image=_make_face_image(),
        enrollment_method="upload"
    )
    gallery = enrollment_service.load_active_gallery()
    assert len(gallery) == 1
    identity, embedding = gallery[0]
    assert identity == "2021001"
    assert embedding.encoding.shape == (128,)


def test_unknown_not_in_gallery(enrollment_service, sample_student):
    enrollment_service.add_face_sample(
        student_db_id=sample_student,
        face_image=_make_face_image(),
        enrollment_method="upload"
    )
    gallery = enrollment_service.load_active_gallery()
    identities = [g[0] for g in gallery]
    assert "9999999" not in identities


def test_clear_dataset_removes_samples(enrollment_service, sample_student):
    for _ in range(3):
        enrollment_service.add_face_sample(
            student_db_id=sample_student,
            face_image=_make_face_image(),
            enrollment_method="upload"
        )
    assert enrollment_service.get_student_samples_count(sample_student) == 3

    deleted = enrollment_service.clear_student_face_dataset(sample_student)
    assert deleted == 3
    assert enrollment_service.get_student_samples_count(sample_student) == 0


def test_multiple_faces_in_image_rejected(enrollment_service, sample_student):
    enrollment_service._get_detector().detect_faces.return_value = [
        DetectedFace(bbox=(10, 10, 50, 50), confidence=0.9),
        DetectedFace(bbox=(100, 10, 50, 50), confidence=0.85)
    ]
    with pytest.raises(InvalidFaceImageError, match="Multiple faces"):
        enrollment_service.add_face_sample(
            student_db_id=sample_student,
            face_image=_make_face_image(),
            enrollment_method="upload"
        )


def test_embedding_generation_failure_rejected(enrollment_service, sample_student):
    enrollment_service._get_embedder().generate_embedding.return_value = None
    with pytest.raises(InvalidFaceImageError, match="Failed to extract"):
        enrollment_service.add_face_sample(
            student_db_id=sample_student,
            face_image=_make_face_image(),
            enrollment_method="upload"
        )


def test_nonexistent_student_rejected(enrollment_service):
    with pytest.raises(EnrollmentServiceError, match="not found"):
        enrollment_service.add_face_sample(
            student_db_id=9999,
            face_image=_make_face_image(),
            enrollment_method="upload"
        )


def test_add_multiple_face_samples_method(enrollment_service, sample_student):
    images = [_make_face_image() for _ in range(3)]
    ids = enrollment_service.add_multiple_face_samples(
        student_db_id=sample_student,
        face_images=images,
        enrollment_method="upload"
    )
    assert len(ids) == 3
    assert enrollment_service.get_student_samples_count(sample_student) == 3


def test_gallery_multiple_samples_per_student(enrollment_service, sample_student):
    for _ in range(3):
        enrollment_service.add_face_sample(
            student_db_id=sample_student,
            face_image=_make_face_image(),
            enrollment_method="upload"
        )
    gallery = enrollment_service.load_active_gallery()
    assert len(gallery) == 3
    assert all(g[0] == "2021001" for g in gallery)


def test_validate_and_extract_face_none_image(enrollment_service):
    with pytest.raises(InvalidFaceImageError, match="empty"):
        enrollment_service.validate_and_extract_face(None)


def test_validate_and_extract_face_success(enrollment_service):
    face_crop, confidence = enrollment_service.validate_and_extract_face(_make_face_image())
    assert face_crop is not None
    assert confidence == pytest.approx(0.95)


def test_gallery_signature_tracks_gallery_changes(enrollment_service, student_service, sample_student):
    empty = enrollment_service.gallery_signature()
    assert enrollment_service.gallery_signature() == empty          # stable while nothing changes

    enrollment_service.add_face_sample(student_db_id=sample_student, face_image=_make_face_image())
    one_sample = enrollment_service.gallery_signature()
    assert one_sample != empty                                        # new sample

    other = student_service.create_student(student_id="2021002", name="Jane Roe")
    assert enrollment_service.gallery_signature() == one_sample      # a student without samples adds nothing

    enrollment_service._get_embedder().generate_embedding.return_value = _other_face()   # a different person
    enrollment_service.add_face_sample(student_db_id=other, face_image=_make_face_image())
    two_students = enrollment_service.gallery_signature()
    assert two_students != one_sample

    student_service.deactivate_student(other)
    assert enrollment_service.gallery_signature() == one_sample      # deactivated student leaves the gallery

    enrollment_service.clear_student_face_dataset(sample_student)
    assert enrollment_service.gallery_signature() == empty


def _gallery_cache(enrollment_service):
    return enrollment_service._gallery_cache_path(enrollment_service._gallery_backend())


def _count_sample_reads(enrollment_service, monkeypatch):
    """List that receives the key of every sample read from its own encoding file."""
    reads = []
    original = enrollment_service._read_sample_encoding

    def counting(key, backend):
        reads.append(key)
        return original(key, backend)

    monkeypatch.setattr(enrollment_service, "_read_sample_encoding", counting)
    return reads


def _add_samples(enrollment_service, student_db_id, count):
    for _ in range(count):
        enrollment_service.save_face_sample(
            enrollment_service.student_repo.get_by_id(student_db_id), np.zeros((20, 20, 3), dtype=np.uint8),
            FaceEmbedding(np.random.randn(128)), 0.9)


def test_gallery_is_read_from_the_cache_file_while_unchanged(enrollment_service, sample_student, temp_dirs):
    """Reading one file per sample took ~44 s for 11,200 samples after a restart."""
    _add_samples(enrollment_service, sample_student, 2)
    first = enrollment_service.load_active_gallery()
    assert _gallery_cache(enrollment_service).exists()

    enc_dir, _ = temp_dirs
    for sample_file in (enc_dir / "2021001").glob("*.npy"):
        sample_file.unlink()                            # the cache alone must be enough now
    cached = enrollment_service.load_active_gallery()

    assert [identity for identity, _ in cached] == [identity for identity, _ in first] == ["2021001"] * 2
    for (_, a), (_, b) in zip(cached, first):
        np.testing.assert_array_equal(a.encoding, b.encoding)


def test_gallery_load_reads_only_samples_missing_from_the_cache(enrollment_service, student_service,
                                                                sample_student, monkeypatch):
    _add_samples(enrollment_service, sample_student, 3)
    enrollment_service.load_active_gallery()
    reads = _count_sample_reads(enrollment_service, monkeypatch)

    other = student_service.create_student(student_id="2021002", name="Jane Roe")
    _add_samples(enrollment_service, other, 1)
    gallery = enrollment_service.load_active_gallery()
    assert len(reads) == 1                              # only the new sample, not all four
    assert sorted(i for i, _ in gallery) == ["2021001"] * 3 + ["2021002"]

    reads.clear()
    student_service.deactivate_student(other)
    assert [i for i, _ in enrollment_service.load_active_gallery()] == ["2021001"] * 3
    assert reads == []                                  # leaving needs no file reads
    student_service.activate_student(other)
    assert len(enrollment_service.load_active_gallery()) == 4
    assert len(reads) == 1                              # returning reads only that student's sample

    enrollment_service.clear_student_face_dataset(sample_student)
    assert [i for i, _ in enrollment_service.load_active_gallery()] == ["2021002"]


def test_rebuild_reads_every_sample_again(enrollment_service, sample_student, monkeypatch):
    _add_samples(enrollment_service, sample_student, 3)
    enrollment_service.load_active_gallery()
    reads = _count_sample_reads(enrollment_service, monkeypatch)
    assert enrollment_service.prepare_recognition_data(rebuild=True) == 3
    assert len(reads) == 3


def test_gallery_load_reports_progress_only_for_files_it_reads(enrollment_service, sample_student):
    _add_samples(enrollment_service, sample_student, 3)
    calls = []
    enrollment_service.load_active_gallery(progress=lambda done, total: calls.append((done, total)))
    assert calls[0] == (0, 3) and calls[-1] == (3, 3)

    calls.clear()
    enrollment_service.load_active_gallery(progress=lambda done, total: calls.append((done, total)))
    assert calls == []                                  # all from the cache


def test_gallery_cache_status(enrollment_service, sample_student):
    assert enrollment_service.gallery_cache_status() == {"samples": 0, "pending": 0}
    _add_samples(enrollment_service, sample_student, 2)
    assert enrollment_service.gallery_cache_status() == {"samples": 2, "pending": 2}
    enrollment_service.prepare_recognition_data()
    assert enrollment_service.gallery_cache_status() == {"samples": 2, "pending": 0}


def test_adding_a_face_updates_an_existing_cache_only(enrollment_service, sample_student, monkeypatch):
    enrollment_service.add_face_sample(student_db_id=sample_student, face_image=_make_face_image())
    assert not _gallery_cache(enrollment_service).exists()   # no cache: building one is the button's job

    enrollment_service.prepare_recognition_data()
    enrollment_service.add_face_sample(student_db_id=sample_student, face_image=_make_face_image())
    assert enrollment_service.gallery_cache_status() == {"samples": 2, "pending": 0}


def test_missing_sample_file_is_not_read_again(enrollment_service, sample_student, temp_dirs, monkeypatch):
    _add_samples(enrollment_service, sample_student, 2)
    enc_dir, _ = temp_dirs
    next((enc_dir / "2021001").glob("*.npy")).unlink()
    assert len(enrollment_service.load_active_gallery()) == 1
    reads = _count_sample_reads(enrollment_service, monkeypatch)
    assert len(enrollment_service.load_active_gallery()) == 1
    assert reads == []                                  # remembered as missing; a rebuild checks again


def test_unreadable_or_old_gallery_cache_is_rebuilt(enrollment_service, sample_student):
    _add_samples(enrollment_service, sample_student, 1)
    cache = _gallery_cache(enrollment_service)
    for content in (b"not a numpy file", None):
        if content is None:                             # the first cache format (no per-sample keys)
            with open(cache, "wb") as f:
                np.savez(f, signature=np.array("x"), identities=np.array(["2021001"]),
                         encodings=np.zeros((1, 128)))
        else:
            cache.write_bytes(content)
        assert [i for i, _ in enrollment_service.load_active_gallery()] == ["2021001"]
        assert enrollment_service.gallery_cache_status()["pending"] == 0   # rewritten


# ============================================================================
# Duplicate-face checks
# ============================================================================

@pytest.fixture
def two_students(enrollment_service, student_service, sample_student):
    """2021001 enrolled with the fixture's face; 2021002 without face data."""
    enrollment_service.add_face_sample(student_db_id=sample_student, face_image=_make_face_image())
    other = student_service.create_student(student_id="2021002", name="Jane Roe")
    return sample_student, other


def _set_face(service, embedding):
    service._get_embedder().generate_embedding.return_value = embedding


def test_the_same_face_for_another_student_is_refused(enrollment_service, two_students):
    first, other = two_students
    with pytest.raises(DuplicateFaceError, match=r"already enrolled for John Doe \(2021001\)") as raised:
        enrollment_service.add_face_sample(student_db_id=other, face_image=_make_face_image())
    assert raised.value.check.other_student_id == "2021001"
    # Not even with confirmation: it would make attendance reject both students
    with pytest.raises(DuplicateFaceError):
        enrollment_service.add_face_sample(student_db_id=other, face_image=_make_face_image(), confirmed=True)
    assert enrollment_service.get_student_samples_count(other) == 0


def test_a_similar_face_needs_confirmation(enrollment_service, two_students):
    first, other = two_students
    enrolled = enrollment_service._get_embedder().generate_embedding.return_value
    _set_face(enrollment_service, _near(enrolled, 0.4))      # under 0.45 (0.75 x 0.6), above the 0.3 duplicate limit
    with pytest.raises(FaceNeedsConfirmationError, match="looks like John Doe") as raised:
        enrollment_service.add_face_sample(student_db_id=other, face_image=_make_face_image())
    assert raised.value.check.status == "similar"
    assert enrollment_service.get_student_samples_count(other) == 0

    enrollment_service.add_face_sample(student_db_id=other, face_image=_make_face_image(), confirmed=True)
    assert enrollment_service.get_student_samples_count(other) == 1


def test_a_face_unlike_the_students_own_samples_needs_confirmation(enrollment_service, two_students):
    first, _ = two_students
    _set_face(enrollment_service, _other_face())
    with pytest.raises(FaceNeedsConfirmationError, match="does not match this student's other face samples"):
        enrollment_service.add_face_sample(student_db_id=first, face_image=_make_face_image())
    enrollment_service.add_face_sample(student_db_id=first, face_image=_make_face_image(), confirmed=True)
    assert enrollment_service.get_student_samples_count(first) == 2


def test_more_samples_of_the_same_student_need_no_confirmation(enrollment_service, two_students):
    first, _ = two_students
    enrolled = enrollment_service._get_embedder().generate_embedding.return_value
    _set_face(enrollment_service, _near(enrolled, 0.2))
    enrollment_service.add_face_sample(student_db_id=first, face_image=_make_face_image())
    assert enrollment_service.get_student_samples_count(first) == 2


def test_the_face_index_is_not_rebuilt_for_this_process_s_own_enrollments(
        enrollment_service, student_service, two_students, monkeypatch):
    """Checking a face must not reload the gallery each time (performance)."""
    first, other = two_students
    loads = []
    real_load = enrollment_service.load_active_gallery
    monkeypatch.setattr(enrollment_service, "load_active_gallery",
                        lambda *a, **k: loads.append(1) or real_load(*a, **k))
    builds = []
    real_index = enrollment_module.FaceIndex
    monkeypatch.setattr(enrollment_module, "FaceIndex", lambda *a: builds.append(1) or real_index(*a))

    for sid in ("2021003", "2021004", "2021005"):
        new = student_service.create_student(student_id=sid, name=f"Student {sid}")
        _set_face(enrollment_service, _other_face())
        enrollment_service.add_face_sample(student_db_id=new, face_image=_make_face_image())
    assert builds == []                     # built once (by the fixture), then kept in step

    # A change this process did not record (here: a reset) rebuilds it once
    enrollment_service.clear_student_face_dataset(first)
    _set_face(enrollment_service, _other_face())
    enrollment_service.add_face_sample(student_db_id=other, face_image=_make_face_image())
    assert builds == [1]


def test_find_duplicate_faces_lists_students_sharing_a_face(enrollment_service, two_students):
    first, other = two_students
    enrolled = enrollment_service._get_embedder().generate_embedding.return_value
    # Enrolled before the checks existed (or by another tool): straight to save_face_sample
    student = enrollment_service.student_repo.get_by_id(other)
    enrollment_service.save_face_sample(student, np.zeros((10, 10, 3), np.uint8), enrolled, 0.9)

    pairs = enrollment_service.find_duplicate_faces()
    assert [(p.student_a, p.student_b, p.same_face) for p in pairs] == [("2021001", "2021002", True)]
