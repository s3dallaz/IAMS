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

from src.services.enrollment_service import (
    EnrollmentService,
    EnrollmentServiceError,
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

    enrollment_service.add_face_sample(student_db_id=other, face_image=_make_face_image())
    two_students = enrollment_service.gallery_signature()
    assert two_students != one_sample

    student_service.deactivate_student(other)
    assert enrollment_service.gallery_signature() == one_sample      # deactivated student leaves the gallery

    enrollment_service.clear_student_face_dataset(sample_student)
    assert enrollment_service.gallery_signature() == empty
