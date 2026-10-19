"""
Tests for the OpenCV YuNet detector / SFace recogniser backends and the
automatic creation of SFace encodings from saved enrollment photos.

Model-based tests run only when the model files are in the project root.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import config
from src.computer_vision.face_detector import DetectedFace, FaceDetector
from src.computer_vision.face_embedder import FaceEmbedder
from src.computer_vision.face_recognizer import FaceRecognizer
from src.database.schema import initialize_database
from src.services.enrollment_service import EnrollmentService, encoding_file_for_backend
from src.services.student_service import StudentService

needs_yunet = pytest.mark.skipif(not Path(config.YUNET_MODEL_PATH).exists(), reason="YuNet model not downloaded")
needs_sface = pytest.mark.skipif(not Path(config.SFACE_MODEL_PATH).exists(), reason="SFace model not downloaded")

FRAME = (np.random.default_rng(1).random((240, 320, 3)) * 255).astype(np.uint8)
LANDMARKS = np.array([[130, 100], [190, 100], [160, 130], [138, 160], [182, 160]], dtype=np.float32)


# -- detector ------------------------------------------------------------------------

@needs_yunet
def test_yunet_rows_become_clipped_faces_with_landmarks():
    detector = FaceDetector(backend="yunet")
    assert detector.backend == "yunet"
    row = np.array([[-5, 10, 100, 300, *LANDMARKS.reshape(-1), 0.93]], dtype=np.float32)
    detector._yunet = MagicMock()
    detector._yunet.detect.return_value = (1, row)

    faces = detector.detect_faces(FRAME)

    assert len(faces) == 1
    assert faces[0].bbox == (0, 10, 100, 230)          # clipped to the 320x240 frame
    assert faces[0].confidence == pytest.approx(0.93)
    np.testing.assert_allclose(faces[0].landmarks, LANDMARKS)


@needs_yunet
def test_yunet_finds_nothing_in_noise():
    assert FaceDetector(backend="yunet").detect_faces(FRAME) == []


def test_detector_falls_back_to_mediapipe_without_the_yunet_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "YUNET_MODEL_PATH", tmp_path / "missing.onnx")
    assert FaceDetector(backend="yunet").backend == "mediapipe"


def test_yunet_row_layout_for_alignment():
    face = DetectedFace((10, 20, 30, 40), 0.9, LANDMARKS)
    row = face.yunet_row()
    assert row.shape == (15,) and row.dtype == np.float32
    assert list(row[:4]) == [10, 20, 30, 40] and row[14] == pytest.approx(0.9)


# -- embedder --------------------------------------------------------------------------

@needs_sface
def test_sface_embedding_from_a_crop_is_unit_length():
    embedding = FaceEmbedder(backend="sface").generate_embedding(FRAME[60:200, 100:240])
    assert embedding.encoding.shape == (128,)
    assert np.linalg.norm(embedding.encoding) == pytest.approx(1.0)


@needs_sface
def test_sface_aligns_on_landmarks_when_embedding_from_the_frame():
    embedder = FaceEmbedder(backend="sface")
    face = DetectedFace((100, 60, 120, 140), 0.95, LANDMARKS)
    first = embedder.generate_embedding_from_frame(FRAME, face)
    again = embedder.generate_embedding_from_frame(FRAME, face)
    assert np.linalg.norm(first.encoding) == pytest.approx(1.0)
    np.testing.assert_allclose(first.encoding, again.encoding)   # deterministic


def test_embedder_falls_back_to_dlib_without_the_sface_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SFACE_MODEL_PATH", tmp_path / "missing.onnx")
    assert FaceEmbedder(backend="sface").backend == "dlib"


# -- recognizer ----------------------------------------------------------------------------

def test_recognizer_uses_the_thresholds_of_its_model():
    sface = FaceRecognizer(backend="sface")
    dlib = FaceRecognizer(backend="dlib")
    assert sface.threshold == config.SFACE_RECOGNITION_THRESHOLD
    assert sface.margin == config.SFACE_RECOGNITION_MARGIN
    assert dlib.threshold == config.FACE_RECOGNITION_THRESHOLD
    assert FaceRecognizer(threshold=0.5, backend="sface").threshold == 0.5


# -- encoding files and automatic SFace migration ---------------------------------------------

@pytest.mark.parametrize("name, backend, expected", [
    ("enc_123.npy", "sface", "enc_sface_123.npy"),
    ("enc_sface_123.npy", "sface", "enc_sface_123.npy"),
    ("enc_sface_123.npy", "dlib", "enc_123.npy"),
    ("enc_123.npy", "dlib", "enc_123.npy"),
])
def test_encoding_file_for_backend(name, backend, expected):
    assert encoding_file_for_backend(Path("/data") / name, backend).name == expected


@pytest.fixture
def enrolled_with_dlib():
    """A student enrolled with dlib: enc_<id>.npy + the face_<id>.jpg photo, in temp folders."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Path(f.name)
    initialize_database(db)
    enc_dir, img_dir = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    StudentService(db).create_student(student_id="2021001", name="Alice", department="CS")
    service = EnrollmentService(database_path=db, encodings_dir=enc_dir, images_dir=img_dir)
    student = service.student_repo.get_all(active_only=True)[0]
    (enc_dir / "2021001").mkdir()
    (img_dir / "2021001").mkdir()
    dlib_file = enc_dir / "2021001" / "enc_777.npy"
    np.save(dlib_file, np.random.rand(128))
    cv2.imwrite(str(img_dir / "2021001" / "face_777.jpg"), FRAME[60:200, 100:240])
    service.enrollment_repo.create_enrollment(student_id=student["id"], encoding_path=str(dlib_file),
                                              enrollment_method="camera", image_count=1,
                                              quality_score=0.9, view_angle="frontal", is_active=True)
    yield service, dlib_file, img_dir
    shutil.rmtree(enc_dir, ignore_errors=True)
    shutil.rmtree(img_dir, ignore_errors=True)
    db.unlink(missing_ok=True)


@needs_sface
def test_gallery_creates_missing_sface_encodings_from_the_photos(enrolled_with_dlib, monkeypatch):
    monkeypatch.setattr(config, "FACE_RECOGNITION_BACKEND", "sface")
    service, dlib_file, _ = enrolled_with_dlib
    dlib_before = np.load(dlib_file)

    gallery = service.load_active_gallery()

    sface_file = dlib_file.with_name("enc_sface_777.npy")
    assert sface_file.exists()
    np.testing.assert_array_equal(np.load(dlib_file), dlib_before)   # dlib data untouched
    assert len(gallery) == 1 and gallery[0][0] == "2021001"
    assert np.linalg.norm(gallery[0][1].encoding) == pytest.approx(1.0)


@needs_sface
def test_sample_without_a_photo_is_skipped(enrolled_with_dlib, monkeypatch):
    monkeypatch.setattr(config, "FACE_RECOGNITION_BACKEND", "sface")
    service, dlib_file, img_dir = enrolled_with_dlib
    (img_dir / "2021001" / "face_777.jpg").unlink()
    assert service.load_active_gallery() == []
    assert not dlib_file.with_name("enc_sface_777.npy").exists()


def test_dlib_gallery_still_loads_the_dlib_encodings(enrolled_with_dlib):
    service, *_ = enrolled_with_dlib                     # conftest: dlib backend
    gallery = service.load_active_gallery()
    assert len(gallery) == 1


@needs_sface
def test_new_enrollment_with_sface_writes_a_tagged_encoding(enrolled_with_dlib, monkeypatch):
    service, dlib_file, _ = enrolled_with_dlib
    service.face_detector = MagicMock()
    service.face_detector.detect_faces.return_value = [DetectedFace((100, 60, 120, 140), 0.95, LANDMARKS)]
    service.face_embedder = FaceEmbedder(backend="sface")
    student = service.student_repo.get_all(active_only=True)[0]

    service.add_face_sample(student["id"], FRAME)

    tagged = list(dlib_file.parent.glob("enc_sface_*.npy"))
    assert len(tagged) == 1
    assert np.linalg.norm(np.load(tagged[0])) == pytest.approx(1.0)
