"""
Face Enrollment & Dataset Management Service.

Provides business logic for enrolling student face images, generating 128D embeddings,
persisting embeddings to disk/database, updating face datasets for changed appearances,
and constructing the active recognition gallery.
"""

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

import config
from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbedding, FaceEmbeddingError
from src.data.repositories.enrollment_repository import EnrollmentRepository
from src.data.repositories.student_repository import StudentRepository
from src.services.face_index import SIMILAR_WARN_FRACTION, DuplicatePair, FaceCheck, FaceIndex
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Gallery cache file layout; a file with another version is rebuilt
GALLERY_CACHE_VERSION = 2

# Duplicate-check index per (database, encodings folder, backend), shared by
# every EnrollmentService of this process (the pages create a new service on
# each rerun): {key: (gallery fingerprint, FaceIndex)}. Built from the gallery
# cache the first time a face is checked, then kept in step with the samples
# this process saves, so a check costs one ~25 ms query plus ~7 ms of maths.
_face_indexes: Dict[tuple, Tuple[tuple, FaceIndex]] = {}
_face_indexes_lock = threading.Lock()

# progress(done, total) while a gallery load reads sample files
GalleryProgress = Callable[[int, int], None]


def _throttled(progress: Optional[GalleryProgress], total: int, interval: float = 0.1) -> Callable[[int], None]:
    """
    report(done) that calls progress(done, total) at most every `interval`
    seconds, plus at the start and the end (a progress bar updated for each
    of 11,000 files would slow the load down).
    """
    if progress is None or total == 0:
        return lambda done: None
    progress(0, total)
    last = time.perf_counter()

    def report(done: int) -> None:
        nonlocal last
        now = time.perf_counter()
        if done == total or now - last >= interval:
            last = now
            progress(done, total)
    return report


def encoding_file_for_backend(encoding_path: Path, backend: str) -> Path:
    """
    The encoding file of `backend` for an enrollment sample. dlib encodings
    are enc_<id>.npy and SFace ones enc_sface_<id>.npy in the same folder;
    the database keeps whichever path the sample was enrolled with.
    """
    name = encoding_path.name
    if backend == "sface":
        if name.startswith("enc_sface_"):
            return encoding_path
        return encoding_path.with_name("enc_sface_" + name[len("enc_"):] if name.startswith("enc_") else "sface_" + name)
    if name.startswith("enc_sface_"):
        return encoding_path.with_name("enc_" + name[len("enc_sface_"):])
    return encoding_path


class EnrollmentServiceError(Exception):
    """Base exception for enrollment service errors."""
    pass


class InvalidFaceImageError(EnrollmentServiceError):
    """Raised when an uploaded/captured face image is invalid or has no detectable face."""
    pass


class DuplicateFaceError(InvalidFaceImageError):
    """The face is already enrolled for another student (almost certainly the same person)."""

    def __init__(self, message: str, check: FaceCheck):
        super().__init__(message)
        self.check = check


class FaceNeedsConfirmationError(EnrollmentServiceError):
    """
    The face looks like another student's, or not like this student's own
    samples. Nothing was saved; add_face_sample(..., confirmed=True) saves it.
    """

    def __init__(self, message: str, check: FaceCheck):
        super().__init__(message)
        self.check = check


class EnrollmentService:
    """
    Service layer for student face dataset and gallery management.
    
    Handles:
    - Validating face images (detectability, single-face crop, quality)
    - Generating 128D FaceEmbedding
    - Saving face embeddings and sample crops to disk
    - Managing student enrollments in the database (supports multiple samples & appearance updates)
    - Constructing known gallery: List[Tuple[student_id_str, FaceEmbedding]]
    """
    
    def __init__(
        self,
        database_path: Optional[Path] = None,
        encodings_dir: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        face_detector: Optional[FaceDetector] = None,
        face_embedder: Optional[FaceEmbedder] = None
    ):
        """
        Initialize the enrollment service.
        """
        self.database_path = database_path
        self.encodings_dir = encodings_dir or config.FACE_ENCODINGS_DIR
        self.images_dir = images_dir or config.ENROLLMENT_IMAGES_DIR
        
        # Ensure target directories exist
        self.encodings_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.enrollment_repo = EnrollmentRepository(database_path)
        self.student_repo = StudentRepository(database_path)
        
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        
    def _get_detector(self) -> FaceDetector:
        if self.face_detector is None:
            self.face_detector = FaceDetector()
        return self.face_detector
        
    def _get_embedder(self) -> FaceEmbedder:
        if self.face_embedder is None:
            self.face_embedder = FaceEmbedder()
        return self.face_embedder
        
    def validate_and_extract_face(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Validate an image and extract the primary face crop and confidence score.
        
        Args:
            image: BGR numpy image
            
        Returns:
            Tuple of (cropped_face_bgr, detection_confidence)
            
        Raises:
            InvalidFaceImageError: If image is empty, has no face, or has multiple ambiguous faces
        """
        face, face_crop = self._detect_single_face(image)
        return face_crop, float(face.confidence)

    def _detect_single_face(self, image: np.ndarray) -> Tuple[DetectedFace, np.ndarray]:
        """The single face in `image` and its crop; raises InvalidFaceImageError otherwise."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise InvalidFaceImageError("Input image cannot be empty or invalid")

        detector = self._get_detector()
        detected_faces = detector.detect_faces(image)

        if not detected_faces:
            raise InvalidFaceImageError("No face detected in the provided image. Please ensure good lighting and face visibility.")
            
        if len(detected_faces) > 1:
            raise InvalidFaceImageError(f"Multiple faces detected ({len(detected_faces)}). Please provide an image with only one student face.")
            
        face = detected_faces[0]
        x, y, w, h = face.bbox
        h_img, w_img = image.shape[:2]
        
        crop_x = max(0, min(x, w_img - 1))
        crop_y = max(0, min(y, h_img - 1))
        crop_w = max(1, min(w, w_img - crop_x))
        crop_h = max(1, min(h, h_img - crop_y))
        
        face_crop = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        return face, face_crop
        
    def add_face_sample(
        self,
        student_db_id: int,
        face_image: np.ndarray,
        enrollment_method: str = 'camera',
        view_angle: str = 'unknown',
        confirmed: bool = False
    ) -> int:
        """
        Add a single face sample for a student, generate embedding, and record in database.
        Supports adding multiple samples over time (e.g. changed appearance).

        Args:
            student_db_id: Student database ID
            face_image: Full or cropped face image as numpy array
            enrollment_method: 'camera' or 'upload'
            view_angle: Face angle ('frontal', 'left', 'right', 'unknown')
            confirmed: the user chose to enroll a face that needed confirmation
                (FaceNeedsConfirmationError); a duplicate is refused regardless

        Returns:
            Enrollment record database ID

        Raises:
            DuplicateFaceError: the face is already enrolled for another student
            FaceNeedsConfirmationError: similar to another student, or unlike
                this student's own samples, and not `confirmed`
        """
        student = self.student_repo.get_by_id(student_db_id)
        if not student:
            raise EnrollmentServiceError(f"Student with DB ID {student_db_id} not found")

        # Steps 1-2: validate the face, crop it and generate its embedding
        face, face_crop, embedding = self.prepare_face_sample(face_image)

        # Compare with every enrolled face before anything is saved
        check = self.check_face(embedding, student['student_id'])
        if check.status == "duplicate":
            raise DuplicateFaceError(self.describe_face_check(check), check)
        if check.needs_confirmation and not confirmed:
            raise FaceNeedsConfirmationError(self.describe_face_check(check), check)

        # Steps 3-4: save the files and the enrollment record
        enrollment_id = self.save_face_sample(student, face_crop, embedding, float(face.confidence),
                                              enrollment_method=enrollment_method, view_angle=view_angle)
        # The attendance engine then finds the new sample in the cache instead
        # of reading it at start-up (bulk imports update the cache once at the end)
        self.update_gallery_cache()
        return enrollment_id

    def prepare_face_sample(self, face_image: np.ndarray) -> Tuple[DetectedFace, np.ndarray, FaceEmbedding]:
        """
        The single face in `face_image`, its crop and its 128D embedding
        (SFace: aligned on the face's landmarks), without saving anything.

        Raises:
            InvalidFaceImageError: no face, several faces, or no embedding
        """
        face, face_crop = self._detect_single_face(face_image)
        embedder = self._get_embedder()
        try:
            if self._uses_sface():
                embedding = embedder.generate_embedding_from_frame(face_image, face)
            else:
                embedding = embedder.generate_embedding(face_crop)
            if embedding is None:
                raise InvalidFaceImageError("Failed to extract face embedding from the detected face")
        except FaceEmbeddingError as e:
            raise InvalidFaceImageError(f"Embedding error: {e}")
        return face, face_crop, embedding

    def _uses_sface(self) -> bool:
        return getattr(self._get_embedder(), "backend", None) == "sface"

    def save_face_sample(
        self,
        student: Dict[str, Any],
        face_crop: np.ndarray,
        embedding: FaceEmbedding,
        confidence: float,
        enrollment_method: str = 'camera',
        view_angle: str = 'unknown'
    ) -> int:
        """Save a prepared face sample (embedding + crop) for `student` and record it. Returns the enrollment ID."""
        student_db_id = student['id']
        student_id_str = student['student_id']
        sface = self._uses_sface()

        # Step 3: Save embedding and sample image to student's folder
        student_enc_dir = self.encodings_dir / str(student_id_str)
        student_img_dir = self.images_dir / str(student_id_str)
        student_enc_dir.mkdir(parents=True, exist_ok=True)
        student_img_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = f"{int(np.round(cv2.getTickCount()))}"
        # SFace and dlib embeddings are not comparable: tag the file with the model
        encoding_filename = f"enc_sface_{timestamp_str}.npy" if sface else f"enc_{timestamp_str}.npy"
        image_filename = f"face_{timestamp_str}.jpg"
        
        encoding_file_path = student_enc_dir / encoding_filename
        image_file_path = student_img_dir / image_filename
        
        # Save .npy embedding array
        np.save(str(encoding_file_path), embedding.encoding)
        
        # Save sample image crop
        cv2.imwrite(str(image_file_path), face_crop)
        
        # Step 4: Record enrollment in database
        enrollment_id = self.enrollment_repo.create_enrollment(
            student_id=student_db_id,
            encoding_path=str(encoding_file_path),
            enrollment_method=enrollment_method,
            image_count=1,
            quality_score=confidence,
            view_angle=view_angle,
            is_active=True
        )
        if student.get('is_active', 1):
            self._add_to_face_index(enrollment_id, student, embedding.encoding)

        logger.info(
            f"Enrolled face sample for student {student_id_str} (DB ID: {student_db_id}, "
            f"Enrollment ID: {enrollment_id})"
        )
        return enrollment_id
        
    # ------------------------------------------------------------------
    # Duplicate-face checks
    # ------------------------------------------------------------------

    def recognition_threshold(self) -> float:
        """Distance below which live attendance treats two faces as the same person."""
        backend = self._gallery_backend()
        return config.SFACE_RECOGNITION_THRESHOLD if backend == "sface" else config.FACE_RECOGNITION_THRESHOLD

    def _face_index_key(self) -> tuple:
        return (str(self.database_path), str(self.encodings_dir), self._gallery_backend())

    def _with_face_index(self, use: Callable[[FaceIndex], Any]) -> Any:
        """
        Run use(index) on the up-to-date duplicate-check index. The index is
        rebuilt (from the gallery cache, ~1 s for 65,000 samples) only when the
        gallery changed in a way this process did not record itself.
        """
        key = self._face_index_key()
        with _face_indexes_lock:
            # Fingerprint BEFORE loading: a sample added during the load makes it stale, never missed
            fingerprint = self.enrollment_repo.get_active_gallery_fingerprint()
            entry = _face_indexes.get(key)
            if entry is None or entry[0] != fingerprint:
                started = time.perf_counter()
                gallery = self.load_active_gallery()
                entry = (fingerprint, FaceIndex((sid, embedding.encoding) for sid, embedding in gallery))
                _face_indexes[key] = entry
                logger.info(f"Built the duplicate-face index: {entry[1].size} samples "
                            f"in {time.perf_counter() - started:.2f} s")
            return use(entry[1])

    def _add_to_face_index(self, enrollment_id: int, student: Dict[str, Any], encoding: np.ndarray) -> None:
        """
        Keep a built index in step with a sample this process just saved, so the
        next check does not rebuild it. The fingerprint is advanced by exactly
        this sample: if anything else changed the gallery it still differs.
        """
        key = self._face_index_key()
        with _face_indexes_lock:
            entry = _face_indexes.get(key)
            if entry is None:
                return
            count, max_id, id_sum, student_sum, length_sum = entry[0]
            entry[1].add(student['student_id'], encoding)
            _face_indexes[key] = ((count + 1, max(max_id, enrollment_id), id_sum + enrollment_id,
                                   student_sum + student['id'], length_sum + len(student['student_id'])),
                                  entry[1])

    def check_face(self, embedding: FaceEmbedding, student_id_str: str) -> FaceCheck:
        """How a new face sample of `student_id_str` compares with every enrolled face."""
        threshold = self.recognition_threshold()
        return self._with_face_index(lambda index: index.check(embedding.encoding, student_id_str, threshold))

    def describe_face_check(self, check: FaceCheck) -> str:
        """A message for the user about a duplicate / similar / mismatching face."""
        threshold = self.recognition_threshold()
        other = ""
        if check.other_student_id is not None:
            record = self.student_repo.get_by_student_id(check.other_student_id)
            name = record['name'] if record else "another student"
            other = f"{name} ({check.other_student_id})"
        if check.status == "duplicate":
            return (f"This face is already enrolled for {other} (distance {check.other_distance:.2f}). "
                    f"One person enrolled under two student IDs is not recognized at attendance - neither "
                    f"student is marked. If the face was enrolled for the wrong student, reset that "
                    f"student's face data first.")
        if check.status == "similar":
            return (f"This face looks like {other} (distance {check.other_distance:.2f}; photos of the same "
                    f"person are usually closer than {SIMILAR_WARN_FRACTION * threshold:.2f}). Enroll it only "
                    f"if they are different people; faces this close can be rejected as ambiguous at attendance.")
        if check.status == "mismatch":
            return (f"This face does not match this student's other face samples (closest distance "
                    f"{check.own_distance:.2f}, the same-person limit is {threshold:.2f}). Check it is the "
                    f"right person; enroll it anyway only if their appearance changed.")
        return ""

    def find_duplicate_faces(self, progress: Optional[GalleryProgress] = None) -> List[DuplicatePair]:
        """Every pair of different students whose faces are similar (see face_index), closest first."""
        threshold = self.recognition_threshold()
        # Scan a copy, so checks in other sessions are not held up for the whole scan
        snapshot = self._with_face_index(lambda index: index.copy())
        return snapshot.duplicate_pairs(threshold, progress=progress)

    def add_multiple_face_samples(
        self,
        student_db_id: int,
        face_images: List[np.ndarray],
        enrollment_method: str = 'camera'
    ) -> List[int]:
        """
        Add multiple face samples for a student.
        
        Args:
            student_db_id: Student database ID
            face_images: List of image numpy arrays
            enrollment_method: 'camera' or 'upload'
            
        Returns:
            List of created enrollment IDs
        """
        if not face_images:
            raise EnrollmentServiceError("No face images provided")
            
        enrollment_ids = []
        for idx, img in enumerate(face_images):
            try:
                eid = self.add_face_sample(student_db_id, img, enrollment_method)
                enrollment_ids.append(eid)
            except Exception as e:
                logger.warning(f"Skipped sample {idx + 1} due to error: {e}")
                
        if not enrollment_ids:
            raise InvalidFaceImageError("None of the provided face samples could be enrolled. Please check face clarity.")
            
        return enrollment_ids
        
    def get_student_samples_count(self, student_db_id: int) -> int:
        """Get the number of active enrolled face samples for a student."""
        enrollments = self.enrollment_repo.get_by_student_id(student_db_id, active_only=True)
        return len(enrollments)

    def get_all_samples_counts(self) -> Dict[int, int]:
        """Active face sample count per student DB ID, in one query (students without samples are absent)."""
        counts: Dict[int, int] = {}
        for record in self.enrollment_repo.get_all(active_only=True):
            counts[record['student_id']] = counts.get(record['student_id'], 0) + 1
        return counts
        
    def get_student_enrollment_coverage(self, student_db_id: int) -> Dict[str, bool]:
        """
        Get enrollment coverage information for a student.
        
        Args:
            student_db_id: Student database ID
            
        Returns:
            Dict with coverage info: {'frontal': bool, 'left': bool, 'right': bool, 'total_samples': int}
        """
        enrollments = self.enrollment_repo.get_by_student_id(student_db_id, active_only=True)
        
        angles_present = set()
        for enrollment in enrollments:
            angle = enrollment.get('view_angle', 'unknown')
            if angle in ['frontal', 'left', 'right']:
                angles_present.add(angle)
        
        return {
            'frontal': 'frontal' in angles_present,
            'left': 'left' in angles_present, 
            'right': 'right' in angles_present,
            'total_samples': len(enrollments),
            'angles_present': list(angles_present)
        }
        
    def load_active_gallery(
        self,
        progress: Optional[GalleryProgress] = None,
        rebuild: bool = False
    ) -> List[Tuple[str, FaceEmbedding]]:
        """
        Load all active face embeddings for all active students.

        The encodings are kept in one cache file (gallery_cache_<backend>.npz
        in the encodings folder), one row per enrollment sample. A load reads
        only the samples that are not in the cache yet and saves the updated
        cache: after a restart, reading 11,200 sample files took ~44 s (cold
        disk cache, antivirus scanning every file) and timed the engine out,
        while the cache loads in ~30 ms. Removed or deactivated samples are
        simply left out.

        Args:
            progress: called as progress(done, total) while sample files are
                read (not called when everything comes from the cache)
            rebuild: ignore the cache and read every sample file again

        Returns:
            List of (student_id_string, FaceEmbedding) tuples ready for FaceRecognizer
        """
        backend = self._gallery_backend()
        keys = self.enrollment_repo.get_active_gallery_keys()
        cached = {} if rebuild else self._read_gallery_cache(backend)

        to_read = [key for key in keys if key not in cached]
        rows: Dict[tuple, Optional[np.ndarray]] = dict(cached)
        unreadable = set()
        report = _throttled(progress, len(to_read))
        for done, key in enumerate(to_read, start=1):
            encoding, readable = self._read_sample_encoding(key, backend)
            rows[key] = encoding
            if not readable:
                unreadable.add(key)  # try again next time instead of caching the failure
            report(done)

        gallery = [(key[2], FaceEmbedding(rows[key])) for key in keys if rows[key] is not None]
        if to_read or len(cached) != len(keys):
            self._write_gallery_cache(backend, [key for key in keys if key not in unreadable], rows)
        logger.info(f"Loaded active recognition gallery with {len(gallery)} face samples ({backend}, "
                    f"{len(to_read)} read from their files, {len(keys) - len(to_read)} from the cache)")
        return gallery

    def update_gallery_cache(self) -> None:
        """
        Bring an existing gallery cache up to date after an enrollment change
        (reads only the new samples). Without a cache it does nothing: building
        one reads every sample, which prepare_recognition_data() does with a
        progress bar.
        """
        if not self._gallery_cache_path(self._gallery_backend()).exists():
            return
        try:
            self.load_active_gallery()
        except Exception as e:
            logger.warning(f"Could not update the gallery cache: {e}")

    def prepare_recognition_data(self, progress: Optional[GalleryProgress] = None,
                                 rebuild: bool = False) -> int:
        """Build the gallery cache ahead of an attendance session. Returns the number of face samples."""
        return len(self.load_active_gallery(progress=progress, rebuild=rebuild))

    def gallery_cache_status(self) -> Dict[str, int]:
        """
        {'samples': active face samples, 'pending': samples not in the gallery
        cache yet}. Cheap: one SQL query and the cache's keys, no encodings.
        """
        keys = self.enrollment_repo.get_active_gallery_keys()
        cached = self._read_gallery_cache(self._gallery_backend(), keys_only=True)
        return {"samples": len(keys), "pending": sum(1 for key in keys if key not in cached)}

    def _read_sample_encoding(self, key: tuple, backend: str) -> Tuple[Optional[np.ndarray], bool]:
        """
        (encoding or None, readable) of one sample from its own file. A missing
        file or wrong shape is a final None; readable is False only for a read
        error, which is worth retrying.
        """
        _, encoding_path, student_id_str = key
        path = encoding_file_for_backend(Path(encoding_path), backend)
        if not path.exists() and backend == "sface":
            path = self._create_sface_encoding(path, student_id_str) or path
        if not path.exists():
            logger.debug(f"Encoding file missing: {path}")
            return None, True
        try:
            encoding = np.load(str(path))
        except Exception as e:
            logger.warning(f"Could not load encoding from {path}: {e}")
            return None, False
        return (encoding if encoding.shape == (128,) else None), True

    def _gallery_cache_path(self, backend: str) -> Path:
        return self.encodings_dir / f"gallery_cache_{backend}.npz"

    def _read_gallery_cache(self, backend: str, keys_only: bool = False) -> Dict[tuple, Optional[np.ndarray]]:
        """
        Cached rows as {(enrollment id, encoding path, student_id): encoding,
        or None for a sample without a usable encoding}; {} if there is no
        usable cache. keys_only skips reading the encodings (values are None).
        """
        path = self._gallery_cache_path(backend)
        if not path.exists():
            return {}
        try:
            with np.load(str(path), allow_pickle=False) as data:
                if "version" not in data.files or int(data["version"]) != GALLERY_CACHE_VERSION:
                    logger.info(f"Gallery cache {path.name} has an older format; it will be rebuilt")
                    return {}
                ids, paths, identities = data["enrollment_ids"], data["paths"], data["identities"]
                valid = encodings = None
                if not keys_only:
                    valid, encodings = data["valid"], data["encodings"]
            if not len(ids) == len(paths) == len(identities):
                raise ValueError("key arrays differ in length")
            keys = list(zip(map(int, ids.tolist()), paths.tolist(), identities.tolist()))
            if keys_only:
                return dict.fromkeys(keys)
            if encodings.shape != (len(keys), 128) or len(valid) != len(keys):
                raise ValueError(f"unexpected shapes {encodings.shape} / {valid.shape} for {len(keys)} rows")
            return {key: (encodings[n] if valid[n] else None) for n, key in enumerate(keys)}
        except Exception as e:
            logger.warning(f"Ignoring unreadable gallery cache {path.name}: {e}")
            return {}

    def _write_gallery_cache(self, backend: str, keys: List[tuple],
                             rows: Dict[tuple, Optional[np.ndarray]]) -> None:
        """
        Save the rows of `keys` as one file. Encodings are stored as float64
        (they are float64 already; recognition computes in float64 anyway).
        A failure only costs the next load its speed.
        """
        path = self._gallery_cache_path(backend)
        # Unique temporary name, then an atomic replace: a reader never sees a half-written file
        tmp_path = path.with_name(f"{path.stem}.{os.getpid()}-{threading.get_ident()}.tmp")
        encodings = np.zeros((len(keys), 128), dtype=np.float64)
        valid = np.zeros(len(keys), dtype=bool)
        for n, key in enumerate(keys):
            if rows[key] is not None:
                encodings[n] = rows[key]
                valid[n] = True
        try:
            with open(tmp_path, "wb") as f:
                np.savez(f, version=np.array(GALLERY_CACHE_VERSION),
                         enrollment_ids=np.array([key[0] for key in keys], dtype=np.int64),
                         paths=np.array([key[1] for key in keys], dtype=str),
                         identities=np.array([key[2] for key in keys], dtype=str),
                         valid=valid, encodings=encodings)
            os.replace(tmp_path, path)
        except OSError as e:
            logger.warning(f"Could not save the gallery cache {path.name}: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def gallery_signature(self) -> str:
        """
        Fingerprint of what load_active_gallery() would load: changes when a
        sample is added, removed or (de)activated, or a student is
        (de)activated or renamed. One SQL query (~tens of ms), cheaper than
        even a cached gallery load.
        """
        digest = hashlib.blake2b(self._gallery_backend().encode(), digest_size=16)
        for key in self.enrollment_repo.get_active_gallery_keys():
            digest.update(repr(key).encode())
        return digest.hexdigest()

    def _gallery_backend(self) -> str:
        """Embedding model of the gallery: the embedder's, or SFace if configured and available."""
        if self.face_embedder is not None:
            return "sface" if getattr(self.face_embedder, "backend", None) == "sface" else "dlib"
        if config.FACE_RECOGNITION_BACKEND == "sface" and Path(config.SFACE_MODEL_PATH).exists():
            return "sface"
        return "dlib"

    def _create_sface_encoding(self, sface_path: Path, student_id_str: str) -> Optional[Path]:
        """
        Create a missing SFace encoding from the saved enrollment photo
        (face_<id>.jpg) of a sample enrolled with dlib. Only adds a file; the
        dlib encoding is kept. Returns the new path, or None if not possible.
        """
        sample_id = sface_path.stem[len("enc_sface_"):]
        photo_path = self.images_dir / str(student_id_str) / f"face_{sample_id}.jpg"
        photo = cv2.imread(str(photo_path)) if photo_path.exists() else None
        if photo is None:
            logger.warning(f"No enrollment photo {photo_path} to create the SFace encoding from; "
                           f"re-enroll student {student_id_str} to include this sample")
            return None
        try:
            embedding = self._get_embedder().generate_embedding(photo)
            if embedding is None:
                return None
            np.save(str(sface_path), embedding.encoding)
            logger.info(f"Created SFace encoding {sface_path.name} for student {student_id_str} "
                        f"from the enrollment photo")
            return sface_path
        except Exception as e:
            logger.warning(f"Could not create the SFace encoding for {photo_path}: {e}")
            return None
        
    def clear_student_face_dataset(self, student_db_id: int) -> int:
        """
        Clear all face enrollments for a student (allows complete dataset refresh).
        
        Args:
            student_db_id: Student database ID
            
        Returns:
            Number of deleted records
        """
        student = self.student_repo.get_by_id(student_db_id)
        if not student:
            raise EnrollmentServiceError(f"Student with DB ID {student_db_id} not found")
            
        student_id_str = student['student_id']
        
        # Delete from repository
        deleted_count = self.enrollment_repo.delete_by_student_id(student_db_id)
        
        # Clean files
        student_enc_dir = self.encodings_dir / str(student_id_str)
        if student_enc_dir.exists():
            for f in student_enc_dir.glob("*.npy"):
                try:
                    f.unlink()
                except Exception:
                    pass
                    
        logger.info(f"Cleared {deleted_count} face sample(s) for student {student_id_str}")
        return deleted_count
