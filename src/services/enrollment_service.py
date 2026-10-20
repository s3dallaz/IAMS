"""
Face Enrollment & Dataset Management Service.

Provides business logic for enrolling student face images, generating 128D embeddings,
persisting embeddings to disk/database, updating face datasets for changed appearances,
and constructing the active recognition gallery.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

import config
from src.computer_vision.face_detector import FaceDetector, DetectedFace
from src.computer_vision.face_embedder import FaceEmbedder, FaceEmbedding, FaceEmbeddingError
from src.data.repositories.enrollment_repository import EnrollmentRepository
from src.data.repositories.student_repository import StudentRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        view_angle: str = 'unknown'
    ) -> int:
        """
        Add a single face sample for a student, generate embedding, and record in database.
        Supports adding multiple samples over time (e.g. changed appearance).
        
        Args:
            student_db_id: Student database ID
            face_image: Full or cropped face image as numpy array
            enrollment_method: 'camera' or 'upload'
            view_angle: Face angle ('frontal', 'left', 'right', 'unknown')
            
        Returns:
            Enrollment record database ID
        """
        student = self.student_repo.get_by_id(student_db_id)
        if not student:
            raise EnrollmentServiceError(f"Student with DB ID {student_db_id} not found")

        # Steps 1-2: validate the face, crop it and generate its embedding
        face, face_crop, embedding = self.prepare_face_sample(face_image)

        # Steps 3-4: save the files and the enrollment record
        return self.save_face_sample(student, face_crop, embedding, float(face.confidence),
                                     enrollment_method=enrollment_method, view_angle=view_angle)

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
        
        logger.info(
            f"Enrolled face sample for student {student_id_str} (DB ID: {student_db_id}, "
            f"Enrollment ID: {enrollment_id})"
        )
        return enrollment_id
        
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
        
    def load_active_gallery(self) -> List[Tuple[str, FaceEmbedding]]:
        """
        Load all active face embeddings from disk/database for all active students.
        
        Returns:
            List of (student_id_string, FaceEmbedding) tuples ready for FaceRecognizer
        """
        gallery: List[Tuple[str, FaceEmbedding]] = []
        backend = self._gallery_backend()

        # Get all active students
        students = self.student_repo.get_all(active_only=True)
        student_map = {s['id']: s['student_id'] for s in students}
        
        enrollments = self.enrollment_repo.get_all(active_only=True)
        
        for record in enrollments:
            student_db_id = record['student_id']
            if student_db_id not in student_map:
                continue  # Student inactive or deleted
                
            student_id_str = student_map[student_db_id]
            encoding_path = encoding_file_for_backend(Path(record['encoding_path']), backend)
            if not encoding_path.exists() and backend == "sface":
                encoding_path = self._create_sface_encoding(encoding_path, student_id_str) or encoding_path

            if encoding_path.exists():
                try:
                    encoding_arr = np.load(str(encoding_path))
                    if encoding_arr.shape == (128,):
                        gallery.append((student_id_str, FaceEmbedding(encoding_arr)))
                except Exception as e:
                    logger.warning(f"Could not load encoding from {encoding_path}: {e}")
            else:
                logger.debug(f"Encoding file missing: {encoding_path}")
                
        logger.info(f"Loaded active recognition gallery with {len(gallery)} face samples ({backend})")
        return gallery

    def gallery_signature(self) -> str:
        """
        Fingerprint of what load_active_gallery() would load: changes when a
        sample is added, removed or (de)activated, or a student is
        (de)activated or renamed. One SQL query (~tens of ms), while loading
        the gallery reads every encoding file (~2.6 s for 11,000 samples).
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
