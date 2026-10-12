"""
Face Enrollment & Dataset Management Service.

Provides business logic for enrolling student face images, generating 128D embeddings,
persisting embeddings to disk/database, updating face datasets for changed appearances,
and constructing the active recognition gallery.
"""

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
        return face_crop, float(face.confidence)
        
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
            
        student_id_str = student['student_id']
        
        # Step 1: Validate face and extract crop
        face_crop, confidence = self.validate_and_extract_face(face_image)
        
        # Step 2: Generate 128D embedding
        embedder = self._get_embedder()
        try:
            embedding = embedder.generate_embedding(face_crop)
            if embedding is None:
                raise InvalidFaceImageError("Failed to extract face embedding from the detected face")
        except FaceEmbeddingError as e:
            raise InvalidFaceImageError(f"Embedding error: {e}")
            
        # Step 3: Save embedding and sample image to student's folder
        student_enc_dir = self.encodings_dir / str(student_id_str)
        student_img_dir = self.images_dir / str(student_id_str)
        student_enc_dir.mkdir(parents=True, exist_ok=True)
        student_img_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = f"{int(np.round(cv2.getTickCount()))}"
        encoding_filename = f"enc_{timestamp_str}.npy"
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
        
        # Get all active students
        students = self.student_repo.get_all(active_only=True)
        student_map = {s['id']: s['student_id'] for s in students}
        
        enrollments = self.enrollment_repo.get_all(active_only=True)
        
        for record in enrollments:
            student_db_id = record['student_id']
            if student_db_id not in student_map:
                continue  # Student inactive or deleted
                
            student_id_str = student_map[student_db_id]
            encoding_path_str = record['encoding_path']
            encoding_path = Path(encoding_path_str)
            
            if encoding_path.exists():
                try:
                    encoding_arr = np.load(str(encoding_path))
                    if encoding_arr.shape == (128,):
                        gallery.append((student_id_str, FaceEmbedding(encoding_arr)))
                except Exception as e:
                    logger.warning(f"Could not load encoding from {encoding_path}: {e}")
            else:
                logger.debug(f"Encoding file missing: {encoding_path}")
                
        logger.info(f"Loaded active recognition gallery with {len(gallery)} face samples")
        return gallery
        
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
