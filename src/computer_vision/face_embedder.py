"""
Face Embedder for the Intelligent Attendance Monitoring System.

Generates 128-dimensional face encodings for recognition with OpenCV's SFace
model (default, ~5 ms, face aligned on its landmarks first) or dlib via the
face_recognition library (~240 ms) - config.FACE_RECOGNITION_BACKEND.
The two models' encodings are not comparable with each other.
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import face_recognition

import config

from src.utils.logger import get_logger
from src.utils.performance_timer import time_operation


class FaceEmbeddingError(Exception):
    """Exception raised when face embedding generation fails."""
    pass


class FaceEmbedding:
    """
    Represents a face embedding (128-dimensional encoding).
    
    Attributes:
        encoding: 128-dimensional numpy array representing the face
    """
    
    def __init__(self, encoding: np.ndarray):
        """
        Initialize a face embedding.
        
        Args:
            encoding: 128-dimensional face encoding
            
        Raises:
            ValueError: If encoding is not 128-dimensional
        """
        if encoding.shape != (128,):
            raise ValueError(
                f"Face encoding must be 128-dimensional, got shape {encoding.shape}"
            )
        self.encoding = encoding
    
    def __repr__(self) -> str:
        return f"FaceEmbedding(shape={self.encoding.shape}, dtype={self.encoding.dtype})"


class FaceEmbedder:
    """
    Generates face embeddings from face images using face_recognition library.
    
    Accepts face images (typically crops from face detection) and generates
    128-dimensional encodings suitable for face recognition/matching.
    Uses dlib's face recognition model under the hood.
    """
    
    def __init__(self, backend: Optional[str] = None):
        """
        Initialize the FaceEmbedder.

        Args:
            backend: "sface" or "dlib"; None = config.FACE_RECOGNITION_BACKEND.
                SFace falls back to dlib if its model file is missing.
        """
        self.logger = get_logger(__name__)
        self.backend = (backend or config.FACE_RECOGNITION_BACKEND).lower()
        self._sface = None
        self._crop_detector = None  # YuNet, to find landmarks inside a face crop (lazy)
        if self.backend == "sface":
            model_path = Path(config.SFACE_MODEL_PATH)
            try:
                if not model_path.exists():
                    raise FileNotFoundError(model_path)
                self._sface = cv2.FaceRecognizerSF.create(str(model_path), "")
            except Exception as e:
                self.logger.warning(f"SFace model unavailable ({e}); using dlib face encodings")
                self.backend = "dlib"
        self.logger.info(f"FaceEmbedder initialized ({self.backend})")

    def generate_embedding_from_frame(self, frame: np.ndarray, face) -> Optional[FaceEmbedding]:
        """
        Embedding of a detected face in a full frame. With SFace and YuNet
        landmarks the face is aligned straight from the frame (best quality);
        otherwise the face box is cropped and passed to generate_embedding().
        """
        if self._sface is not None and getattr(face, "landmarks", None) is not None:
            with time_operation("face_embedding"):
                aligned = self._sface.alignCrop(frame, face.yunet_row())
                return self._sface_embedding(aligned)
        x, y, w, h = face.bbox
        return self.generate_embedding(frame[max(0, y):y + h, max(0, x):x + w])

    def _sface_embedding(self, aligned_face: np.ndarray) -> FaceEmbedding:
        """SFace feature of a 112x112 aligned BGR face, scaled to unit length."""
        feature = self._sface.feature(aligned_face).reshape(-1).astype(np.float64)
        norm = np.linalg.norm(feature)
        if norm == 0:
            raise FaceEmbeddingError("SFace returned an empty feature")
        return FaceEmbedding(feature / norm)

    def _sface_from_crop(self, face_crop: np.ndarray) -> Optional[FaceEmbedding]:
        """
        SFace embedding of a face crop (e.g. a saved enrollment photo): find the
        landmarks with YuNet on the padded crop and align; if YuNet finds no
        face, use the crop resized to 112x112.
        """
        bgr = face_crop
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        elif bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        pad = max(8, int(0.3 * max(h, w)))
        padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if self._crop_detector is None and Path(config.YUNET_MODEL_PATH).exists():
            self._crop_detector = cv2.FaceDetectorYN.create(
                str(config.YUNET_MODEL_PATH), "", (padded.shape[1], padded.shape[0]),
                score_threshold=0.5, nms_threshold=0.3, top_k=5
            )
        faces = None
        if self._crop_detector is not None:
            self._crop_detector.setInputSize((padded.shape[1], padded.shape[0]))
            _, faces = self._crop_detector.detect(padded)
        with time_operation("face_embedding"):
            if faces is not None and len(faces):
                largest = max(faces, key=lambda row: row[2] * row[3])
                aligned = self._sface.alignCrop(padded, largest)
            else:
                self.logger.debug("No landmarks found in face crop; using it unaligned")
                aligned = cv2.resize(bgr, (112, 112))
            return self._sface_embedding(aligned)

    def generate_embedding(self, face_image: np.ndarray) -> Optional[FaceEmbedding]:
        """
        Generate a face embedding from a face image.
        
        Args:
            face_image: Input face image as numpy array (RGB or BGR format).
                       Should contain a single face, typically a crop from face detection.
        
        Returns:
            FaceEmbedding object containing 128-dimensional encoding,
            or None if no face could be encoded.
        
        Raises:
            FaceEmbeddingError: If input validation fails or multiple faces detected
        """
        # Input validation
        if face_image is None:
            raise FaceEmbeddingError("Input image cannot be None")
        
        if not isinstance(face_image, np.ndarray):
            raise FaceEmbeddingError(
                f"Input must be a numpy array, got {type(face_image).__name__}"
            )
        
        if face_image.size == 0:
            raise FaceEmbeddingError("Input image cannot be empty")
        
        # Validate image shape
        if len(face_image.shape) not in [2, 3]:
            raise FaceEmbeddingError(
                f"Invalid image shape: {face_image.shape}. "
                "Expected 2D (grayscale) or 3D (color) array"
            )
        
        if self._sface is not None:
            try:
                return self._sface_from_crop(face_image)
            except FaceEmbeddingError:
                raise
            except Exception as e:
                self.logger.error(f"Error during face encoding: {e}")
                raise FaceEmbeddingError(f"Failed to generate face encoding: {e}")

        # Convert to RGB if needed
        try:
            rgb_image = self._ensure_rgb(face_image)
        except Exception as e:
            raise FaceEmbeddingError(f"Failed to convert image to RGB: {e}")
        
        # Ensure array is contiguous for dlib compatibility
        # Non-contiguous arrays can cause dlib to fail or throw exceptions
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image)
        
        # Generate face encodings
        try:
            # PERFORMANCE: Use small crop from MediaPipe detection instead of full frame.
            # Benchmark shows: crop=11ms vs full_frame=138ms vs full_frame_with_bbox=248ms
            #
            # ROBUSTNESS FIX: Pass known_face_locations to bypass dlib's internal face detection.
            # MediaPipe already detected the face and provided the crop. Telling dlib where
            # the face is prevents failures when the crop is rotated/turned (head turn scenario).
            # Format: [(top, right, bottom, left)] in CSS order
            h, w = rgb_image.shape[:2]
            known_face_locations = [(0, w, h, 0)]  # Entire crop is the face
            
            with time_operation("dlib_face_encoding"):
                encodings = face_recognition.face_encodings(rgb_image, known_face_locations=known_face_locations)
            
            if len(encodings) == 0:
                # No face detected in the provided image - log at DEBUG to reduce spam
                # since this happens frequently during real-time processing
                self.logger.debug("No face detected in image for encoding")
                return None
            
            if len(encodings) > 1:
                # Multiple faces detected - ambiguous input
                raise FaceEmbeddingError(
                    f"Expected single face but detected {len(encodings)} faces. "
                    "Provide a single-face crop from face detection."
                )
            
            # Single face encoding - expected case
            encoding = encodings[0]
            self.logger.debug("Successfully generated face embedding")
            return FaceEmbedding(encoding)
            
        except FaceEmbeddingError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Catch any other errors from face_recognition
            self.logger.error(f"Error during face encoding: {e}")
            raise FaceEmbeddingError(f"Failed to generate face encoding: {e}")
    
    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image is in RGB format.
        
        Args:
            image: Input image (grayscale or color)
        
        Returns:
            RGB image
        """
        # Grayscale - convert to RGB
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)
        
        # Already has 3 channels - assume BGR from OpenCV, convert to RGB
        if image.shape[2] == 3:
            return image[:, :, ::-1]  # BGR to RGB
        
        # 4 channels (RGBA) - drop alpha channel and return RGB
        if image.shape[2] == 4:
            return image[:, :, :3][:, :, ::-1]  # BGRA to RGB
        
        # Unexpected number of channels
        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
