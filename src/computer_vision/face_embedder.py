"""
Face Embedder for the Intelligent Attendance Monitoring System.

Provides face embedding generation using face_recognition library.
Generates 128-dimensional face encodings from face images for recognition purposes.
"""

from typing import Optional
import numpy as np
import face_recognition

from src.utils.logger import get_logger


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
    
    def __init__(self):
        """Initialize the FaceEmbedder."""
        self.logger = get_logger(__name__)
        self.logger.info("FaceEmbedder initialized")
    
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
            # face_recognition.face_encodings returns a list of encodings
            # (one per face detected in the image)
            encodings = face_recognition.face_encodings(rgb_image)
            
            if len(encodings) == 0:
                # No face detected in the provided image
                self.logger.warning("No face detected in image for encoding")
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
