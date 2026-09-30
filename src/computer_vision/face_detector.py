"""
Face Detector for the Intelligent Attendance Monitoring System.

Provides face detection capabilities using MediaPipe Face Detection.
Detects faces in images/frames and returns bounding boxes with confidence scores.
"""

from typing import List, Optional, Tuple
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

import config
from src.utils.logger import get_logger


class DetectedFace:
    """
    Represents a single detected face.
    
    Attributes:
        bbox: Bounding box as (x, y, width, height) in pixels
        confidence: Detection confidence score (0.0 to 1.0)
    """
    
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float):
        """
        Initialize a detected face.
        
        Args:
            bbox: Bounding box (x, y, width, height) in pixels
            confidence: Detection confidence score
        """
        self.bbox = bbox
        self.confidence = confidence
    
    def __repr__(self) -> str:
        return f"DetectedFace(bbox={self.bbox}, confidence={self.confidence:.3f})"


class FaceDetector:
    """
    Detects faces in images using MediaPipe Face Detection.
    
    Provides face detection capabilities with configurable confidence thresholds.
    Returns bounding boxes and confidence scores for detected faces.
    Uses MediaPipe's BlazeFace model for face detection.
    """
    
    def __init__(self, min_detection_confidence: Optional[float] = None):
        """
        Initialize the FaceDetector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold for detection.
                                     If None, uses config.FACE_DETECTION_CONFIDENCE
        """
        self.min_detection_confidence = (
            min_detection_confidence 
            if min_detection_confidence is not None 
            else config.FACE_DETECTION_CONFIDENCE
        )
        
        self.logger = get_logger(__name__)
        self._detector = None
        self._initialize_detector()
    
    def _initialize_detector(self) -> None:
        """Initialize MediaPipe face detector."""
        try:
            # Locate the model file
            model_path = self._get_model_path()
            
            # Configure MediaPipe Face Detector options
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.min_detection_confidence
            )
            self._detector = vision.FaceDetector.create_from_options(options)
            self.logger.info(
                f"FaceDetector initialized with confidence threshold: "
                f"{self.min_detection_confidence}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe face detector: {e}")
            self._detector = None
    
    def _get_model_path(self) -> Path:
        """
        Get the path to the MediaPipe face detection model.
        
        Returns:
            Path to the model file
        
        Raises:
            FileNotFoundError: If model file is not found
        """
        # Check project root for model file
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "blaze_face_short_range.tflite"
        
        if model_path.exists():
            return model_path
        
        raise FileNotFoundError(
            f"MediaPipe face detection model not found at {model_path}. "
            "Please download from: "
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        )
    
    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
        
        Returns:
            List[DetectedFace]: List of detected faces with bounding boxes and confidence.
                               Empty list if no faces detected or on error.
        """
        if self._detector is None:
            self.logger.error("Face detector not initialized")
            return []
        
        if image is None or not isinstance(image, np.ndarray):
            self.logger.error("Invalid input: image must be a numpy array")
            return []
        
        if image.size == 0:
            self.logger.error("Invalid input: empty image")
            return []
        
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            image_rgb = self._convert_to_rgb(image)
            
            # Get image dimensions
            height, width = image_rgb.shape[:2]
            
            # Create MediaPipe Image object
            mp_image = Image(
                image_format=ImageFormat.SRGB,
                data=image_rgb
            )
            
            # Perform detection
            detection_result = self._detector.detect(mp_image)
            
            if detection_result.detections is None or len(detection_result.detections) == 0:
                return []
            
            # Convert detections to DetectedFace objects
            detected_faces = []
            for detection in detection_result.detections:
                bbox = self._extract_bounding_box(detection, width, height)
                confidence = detection.categories[0].score
                detected_faces.append(DetectedFace(bbox, confidence))
            
            self.logger.debug(f"Detected {len(detected_faces)} face(s)")
            return detected_faces
            
        except Exception as e:
            self.logger.error(f"Error during face detection: {e}")
            return []
    
    def _convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to RGB format if needed.
        
        Args:
            image: Input image (assumed BGR from OpenCV)
        
        Returns:
            RGB image
        """
        # Check if image is grayscale
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)
        
        # Check if image has 3 channels (assume BGR from OpenCV)
        if image.shape[2] == 3:
            return image[:, :, ::-1]  # BGR to RGB
        
        # Already RGB or RGBA
        return image
    
    def _extract_bounding_box(
        self, 
        detection, 
        image_width: int, 
        image_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Extract and convert bounding box from MediaPipe detection.
        
        MediaPipe returns pixel coordinates directly.
        This ensures they are within image bounds.
        
        Args:
            detection: MediaPipe detection object
            image_width: Image width in pixels
            image_height: Image height in pixels
        
        Returns:
            Tuple[int, int, int, int]: (x, y, width, height) in pixels
        """
        bbox = detection.bounding_box
        
        # MediaPipe returns pixel coordinates directly
        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        width = int(bbox.width)
        height = int(bbox.height)
        
        # Clamp coordinates to image bounds
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        
        # Adjust width/height to stay within bounds
        width = min(width, image_width - x)
        height = min(height, image_height - y)
        
        return (x, y, width, height)
    
    def close(self) -> None:
        """
        Close the face detector and release resources.
        
        Safe to call multiple times.
        """
        if self._detector is not None:
            try:
                self._detector.close()
                self.logger.info("Face detector closed")
            except Exception as e:
                self.logger.error(f"Error closing face detector: {e}")
            finally:
                self._detector = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close detector."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor: ensure detector is closed."""
        self.close()
