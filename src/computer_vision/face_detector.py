"""
Face Detector for the Intelligent Attendance Monitoring System.

Provides face detection with OpenCV's YuNet model (default) or MediaPipe
BlazeFace (config.FACE_DETECTOR_BACKEND). Detects faces in images/frames and
returns bounding boxes with confidence scores; YuNet also returns 5 facial
landmarks, which the SFace recogniser uses to align the face.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

import config
from src.computer_vision.detection_diagnostics import get_diagnostics
from src.utils.logger import get_logger


class DetectedFace:
    """
    Represents a single detected face.
    
    Attributes:
        bbox: Bounding box as (x, y, width, height) in pixels
        confidence: Detection confidence score (0.0 to 1.0)
        landmarks: 5x2 array of (x, y) points - right eye, left eye, nose tip,
            right and left mouth corner - from YuNet, or None (MediaPipe)
    """

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        landmarks: Optional[np.ndarray] = None
    ):
        """
        Initialize a detected face.

        Args:
            bbox: Bounding box (x, y, width, height) in pixels
            confidence: Detection confidence score
            landmarks: Optional 5x2 facial landmarks (YuNet)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks = landmarks

    def yunet_row(self) -> np.ndarray:
        """The face in YuNet's 15-value layout (box, landmarks, score) used by SFace alignment."""
        return np.array([*self.bbox, *self.landmarks.reshape(-1), self.confidence], dtype=np.float32)

    def __repr__(self) -> str:
        return f"DetectedFace(bbox={self.bbox}, confidence={self.confidence:.3f})"


def select_foreground_face(
    bboxes: List[Tuple[int, int, int, int]],
    max_background_ratio: Optional[float] = None
) -> Optional[int]:
    """
    Index of the one face in front of the camera, or None.

    A face narrower than `max_background_ratio` times the widest face is
    someone in the background and is ignored. None when there is no face, or
    when two or more faces are about the same size (nobody can tell whose
    blink or head turn it is, so nobody may be checked).

    Args:
        bboxes: Face boxes (x, y, width, height)
        max_background_ratio: Default config.BACKGROUND_FACE_MAX_RATIO; 0 =
            every face counts, so only a single face in view qualifies
    """
    if not bboxes:
        return None
    ratio = config.BACKGROUND_FACE_MAX_RATIO if max_background_ratio is None else max_background_ratio
    widths = [box[2] for box in bboxes]
    main = max(range(len(widths)), key=widths.__getitem__)
    for index, width in enumerate(widths):
        if index != main and width >= ratio * widths[main]:
            return None
    return main


class FaceDetector:
    """
    Detects faces in images with YuNet (OpenCV DNN) or MediaPipe BlazeFace.

    Provides face detection capabilities with configurable confidence thresholds.
    Returns bounding boxes and confidence scores for detected faces.
    YuNet (default) also detects turned faces and returns 5 landmarks per face.
    """

    def __init__(self, min_detection_confidence: Optional[float] = None, backend: Optional[str] = None):
        """
        Initialize the FaceDetector.

        Args:
            min_detection_confidence: Minimum confidence threshold for detection.
                                     If None, uses config.FACE_DETECTION_CONFIDENCE
            backend: "yunet" or "mediapipe"; None = config.FACE_DETECTOR_BACKEND.
                YuNet falls back to MediaPipe if its model file is missing.
        """
        self.backend = (backend or config.FACE_DETECTOR_BACKEND).lower()
        self._yunet = None
        self._yunet_size: Optional[Tuple[int, int]] = None
        self.min_detection_confidence = (
            min_detection_confidence 
            if min_detection_confidence is not None 
            else config.FACE_DETECTION_CONFIDENCE
        )
        
        self.logger = get_logger(__name__)
        self._detector = None
        if self.backend == "yunet":
            self._initialize_yunet()
        if self._yunet is None:
            self.backend = "mediapipe"
            self._initialize_detector()

    def _initialize_yunet(self) -> None:
        """Initialize OpenCV's YuNet face detector (falls back to MediaPipe on failure)."""
        model_path = Path(config.YUNET_MODEL_PATH)
        if not model_path.exists():
            self.logger.warning(f"YuNet model not found at {model_path}; using MediaPipe face detection")
            return
        try:
            self._yunet = cv2.FaceDetectorYN.create(
                str(model_path), "", (320, 320),
                score_threshold=self.min_detection_confidence, nms_threshold=0.3, top_k=50
            )
            self._yunet_size = (320, 320)
            self.logger.info(f"FaceDetector initialized (YuNet, confidence threshold {self.min_detection_confidence})")
        except Exception as e:
            self.logger.error(f"Failed to initialize YuNet: {e}; using MediaPipe face detection")
            self._yunet = None

    def _detect_yunet(self, image: np.ndarray) -> List[DetectedFace]:
        """YuNet detection: boxes clipped to the image, plus 5 landmarks per face."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        height, width = image.shape[:2]
        if self._yunet_size != (width, height):
            self._yunet.setInputSize((width, height))
            self._yunet_size = (width, height)
        _, faces = self._yunet.detect(image)
        detected_faces = []
        for row in faces if faces is not None else []:
            x, y = max(0, int(row[0])), max(0, int(row[1]))
            w = min(int(row[2]), width - x)
            h = min(int(row[3]), height - y)
            if w <= 0 or h <= 0:
                continue
            detected_faces.append(DetectedFace((x, y, w, h), float(row[14]), row[4:14].reshape(5, 2).copy()))
        get_diagnostics().record(image.shape, detected_faces, stage="detection")
        return detected_faces

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
        if self._detector is None and self._yunet is None:
            self.logger.error("Face detector not initialized")
            return []

        if image is None or not isinstance(image, np.ndarray):
            self.logger.error("Invalid input: image must be a numpy array")
            return []

        if image.size == 0:
            self.logger.error("Invalid input: empty image")
            return []

        if self._yunet is not None:
            try:
                return self._detect_yunet(image)
            except Exception as e:
                self.logger.error(f"Error during face detection: {e}")
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
                # Diagnostics: a zero-detection frame is the key measurement for
                # diagnosing distance-related detection loss.
                get_diagnostics().record(image_rgb.shape, [], stage="detection")
                return []
            
            # Convert detections to DetectedFace objects
            detected_faces = []
            for detection in detection_result.detections:
                bbox = self._extract_bounding_box(detection, width, height)
                confidence = detection.categories[0].score
                detected_faces.append(DetectedFace(bbox, confidence))
            
            self.logger.debug(f"Detected {len(detected_faces)} face(s)")
            # Diagnostics: record box size + confidence for this frame.
            get_diagnostics().record(image_rgb.shape, detected_faces, stage="detection")
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
