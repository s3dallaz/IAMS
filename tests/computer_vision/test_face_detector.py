"""
Tests for FaceDetector.

Tests face detection functionality without requiring a physical camera.
Uses generated test images and mocking for MediaPipe detector.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config
from src.computer_vision.face_detector import FaceDetector, DetectedFace


class TestDetectedFace:
    """Test DetectedFace data class."""
    
    def test_init(self):
        """Test DetectedFace initialization."""
        bbox = (100, 50, 200, 250)
        confidence = 0.95
        
        face = DetectedFace(bbox, confidence)
        
        assert face.bbox == bbox
        assert face.confidence == confidence
    
    def test_repr(self):
        """Test DetectedFace string representation."""
        face = DetectedFace((10, 20, 30, 40), 0.876)
        
        repr_str = repr(face)
        
        assert "DetectedFace" in repr_str
        assert "0.876" in repr_str


class TestFaceDetectorInitialization:
    """Test FaceDetector initialization."""
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_init_with_default_confidence(self, mock_exists, mock_create):
        """Test initialization with default confidence from config."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        detector = FaceDetector()
        
        assert detector.min_detection_confidence == config.FACE_DETECTION_CONFIDENCE
        mock_create.assert_called_once()
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_init_with_custom_confidence(self, mock_exists, mock_create):
        """Test initialization with custom confidence threshold."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        custom_confidence = 0.85
        detector = FaceDetector(min_detection_confidence=custom_confidence)
        
        assert detector.min_detection_confidence == custom_confidence
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_init_exception_handling(self, mock_exists, mock_create):
        """Test exception handling during initialization."""
        mock_exists.return_value = True
        mock_create.side_effect = Exception("MediaPipe initialization error")
        
        detector = FaceDetector()
        
        assert detector._detector is None


class TestFaceDetectorDetection:
    """Test face detection functionality."""
    
    def _create_mock_detection(self, origin_x, origin_y, width, height, confidence):
        """Helper to create a mock MediaPipe detection."""
        detection = Mock()
        detection.bounding_box = Mock()
        detection.bounding_box.origin_x = origin_x
        detection.bounding_box.origin_y = origin_y
        detection.bounding_box.width = width
        detection.bounding_box.height = height
        detection.categories = [Mock()]
        detection.categories[0].score = confidence
        return detection
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_single_face(self, mock_exists, mock_create):
        """Test detection of a single face."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        # Create mock detection result (pixel coordinates)
        mock_detection = self._create_mock_detection(
            origin_x=100, origin_y=120, width=150, height=180, confidence=0.95
        )
        mock_result = Mock()
        mock_result.detections = [mock_detection]
        mock_detector.detect.return_value = mock_result
        
        # Create test image (640x480)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector()
        faces = detector.detect_faces(test_image)
        
        assert len(faces) == 1
        assert isinstance(faces[0], DetectedFace)
        assert faces[0].bbox == (100, 120, 150, 180)
        assert faces[0].confidence == 0.95
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_multiple_faces(self, mock_exists, mock_create):
        """Test detection of multiple faces."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        # Create multiple mock detections
        mock_detection1 = self._create_mock_detection(
            origin_x=50, origin_y=60, width=100, height=120, confidence=0.92
        )
        mock_detection2 = self._create_mock_detection(
            origin_x=300, origin_y=200, width=120, height=140, confidence=0.88
        )
        mock_result = Mock()
        mock_result.detections = [mock_detection1, mock_detection2]
        mock_detector.detect.return_value = mock_result
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector()
        faces = detector.detect_faces(test_image)
        
        assert len(faces) == 2
        assert all(isinstance(face, DetectedFace) for face in faces)
        assert faces[0].confidence == 0.92
        assert faces[1].confidence == 0.88
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_no_faces(self, mock_exists, mock_create):
        """Test detection when no faces are present."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        # No detections
        mock_result = Mock()
        mock_result.detections = None
        mock_detector.detect.return_value = mock_result
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector()
        faces = detector.detect_faces(test_image)
        
        assert len(faces) == 0
        assert faces == []


class TestFaceDetectorInputValidation:
    """Test input validation and error handling."""
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_with_none_image(self, mock_exists, mock_create):
        """Test detection with None as input."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        detector = FaceDetector()
        faces = detector.detect_faces(None)
        
        assert faces == []
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_with_empty_image(self, mock_exists, mock_create):
        """Test detection with empty image."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        empty_image = np.array([])
        
        detector = FaceDetector()
        faces = detector.detect_faces(empty_image)
        
        assert faces == []
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_with_invalid_type(self, mock_exists, mock_create):
        """Test detection with invalid input type."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        detector = FaceDetector()
        faces = detector.detect_faces("not an image")
        
        assert faces == []
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_when_detector_not_initialized(self, mock_exists, mock_create):
        """Test detection when detector failed to initialize."""
        mock_exists.return_value = True
        mock_create.side_effect = Exception("Init failed")
        
        detector = FaceDetector()
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(test_image)
        
        assert faces == []
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_exception_during_processing(self, mock_exists, mock_create):
        """Test exception handling during face detection."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        mock_detector.detect.side_effect = Exception("Processing error")
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector()
        faces = detector.detect_faces(test_image)
        
        assert faces == []


class TestFaceDetectorImageFormats:
    """Test handling of different image formats."""
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_with_bgr_image(self, mock_exists, mock_create):
        """Test detection with BGR image (OpenCV default)."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        mock_result = Mock()
        mock_result.detections = []
        mock_detector.detect.return_value = mock_result
        
        # BGR image (3 channels)
        bgr_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector()
        faces = detector.detect_faces(bgr_image)
        
        # Should process without error
        assert isinstance(faces, list)
        mock_detector.detect.assert_called_once()
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_detect_with_grayscale_image(self, mock_exists, mock_create):
        """Test detection with grayscale image."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        mock_result = Mock()
        mock_result.detections = []
        mock_detector.detect.return_value = mock_result
        
        # Grayscale image (2D)
        gray_image = np.zeros((480, 640), dtype=np.uint8)
        
        detector = FaceDetector()
        faces = detector.detect_faces(gray_image)
        
        # Should convert and process without error
        assert isinstance(faces, list)
        mock_detector.detect.assert_called_once()


class TestFaceDetectorBoundingBoxClamping:
    """Test bounding box clamping to image bounds."""
    
    def _create_mock_detection(self, origin_x, origin_y, width, height, confidence):
        """Helper to create a mock MediaPipe detection."""
        detection = Mock()
        detection.bounding_box = Mock()
        detection.bounding_box.origin_x = origin_x
        detection.bounding_box.origin_y = origin_y
        detection.bounding_box.width = width
        detection.bounding_box.height = height
        detection.categories = [Mock()]
        detection.categories[0].score = confidence
        return detection
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_bbox_clamping_negative_coordinates(self, mock_exists, mock_create):
        """Test bounding box clamping for negative coordinates."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        # Face partially outside image (negative coords)
        mock_detection = self._create_mock_detection(
            origin_x=-10, origin_y=-5, width=100, height=120, confidence=0.8
        )
        mock_result = Mock()
        mock_result.detections = [mock_detection]
        mock_detector.detect.return_value = mock_result
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector()
        faces = detector.detect_faces(test_image)
        
        assert len(faces) == 1
        # Coordinates should be clamped to 0
        x, y, w, h = faces[0].bbox
        assert x >= 0
        assert y >= 0
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_bbox_clamping_exceeds_bounds(self, mock_exists, mock_create):
        """Test bounding box clamping when exceeding image bounds."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        # Face extending beyond image bounds
        mock_detection = self._create_mock_detection(
            origin_x=600, origin_y=450, width=100, height=100, confidence=0.85
        )
        mock_result = Mock()
        mock_result.detections = [mock_detection]
        mock_detector.detect.return_value = mock_result
        
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector()
        faces = detector.detect_faces(test_image)
        
        assert len(faces) == 1
        x, y, w, h = faces[0].bbox
        # Width and height should be adjusted to stay within bounds
        assert x + w <= 640
        assert y + h <= 480


class TestFaceDetectorResourceManagement:
    """Test resource cleanup and management."""
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_close(self, mock_exists, mock_create):
        """Test closing the detector."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        detector = FaceDetector()
        assert detector._detector is not None
        
        detector.close()
        
        mock_detector.close.assert_called_once()
        assert detector._detector is None
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_close_when_not_initialized(self, mock_exists, mock_create):
        """Test closing when detector not initialized."""
        mock_exists.return_value = True
        mock_create.side_effect = Exception("Init failed")
        
        detector = FaceDetector()
        assert detector._detector is None
        
        # Should not raise exception
        detector.close()
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_close_multiple_times(self, mock_exists, mock_create):
        """Test closing detector multiple times."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        detector = FaceDetector()
        detector.close()
        detector.close()  # Should not raise exception
        
        mock_detector.close.assert_called_once()
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_context_manager(self, mock_exists, mock_create):
        """Test context manager usage."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        with FaceDetector() as detector:
            assert detector._detector is not None
        
        mock_detector.close.assert_called_once()
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_context_manager_with_exception(self, mock_exists, mock_create):
        """Test context manager closes detector even with exception."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        with pytest.raises(ValueError):
            with FaceDetector() as detector:
                raise ValueError("Test exception")
        
        mock_detector.close.assert_called_once()


class TestFaceDetectorConfiguration:
    """Test configuration handling."""
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_uses_config_threshold(self, mock_exists, mock_create):
        """Test that detector uses configured threshold."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        detector = FaceDetector()
        
        assert detector.min_detection_confidence == config.FACE_DETECTION_CONFIDENCE
        assert detector.min_detection_confidence == 0.7
    
    @patch('src.computer_vision.face_detector.vision.FaceDetector.create_from_options')
    @patch('src.computer_vision.face_detector.Path.exists')
    def test_custom_threshold_overrides_config(self, mock_exists, mock_create):
        """Test that custom threshold overrides config."""
        mock_exists.return_value = True
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        custom_threshold = 0.5
        detector = FaceDetector(min_detection_confidence=custom_threshold)
        
        assert detector.min_detection_confidence == custom_threshold
        assert detector.min_detection_confidence != config.FACE_DETECTION_CONFIDENCE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
