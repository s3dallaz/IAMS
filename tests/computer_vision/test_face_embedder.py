"""
Tests for FaceEmbedder.

Tests face embedding generation functionality without requiring a physical camera.
Uses generated test images and controlled test fixtures.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.computer_vision.face_embedder import (
    FaceEmbedder, 
    FaceEmbedding, 
    FaceEmbeddingError
)


class TestFaceEmbedding:
    """Test FaceEmbedding data class."""
    
    def test_init_with_valid_encoding(self):
        """Test FaceEmbedding initialization with valid 128-dimensional encoding."""
        encoding = np.random.rand(128)
        
        embedding = FaceEmbedding(encoding)
        
        assert embedding.encoding.shape == (128,)
        assert np.array_equal(embedding.encoding, encoding)
    
    def test_init_with_invalid_shape(self):
        """Test FaceEmbedding initialization with invalid shape."""
        invalid_encoding = np.random.rand(64)  # Wrong dimension
        
        with pytest.raises(ValueError, match="must be 128-dimensional"):
            FaceEmbedding(invalid_encoding)
    
    def test_init_with_2d_array(self):
        """Test FaceEmbedding initialization with 2D array."""
        invalid_encoding = np.random.rand(128, 1)
        
        with pytest.raises(ValueError, match="must be 128-dimensional"):
            FaceEmbedding(invalid_encoding)
    
    def test_repr(self):
        """Test FaceEmbedding string representation."""
        encoding = np.random.rand(128)
        embedding = FaceEmbedding(encoding)
        
        repr_str = repr(embedding)
        
        assert "FaceEmbedding" in repr_str
        assert "128" in repr_str


class TestFaceEmbedderInitialization:
    """Test FaceEmbedder initialization."""
    
    def test_init(self):
        """Test FaceEmbedder initialization."""
        embedder = FaceEmbedder()
        
        assert embedder is not None
        assert embedder.logger is not None


class TestFaceEmbedderValidSingleFace:
    """Test FaceEmbedder with valid single-face input."""
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_embedding_single_face(self, mock_encodings):
        """Test successful embedding generation with single face."""
        # Mock face_recognition to return a single 128-dimensional encoding
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(test_image)
        
        assert result is not None
        assert isinstance(result, FaceEmbedding)
        assert result.encoding.shape == (128,)
        assert np.array_equal(result.encoding, mock_encoding)
        mock_encodings.assert_called_once()
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_embedding_output_shape(self, mock_encodings):
        """Test that embedding output has correct shape."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        test_image = np.zeros((150, 150, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(test_image)
        
        assert result.encoding.shape == (128,)
        assert result.encoding.dtype == np.float64
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_embedding_with_rgb_image(self, mock_encodings):
        """Test embedding generation with RGB image."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        rgb_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(rgb_image)
        
        assert result is not None
        assert isinstance(result, FaceEmbedding)


class TestFaceEmbedderInputValidation:
    """Test FaceEmbedder input validation."""
    
    def test_generate_with_none_input(self):
        """Test embedding generation with None input."""
        embedder = FaceEmbedder()
        
        with pytest.raises(FaceEmbeddingError, match="cannot be None"):
            embedder.generate_embedding(None)
    
    def test_generate_with_invalid_type(self):
        """Test embedding generation with invalid input type."""
        embedder = FaceEmbedder()
        
        with pytest.raises(FaceEmbeddingError, match="must be a numpy array"):
            embedder.generate_embedding([1, 2, 3])
    
    def test_generate_with_empty_array(self):
        """Test embedding generation with empty array."""
        embedder = FaceEmbedder()
        empty_image = np.array([])
        
        with pytest.raises(FaceEmbeddingError, match="cannot be empty"):
            embedder.generate_embedding(empty_image)
    
    def test_generate_with_invalid_shape(self):
        """Test embedding generation with invalid image shape."""
        embedder = FaceEmbedder()
        # 1D array is not valid
        invalid_image = np.random.rand(100)
        
        with pytest.raises(FaceEmbeddingError, match="Invalid image shape"):
            embedder.generate_embedding(invalid_image)
    
    def test_generate_with_4d_array(self):
        """Test embedding generation with 4D array."""
        embedder = FaceEmbedder()
        # 4D array is not valid for single image
        invalid_image = np.random.rand(10, 200, 200, 3)
        
        with pytest.raises(FaceEmbeddingError, match="Invalid image shape"):
            embedder.generate_embedding(invalid_image)


class TestFaceEmbedderNoFaceDetected:
    """Test FaceEmbedder when no face is detected."""
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_no_face(self, mock_encodings):
        """Test embedding generation when no face is detected."""
        # face_recognition returns empty list when no face found
        mock_encodings.return_value = []
        
        embedder = FaceEmbedder()
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(test_image)
        
        assert result is None
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_blank_image(self, mock_encodings):
        """Test embedding generation with blank image (no face)."""
        mock_encodings.return_value = []
        
        embedder = FaceEmbedder()
        blank_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = embedder.generate_embedding(blank_image)
        
        assert result is None


class TestFaceEmbedderMultipleFaces:
    """Test FaceEmbedder with multiple faces (ambiguous input)."""
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_multiple_faces(self, mock_encodings):
        """Test embedding generation rejects multiple faces."""
        # Simulate detecting 2 faces
        mock_encoding1 = np.random.rand(128)
        mock_encoding2 = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding1, mock_encoding2]
        
        embedder = FaceEmbedder()
        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        
        with pytest.raises(FaceEmbeddingError, match="Expected single face but detected 2"):
            embedder.generate_embedding(test_image)
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_three_faces(self, mock_encodings):
        """Test embedding generation rejects three faces."""
        mock_encodings.return_value = [
            np.random.rand(128),
            np.random.rand(128),
            np.random.rand(128)
        ]
        
        embedder = FaceEmbedder()
        test_image = np.zeros((600, 600, 3), dtype=np.uint8)
        
        with pytest.raises(FaceEmbeddingError, match="Expected single face but detected 3"):
            embedder.generate_embedding(test_image)


class TestFaceEmbedderImageFormats:
    """Test FaceEmbedder with different image formats."""
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_bgr_image(self, mock_encodings):
        """Test embedding generation with BGR image (OpenCV default)."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        bgr_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(bgr_image)
        
        assert result is not None
        # Verify BGR to RGB conversion occurred
        mock_encodings.assert_called_once()
        call_args = mock_encodings.call_args[0][0]
        # The conversion should have happened (BGR -> RGB reversal)
        assert call_args.shape == (200, 200, 3)
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_grayscale_image(self, mock_encodings):
        """Test embedding generation with grayscale image."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        gray_image = np.zeros((200, 200), dtype=np.uint8)
        
        result = embedder.generate_embedding(gray_image)
        
        assert result is not None
        # Verify grayscale to RGB conversion occurred
        mock_encodings.assert_called_once()
        call_args = mock_encodings.call_args[0][0]
        assert call_args.shape == (200, 200, 3)
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_rgba_image(self, mock_encodings):
        """Test embedding generation with RGBA image."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        rgba_image = np.zeros((200, 200, 4), dtype=np.uint8)
        
        result = embedder.generate_embedding(rgba_image)
        
        assert result is not None
        # Verify RGBA to RGB conversion occurred
        mock_encodings.assert_called_once()
        call_args = mock_encodings.call_args[0][0]
        assert call_args.shape == (200, 200, 3)
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_invalid_channels(self, mock_encodings):
        """Test embedding generation with invalid number of channels."""
        embedder = FaceEmbedder()
        # 5 channels is not standard
        invalid_image = np.zeros((200, 200, 5), dtype=np.uint8)
        
        with pytest.raises(FaceEmbeddingError, match="Failed to convert image to RGB"):
            embedder.generate_embedding(invalid_image)


class TestFaceEmbedderErrorHandling:
    """Test FaceEmbedder error handling."""
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_face_recognition_error(self, mock_encodings):
        """Test handling of face_recognition library errors."""
        mock_encodings.side_effect = RuntimeError("face_recognition internal error")
        
        embedder = FaceEmbedder()
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        with pytest.raises(FaceEmbeddingError, match="Failed to generate face encoding"):
            embedder.generate_embedding(test_image)
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_preserves_custom_exceptions(self, mock_encodings):
        """Test that FaceEmbeddingError exceptions are preserved."""
        embedder = FaceEmbedder()
        
        # Test that validation errors are raised correctly
        with pytest.raises(FaceEmbeddingError, match="cannot be None"):
            embedder.generate_embedding(None)


class TestFaceEmbedderDifferentImageSizes:
    """Test FaceEmbedder with different image sizes."""
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_small_image(self, mock_encodings):
        """Test embedding generation with small image."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        small_image = np.zeros((50, 50, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(small_image)
        
        assert result is not None
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_large_image(self, mock_encodings):
        """Test embedding generation with large image."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        large_image = np.zeros((1920, 1080, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(large_image)
        
        assert result is not None
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_non_square_image(self, mock_encodings):
        """Test embedding generation with non-square image."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        rect_image = np.zeros((300, 150, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(rect_image)
        
        assert result is not None


class TestFaceEmbedderDataTypes:
    """Test FaceEmbedder with different data types."""
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_uint8_image(self, mock_encodings):
        """Test embedding generation with uint8 image."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        uint8_image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(uint8_image)
        
        assert result is not None
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_float32_image(self, mock_encodings):
        """Test embedding generation with float32 image."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        float32_image = np.zeros((200, 200, 3), dtype=np.float32)
        
        result = embedder.generate_embedding(float32_image)
        
        assert result is not None


class TestFaceEmbedderArrayContiguity:
    """Test FaceEmbedder with non-contiguous arrays (regression tests)."""
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_non_contiguous_array(self, mock_encodings):
        """Test embedding generation with non-contiguous array from slicing."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        
        # Create a non-contiguous array by slicing
        full_image = np.zeros((400, 400, 3), dtype=np.uint8)
        # Slice creates a view which may be non-contiguous
        sliced_image = full_image[100:300, 100:300, :]
        
        result = embedder.generate_embedding(sliced_image)
        
        assert result is not None
        # Verify that face_recognition was called with a contiguous array
        mock_encodings.assert_called_once()
        call_arg = mock_encodings.call_args[0][0]
        assert call_arg.flags['C_CONTIGUOUS'], "Array passed to face_recognition should be contiguous"
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_reversed_channels(self, mock_encodings):
        """Test embedding generation with BGR to RGB reversal (non-contiguous)."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        
        # BGR image (typical from OpenCV)
        bgr_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        result = embedder.generate_embedding(bgr_image)
        
        assert result is not None
        # Verify the array passed to face_recognition is contiguous
        mock_encodings.assert_called_once()
        call_arg = mock_encodings.call_args[0][0]
        assert call_arg.flags['C_CONTIGUOUS'], "RGB array should be made contiguous"
    
    @patch('src.computer_vision.face_embedder.face_recognition.face_encodings')
    def test_generate_with_image_crop(self, mock_encodings):
        """Test embedding generation with face crop (typical use case)."""
        mock_encoding = np.random.rand(128)
        mock_encodings.return_value = [mock_encoding]
        
        embedder = FaceEmbedder()
        
        # Simulate typical face detection scenario
        full_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Crop face region (creates non-contiguous view)
        x, y, w, h = 150, 100, 200, 200
        face_crop = full_image[y:y+h, x:x+w]
        
        # Verify it's non-contiguous before processing
        assert not face_crop.flags['C_CONTIGUOUS'], "Face crop should be non-contiguous"
        
        result = embedder.generate_embedding(face_crop)
        
        assert result is not None
        # After processing, the array passed should be contiguous
        call_arg = mock_encodings.call_args[0][0]
        assert call_arg.flags['C_CONTIGUOUS'], "Should convert to contiguous for dlib"
