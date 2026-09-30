"""
Tests for FaceRecognizer.

Tests face recognition functionality without requiring physical camera or database.
Uses synthetic embeddings and controlled test fixtures.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.computer_vision.face_embedder import FaceEmbedding
from src.computer_vision.face_recognizer import FaceRecognizer, RecognitionResult


class TestRecognitionResult:
    """Test RecognitionResult data class."""
    
    def test_init_with_match(self):
        """Test RecognitionResult initialization with a match."""
        result = RecognitionResult(identity="Alice", distance=0.35, is_match=True)
        
        assert result.identity == "Alice"
        assert result.distance == 0.35
        assert result.is_match is True
    
    def test_init_with_no_match(self):
        """Test RecognitionResult initialization without a match."""
        result = RecognitionResult(identity="Unknown", distance=0.85, is_match=False)
        
        assert result.identity == "Unknown"
        assert result.distance == 0.85
        assert result.is_match is False
    
    def test_repr(self):
        """Test RecognitionResult string representation."""
        result = RecognitionResult(identity="Bob", distance=0.42, is_match=True)
        
        repr_str = repr(result)
        
        assert "RecognitionResult" in repr_str
        assert "Bob" in repr_str
        assert "0.42" in repr_str


class TestFaceRecognizerInitialization:
    """Test FaceRecognizer initialization."""
    
    def test_init_default_threshold(self):
        """Test FaceRecognizer initialization with default threshold."""
        recognizer = FaceRecognizer()
        
        assert recognizer is not None
        assert recognizer.threshold == 0.6  # Default from config
        assert recognizer.logger is not None
    
    def test_init_custom_threshold(self):
        """Test FaceRecognizer initialization with custom threshold."""
        recognizer = FaceRecognizer(threshold=0.5)
        
        assert recognizer.threshold == 0.5
    
    def test_init_zero_threshold(self):
        """Test FaceRecognizer initialization with zero threshold."""
        recognizer = FaceRecognizer(threshold=0.0)
        
        assert recognizer.threshold == 0.0
    
    def test_init_high_threshold(self):
        """Test FaceRecognizer initialization with high threshold."""
        recognizer = FaceRecognizer(threshold=1.5)
        
        assert recognizer.threshold == 1.5


class TestFaceRecognizerExactMatch:
    """Test FaceRecognizer with exact embedding matches."""
    
    def test_exact_match_zero_distance(self):
        """Test recognition with identical embeddings (distance = 0)."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create identical embeddings
        encoding = np.random.rand(128)
        query = FaceEmbedding(encoding.copy())
        gallery = [("Alice", FaceEmbedding(encoding.copy()))]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is True
        assert result.identity == "Alice"
        assert result.distance < 1e-10  # Essentially zero


class TestFaceRecognizerSuccessfulRecognition:
    """Test FaceRecognizer with successful matches."""
    
    def test_recognize_below_threshold(self):
        """Test recognition with distance below threshold."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create embeddings with small difference
        query_enc = np.random.rand(128)
        gallery_enc = query_enc + np.random.rand(128) * 0.01  # Small perturbation
        
        query = FaceEmbedding(query_enc)
        gallery = [("Bob", FaceEmbedding(gallery_enc))]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is True
        assert result.identity == "Bob"
        assert result.distance < 0.6
    
    def test_recognize_at_threshold(self):
        """Test recognition with distance exactly at threshold."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create embeddings with specific distance
        query_enc = np.zeros(128)
        gallery_enc = np.zeros(128)
        gallery_enc[0] = 0.6  # Distance will be exactly 0.6
        
        query = FaceEmbedding(query_enc)
        gallery = [("Charlie", FaceEmbedding(gallery_enc))]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is True  # At threshold is considered a match
        assert result.identity == "Charlie"
        assert abs(result.distance - 0.6) < 1e-10


class TestFaceRecognizerUnknownFace:
    """Test FaceRecognizer with unknown faces (no match)."""
    
    def test_recognize_above_threshold(self):
        """Test recognition with distance above threshold."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create embeddings with large difference
        query_enc = np.random.rand(128)
        gallery_enc = np.random.rand(128) + 5.0  # Large difference
        
        query = FaceEmbedding(query_enc)
        gallery = [("David", FaceEmbedding(gallery_enc))]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is False
        assert result.identity == "Unknown"
        assert result.distance > 0.6
    
    def test_recognize_just_above_threshold(self):
        """Test recognition with distance slightly above threshold."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create embeddings with distance just above threshold
        query_enc = np.zeros(128)
        gallery_enc = np.zeros(128)
        gallery_enc[0] = 0.61  # Distance = 0.61, just above threshold
        
        query = FaceEmbedding(query_enc)
        gallery = [("Eve", FaceEmbedding(gallery_enc))]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is False
        assert result.identity == "Unknown"
        assert result.distance > 0.6


class TestFaceRecognizerMultipleIdentities:
    """Test FaceRecognizer with multiple gallery entries."""
    
    def test_select_closest_match(self):
        """Test that recognizer selects the closest matching identity."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create query and multiple gallery entries
        query_enc = np.random.rand(128)
        
        # Create embeddings with different distances
        close_enc = query_enc + np.random.rand(128) * 0.01  # Very close
        medium_enc = query_enc + np.random.rand(128) * 0.3  # Medium distance
        far_enc = query_enc + np.random.rand(128) * 0.5  # Farther
        
        query = FaceEmbedding(query_enc)
        gallery = [
            ("Far", FaceEmbedding(far_enc)),
            ("Close", FaceEmbedding(close_enc)),  # Should match this
            ("Medium", FaceEmbedding(medium_enc))
        ]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is True
        assert result.identity == "Close"
    
    def test_multiple_identities_none_match(self):
        """Test with multiple gallery entries but none below threshold."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        query_enc = np.random.rand(128)
        
        # All far from query
        gallery = [
            ("Alice", FaceEmbedding(np.random.rand(128) + 5.0)),
            ("Bob", FaceEmbedding(np.random.rand(128) + 6.0)),
            ("Charlie", FaceEmbedding(np.random.rand(128) + 7.0))
        ]
        
        query = FaceEmbedding(query_enc)
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is False
        assert result.identity == "Unknown"


class TestFaceRecognizerEmptyGallery:
    """Test FaceRecognizer with empty gallery."""
    
    def test_empty_gallery_returns_unknown(self):
        """Test recognition with empty gallery."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        query = FaceEmbedding(np.random.rand(128))
        gallery = []
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is False
        assert result.identity == "Unknown"
        assert result.distance == float('inf')


class TestFaceRecognizerInputValidation:
    """Test FaceRecognizer input validation."""
    
    def test_none_query_embedding(self):
        """Test that None query embedding raises ValueError."""
        recognizer = FaceRecognizer(threshold=0.6)
        gallery = [("Alice", FaceEmbedding(np.random.rand(128)))]
        
        with pytest.raises(ValueError, match="Query embedding cannot be None"):
            recognizer.recognize(None, gallery)
    
    def test_invalid_query_dimension(self):
        """Test that invalid query dimension raises ValueError."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create invalid embedding (wrong dimension)
        invalid_enc = np.random.rand(64)  # Should be 128
        
        # FaceEmbedding constructor will raise ValueError
        with pytest.raises(ValueError, match="128-dimensional"):
            FaceEmbedding(invalid_enc)
    
    def test_none_gallery_embedding(self):
        """Test that None gallery embedding raises ValueError."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        query = FaceEmbedding(np.random.rand(128))
        gallery = [("Alice", None)]  # Invalid: None embedding
        
        with pytest.raises(ValueError, match="Gallery embedding.*is None"):
            recognizer.recognize(query, gallery)
    
    def test_invalid_gallery_dimension(self):
        """Test that invalid gallery embedding dimension raises ValueError."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        query = FaceEmbedding(np.random.rand(128))
        
        # Create gallery with invalid embedding dimension
        invalid_enc = np.random.rand(64)
        
        # FaceEmbedding constructor will catch this
        with pytest.raises(ValueError, match="128-dimensional"):
            FaceEmbedding(invalid_enc)


class TestFaceRecognizerDeterministicBehavior:
    """Test FaceRecognizer deterministic behavior with ties."""
    
    def test_equal_distance_selects_first(self):
        """Test that equal distances select the first occurrence."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create query and two gallery entries with identical distance
        query_enc = np.zeros(128)
        
        # Both have distance = 0.3
        enc1 = np.zeros(128)
        enc1[0] = 0.3
        enc2 = np.zeros(128)
        enc2[0] = 0.3
        
        query = FaceEmbedding(query_enc)
        gallery = [
            ("First", FaceEmbedding(enc1)),
            ("Second", FaceEmbedding(enc2))
        ]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is True
        assert result.identity == "First"  # First occurrence
        assert abs(result.distance - 0.3) < 1e-10


class TestFaceRecognizerCustomThreshold:
    """Test FaceRecognizer with custom thresholds."""
    
    def test_strict_threshold(self):
        """Test with strict (low) threshold."""
        recognizer = FaceRecognizer(threshold=0.3)
        
        # Distance = 0.4 (above 0.3 threshold)
        query_enc = np.zeros(128)
        gallery_enc = np.zeros(128)
        gallery_enc[0] = 0.4
        
        query = FaceEmbedding(query_enc)
        gallery = [("Alice", FaceEmbedding(gallery_enc))]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is False
        assert result.identity == "Unknown"
    
    def test_lenient_threshold(self):
        """Test with lenient (high) threshold."""
        recognizer = FaceRecognizer(threshold=1.0)
        
        # Distance = 0.8 (below 1.0 threshold)
        query_enc = np.zeros(128)
        gallery_enc = np.zeros(128)
        gallery_enc[0] = 0.8
        
        query = FaceEmbedding(query_enc)
        gallery = [("Bob", FaceEmbedding(gallery_enc))]
        
        result = recognizer.recognize(query, gallery)
        
        assert result.is_match is True
        assert result.identity == "Bob"


class TestFaceRecognizerDistanceComputation:
    """Test Euclidean distance computation."""
    
    def test_distance_computation_orthogonal(self):
        """Test distance with orthogonal vectors."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Create orthogonal unit vectors
        query_enc = np.zeros(128)
        query_enc[0] = 1.0
        
        gallery_enc = np.zeros(128)
        gallery_enc[1] = 1.0
        
        query = FaceEmbedding(query_enc)
        gallery = [("Orthogonal", FaceEmbedding(gallery_enc))]
        
        result = recognizer.recognize(query, gallery)
        
        # Distance = sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414
        assert abs(result.distance - np.sqrt(2)) < 1e-10
    
    def test_distance_computation_known_value(self):
        """Test distance with known vectors for verification."""
        recognizer = FaceRecognizer(threshold=0.6)
        
        # Simple vectors with known distance
        query_enc = np.zeros(128)
        gallery_enc = np.zeros(128)
        gallery_enc[0] = 3.0
        gallery_enc[1] = 4.0
        # Distance = sqrt(3^2 + 4^2) = 5.0
        
        query = FaceEmbedding(query_enc)
        gallery = [("Known", FaceEmbedding(gallery_enc))]
        
        result = recognizer.recognize(query, gallery)
        
        assert abs(result.distance - 5.0) < 1e-10
