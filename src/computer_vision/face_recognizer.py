"""
Face Recognizer for the Intelligent Attendance Monitoring System.

Provides face recognition by comparing face embeddings using Euclidean distance.
Matches query faces against a gallery of known identities.
"""

from typing import Optional, List, Tuple
import numpy as np

import config
from src.computer_vision.face_embedder import FaceEmbedding
from src.utils.logger import get_logger


class RecognitionResult:
    """
    Represents the result of a face recognition attempt.
    
    Attributes:
        identity: The matched identity name, or "Unknown" if no match
        distance: The Euclidean distance to the matched embedding
        is_match: True if a match was found below threshold, False otherwise
    """
    
    def __init__(self, identity: str, distance: float, is_match: bool):
        """
        Initialize a recognition result.
        
        Args:
            identity: Identity name or "Unknown"
            distance: Distance to the matched embedding
            is_match: Whether this is a valid match
        """
        self.identity = identity
        self.distance = distance
        self.is_match = is_match
    
    def __repr__(self) -> str:
        match_str = "Match" if self.is_match else "No match"
        return f"RecognitionResult(identity='{self.identity}', distance={self.distance:.4f}, {match_str})"


class FaceRecognizer:
    """
    Recognizes faces by comparing embeddings against a gallery of known identities.
    
    Uses Euclidean distance for comparison with a configurable threshold.
    Returns the closest matching identity or "Unknown" if no match is found.
    """
    
    def __init__(self, threshold: Optional[float] = None):
        """
        Initialize the FaceRecognizer.
        
        Args:
            threshold: Distance threshold for recognition. Distances below this
                      are considered matches. If None, uses config.FACE_RECOGNITION_THRESHOLD
        """
        self.threshold = threshold if threshold is not None else config.FACE_RECOGNITION_THRESHOLD
        self.logger = get_logger(__name__)
        self.logger.info(f"FaceRecognizer initialized with threshold: {self.threshold}")
    
    def recognize(
        self,
        query_embedding: FaceEmbedding,
        gallery: List[Tuple[str, FaceEmbedding]]
    ) -> RecognitionResult:
        """
        Recognize a face by comparing against a gallery of known embeddings.
        
        Args:
            query_embedding: The face embedding to identify
            gallery: List of (identity, embedding) tuples representing known faces
        
        Returns:
            RecognitionResult with the matched identity, distance, and match status
        
        Raises:
            ValueError: If query_embedding is None or has invalid shape
            ValueError: If gallery contains invalid embeddings
        """
        # Validate query embedding
        if query_embedding is None:
            raise ValueError("Query embedding cannot be None")
        
        if query_embedding.encoding.shape != (128,):
            raise ValueError(
                f"Query embedding must be 128-dimensional, got {query_embedding.encoding.shape}"
            )
        
        # Handle empty gallery
        if not gallery:
            self.logger.debug("Empty gallery, returning Unknown")
            return RecognitionResult(identity="Unknown", distance=float('inf'), is_match=False)
        
        # Validate gallery embeddings
        for idx, (identity, embedding) in enumerate(gallery):
            if embedding is None:
                raise ValueError(f"Gallery embedding at index {idx} (identity '{identity}') is None")
            if embedding.encoding.shape != (128,):
                raise ValueError(
                    f"Gallery embedding at index {idx} (identity '{identity}') must be "
                    f"128-dimensional, got {embedding.encoding.shape}"
                )
        
        # Compute distances to all gallery embeddings and aggregate by identity
        from collections import defaultdict
        identity_distances = defaultdict(list)
        
        for identity, embedding in gallery:
            distance = self._compute_distance(query_embedding.encoding, embedding.encoding)
            identity_distances[identity].append(distance)
        
        # For each identity, use the minimum distance across all their samples
        # This ensures multi-sample students get the best possible match
        identity_min_distances = []
        for identity, distances in identity_distances.items():
            min_distance = min(distances)
            identity_min_distances.append((identity, min_distance))
        
        # Find the closest identity based on their best sample
        best_identity, best_distance = min(identity_min_distances, key=lambda x: x[1])
        
        # Check if distance is below threshold
        if best_distance <= self.threshold:
            self.logger.debug(
                f"Match found: {best_identity} (distance: {best_distance:.4f})"
            )
            return RecognitionResult(
                identity=best_identity,
                distance=best_distance,
                is_match=True
            )
        else:
            self.logger.debug(
                f"No match found. Closest: {best_identity} (distance: {best_distance:.4f}, "
                f"threshold: {self.threshold})"
            )
            return RecognitionResult(
                identity="Unknown",
                distance=best_distance,
                is_match=False
            )
    
    def _compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two face embeddings.
        
        Args:
            embedding1: First 128-dimensional embedding
            embedding2: Second 128-dimensional embedding
        
        Returns:
            Euclidean distance (L2 norm of difference)
        """
        return float(np.linalg.norm(embedding1 - embedding2))
