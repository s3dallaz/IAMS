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
        distance: The Euclidean distance to the match, or to the closest
            conflicting identity when no confident match was found.
            float('inf') for an empty gallery
        is_match: True if a confident, unambiguous match was found
        second_best_distance: Distance to the closest embedding that belongs to
            a DIFFERENT identity, or float('inf') when the gallery holds only
            one distinct identity. Used by the ambiguity check
        reason: Short code explaining the outcome - "matched",
            "above_threshold", "ambiguous_match" or "empty_gallery"
    """

    def __init__(
        self,
        identity: str,
        distance: float,
        is_match: bool,
        second_best_distance: float = float('inf'),
        reason: str = ""
    ):
        """
        Initialize a recognition result.

        Args:
            identity: Identity name or "Unknown"
            distance: Distance to the matched embedding
            is_match: Whether this is a valid match
            second_best_distance: Distance to the runner-up identity, if any
            reason: Short code describing why the outcome was reached
        """
        self.identity = identity
        self.distance = distance
        self.is_match = is_match
        self.second_best_distance = second_best_distance
        self.reason = reason
    
    def __repr__(self) -> str:
        match_str = "Match" if self.is_match else "No match"
        return f"RecognitionResult(identity='{self.identity}', distance={self.distance:.4f}, {match_str})"


class FaceRecognizer:
    """
    Recognizes faces by comparing embeddings against a gallery of known identities.
    
    Uses Euclidean distance for comparison with a configurable threshold.
    Matches closer than the threshold are additionally required to be
    clearly ahead of the runner-up identity (FACE_RECOGNITION_MARGIN),
    otherwise they are treated as "Unknown".
    """

    def __init__(self, threshold: Optional[float] = None, margin: Optional[float] = None):
        """
        Initialize the FaceRecognizer.

        Args:
            threshold: Distance threshold for recognition. Distances below this
                      are considered matches. If None, uses config.FACE_RECOGNITION_THRESHOLD
            margin: Minimum required distance gap to a DIFFERENT enrolled
                    identity. Below this the match is ambiguous. If None, uses
                    config.FACE_RECOGNITION_MARGIN
        """
        self.threshold = threshold if threshold is not None else config.FACE_RECOGNITION_THRESHOLD
        self.margin = margin if margin is not None else config.FACE_RECOGNITION_MARGIN
        self.logger = get_logger(__name__)
        self.logger.info(
            f"FaceRecognizer initialized with threshold: {self.threshold}, "
            f"ambiguous-match margin: {self.margin}"
        )
    
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
            return RecognitionResult(
                identity="Unknown",
                distance=float('inf'),
                is_match=False,
                reason="empty_gallery"
            )
        
        # Validate gallery embeddings
        for idx, (identity, embedding) in enumerate(gallery):
            if embedding is None:
                raise ValueError(f"Gallery embedding at index {idx} (identity '{identity}') is None")
            if embedding.encoding.shape != (128,):
                raise ValueError(
                    f"Gallery embedding at index {idx} (identity '{identity}') must be "
                    f"128-dimensional, got {embedding.encoding.shape}"
                )
        
        # Compute distances to all gallery embeddings
        distances = []
        for identity, embedding in gallery:
            distance = self._compute_distance(query_embedding.encoding, embedding.encoding)
            distances.append((identity, distance))
        
        # Find the closest match (deterministic: first occurrence if tied)
        best_identity, best_distance = min(distances, key=lambda x: x[1])

        # Best distance belonging to a DIFFERENT identity. Embeddings of the
        # same student are one identity, so the runner-up must come from
        # someone else - do not fabricate a conflict from the same person.
        second_best_distance = float('inf')
        for identity, distance in distances:
            if identity == best_identity:
                continue
            if distance < second_best_distance:
                second_best_distance = distance

        if best_distance > self.threshold:
            self.logger.debug(
                f"No match. Closest: {best_identity} (distance: {best_distance:.4f}, "
                f"threshold: {self.threshold}, second best: {second_best_distance:.4f})"
            )
            return RecognitionResult(
                identity="Unknown",
                distance=best_distance,
                is_match=False,
                second_best_distance=second_best_distance,
                reason="above_threshold"
            )

        # Ambiguity guard: someone else is nearly as close as the winner.
        # Compare against the runner-up only when it exists (float('inf')
        # means a single-identity gallery, so the guard is skipped).
        if second_best_distance - best_distance < self.margin:
            self.logger.debug(
                f"Ambiguous match rejected: {best_identity} "
                f"(distance: {best_distance:.4f}) vs {second_best_distance:.4f} "
                f"(margin: {self.margin})"
            )
            return RecognitionResult(
                identity="Unknown",
                distance=best_distance,
                is_match=False,
                second_best_distance=second_best_distance,
                reason="ambiguous_match"
            )

        self.logger.debug(
            f"Match found: {best_identity} (distance: {best_distance:.4f}, "
            f"second best: {second_best_distance:.4f}, margin: {self.margin})"
        )
        return RecognitionResult(
            identity=best_identity,
            distance=best_distance,
            is_match=True,
            second_best_distance=second_best_distance,
            reason="matched"
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
