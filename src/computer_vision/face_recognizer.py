"""
Face Recognizer for the Intelligent Attendance Monitoring System.

Provides face recognition by comparing face embeddings using Euclidean distance.
Matches query faces against a gallery of known identities.
"""

from collections.abc import Sequence
from typing import Iterable, Optional, List, Tuple, Union
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


class FaceGallery(Sequence):
    """
    Gallery of known faces prepared for fast matching.

    Behaves like the list of (identity, FaceEmbedding) tuples it is built
    from, and also keeps every embedding in one (N, 128) matrix with an
    identity code per row, so recognize() computes all distances in one
    NumPy call instead of a Python loop (~27 ms -> <1 ms per face at 11k
    samples). Build it once per gallery change, not per frame -
    RealTimeAttendanceManager.set_gallery() does.
    """

    def __init__(self, entries: Iterable[Tuple[str, FaceEmbedding]]):
        """
        Raises:
            ValueError: If an embedding is None or not 128-dimensional
        """
        self._entries = tuple(entries)
        for idx, (identity, embedding) in enumerate(self._entries):
            if embedding is None:
                raise ValueError(f"Gallery embedding at index {idx} (identity '{identity}') is None")
            if embedding.encoding.shape != (128,):
                raise ValueError(
                    f"Gallery embedding at index {idx} (identity '{identity}') must be "
                    f"128-dimensional, got {embedding.encoding.shape}"
                )

        code_of = {}
        self.codes = np.array([code_of.setdefault(identity, len(code_of)) for identity, _ in self._entries],
                              dtype=np.int64)
        self.identities: List[str] = list(code_of)
        if self._entries:
            self.matrix = np.stack([embedding.encoding for _, embedding in self._entries]).astype(np.float64)
        else:
            self.matrix = np.zeros((0, 128), dtype=np.float64)
        self.sq_norms = np.einsum("ij,ij->i", self.matrix, self.matrix)
        self.max_sq_norm = float(self.sq_norms.max()) if len(self.sq_norms) else 0.0

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

    def same_content(self, other: "FaceGallery") -> bool:
        """True if `other` holds the same identities and embeddings in the same order."""
        return (isinstance(other, FaceGallery) and len(self) == len(other)
                and self.identities == other.identities
                and np.array_equal(self.codes, other.codes)
                and np.array_equal(self.matrix, other.matrix))

    def approx_sq_distances(self, query: np.ndarray) -> np.ndarray:
        """Squared distances to every entry, |m|^2 - 2 m.q + |q|^2 (rounding error ~1e-15 * scale)."""
        query = np.asarray(query, dtype=np.float64)
        sq = self.sq_norms - 2.0 * (self.matrix @ query) + float(query @ query)
        return np.maximum(sq, 0.0, out=sq)


class FaceRecognizer:
    """
    Recognizes faces by comparing embeddings against a gallery of known identities.
    
    Uses Euclidean distance for comparison with a configurable threshold.
    Matches closer than the threshold are additionally required to be
    clearly ahead of the runner-up identity (FACE_RECOGNITION_MARGIN),
    otherwise they are treated as "Unknown".
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
        backend: Optional[str] = None
    ):
        """
        Initialize the FaceRecognizer.

        Args:
            threshold: Distance threshold for recognition. Distances below this
                      are considered matches. If None, uses the backend's
                      config value (FACE_RECOGNITION_THRESHOLD for dlib,
                      SFACE_RECOGNITION_THRESHOLD for SFace)
            margin: Minimum required distance gap to a DIFFERENT enrolled
                    identity. Below this the match is ambiguous. If None, uses
                    the backend's config value
            backend: Embedding model the distances come from ("sface" or
                    "dlib"); None = config.FACE_RECOGNITION_BACKEND
        """
        self.backend = (backend or config.FACE_RECOGNITION_BACKEND).lower()
        if self.backend == "sface":
            default_threshold, default_margin = config.SFACE_RECOGNITION_THRESHOLD, config.SFACE_RECOGNITION_MARGIN
        else:
            default_threshold, default_margin = config.FACE_RECOGNITION_THRESHOLD, config.FACE_RECOGNITION_MARGIN
        self.threshold = threshold if threshold is not None else default_threshold
        self.margin = margin if margin is not None else default_margin
        self.logger = get_logger(__name__)
        self.logger.info(
            f"FaceRecognizer initialized with threshold: {self.threshold}, "
            f"ambiguous-match margin: {self.margin}"
        )
    
    def recognize(
        self,
        query_embedding: FaceEmbedding,
        gallery: Union[FaceGallery, List[Tuple[str, FaceEmbedding]]]
    ) -> RecognitionResult:
        """
        Recognize a face by comparing against a gallery of known embeddings.
        
        Args:
            query_embedding: The face embedding to identify
            gallery: FaceGallery, or a list of (identity, embedding) tuples
                representing known faces (prepared on every call - pass a
                FaceGallery when recognising many faces against one gallery)
        
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
        
        if not isinstance(gallery, FaceGallery):
            gallery = FaceGallery(gallery)  # validates the embeddings

        # All distances at once. The fast formula is only used to find the
        # near-ties; the exact distance of those few entries (the same
        # formula as _compute_distance) decides, so results, thresholds and
        # tie-breaking are identical to comparing every entry one by one.
        query = query_embedding.encoding
        approx = gallery.approx_sq_distances(query)
        tolerance = 1e-5 * (float(np.dot(query, query)) + gallery.max_sq_norm)

        # Closest match (deterministic: first occurrence if tied)
        best_index, best_distance = self._closest(query, gallery, approx, tolerance)
        best_identity = gallery[best_index][0]

        # Best distance belonging to a DIFFERENT identity. Embeddings of the
        # same student are one identity, so the runner-up must come from
        # someone else - do not fabricate a conflict from the same person.
        others = np.where(gallery.codes != gallery.codes[best_index], approx, np.inf)
        _, second_best_distance = self._closest(query, gallery, others, tolerance)

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
    
    def _closest(
        self,
        query: np.ndarray,
        gallery: FaceGallery,
        approx_sq: np.ndarray,
        tolerance: float
    ) -> Tuple[Optional[int], float]:
        """
        Index and exact distance of the closest entry; the first one wins a tie.

        Entries whose approximate squared distance is within `tolerance` of
        the lowest are compared by exact distance; entries marked np.inf are
        skipped. (None, inf) when every entry is skipped.
        """
        lowest = approx_sq.min()
        if not np.isfinite(lowest):
            return None, float('inf')
        best_index, best_distance = None, float('inf')
        for index in np.flatnonzero(approx_sq <= lowest + tolerance):
            distance = self._compute_distance(query, gallery[index][1].encoding)
            if distance < best_distance:
                best_index, best_distance = int(index), distance
        return best_index, best_distance

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
