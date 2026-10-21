"""
In-memory index of enrolled face embeddings for the duplicate-face checks.

A new face sample is compared with every enrolled face before it is saved:
- closer than DUPLICATE_BLOCK_FRACTION x the recognition threshold to ANOTHER
  student: almost certainly the same person (one person under two IDs) and
  refused, because live attendance then rejects that face as ambiguous and
  marks neither student;
- closer than SIMILAR_WARN_FRACTION x the threshold to another student:
  similar, saved only after the user confirms;
- not within the threshold of the student's OWN earlier samples: maybe the
  wrong person, saved only after the user confirms.

The same rules are used by the manual enrollment (upload / camera), the ZIP
bulk import and the registry-wide duplicate scan.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# A face closer than this fraction of the recognition threshold to ANOTHER
# student's face is treated as the same person (e.g. one photo under two IDs)
DUPLICATE_BLOCK_FRACTION = 0.5
# Closer than this fraction it is "similar" and needs confirmation. Not the
# full threshold: in a gallery of 65,000 SFace samples the closest other
# student is within the threshold for ~95% of faces, while two photos of the
# same person are within 0.75 x threshold ~85% of the time (median 0.6 x).
SIMILAR_WARN_FRACTION = 0.75


@dataclass
class FaceCheck:
    """How a new face sample compares with the enrolled faces."""
    status: str                          # ok | duplicate | similar | mismatch
    other_student_id: Optional[str] = None
    other_distance: float = float("inf")
    own_distance: Optional[float] = None  # closest of the student's own samples

    @property
    def needs_confirmation(self) -> bool:
        return self.status in ("similar", "mismatch")


def classify_face(own: Optional[float], other_id: Optional[str], other_distance: float,
                  threshold: float) -> FaceCheck:
    """The FaceCheck for the distances returned by FaceIndex.compare()."""
    check = FaceCheck("ok", other_id, other_distance, own)
    if other_id is not None and other_distance < DUPLICATE_BLOCK_FRACTION * threshold:
        check.status = "duplicate"
    elif own is not None and own >= threshold:
        check.status = "mismatch"        # unlike their own photos (maybe also like someone else's)
    elif other_id is not None and other_distance < SIMILAR_WARN_FRACTION * threshold:
        check.status = "similar"
    return check


@dataclass
class DuplicatePair:
    """Two students with faces closer than SIMILAR_WARN_FRACTION x the recognition threshold."""
    student_a: str
    student_b: str
    distance: float
    same_face: bool                       # below the duplicate-block distance


class FaceIndex:
    """
    Embeddings of enrolled faces with their students. The arrays grow in place
    (doubling, like a list), so adding a face does not copy the whole index and
    a check stays one matrix-vector product even with tens of thousands of faces.
    """

    def __init__(self, entries: Iterable[Tuple[str, np.ndarray]] = ()):
        self.size = 0
        self.matrix = np.zeros((0, 0))
        self.sq_norms = np.zeros(0)
        self.labels = np.zeros(0, dtype=np.int64)    # row -> index into self.student_ids
        self.student_ids: List[str] = []
        self._label_of: Dict[str, int] = {}
        entries = list(entries)
        if entries:
            self._load(entries)

    def _label(self, student_id: str) -> int:
        if student_id not in self._label_of:
            self._label_of[student_id] = len(self.student_ids)
            self.student_ids.append(student_id)
        return self._label_of[student_id]

    def _load(self, entries: List[Tuple[str, np.ndarray]]) -> None:
        """Fill an empty index in one go (no per-row array writes)."""
        self.matrix = np.stack([encoding for _, encoding in entries]).astype(np.float64, copy=False)
        self.matrix = self.matrix.reshape(len(entries), -1)
        self.sq_norms = np.einsum("ij,ij->i", self.matrix, self.matrix)
        self.labels = np.array([self._label(student_id) for student_id, _ in entries], dtype=np.int64)
        self.size = len(entries)

    def copy(self) -> "FaceIndex":
        clone = FaceIndex()
        clone.size = self.size
        clone.matrix = self.matrix[:self.size].copy()
        clone.sq_norms = self.sq_norms[:self.size].copy()
        clone.labels = self.labels[:self.size].copy()
        clone.student_ids = list(self.student_ids)
        clone._label_of = dict(self._label_of)
        return clone

    def add(self, student_id: str, encoding: np.ndarray) -> None:
        encoding = np.asarray(encoding, dtype=np.float64).ravel()
        if self.size == len(self.matrix):
            capacity = max(1024, 2 * len(self.matrix))
            matrix = np.zeros((capacity, len(encoding)))
            if self.size:
                matrix[:self.size] = self.matrix[:self.size]
            self.matrix = matrix
            self.sq_norms = np.resize(self.sq_norms, capacity)
            self.labels = np.resize(self.labels, capacity)
        self.matrix[self.size] = encoding
        self.sq_norms[self.size] = encoding @ encoding
        self.labels[self.size] = self._label(student_id)
        self.size += 1

    def compare(self, encoding: np.ndarray, student_id: str) -> Tuple[Optional[float], Optional[str], float]:
        """
        Returns:
            (closest distance to this student's own faces or None,
             the closest OTHER student or None, the distance to that student)
        """
        if not self.size:
            return None, None, float("inf")
        encoding = np.asarray(encoding, dtype=np.float64).ravel()
        # |a - b|^2 = |a|^2 - 2 a.b + |b|^2: no (faces x dims) temporary array
        sq = self.sq_norms[:self.size] - 2.0 * (self.matrix[:self.size] @ encoding) + encoding @ encoding
        distances = np.sqrt(np.maximum(sq, 0.0))
        label = self._label_of.get(student_id, -1)
        own_mask = self.labels[:self.size] == label
        own = float(distances[own_mask].min()) if own_mask.any() else None
        others = np.where(own_mask, np.inf, distances)
        closest = int(np.argmin(others))
        if not np.isfinite(others[closest]):
            return own, None, float("inf")
        return own, self.student_ids[self.labels[closest]], float(others[closest])

    def check(self, encoding: np.ndarray, student_id: str, threshold: float) -> FaceCheck:
        return classify_face(*self.compare(encoding, student_id), threshold)

    def duplicate_pairs(self, threshold: float, block: int = 256,
                        progress: Optional[Callable[[int, int], None]] = None) -> List[DuplicatePair]:
        """
        Every pair of different students whose closest faces are similar
        (SIMILAR_WARN_FRACTION x threshold), closest first. Compares all faces
        with each other in blocks of rows (float32, upper triangle only), so
        memory stays ~block x faces.
        """
        n = self.size
        if n < 2:
            return []
        matrix = self.matrix[:n].astype(np.float32)
        sq_norms = self.sq_norms[:n].astype(np.float32)
        labels = self.labels[:n]
        limit = np.float32(SIMILAR_WARN_FRACTION * threshold) ** 2
        n_labels = len(self.student_ids)
        found_codes, found_distances = [], []
        for start in range(0, n, block):
            stop = min(start + block, n)
            # Rows start..stop against rows start..n (each pair once)
            sq = (sq_norms[start:stop, None] + sq_norms[None, start:]
                  - 2.0 * (matrix[start:stop] @ matrix[start:].T))
            rows, cols = np.nonzero(sq < limit)
            cols += start
            keep = labels[start + rows] != labels[cols]
            rows, cols = rows[keep], cols[keep]
            if len(rows):
                a, b = labels[start + rows], labels[cols]
                # One code per student pair, so the closest faces per pair can be kept with numpy
                found_codes.append(np.minimum(a, b) * n_labels + np.maximum(a, b))
                found_distances.append(np.sqrt(np.maximum(sq[rows, cols - start], 0.0)))
            if progress:   # by comparisons done: the first blocks hold the most
                done = n * stop - (stop * stop - stop) // 2
                progress(done, n * n - (n * n - n) // 2)
        if not found_codes:
            return []
        codes, distances = np.concatenate(found_codes), np.concatenate(found_distances)
        order = np.lexsort((distances, codes))          # by pair, closest first
        codes, distances = codes[order], distances[order]
        first = np.concatenate(([True], codes[1:] != codes[:-1]))
        codes, distances = codes[first], distances[first]
        order = np.argsort(distances, kind="stable")
        block_limit = DUPLICATE_BLOCK_FRACTION * threshold
        return [DuplicatePair(self.student_ids[int(code) // n_labels], self.student_ids[int(code) % n_labels],
                              float(distance), bool(distance < block_limit))
                for code, distance in zip(codes[order], distances[order])]
