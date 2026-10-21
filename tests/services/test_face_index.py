"""Tests for the in-memory face index behind the duplicate-face checks."""

import numpy as np
import pytest

from src.services.face_index import DUPLICATE_BLOCK_FRACTION, SIMILAR_WARN_FRACTION, FaceIndex, classify_face

THRESHOLD = 0.6


def test_compare_matches_a_brute_force_search():
    """The index grows in place past its first capacity and still finds the right faces."""
    rng = np.random.default_rng(0)
    entries = [(f"s{i % 700}", rng.standard_normal(128)) for i in range(1000)]
    index = FaceIndex(entries)
    ids, vectors = [sid for sid, _ in entries], [vector for _, vector in entries]
    for _ in range(1500):                                # crosses the 1024 and 2048 capacities
        sid, vector = f"s{rng.integers(0, 900)}", rng.standard_normal(128)
        index.add(sid, vector)
        ids.append(sid)
        vectors.append(vector)

    for student_id in ("s5", "s850", "new-student"):
        query = rng.standard_normal(128)
        distances = np.linalg.norm(np.array(vectors) - query, axis=1)
        own_mask = np.array(ids) == student_id
        own, other_id, other_distance = index.compare(query, student_id)
        assert own == (pytest.approx(distances[own_mask].min()) if own_mask.any() else None)
        closest = int(np.argmin(np.where(own_mask, np.inf, distances)))
        assert other_id == ids[closest] and other_distance == pytest.approx(distances[closest])


@pytest.mark.parametrize("own, other, status", [
    (None, None, "ok"),                                            # first face, empty gallery
    (0.2, 0.9, "ok"),                                              # like their own photos only
    (None, 0.25, "duplicate"),                                     # someone else's face
    (0.1, 0.25, "duplicate"),                                      # a duplicate is refused regardless
    (None, 0.4, "similar"),
    (None, 0.5, "ok"),                                             # within the threshold, but not close
    (0.7, 0.9, "mismatch"),                                        # unlike their own photos
    (0.7, 0.4, "mismatch"),
])
def test_classify_face(own, other, status):
    other_id = "other" if other is not None else None
    assert classify_face(own, other_id, other if other is not None else float("inf"), THRESHOLD).status == status


def test_duplicate_pairs_matches_a_brute_force_search():
    rng = np.random.default_rng(3)
    base = rng.standard_normal((40, 128)) * 0.2           # 40 people; nearby faces for some pairs
    entries = []
    for i in range(700):
        person = i % 40
        entries.append((f"s{i % 300}", base[person] + rng.standard_normal(128) * 0.01))
    index = FaceIndex(entries)
    pairs = index.duplicate_pairs(THRESHOLD, block=64)     # several blocks

    limit = SIMILAR_WARN_FRACTION * THRESHOLD
    expected = {}
    for i, (a, va) in enumerate(entries):
        for b, vb in entries[i + 1:]:
            if a != b and np.linalg.norm(va - vb) < limit:
                key = tuple(sorted((a, b), key=lambda sid: index._label_of[sid]))
                expected[key] = min(expected.get(key, np.inf), np.linalg.norm(va - vb))
    assert {(p.student_a, p.student_b): pytest.approx(p.distance, abs=1e-4) for p in pairs} == expected
    assert [p.distance for p in pairs] == sorted(p.distance for p in pairs)
    assert all(p.same_face == (p.distance < DUPLICATE_BLOCK_FRACTION * THRESHOLD) for p in pairs)
