"""Reference galleries. Similarity alone never promotes a trusted reference."""

from __future__ import annotations

import numpy as np

from app.perception.assignment import cosine


def reference_state(*, human: bool, temporal_strong: bool, assigned_reid: bool) -> str | None:
    """Trusted requires a person or strong short-gap temporal evidence.

    An appearance-only re-identification stays a candidate. An unmatched or
    ambiguous observation is not stored on the candidate's gallery.
    """
    if human or temporal_strong:
        return "trusted"
    if assigned_reid:
        return "candidate"
    return None


def gallery_cosine(vectors: list[np.ndarray], query: np.ndarray | None) -> float | None:
    if query is None or not vectors:
        return None
    scores = [cosine(query, item) for item in vectors]
    usable = [score for score in scores if score is not None]
    if not usable:
        return None
    return max(usable)


def store_reference(
    vectors: list[np.ndarray],
    metas: list[dict],
    vector: np.ndarray,
    meta: dict,
    limit: int,
) -> None:
    quality = float(meta.get("quality") or 0)
    if len(vectors) >= limit:
        worst = min(range(len(metas)), key=lambda index: float(metas[index].get("quality") or 0))
        if quality <= float(metas[worst].get("quality") or 0):
            return
        vectors.pop(worst)
        metas.pop(worst)
    vectors.append(np.asarray(vector, dtype=np.float32))
    metas.append(meta)


def promote_candidates(trusted: list, trusted_meta: list, candidates: list, candidate_meta: list, limit: int) -> None:
    """Move candidate references into the trusted gallery after a person confirms them."""
    for vector, meta in zip(candidates, candidate_meta, strict=True):
        confirmed = dict(meta)
        confirmed["state"] = "trusted"
        confirmed["confirmed"] = True
        store_reference(trusted, trusted_meta, vector, confirmed, limit)
    candidates.clear()
    candidate_meta.clear()
