"""Metrics for formation, resistance, persistence, and decision transfer."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np


def _cosine(model, left: str, right: str) -> float:
    vectors = model.encode([left, right], normalize_embeddings=True, show_progress_bar=False)
    return float(np.dot(vectors[0], vectors[1]))


def semantic_occupancy(hits: List[Dict], reference_claim: str, label: str,
                       model) -> Optional[float]:
    """Rank-weighted semantic occupancy of one labelled claim in top-k."""
    if not hits:
        return None
    numerator = denominator = 0.0
    for hit in hits:
        rank_weight = 1.0 / math.log2(hit["retrieval_rank"] + 1.0)
        affinity = max(0.0, _cosine(model, hit["text"], reference_claim))
        denominator += rank_weight
        if hit.get("supplier_id") == label:
            numerator += rank_weight * affinity
    return numerator / denominator if denominator else None


def echo_density(hits: List[Dict], echo_claim: str, model) -> Optional[float]:
    """Rank-weighted semantic occupancy of Echo-labelled memories in top-k."""
    return semantic_occupancy(hits, echo_claim, "ECHO", model)


def fit_persistence_half_life(delay_series: Iterable[Dict]) -> Dict:
    points = [(float(row["delay"]), float(row["ED"])) for row in delay_series
              if row.get("ED") is not None and row["ED"] > 0]
    if len(points) < 2:
        return {"PHL": None, "decay_rate": None, "fit_r2": None, "censored": True}
    x = np.asarray([p[0] for p in points], dtype=float)
    y = np.asarray([p[1] for p in points], dtype=float)
    y0 = max(y[0], 1e-9)
    log_ratio = np.log(np.clip(y / y0, 1e-9, None))
    slope = float(np.sum(x * log_ratio) / max(np.sum(x * x), 1e-12))
    decay_rate = max(0.0, -slope)
    predicted = -decay_rate * x
    residual = float(np.sum((log_ratio - predicted) ** 2))
    total = float(np.sum((log_ratio - np.mean(log_ratio)) ** 2))
    r2 = None if total <= 1e-12 else 1.0 - residual / total
    if decay_rate <= 1e-12:
        return {"PHL": None, "decay_rate": 0.0, "fit_r2": r2, "censored": True}
    half_life = math.log(2.0) / decay_rate
    return {
        "PHL": half_life,
        "decay_rate": decay_rate, "fit_r2": r2,
        "censored": bool(half_life > max(x)),
    }


def windowed_echo_persistence(delay_series: Iterable[Dict]) -> Optional[float]:
    """Normalized area under the ED-by-delay curve within the observed window.

    Delay points must be consecutive, beginning at session 1.  Because ED is in
    [0, 1], the linearly normalized trapezoidal area is also in [0, 1].  Unlike
    half-life, WEP remains identifiable when Echo never halves in the window.
    """
    points = sorted(
        (int(row["delay"]), float(row["ED"]))
        for row in delay_series
        if row.get("ED") is not None
    )
    if len(points) < 2:
        return None
    delays = [point[0] for point in points]
    if delays != list(range(1, delays[-1] + 1)):
        raise ValueError("WEP requires consecutive delay points beginning at 1")
    area = sum(
        (right_delay - left_delay) * (left_ed + right_ed) / 2.0
        for (left_delay, left_ed), (right_delay, right_ed)
        in zip(points, points[1:])
    )
    width = points[-1][0] - points[0][0]
    return max(0.0, min(1.0, area / width)) if width > 0 else None


def decision_contamination(answer: str, echo_answer: str,
                           correct_answer: str, model,
                           threshold: float = 0.5,
                           predicted_label: str | None = None) -> Dict:
    echo_similarity = _cosine(model, answer, echo_answer)
    correct_similarity = _cosine(model, answer, correct_answer)
    margin = max(-60.0, min(60.0, (correct_similarity - echo_similarity) / 0.08))
    score = 1.0 / (1.0 + math.exp(margin))
    return {
        "DCE": score,
        "contaminated": bool(
            score >= threshold and echo_similarity >= correct_similarity
            and predicted_label in (None, "ECHO")
        ),
        "echo_similarity": echo_similarity,
        "correct_similarity": correct_similarity,
        "threshold": threshold,
        "predicted_label": predicted_label,
    }


def reconsolidation_resistance(pre_ed: Optional[float], post_ed: Optional[float],
                               correction_occupancy: Optional[float]) -> Optional[float]:
    """Residual Echo after correction, normalized by its pre-update strength.

    Correction occupancy prevents a high score from being attributed merely to
    an empty retrieval result. Paired aggregation additionally subtracts control.
    """
    if pre_ed is None or post_ed is None or correction_occupancy is None:
        return None
    if pre_ed <= 1e-12:
        return 0.0
    survival = min(1.0, post_ed / pre_ed)
    competition = post_ed / max(post_ed + correction_occupancy, 1e-12)
    return 0.5 * survival + 0.5 * competition
