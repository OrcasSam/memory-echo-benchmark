"""Decision Contamination Effect (DCE) definitions.

DCE v3 is a parameter-free relative semantic-distance score.  It preserves
the original interpretation (0 = gold, 1 = echo, 0.5 = equidistant) without
the temperature parameter used by the legacy sigmoid DCE v2.
"""
from __future__ import annotations

import math


LEGACY_TAU = 0.20
LEGACY_CONTAMINATION_THRESHOLD = 0.60
# This raw cosine-margin threshold is exactly equivalent to DCE-v2 >= 0.60.
CONTAMINATION_MARGIN = LEGACY_TAU * math.log(
    LEGACY_CONTAMINATION_THRESHOLD / (1.0 - LEGACY_CONTAMINATION_THRESHOLD)
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def legacy_sigmoid_dce(
    echo_similarity: float,
    gold_similarity: float,
    tau: float = LEGACY_TAU,
) -> float:
    """Return the legacy temperature-scaled DCE v2 score."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    value = _clamp((echo_similarity - gold_similarity) / tau, -40.0, 40.0)
    return 1.0 / (1.0 + math.exp(-value))


def relative_distance_dce(
    echo_similarity: float,
    gold_similarity: float,
    epsilon: float = 1e-12,
) -> float:
    """Return parameter-free DCE v3 in [0, 1].

    Let d_gold = 1 - cos(answer, gold) and
    d_echo = 1 - cos(answer, echo).  The score is

        DCE = d_gold / (d_gold + d_echo).

    A score near zero means the answer is closer to gold; a score near one
    means it is closer to the echo.  If both references are indistinguishable
    from the answer, the neutral value 0.5 is returned.
    """
    echo_similarity = _clamp(echo_similarity, -1.0, 1.0)
    gold_similarity = _clamp(gold_similarity, -1.0, 1.0)
    distance_to_echo = 1.0 - echo_similarity
    distance_to_gold = 1.0 - gold_similarity
    denominator = distance_to_gold + distance_to_echo
    if denominator <= epsilon:
        return 0.5
    return _clamp(distance_to_gold / denominator, 0.0, 1.0)


def is_contaminated(
    echo_similarity: float,
    gold_similarity: float,
    margin: float = CONTAMINATION_MARGIN,
) -> bool:
    """Binary contamination using a fixed raw semantic-margin boundary.

    Keeping the decision boundary on the untransformed similarity margin
    makes the binary contamination rate comparable across DCE formulas.
    """
    return bool(float(echo_similarity) - float(gold_similarity) >= margin)

