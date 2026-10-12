"""Metrics required by Tables A--C of the unified protocol."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from memory_echo_lifecycle.metrics import (
    echo_density,
    reconsolidation_resistance,
    semantic_occupancy,
    windowed_echo_persistence,
)


def _cosine(model, left: str, right: str) -> float:
    vectors = model.encode(
        [left, right], normalize_embeddings=True, show_progress_bar=False
    )
    return max(0.0, float(np.dot(vectors[0], vectors[1])))


def storage_occupancy(records: Iterable[dict], reference: str,
                      label: str, model) -> float:
    """Semantic claim mass in storage, independent of retrieval rank."""
    records = list(records)
    if not records:
        return 0.0
    numerator = sum(
        _cosine(model, record.get("text", ""), reference)
        for record in records
        if record.get("supplier_id") == label
    )
    return numerator / len(records)


def storage_rri(before_records: list[dict], after_records: list[dict],
                echo_claim: str, correction_claim: str, model) -> float:
    pre_echo = storage_occupancy(before_records, echo_claim, "ECHO", model)
    post_echo = storage_occupancy(after_records, echo_claim, "ECHO", model)
    post_correction = storage_occupancy(
        after_records, correction_claim, "CORRECTION", model
    )
    return reconsolidation_resistance(pre_echo, post_echo, post_correction) or 0.0


def correction_retrieved(hits: list[dict]) -> bool:
    return any(hit.get("supplier_id") == "CORRECTION" for hit in hits)


def old_retrieval_wins(hits: list[dict]) -> bool:
    weights = {"ECHO": 0.0, "CORRECTION": 0.0}
    for hit in hits:
        label = hit.get("supplier_id")
        if label in weights:
            weights[label] += 1.0 / math.log2(hit["retrieval_rank"] + 1.0)
    return weights["ECHO"] > weights["CORRECTION"]


__all__ = [
    "correction_retrieved", "echo_density", "old_retrieval_wins",
    "reconsolidation_resistance", "semantic_occupancy", "storage_rri",
    "windowed_echo_persistence",
]
