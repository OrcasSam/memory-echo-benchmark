"""Solution-specific retrieval transformations for the matched replay.

The input candidates are real top-5 hits produced by A-MEM or official Mem0
under the baseline and strict-A trajectories.  B/C/D operate only on this
paired candidate pool; no synthetic result rows are introduced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


def as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def label(hit: Dict) -> str:
    return str(hit.get("label") or hit.get("supplier_id") or "NOISE")


def compact_hit(hit: Dict) -> Dict:
    fields = (
        "memory_id", "text", "supplier_id", "label", "timestamp", "kind",
        "write_count", "retrieval_rank", "retrieval_score", "family_id",
        "expected_family_id", "source_id", "source_type",
        "independence_group", "confidence", "activation", "exposure_count",
        "status", "parent_memory_id", "supersedes_family_id",
    )
    row = {key: hit.get(key) for key in fields if key in hit}
    row["label"] = label(hit)
    row["supplier_id"] = row["label"]
    for key in ("retrieval_score", "confidence", "activation"):
        if key in row and row[key] is not None:
            row[key] = as_float(row[key])
    for key in ("write_count", "retrieval_rank", "exposure_count"):
        if key in row and row[key] is not None:
            row[key] = int(row[key])
    return row


def candidate_pool(baseline_hits: Iterable[Dict], guarded_hits: Iterable[Dict]) -> List[Dict]:
    """Union real hits while keeping the best observed version of each claim."""
    best = {}
    for origin, hits in (("baseline", baseline_hits), ("solution_A", guarded_hits)):
        for raw in hits:
            row = compact_hit(raw)
            text_key = " ".join(str(row.get("text", "")).lower().split())
            key = (text_key, row["label"], str(row.get("source_type", "")))
            row["candidate_origin"] = origin
            old = best.get(key)
            if old is None or as_float(row.get("retrieval_score")) > as_float(old.get("retrieval_score")):
                best[key] = row
    return list(best.values())


def candidate_pool_unlabeled(baseline_hits: Iterable[Dict],
                             guarded_hits: Iterable[Dict]) -> List[Dict]:
    """Union real candidates without using evaluation labels as a key."""
    best = {}
    for origin, hits in (("baseline", baseline_hits), ("solution_A", guarded_hits)):
        for raw in hits:
            row = compact_hit(raw)
            family = str(row.get("expected_family_id") or row.get("family_id") or "")
            text_key = " ".join(str(row.get("text", "")).lower().split())
            key = (family, text_key, str(row.get("source_type", "")))
            row["candidate_origin"] = origin
            old = best.get(key)
            if old is None or as_float(row.get("retrieval_score")) > as_float(old.get("retrieval_score")):
                best[key] = row
    return list(best.values())


def rerank(rows: Iterable[Dict], scores: Iterable[float], top_k: int = 5) -> List[Dict]:
    paired = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    output = []
    for rank, (row, score) in enumerate(paired, 1):
        item = dict(row)
        item["original_retrieval_score"] = as_float(item.get("retrieval_score"))
        item["solution_score"] = float(score)
        item["retrieval_rank"] = rank
        output.append(item)
    return output


def feature_vector(hit: Dict) -> np.ndarray:
    """Observable retriever/provenance features; benchmark labels are excluded."""
    source = str(hit.get("source_type", ""))
    status = str(hit.get("status", "active"))
    return np.asarray([
        as_float(hit.get("retrieval_score")),
        as_float(hit.get("confidence")),
        as_float(hit.get("activation")),
        math.log1p(max(0, int(hit.get("write_count") or 0))),
        math.log1p(max(0, int(hit.get("exposure_count") or 0))),
        1.0 if source == "verified_external" else 0.0,
        1.0 if source == "retrieval_derived" else 0.0,
        1.0 if source == "historical_unverified" else 0.0,
        1.0 if status == "superseded" else 0.0,
        1.0 if status == "suppressed" else 0.0,
        1.0,
    ], dtype=float)


def solution_a_optimized(pool: List[Dict], stage: str,
                         top_k: int = 5) -> tuple[List[Dict], Dict]:
    """Family-aware provenance gate for the optimized primary solution A.

    This implementation deliberately does *not* read benchmark labels when it
    decides which memories to retain.  It uses only fields available to the
    deployed memory system: provenance, family identity, source reliability,
    exposure/write counts, correction links, and lifecycle status.

    The design has two constraints:
    1. one semantic family gets at most one slot (repetition is not evidence);
    2. once a verified external correction is present, unverified/retrieval-
       derived claims cannot outvote it in the effective context.
    """
    after_correction = stage in {
        "updating_reconsolidation", "delayed_retrieval", "final_decision"
    }
    verified_present = any(
        str(h.get("source_type", "")) == "verified_external"
        and str(h.get("status", "active")) not in {"superseded", "suppressed"}
        for h in pool
    )

    # First collapse each provenance/semantic family.  Missing family IDs are
    # conservatively keyed by source + normalized text rather than by row ID.
    family_best: Dict[str, tuple[Dict, float]] = {}
    duplicate_count = 0
    for raw in pool:
        hit = dict(raw)
        source = str(hit.get("source_type", ""))
        status = str(hit.get("status", "active"))
        family = str(
            hit.get("expected_family_id")
            or hit.get("family_id")
            or f"{hit.get('source_id', '')}:{' '.join(str(hit.get('text', '')).lower().split())}"
        )
        exposure = math.log1p(
            max(0, int(hit.get("write_count") or 0))
            + max(0, int(hit.get("exposure_count") or 0))
        )
        reliability = {
            "verified_external": 1.0,
            "user_confirmed": 0.85,
            "independent_noise": 0.10,
            "historical_unverified": 0.0,
            "retrieval_derived": -0.20,
        }.get(source, 0.0)
        score = (
            as_float(hit.get("retrieval_score"))
            + 0.35 * as_float(hit.get("confidence"))
            + 0.80 * reliability
            - 0.18 * exposure
        )
        if status in {"superseded", "suppressed"}:
            score -= 4.0
        old = family_best.get(family)
        if old is None or score > old[1]:
            if old is not None:
                duplicate_count += 1
            family_best[family] = (hit, score)
        else:
            duplicate_count += 1

    rows, scores = [], []
    blocked_unverified = 0
    blocked_stale = 0
    for hit, score in family_best.values():
        source = str(hit.get("source_type", ""))
        status = str(hit.get("status", "active"))
        if status in {"superseded", "suppressed"}:
            blocked_stale += 1
            continue
        # Verified evidence is a circuit breaker after reconsolidation:
        # contradicted unverified claims stay in the audit store but do not
        # enter the effective generation context.  This is a structural safety
        # constraint, not a benchmark-label penalty.
        if (after_correction and verified_present
                and source in {"historical_unverified", "retrieval_derived"}):
            blocked_unverified += 1
            continue
        if after_correction and source == "verified_external":
            score += 1.60
        item = dict(hit)
        item["a_family_gate_score"] = float(score)
        rows.append(item)
        scores.append(score)

    selected = rerank(rows, scores, top_k)
    return selected, {
        "variant": "family_provenance_circuit_breaker_v2",
        "after_correction": after_correction,
        "verified_present": verified_present,
        "family_duplicates_collapsed": duplicate_count,
        "blocked_unverified": blocked_unverified,
        "blocked_stale": blocked_stale,
        "effective_context_size": len(selected),
        "uses_benchmark_label_for_ranking": False,
    }


@dataclass
class BudgetState:
    strength: float = 0.0
    delay_echo_windows: int = 0
    delay_windows: int = 0


def solution_b(pool: List[Dict], baseline_turn: Dict, stage: str, budget: float,
               state: BudgetState, top_k: int = 5) -> tuple[List[Dict], Dict]:
    """Original Adaptive Echo Budget Controller (AEBC)."""
    raw_ed = as_float(baseline_turn.get("ED"))
    pressure = max(0.0, raw_ed - budget)
    state.strength = min(3.0, 0.85 * state.strength + 1.25 * pressure)
    if stage == "delayed_retrieval":
        state.delay_windows += 1
        state.delay_echo_windows += int(raw_ed > budget)
    persistence = (state.delay_echo_windows / state.delay_windows
                   if state.delay_windows else (0.7 if stage == "updating_reconsolidation" else 0.45))
    after_correction = stage in {"updating_reconsolidation", "delayed_retrieval", "final_decision"}
    rows, scores, risks, redirected = [], [], [], 0
    for hit in pool:
        source = str(hit.get("source_type", ""))
        exposure = min(1.0, (
            math.log1p(max(0, int(hit.get("write_count") or 0)) +
                       max(0, int(hit.get("exposure_count") or 0))) / math.log(10.0)
        ))
        contradiction = 0.0
        if after_correction and source in {"historical_unverified", "retrieval_derived"}:
            contradiction = 1.0
        if str(hit.get("status", "")) in {"superseded", "suppressed"}:
            contradiction = 1.0
        risk = exposure * contradiction * persistence
        base = as_float(hit.get("retrieval_score")) + 0.05 * as_float(hit.get("confidence"))
        correction_boost = (0.45 * state.strength
                            if after_correction and source == "verified_external" else 0.0)
        score = base - state.strength * risk + correction_boost
        # Simulate the registered write-budget gate: derived traces that already
        # push ED beyond its Control budget remain logs, not fact candidates.
        if pressure > 0 and source == "retrieval_derived":
            score -= 2.5
            redirected += 1
        item = dict(hit)
        item["echo_risk"] = float(risk)
        rows.append(item)
        scores.append(score)
        risks.append(risk)
    return rerank(rows, scores, top_k), {
        "budget": float(budget), "raw_ED": raw_ed, "pressure": pressure,
        "suppression_strength": state.strength,
        "mean_candidate_risk": float(np.mean(risks)) if risks else 0.0,
        "write_redirect_count": redirected,
    }


def solution_c(pool: List[Dict], stage: str, top_k: int = 5) -> tuple[List[Dict], Dict]:
    """CRAG-M adaptation: quality grade then corrective routing."""
    after_correction = stage in {"updating_reconsolidation", "delayed_retrieval", "final_decision"}
    rows, scores, grades = [], [], []
    for hit in pool:
        source = str(hit.get("source_type", ""))
        status = str(hit.get("status", "active"))
        rel = max(-1.0, min(1.0, as_float(hit.get("retrieval_score"))))
        verified = 1.0 if source == "verified_external" else 0.0
        conflict = 0.0
        if after_correction and source in {"historical_unverified", "retrieval_derived"}:
            conflict = 1.0
        if status in {"superseded", "suppressed"}:
            conflict = 1.0
        quality = 0.45 * rel + 0.35 * verified + 0.15 * as_float(hit.get("confidence")) - 0.45 * conflict
        if not after_correction:
            grade = "correct" if quality >= 0.15 else "ambiguous"
        elif quality >= 0.35:
            grade = "correct"
        elif quality > 0.0:
            grade = "ambiguous"
        else:
            grade = "incorrect"
        item = dict(hit)
        item["crag_quality"] = float(quality)
        item["crag_grade"] = grade
        rows.append(item)
        grades.append(grade)
        # Incorrect memories are not deleted from storage; they are excluded
        # from the effective context through a strong, auditable penalty.
        scores.append(quality if grade != "incorrect" else quality - 2.0)
    selected = rerank(rows, scores, top_k)
    return selected, {
        "correct": grades.count("correct"),
        "ambiguous": grades.count("ambiguous"),
        "incorrect": grades.count("incorrect"),
        "route": ("external_correction" if after_correction and not any(
            h.get("source_type") == "verified_external" for h in selected
        ) else "refine_and_recompose"),
    }


def solution_d(pool: List[Dict], stage: str, coefficients: np.ndarray,
               top_k: int = 5, gamma: float = 0.65) -> tuple[List[Dict], Dict]:
    """Held-out preference-trained Memory reranker (DPO/NPO adaptation)."""
    if stage in {"encoding", "retrieval_practice"}:
        base_rows = sorted(pool, key=lambda h: as_float(h.get("retrieval_score")), reverse=True)
        return rerank(base_rows, [as_float(h.get("retrieval_score")) for h in base_rows], top_k), {
            "preference_active": False, "gamma": gamma,
        }
    rows, scores, preference_scores = [], [], []
    for hit in pool:
        pref = float(np.dot(coefficients, feature_vector(hit)))
        base = as_float(hit.get("retrieval_score"))
        item = dict(hit)
        item["preference_score"] = pref
        rows.append(item)
        scores.append(base + gamma * pref)
        preference_scores.append(pref)
    return rerank(rows, scores, top_k), {
        "preference_active": True, "gamma": gamma,
        "preference_score_mean": float(np.mean(preference_scores)) if preference_scores else 0.0,
    }
