"""Prepare paired CoM/LangMem governance contexts for DeepSeek and Kimi.

All five methods replay the same real candidate memories from the completed
strict lifecycle.  No generated answer or benchmark label is used to build a
candidate pool.  The output has 190 samples x 2 frameworks x 3 conditions x
5 methods = 5,700 paired decision inputs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from memory_echo_abcd.solutions import (
    BudgetState, as_float, compact_hit, feature_vector, label,
    solution_a_optimized, solution_b, solution_c, solution_d,
)
from memory_echo_lifecycle.metrics import (
    echo_density, reconsolidation_resistance, semantic_occupancy,
    windowed_echo_persistence,
)
from memory_echo_solution.cached_encoder import CachedEncoder
from scripts.legacy.run_memory_echo_abcd import decision_context, neutral_snippet, retrieval_prediction


ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE = ROOT / "results_memory_echo_ah" / "lifecycle_sample_level.jsonl"
SOURCE = ROOT / "results_memory_echo_amem_vs_mem0_200x30" / "rewritten_dataset.json"
OUTPUT = ROOT / "results_memory_echo_com_lang_methods"
FRAMEWORKS = ("CoM", "LangMem")
CONDITIONS = ("once_only", "read_only_repetition", "retrieve_writeback")
METHODS = ("Baseline", "Budget", "CRAG-M", "PairLR", "FPA")
TOP_K = 5


def model_path() -> str:
    snapshots = (ROOT.parent / ".hf_cache" / "hub" /
                 "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots")
    candidates = sorted(p for p in snapshots.glob("*") if (p / "modules.json").exists())
    return str(candidates[-1]) if candidates else "all-MiniLM-L6-v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def mean(values) -> float:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def flatten_trajectory(trajectory: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    turns: list[tuple[str, dict[str, Any]]] = []
    turns.append(("encoding", trajectory["encoding"]))
    turns.extend(("retrieval_practice", turn) for turn in trajectory["retrieval_practice"])
    turns.append(("pre_correction", trajectory["pre_correction"]))
    turns.append(("updating_reconsolidation", trajectory["post_correction"]))
    turns.extend(("delayed_retrieval", turn) for turn in trajectory["delayed_retrieval"])
    turns.append(("final_decision", trajectory["final_decision"]))
    return turns


def fold_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row["sample_id"] in seen:
            continue
        seen.add(row["sample_id"])
        grouped[row["dataset"]].append(row["sample_id"])
    folds = {}
    for ids in grouped.values():
        for index, sample_id in enumerate(sorted(ids)):
            folds[sample_id] = index % 5
    return folds


def control_budgets(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["condition"] != "once_only":
            continue
        by_stage: dict[str, list[float]] = defaultdict(list)
        for stage, turn in flatten_trajectory(row["trajectory"]):
            by_stage[stage].append(float(turn["ED"]))
        for stage, scores in by_stage.items():
            values[row["framework"]][stage].append(mean(scores))
    return {
        framework: {
            stage: float(np.quantile(scores, 0.95))
            for stage, scores in stages.items()
        }
        for framework, stages in values.items()
    }


def pairlr_coefficients(rows: list[dict[str, Any]], folds: dict[str, int]):
    pairs: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for row in rows:
        final = row["trajectory"]["final_decision"]["retrieved_memories"]
        positives = [compact_hit(hit) for hit in final if label(hit) == "CORRECTION"]
        negatives = [compact_hit(hit) for hit in final if label(hit) == "ECHO"]
        if positives and negatives:
            positive = max(positives, key=lambda h: as_float(h.get("retrieval_score")))
            negative = max(negatives, key=lambda h: as_float(h.get("retrieval_score")))
            pairs[row["framework"]].append(
                (row["sample_id"], feature_vector(positive) - feature_vector(negative))
            )
    coefficients: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for framework in FRAMEWORKS:
        for held_out in range(5):
            x, y = [], []
            for sample_id, difference in pairs[framework]:
                if folds[sample_id] == held_out:
                    continue
                x.extend((difference, -difference))
                y.extend((1, 0))
            if not x:
                coefficients[framework][held_out] = np.zeros(11, dtype=float)
            else:
                model = LogisticRegression(
                    fit_intercept=False, C=1.0, max_iter=2000, random_state=20260826
                )
                model.fit(np.asarray(x), np.asarray(y))
                coefficients[framework][held_out] = model.coef_[0].astype(float)
    return coefficients


def transform(method: str, pool: list[dict[str, Any]], turn: dict[str, Any],
              stage: str, framework: str, budget: float, state: BudgetState,
              coefficients: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if method == "Baseline":
        hits = [compact_hit(hit) for hit in pool[:TOP_K]]
        return hits, {"policy": "real_framework_top_k"}
    if method == "Budget":
        return solution_b(pool, turn, stage, budget, state, TOP_K)
    if method == "CRAG-M":
        return solution_c(pool, stage, TOP_K)
    if method == "PairLR":
        return solution_d(pool, stage, coefficients, TOP_K)
    if method == "FPA":
        return solution_a_optimized(pool, stage, TOP_K)
    raise ValueError(method)


def prepare(lifecycle: Path, source: Path, output: Path) -> dict[str, Any]:
    rows = [row for row in read_jsonl(lifecycle) if row["framework"] in FRAMEWORKS]
    if len(rows) != 190 * 2 * 3:
        raise RuntimeError(f"expected 1140 CoM/LangMem lifecycle rows, found {len(rows)}")
    source_rows = {row["sample_id"]: row for row in json.loads(source.read_text(encoding="utf-8"))}
    folds = fold_map(rows)
    budgets = control_budgets(rows)
    coefficients = pairlr_coefficients(rows, folds)
    encoder = CachedEncoder(SentenceTransformer(model_path(), device=None))

    decisions: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sample = source_rows[row["sample_id"]]
        echo_claim = sample["echo_seed"]["claim"]
        correction_claim = sample["correction"]["claim"]
        for method in METHODS:
            state = BudgetState()
            transformed = []
            for stage, turn in flatten_trajectory(row["trajectory"]):
                pool = [compact_hit(hit) for hit in turn["retrieved_memories"]]
                hits, action = transform(
                    method, pool, turn, stage, row["framework"],
                    budgets[row["framework"]][stage], state,
                    coefficients[row["framework"]][folds[row["sample_id"]]],
                )
                transformed.append({
                    "stage": stage,
                    "index": turn.get("index"),
                    "retrieved_memories": hits,
                    "ED": echo_density(hits, echo_claim, encoder),
                    "correction_occupancy": semantic_occupancy(
                        hits, correction_claim, "CORRECTION", encoder
                    ),
                    "retrieval_prediction": retrieval_prediction(hits),
                    "action": action,
                })
            pre_ed = next(t["ED"] for t in transformed if t["stage"] == "pre_correction")
            post_turn = next(t for t in transformed if t["stage"] == "updating_reconsolidation")
            delay = [
                {"delay": int(t["index"]), "ED": t["ED"]}
                for t in transformed if t["stage"] == "delayed_retrieval"
            ]
            rri = reconsolidation_resistance(
                pre_ed, post_turn["ED"], post_turn["correction_occupancy"]
            )
            wep = windowed_echo_persistence(delay)
            final = next(t for t in transformed if t["stage"] == "final_decision")
            a_variant = "optimized_v3" if method == "FPA" else "none"
            snippets, instruction, audit = decision_context(
                "A" if method == "FPA" else method,
                a_variant,
                final["retrieved_memories"],
                sample["echo_value"], sample["ground_truth_value"],
            )
            evaluation_key = "|".join((
                row["sample_id"], row["framework"], row["condition"], method
            ))
            decisions.append({
                "evaluation_key": evaluation_key,
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "ability": row["ability"],
                "framework": row["framework"],
                "condition": row["condition"],
                "method": method,
                "question": row["decision_input"]["question"],
                "gold_answer": sample["ground_truth_value"],
                "echo_answer": sample["echo_value"],
                "snippets": snippets,
                "decision_instruction": instruction,
                "decision_context_audit": audit,
                "ED": pre_ed,
                "RRI": rri,
                "WEP": wep,
                "final_ED": final["ED"],
            })
            metric_rows.append({
                "evaluation_key": evaluation_key,
                "sample_id": row["sample_id"], "framework": row["framework"],
                "condition": row["condition"], "method": method,
                "ED": pre_ed, "RRI": rri, "WEP": wep, "final_ED": final["ED"],
            })
        if index % 100 == 0:
            print(f"prepared_lifecycle_rows={index}/{len(rows)}", flush=True)

    if len(decisions) != 5700 or len({d["evaluation_key"] for d in decisions}) != 5700:
        raise RuntimeError(f"expected 5700 unique decisions, found {len(decisions)}")
    write_jsonl(output / "decision_inputs.jsonl", decisions)
    write_jsonl(output / "method_metrics.jsonl", metric_rows)
    manifest = {
        "protocol": "com_lang_matched_five_method_v1",
        "samples": 190,
        "frameworks": list(FRAMEWORKS),
        "conditions": list(CONDITIONS),
        "methods": list(METHODS),
        "top_k": TOP_K,
        "decision_inputs": len(decisions),
        "candidate_policy": "same real framework top-k for all five methods",
        "budget_policy": "framework-stage control ED 95th percentile",
        "pairlr_policy": "five-fold dataset-stratified held-out pairwise reranker",
        "fpa_policy": "family/provenance gate plus authority-gated decision context v3",
        "source_lifecycle": str(lifecycle.resolve()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle", type=Path, default=LIFECYCLE)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(prepare(args.lifecycle, args.source, args.output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
