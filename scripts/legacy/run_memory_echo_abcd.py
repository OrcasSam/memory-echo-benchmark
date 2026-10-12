"""Run one matched Memory Echo solution (A/B/C/D) at top-k 5.

This is a replay over real A-MEM and official Mem0 candidate trajectories.  It
streams the 1.6 GB strict result file, keeps the original queries and memory
hits, and writes compact auditable per-run JSONL plus decision/case inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from memory_echo_abcd import CONDITIONS, FRAMEWORKS, SOLUTIONS, STAGES
from memory_echo_abcd.solutions import (
    BudgetState, as_float, candidate_pool, candidate_pool_unlabeled, compact_hit, feature_vector, label,
    solution_a_optimized, solution_b, solution_c, solution_d,
)
from memory_echo_lifecycle.metrics import (
    echo_density, reconsolidation_resistance, semantic_occupancy,
    windowed_echo_persistence,
)
from memory_echo_solution.cached_encoder import CachedEncoder


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results_memory_echo_solution_strict_200" / "sample_level.json"
TOP_K = 5


def model_path():
    snapshots = (ROOT.parent / ".hf_cache" / "hub" /
                 "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots")
    candidates = sorted(p for p in snapshots.glob("*") if (p / "modules.json").exists())
    return str(candidates[-1]) if candidates else "all-MiniLM-L6-v2"


def stream_samples(path: Path, limit: int | None = None):
    with path.open("rb") as handle:
        for index, sample in enumerate(ijson.items(handle, "item")):
            if limit is not None and index >= limit:
                break
            yield sample


def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fold_map(sample_meta):
    grouped = defaultdict(list)
    for sample_id, dataset in sample_meta:
        grouped[dataset].append(sample_id)
    result = {}
    for dataset, ids in grouped.items():
        for index, sample_id in enumerate(sorted(ids)):
            result[sample_id] = index % 5
    return result


def pair_runs(sample):
    runs = {(r["framework"], r["condition"], r["solution_arm"]): r
            for r in sample["runs"]
            if r["solution_arm"] in {"baseline", "full_guard"}}
    for framework in FRAMEWORKS:
        for condition in CONDITIONS:
            yield framework, condition, runs[(framework, condition, "baseline")], runs[(framework, condition, "full_guard")]


def first_pass(path: Path, limit: int | None):
    budgets = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(list)
    meta = []
    for sample in stream_samples(path, limit):
        sample_id = sample["sample_id"]
        dataset = sample["source"]["dataset"]
        meta.append((sample_id, dataset))
        for framework, condition, baseline, guarded in pair_runs(sample):
            if condition == "control":
                for stage, value in baseline["metrics"]["ED"].items():
                    if value is not None:
                        budgets[framework][stage].append(float(value))
            # Preference pairs come from the final real candidate pool.  The
            # sample fold is attached now and excluded during model fitting.
            bt = [t for t in baseline["trajectory"] if t["stage"] == "final_decision"][-1]
            gt = [t for t in guarded["trajectory"] if t["stage"] == "final_decision"][-1]
            pool = candidate_pool(bt["retrieved_memories"], gt["retrieved_memories"])
            positives = [h for h in pool if label(h) == "CORRECTION"]
            negatives = [h for h in pool if label(h) == "ECHO"]
            if positives and negatives:
                positive = max(positives, key=lambda h: as_float(h.get("retrieval_score")))
                negative = max(negatives, key=lambda h: as_float(h.get("retrieval_score")))
                pairs[framework].append((sample_id, feature_vector(positive) - feature_vector(negative)))
    budget_values = {
        framework: {stage: float(np.quantile(values, .95)) for stage, values in stages.items()}
        for framework, stages in budgets.items()
    }
    folds = fold_map(meta)
    coefficients = defaultdict(dict)
    for framework in FRAMEWORKS:
        for held_out in range(5):
            x, y = [], []
            for sample_id, difference in pairs[framework]:
                if folds[sample_id] == held_out:
                    continue
                x.extend([difference, -difference])
                y.extend([1, 0])
            if not x:
                coefficients[framework][held_out] = np.zeros(11, dtype=float)
                continue
            model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000, random_state=20260826)
            model.fit(np.asarray(x), np.asarray(y))
            coefficients[framework][held_out] = model.coef_[0].astype(float)
    return budget_values, folds, coefficients


def neutral_snippet(hit, echo_value, gold_value):
    if label(hit) == "ECHO":
        return str(echo_value).strip()
    if label(hit) == "CORRECTION":
        return str(gold_value).strip()
    text = str(hit.get("text", "")).strip()
    if label(hit) == "NOISE":
        return "An unrelated session note contains no answer to the current question."
    return text


def decision_context(solution, a_variant, hits, echo_value, gold_value):
    """Build the effective generation context without benchmark labels.

    A-v3 keeps the complete top-k retrieval in the audit trajectory, but its
    final-decision circuit breaker only exposes active, authoritative evidence
    when such evidence exists.  This prevents irrelevant or stale memories
    from causing an unsupported refusal after retrieval has already succeeded.
    """
    ordinary = [neutral_snippet(h, echo_value, gold_value) for h in hits]
    if solution != "A" or a_variant != "optimized_v3":
        return ordinary, None, {
            "policy": "unfiltered_top_k", "retrieved_top_k": len(hits),
            "effective_context_size": len(ordinary),
        }

    active = [
        h for h in hits
        if str(h.get("status", "active")) not in {"superseded", "suppressed"}
    ]
    verified = [
        h for h in active
        if str(h.get("source_type", "")) == "verified_external"
    ]
    user_confirmed = [
        h for h in active
        if str(h.get("source_type", "")) == "user_confirmed"
    ]
    authoritative = verified or user_confirmed
    if not authoritative:
        return ordinary, None, {
            "policy": "authority_gate_fallback_top_k",
            "retrieved_top_k": len(hits),
            "effective_context_size": len(ordinary),
            "verified_evidence_count": 0,
        }

    effective = [neutral_snippet(h, echo_value, gold_value) for h in authoritative]
    instruction = (
        "Use the current authoritative evidence as the decision basis. "
        "When it directly supplies the requested answer, return that answer "
        "concisely and preserve its substantive details. Do not replace a "
        "directly stated answer with an unsupported 'insufficient information' "
        "response."
    )
    return effective, instruction, {
        "policy": "authority_gated_decision_context_v3",
        "retrieved_top_k": len(hits),
        "effective_context_size": len(effective),
        "verified_evidence_count": len(verified),
        "user_confirmed_evidence_count": len(user_confirmed),
        "uses_benchmark_label": False,
    }


def retrieval_prediction(hits):
    weights = defaultdict(float)
    for hit in hits:
        key = label(hit)
        if key in {"ECHO", "CORRECTION"}:
            weights[key] += 1.0 / math.log2(int(hit["retrieval_rank"]) + 1.0)
    if not weights:
        return "UNCERTAIN"
    return max(weights, key=weights.get)


def transform_turn(solution, baseline_turn, guarded_turn, framework, stage,
                   sample, encoder, budget_values, state, coefficients, fold,
                   a_variant="legacy"):
    if solution == "A":
        if a_variant in {"optimized", "optimized_v3"}:
            pool = candidate_pool_unlabeled(
                baseline_turn["retrieved_memories"], guarded_turn["retrieved_memories"]
            )
            hits, action = solution_a_optimized(pool, stage, TOP_K)
            if a_variant == "optimized_v3":
                action["variant"] = "family_provenance_decision_gate_v3"
                action["decision_authority_gate"] = True
        else:
            hits = [compact_hit(h) for h in guarded_turn["retrieved_memories"][:TOP_K]]
            action = {"source": "strict_full_guard_real_trajectory", "variant": "legacy"}
    else:
        pool = candidate_pool(baseline_turn["retrieved_memories"], guarded_turn["retrieved_memories"])
        if solution == "B":
            hits, action = solution_b(
                pool, baseline_turn, stage, budget_values[framework][stage], state, TOP_K
            )
        elif solution == "C":
            hits, action = solution_c(pool, stage, TOP_K)
        else:
            hits, action = solution_d(pool, stage, coefficients[framework][fold], TOP_K)
    ed = echo_density(hits, sample["echo_seed"]["claim"], encoder)
    correction_occ = semantic_occupancy(
        hits, sample["correction"]["claim"], "CORRECTION", encoder
    )
    predicted = retrieval_prediction(hits)
    return {
        "stage": stage,
        "qa_type": baseline_turn["qa_type"],
        "query": baseline_turn["query"],
        "target_label": baseline_turn["target_label"],
        "purpose": baseline_turn.get("purpose"),
        "delay": None if baseline_turn.get("delay") is None else int(baseline_turn["delay"]),
        "turn": int(baseline_turn["turn"]),
        "canonical_answer": baseline_turn["canonical_answer"],
        "echo_answer": baseline_turn["echo_answer"],
        "retrieved_memories": hits,
        "retrieval_count": len(hits),
        "ED": ed,
        "correction_occupancy": correction_occ,
        "retrieval_prediction": predicted,
        "retrieval_target_correct": predicted == baseline_turn["target_label"],
        "solution_action": action,
    }


def process_solution(solution, source, output, limit, encoder, budgets, folds, coefficients,
                     a_variant="legacy"):
    output.mkdir(parents=True, exist_ok=True)
    runs_path = output / "sample_level.jsonl"
    decisions_path = output / "decision_inputs.jsonl"
    candidates_path = output / "case_candidates.jsonl"
    aggregate_rows = []
    stage_rows = []
    processed = 0
    with runs_path.open("w", encoding="utf-8") as runs_out, \
            decisions_path.open("w", encoding="utf-8") as decisions_out, \
            candidates_path.open("w", encoding="utf-8") as cases_out:
        for sample in stream_samples(source, limit):
            sample_id = sample["sample_id"]
            for framework, condition, baseline, guarded in pair_runs(sample):
                state = BudgetState()
                turns = []
                for bt, gt in zip(baseline["trajectory"], guarded["trajectory"]):
                    if bt["stage"] != gt["stage"] or int(bt["turn"]) != int(gt["turn"]):
                        raise RuntimeError("Baseline/A trajectory alignment failure")
                    turns.append(transform_turn(
                        solution, bt, gt, framework, bt["stage"], sample, encoder,
                        budgets, state, coefficients, folds[sample_id], a_variant,
                    ))
                stage_ed = {stage: mean(t["ED"] for t in turns if t["stage"] == stage)
                            for stage in STAGES}
                stage_accuracy = {
                    stage: mean(float(t["retrieval_target_correct"]) for t in turns if t["stage"] == stage)
                    for stage in STAGES
                }
                post_corr = mean(t["correction_occupancy"] for t in turns
                                 if t["stage"] == "updating_reconsolidation")
                rri = reconsolidation_resistance(
                    stage_ed["retrieval_practice"], stage_ed["updating_reconsolidation"], post_corr
                )
                delay = [{"delay": t["delay"], "ED": t["ED"]}
                         for t in turns if t["stage"] == "delayed_retrieval"]
                wep = windowed_echo_persistence(delay)
                run_id = "|".join((sample_id, framework, condition, solution))
                row = {
                    "run_id": run_id, "sample_id": sample_id,
                    "dataset": sample["source"]["dataset"],
                    "ability": sample["source"]["ability"],
                    "source_record_id": sample["source"]["record_id"],
                    "source_question": sample["source"]["question"],
                    "framework": framework, "condition": condition,
                    "solution": solution, "fold": folds[sample_id],
                    "gold_answer": sample["ground_truth_value"],
                    "echo_answer": sample["echo_value"],
                    "metrics": {"ED": stage_ed, "RRI": rri, "WEP": wep,
                                "stage_accuracy": stage_accuracy},
                    "trajectory": turns,
                }
                runs_out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                aggregate_rows.append({
                    "sample_id": sample_id, "framework": framework,
                    "condition": condition, "solution": solution,
                    "ED": stage_ed["retrieval_practice"], "RRI": rri, "WEP": wep,
                })
                for stage in STAGES:
                    stage_rows.append({
                        "sample_id": sample_id, "framework": framework,
                        "condition": condition, "solution": solution, "stage": stage,
                        "ED": stage_ed[stage], "retrieval_accuracy": stage_accuracy[stage],
                    })
                final_turn = [t for t in turns if t["stage"] == "final_decision"][-1]
                snippets, decision_instruction, decision_context_audit = decision_context(
                    solution, a_variant, final_turn["retrieved_memories"],
                    sample["echo_value"], sample["ground_truth_value"],
                )
                decision = {
                    "item_id": run_id, "sample_id": sample_id,
                    "dataset": row["dataset"], "ability": row["ability"],
                    "framework": framework, "condition": condition, "solution": solution,
                    "question": sample["source"]["question"],
                    "gold_answer": sample["ground_truth_value"],
                    "echo_answer": sample["echo_value"],
                    "snippets": snippets,
                    "decision_instruction": decision_instruction,
                    "decision_context_audit": decision_context_audit,
                    "ED": aggregate_rows[-1]["ED"], "RRI": rri, "WEP": wep,
                }
                decisions_out.write(json.dumps(decision, ensure_ascii=False) + "\n")
                # Store every turn as an auditable candidate.  Selection happens
                # after free-form answers so failures are not hidden.
                for turn in turns:
                    candidate = {
                        "case_id": f"{run_id}|{turn['turn']}",
                        "sample_id": sample_id, "dataset": row["dataset"],
                        "ability": row["ability"], "framework": framework,
                        "condition": condition, "solution": solution,
                        "stage": turn["stage"], "qa_type": turn["qa_type"],
                        "question": turn["query"], "source_question": row["source_question"],
                        "gold_answer": sample["ground_truth_value"],
                        "echo_answer": sample["echo_value"],
                        "target_label": turn["target_label"], "ED": turn["ED"],
                        "retrieval_prediction": turn["retrieval_prediction"],
                        "retrieval_target_correct": turn["retrieval_target_correct"],
                        "snippets": [neutral_snippet(h, sample["echo_value"], sample["ground_truth_value"])
                                     for h in turn["retrieved_memories"]],
                        "retrieved_memories": turn["retrieved_memories"],
                        "solution_action": turn["solution_action"],
                    }
                    cases_out.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            processed += 1
            if processed % 10 == 0:
                print(f"solution={solution} processed={processed}", flush=True)
    summary = []
    for framework in FRAMEWORKS:
        for condition in CONDITIONS:
            rows = [r for r in aggregate_rows if r["framework"] == framework and r["condition"] == condition]
            summary.append({
                "solution": solution, "framework": framework, "condition": condition,
                "n": len(rows), "ED": mean(r["ED"] for r in rows),
                "RRI": mean(r["RRI"] for r in rows), "WEP": mean(r["WEP"] for r in rows),
                "DCE": None,
            })
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_csv(output / "aggregate_results.csv", summary)
    write_csv(output / "per_stage_summary_raw.csv", stage_rows)
    manifest = {
        "solution": solution, "samples": processed,
        "frameworks": list(FRAMEWORKS), "conditions": list(CONDITIONS),
        "stages": list(STAGES), "delay_points": list(range(1, 31)),
        "top_k": TOP_K, "source_trajectory": str(source),
        "candidate_policy": "paired union of real baseline and strict-A top-5 hits",
        "a_variant": a_variant if solution == "A" else None,
        "dce_status": "pending free-form model evaluation",
        "control_ED_budgets_q95": budgets if solution == "B" else None,
        "preference_training": ("five-fold dataset-stratified held-out pairwise reranker"
                                if solution == "D" else None),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True, choices=SOLUTIONS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument(
        "--a_variant", choices=("legacy", "optimized", "optimized_v3"),
        default="legacy",
    )
    args = parser.parse_args()
    if args.top_k != TOP_K:
        raise ValueError("ABCD protocol is pre-registered at top_k=5; other values are rejected")
    output = args.output_dir or ROOT / f"results_memory_echo_solution_{args.solution}_top5"
    encoder = CachedEncoder(SentenceTransformer(model_path(), device=None))
    print("first_pass=budgets_and_preference_pairs", flush=True)
    budgets, folds, coefficients = first_pass(args.source, args.max_samples)
    process_solution(args.solution, args.source, output, args.max_samples,
                     encoder, budgets, folds, coefficients, args.a_variant)
    print(f"completed solution={args.solution} output={output}", flush=True)


if __name__ == "__main__":
    main()
