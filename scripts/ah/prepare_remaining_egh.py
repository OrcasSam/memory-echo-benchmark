"""Prepare only the missing E/G/H answer inputs plus Kimi baselines for F."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from memory_echo_abcd.solutions import compact_hit, solution_a_optimized
from memory_echo_multimodel.providers import read_jsonl
from scripts.legacy.run_memory_echo_abcd import decision_context, neutral_snippet


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_memory_echo_remaining_egh"
LIFECYCLE = ROOT / "results_memory_echo_ah" / "lifecycle_sample_level.jsonl"
SOURCE = ROOT / "results_memory_echo_amem_vs_mem0_200x30" / "rewritten_dataset.json"
SCENARIO = ROOT / "data" / "scenario_sample_level.jsonl"
FRAMEWORKS = ("A-MEM", "Mem0", "CoM", "LangMem")
ECHO_CONDITIONS = {"read_only_repetition", "retrieve_writeback"}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def family_limit(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = {}
    for raw in hits:
        hit = compact_hit(raw)
        family = str(hit.get("expected_family_id") or hit.get("family_id") or
                     f"{hit.get('source_id','')}:{' '.join(str(hit.get('text','')).lower().split())}")
        old = best.get(family)
        if old is None or float(hit.get("retrieval_score") or 0) > float(old.get("retrieval_score") or 0):
            best[family] = hit
    return sorted(best.values(), key=lambda h: float(h.get("retrieval_score") or 0), reverse=True)[:5]


def provenance_control(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = []
    for raw in hits:
        hit = compact_hit(raw)
        if str(hit.get("status", "active")) in {"superseded", "suppressed"}:
            continue
        if str(hit.get("source_type", "")) == "retrieval_derived":
            continue
        eligible.append(hit)
    return sorted(eligible, key=lambda h: float(h.get("retrieval_score") or 0), reverse=True)[:5]


def e_inputs(lifecycle_rows, samples) -> list[dict[str, Any]]:
    output = []
    for row in lifecycle_rows:
        if row["condition"] not in ECHO_CONDITIONS:
            continue
        sample = samples[row["sample_id"]]
        raw_hits = row["trajectory"]["final_decision"]["retrieved_memories"]
        variants = {
            "Only-F": family_limit(raw_hits),
            "Only-P": provenance_control(raw_hits),
            "F+P": provenance_control(family_limit(raw_hits)),
        }
        for variant, hits in variants.items():
            output.append({
                "evaluation_id": f"E|{row['sample_id']}|{row['framework']}|{row['condition']}|{variant}",
                "experiment": "E", "configuration": variant,
                "sample_id": row["sample_id"], "dataset": row["dataset"],
                "framework": row["framework"], "condition": row["condition"],
                "question": row["decision_input"]["question"],
                "gold_answer": sample["ground_truth_value"],
                "echo_answer": sample["echo_value"],
                "snippets": [neutral_snippet(h, sample["echo_value"], sample["ground_truth_value"]) for h in hits],
                "decision_instruction": None,
            })
    return output


def g_inputs() -> list[dict[str, Any]]:
    output = []
    condition_map = {"repetition": "read_only_repetition", "update": "retrieve_writeback"}
    for row in read_jsonl(SCENARIO):
        if row["condition"] not in condition_map or row["solution"] not in {"Baseline", "A"}:
            continue
        final = [turn for turn in row["trajectory"] if turn["stage"] == "final_decision"][-1]
        hits = final["retrieved_memories"]
        method = "FPA" if row["solution"] == "A" else "Baseline"
        snippets, instruction, audit = decision_context(
            "A" if method == "FPA" else "Baseline",
            "optimized_v3" if method == "FPA" else "none",
            hits, row["echo_value"], row["gold_value"],
        )
        output.append({
            "evaluation_id": f"G|{row['sample_id']}|{row['framework']}|{row['condition']}|{method}",
            "experiment": "G", "task": row["scenario"], "method": method,
            "sample_id": row["sample_id"], "dataset": row["source_dataset"],
            "framework": row["framework"], "condition": condition_map[row["condition"]],
            "question": row["question"], "gold_answer": row["gold_value"],
            "echo_answer": row["echo_value"], "snippets": snippets,
            "decision_instruction": instruction, "decision_context_audit": audit,
        })
    return output


def stable_fraction(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def perturb_hits(raw_hits: list[dict[str, Any]], source_condition: str, sample_id: str):
    hits = [compact_hit(hit) for hit in raw_hits]
    # A non-answering, valid historical record is present in every condition,
    # allowing retention to be measured independently from answer correctness.
    if hits:
        valid = dict(next((h for h in hits if h.get("label") == "NOISE"), hits[-1]))
        valid.update({
            "memory_id": f"valid-history:{sample_id}",
            "text": "A verified historical context note remains valid but does not answer the current question.",
            "label": "VALID_HISTORY", "supplier_id": "VALID_HISTORY",
            "family_id": f"valid-history:{sample_id}",
            "expected_family_id": f"valid-history:{sample_id}",
            "source_id": f"valid-history:{sample_id}", "source_type": "user_confirmed",
            "status": "active", "confidence": 0.85,
        })
        hits.append(valid)
    if source_condition == "family_noise":
        for index, hit in enumerate(hits):
            if stable_fraction(f"{sample_id}|family|{index}") < 0.30:
                hit["family_id"] = f"noisy-family:{sample_id}:{index}"
                hit["expected_family_id"] = hit["family_id"]
    elif source_condition == "correction_missing_or_noisy":
        changed = []
        mode_missing = stable_fraction(f"{sample_id}|correction-mode") < 0.50
        for hit in hits:
            if str(hit.get("source_type")) != "verified_external":
                changed.append(hit)
            elif not mode_missing:
                noisy = dict(hit)
                noisy["source_type"] = "historical_unverified"
                noisy["confidence"] = 0.35
                noisy["family_id"] = f"noisy-correction:{sample_id}"
                noisy["expected_family_id"] = noisy["family_id"]
                changed.append(noisy)
        hits = changed
    elif source_condition == "old_fact_nonconflicting":
        for hit in hits:
            if hit.get("label") == "ECHO":
                hit.update({
                    "text": "A user-confirmed historical fact about a different aspect remains valid and does not conflict with the current answer.",
                    "label": "VALID_HISTORY", "supplier_id": "VALID_HISTORY",
                    "family_id": f"nonconflict:{sample_id}",
                    "expected_family_id": f"nonconflict:{sample_id}",
                    "source_type": "user_confirmed", "status": "active", "confidence": 0.85,
                })
    return hits


def h_inputs(lifecycle_rows, samples) -> list[dict[str, Any]]:
    output = []
    source_conditions = (
        "complete", "family_noise", "correction_missing_or_noisy", "old_fact_nonconflicting",
    )
    for row in lifecycle_rows:
        if row["condition"] != "retrieve_writeback":
            continue
        sample = samples[row["sample_id"]]
        raw_hits = row["trajectory"]["final_decision"]["retrieved_memories"]
        for source_condition in source_conditions:
            pool = perturb_hits(raw_hits, source_condition, row["sample_id"])
            for method in ("Baseline", "FPA"):
                if method == "FPA":
                    served, _ = solution_a_optimized(pool, "final_decision", 5)
                    snippets, instruction, audit = decision_context(
                        "A", "optimized_v3", served,
                        sample["echo_value"], sample["ground_truth_value"],
                    )
                else:
                    served = pool[:5]
                    snippets = [neutral_snippet(h, sample["echo_value"], sample["ground_truth_value"]) for h in served]
                    instruction, audit = None, {"policy": "unfiltered_top_k"}
                valid_retained = any(h.get("label") == "VALID_HISTORY" for h in served)
                output.append({
                    "evaluation_id": f"H|{row['sample_id']}|{row['framework']}|{source_condition}|{method}",
                    "experiment": "H", "source_condition": source_condition,
                    "method": method, "sample_id": row["sample_id"],
                    "dataset": row["dataset"], "framework": row["framework"],
                    "condition": row["condition"],
                    "question": row["decision_input"]["question"],
                    "gold_answer": sample["ground_truth_value"],
                    "echo_answer": sample["echo_value"], "snippets": snippets,
                    "decision_instruction": instruction, "decision_context_audit": audit,
                    "valid_history_retained": valid_retained,
                })
    return output


def kimi_baseline_inputs(lifecycle_rows, samples) -> list[dict[str, Any]]:
    output = []
    for row in lifecycle_rows:
        if row["framework"] not in {"A-MEM", "Mem0"}:
            continue
        sample = samples[row["sample_id"]]
        item = dict(row["decision_input"])
        output.append({
            "evaluation_id": f"kimi-baseline|{row['sample_id']}|{row['framework']}|{row['condition']}",
            "experiment": "F", "method": "Baseline", "sample_id": row["sample_id"],
            "dataset": row["dataset"], "framework": row["framework"],
            "condition": row["condition"], "question": item["question"],
            "gold_answer": sample["ground_truth_value"], "echo_answer": sample["echo_value"],
            "snippets": item["snippets"],
        })
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lifecycle_rows = read_jsonl(LIFECYCLE)
    samples = {row["sample_id"]: row for row in json.loads(SOURCE.read_text(encoding="utf-8"))}
    e = e_inputs(lifecycle_rows, samples)
    g = g_inputs()
    h = h_inputs(lifecycle_rows, samples)
    kimi = kimi_baseline_inputs(lifecycle_rows, samples)
    expected = {"E": 4560, "G": 1200, "H": 6080, "KimiBaseline": 1140}
    actual = {"E": len(e), "G": len(g), "H": len(h), "KimiBaseline": len(kimi)}
    if actual != expected:
        raise RuntimeError(f"input count mismatch: expected={expected} actual={actual}")
    write_jsonl(OUT / "e_inputs.jsonl", e)
    write_jsonl(OUT / "g_inputs.jsonl", g)
    write_jsonl(OUT / "h_inputs.jsonl", h)
    write_jsonl(OUT / "deepseek_inputs.jsonl", e + g + h)
    write_jsonl(OUT / "kimi_baseline_inputs.jsonl", kimi)
    (OUT / "manifest.json").write_text(json.dumps({
        "protocol": "remaining_E_G_H_v1", "counts": actual,
        "deepseek_total": len(e) + len(g) + len(h),
        "source_noise": {"family_noise_rate": 0.30,
                         "correction_missing_vs_noisy": "deterministic 50/50 by sample hash"},
        "reused": ["F+P+Authority raw FPA answers", "Full FPA constrained answers",
                   "A-MEM/Mem0 Kimi FPA answers", "all existing A-D results"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": actual, "deepseek_total": len(e)+len(g)+len(h)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
