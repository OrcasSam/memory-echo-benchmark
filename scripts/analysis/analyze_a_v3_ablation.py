"""Ablate A-v3 context gating and authoritative answer constraint."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import ijson
import numpy as np
from sentence_transformers import SentenceTransformer

from memory_echo_abcd.dce import relative_distance_dce


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_memory_echo_dce_v3_all"
SOURCE = ROOT / "results_memory_echo_solution_strict_200" / "sample_level.json"
OLD = OUT / "solution_A_v2_reference_dce_v3.jsonl"
NEW = OUT / "solution_A_after_dce_v3.jsonl"
SEED = 20260831
BOOTSTRAPS = 10_000


def jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_path():
    snapshots = (ROOT.parent / ".hf_cache" / "hub" /
                 "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots")
    candidates = sorted(p for p in snapshots.glob("*") if (p / "modules.json").exists())
    return str(candidates[-1]) if candidates else "all-MiniLM-L6-v2"


def dev_ids():
    with SOURCE.open("rb") as handle:
        items = ijson.items(handle, "item")
        return {next(items)["sample_id"] for _ in range(10)}


def key(row):
    return row["sample_id"], row["framework"], row["condition"]


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main():
    development = dev_ids()
    old = {key(r): r for r in jsonl(OLD)
           if r["sample_id"] not in development and r["condition"] != "control"}
    new = {key(r): r for r in jsonl(NEW)
           if r["sample_id"] not in development and r["condition"] != "control"}
    if set(old) != set(new) or len(new) != 760:
        raise RuntimeError("A ablation keys do not match locked Echo protocol")

    texts = []
    for row in new.values():
        texts.extend([row["raw_freeform_answer"], row["echo_answer"], row["gold_answer"]])
    unique = list(dict.fromkeys(texts))
    model = SentenceTransformer(model_path(), device=None)
    vectors = model.encode(unique, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=True)
    lookup = {text: vector for text, vector in zip(unique, vectors)}

    raw_scores = {}
    for k, row in new.items():
        answer = lookup[row["raw_freeform_answer"]]
        echo_sim = float(np.dot(answer, lookup[row["echo_answer"]]))
        gold_sim = float(np.dot(answer, lookup[row["gold_answer"]]))
        raw_scores[k] = relative_distance_dce(echo_sim, gold_sim)

    keys = sorted(new)
    stages = {
        "A_v2_family_provenance_gate": np.asarray([old[k]["DCE_v3"] for k in keys]),
        "A_v3_plus_decision_context_raw_answer": np.asarray([raw_scores[k] for k in keys]),
        "A_v3_plus_answer_constraint_final": np.asarray([new[k]["DCE_v3"] for k in keys]),
    }
    rows = []
    baseline_name = "A_v2_family_provenance_gate"
    previous = None
    for name, values in stages.items():
        if previous is None:
            change, low, high = 0.0, None, None
        else:
            reduction = previous - values
            change = float(reduction.mean())
            low, high = bootstrap(reduction)
        rows.append({
            "scope": "locked_190_echo_conditions",
            "ablation_stage": name,
            "paired_n": len(values),
            "DCE_v3": float(values.mean()),
            "incremental_reduction_vs_previous": change,
            "ci95_low": low,
            "ci95_high": high,
            "total_reduction_vs_A_v2": float(stages[baseline_name].mean() - values.mean()),
        })
        previous = values

    with (OUT / "A_v3_ablation_locked_190.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reason_counts = Counter()
    applied = 0
    for row in new.values():
        audit = row.get("authority_answer_constraint") or {}
        reason_counts[str(audit.get("reason") or "not_applied")] += 1
        applied += int(bool(audit.get("applied")))
    audit = {
        "scope": "locked_190_echo_conditions",
        "n": len(new),
        "constraint_applied": applied,
        "constraint_applied_rate": applied / len(new),
        "reason_counts": dict(reason_counts),
        "uses_embedding_score_for_constraint": False,
        "uses_benchmark_label_for_constraint": False,
        "raw_answers_preserved": all("raw_freeform_answer" in r for r in new.values()),
    }
    (OUT / "A_v3_constraint_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ablation": rows, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
