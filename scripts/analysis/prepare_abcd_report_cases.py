"""Pair the selected A/B/C/D cases with their identical no-solution baselines.

The strict experiment file is large, so this script streams it with ijson and
only keeps the 100 sample/framework/condition/turn records used by the report.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import ijson


ROOT = Path(__file__).resolve().parents[2]
COMBINED = ROOT / "results_memory_echo_abcd_top5"
CASES_PATH = COMBINED / "all_stage_cases.json"
STRICT_PATH = ROOT / "results_memory_echo_solution_strict_200" / "sample_level.json"
FREEFORM_BASELINE = (
    ROOT
    / "results_memory_echo_solution_freeform_dce_200"
    / "sample_level_dce_v2.jsonl"
)
OUT_PATH = COMBINED / "paired_stage_cases.json"


def fnum(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def short_memory(memory: dict) -> dict:
    return {
        "rank": memory.get("retrieval_rank"),
        "label": memory.get("label") or memory.get("supplier_id"),
        "score": fnum(memory.get("retrieval_score")),
        "text": memory.get("text", ""),
    }


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    wanted_samples = {case["sample_id"] for case in cases}

    final_freeform = {}
    with FREEFORM_BASELINE.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("solution_arm") != "baseline":
                continue
            key = (row["sample_id"], row["framework"], row["condition"])
            final_freeform[key] = row

    baseline_lookup = {}
    with STRICT_PATH.open("rb") as handle:
        for sample in ijson.items(handle, "item"):
            sample_id = sample.get("sample_id")
            if sample_id not in wanted_samples:
                continue
            for run in sample.get("runs", []):
                if run.get("solution_arm") != "baseline":
                    continue
                framework = run.get("framework")
                condition = run.get("condition")
                if framework not in {"A-MEM", "Mem0"}:
                    continue
                metrics = run.get("metrics", {})
                for turn in run.get("trajectory", []):
                    key = (sample_id, framework, condition, int(turn.get("turn", -1)))
                    baseline_lookup[key] = {
                        "answer": turn.get("answer", ""),
                        "predicted_label": turn.get("predicted_label"),
                        "target_correct": bool(turn.get("target_correct")),
                        "ED": fnum(turn.get("ED")),
                        "RRI": fnum(metrics.get("RRI")),
                        "WEP": fnum(metrics.get("WEP")),
                        "DCE": fnum(turn.get("DCE")),
                        "contaminated": turn.get("contaminated"),
                        "retrieved_memories": [
                            short_memory(m) for m in turn.get("retrieved_memories", [])[:5]
                        ],
                    }

    paired = []
    missing = []
    for case in cases:
        turn = int(case["case_id"].rsplit("|", 1)[-1])
        key = (case["sample_id"], case["framework"], case["condition"], turn)
        before = baseline_lookup.get(key)
        if before is None:
            missing.append(key)
            continue

        if case["stage"] == "final_decision":
            ff = final_freeform.get(key[:3])
            if ff:
                before["answer"] = ff.get("freeform_answer", before["answer"])
                before["DCE"] = fnum(ff.get("DCE_v2"))
                before["contaminated"] = bool(ff.get("contaminated_v2"))
                before["gold_preferred"] = bool(ff.get("gold_preferred"))

        after = {
            "answer": case.get("freeform_answer", ""),
            "answer_correct": bool(case.get("answer_correct")),
            "echo_affected": bool(case.get("echo_affected")),
            "gold_preferred": bool(case.get("gold_preferred")),
            "ED": fnum(case.get("ED")),
            "RRI": fnum(case.get("RRI")),
            "WEP": fnum(case.get("WEP")),
            "DCE": fnum(case.get("DCE")),
            "contaminated": bool(case.get("contaminated")),
            "retrieved_memories": [
                short_memory(m) for m in case.get("retrieved_memories", [])[:5]
            ],
        }
        record = dict(case)
        record["turn"] = turn
        record["before"] = before
        record["after"] = after
        paired.append(record)

    if missing:
        raise RuntimeError(f"Missing {len(missing)} baseline pairs: {missing[:5]}")
    if len(paired) != 100:
        raise RuntimeError(f"Expected 100 paired cases, got {len(paired)}")

    OUT_PATH.write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(paired)} paired cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
