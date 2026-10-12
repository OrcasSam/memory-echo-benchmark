"""Add paired baseline and current A-v3 details to the 100 report cases."""
from __future__ import annotations

import json
from pathlib import Path

import ijson


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results_memory_echo_solution_strict_200" / "sample_level.json"
CASES = ROOT / "results_memory_echo_dce_v3_all" / "stage_cases_all_dce_v3.json"
OUTPUT = ROOT / "results_memory_echo_dce_v3_all" / "stage_cases_all_dce_v3_enriched.json"
A_RUNS = ROOT / "results_memory_echo_solution_A_v3_top5" / "sample_level.jsonl"


def compact_memory(hit):
    return {
        "rank": int(hit.get("retrieval_rank") or 0),
        "label": str(hit.get("label") or hit.get("supplier_id") or "NOISE"),
        "score": float(hit.get("retrieval_score") or 0.0),
        "text": str(hit.get("text") or ""),
    }


def main():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    a_cases = [row for row in cases if row["solution"] == "A"]
    targets = {row["sample_id"] for row in a_cases}
    current_runs = {}
    with A_RUNS.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                current_runs[(row["sample_id"], row["framework"], row["condition"])] = row

    source_samples = {}
    development = set()
    with SOURCE.open("rb") as handle:
        for index, sample in enumerate(ijson.items(handle, "item")):
            if index < 10:
                development.add(sample["sample_id"])
            if sample["sample_id"] in targets:
                source_samples[sample["sample_id"]] = sample
            if len(source_samples) == len(targets) and index >= 10:
                break
    if set(source_samples) != targets:
        raise RuntimeError("not all A case source samples were found")

    for case in a_cases:
        sample = source_samples[case["sample_id"]]
        turn = int(str(case["case_id"]).rsplit("|", 1)[1])
        baseline_run = next(
            run for run in sample["runs"]
            if run["framework"] == case["framework"]
            and run["condition"] == case["condition"]
            and run["solution_arm"] == "baseline"
        )
        baseline_turn = next(t for t in baseline_run["trajectory"] if int(t["turn"]) == turn)
        current_run = current_runs[(case["sample_id"], case["framework"], case["condition"])]
        case["turn"] = turn
        case["split"] = "development" if case["sample_id"] in development else "locked_test"
        case["RRI"] = current_run["metrics"]["RRI"]
        case["WEP"] = current_run["metrics"]["WEP"]
        case["before"] = {
            "answer": str(baseline_turn.get("answer") or ""),
            "predicted_label": baseline_turn.get("predicted_label"),
            "target_correct": bool(baseline_turn.get("target_correct")),
            "ED": None if baseline_turn.get("ED") is None else float(baseline_turn["ED"]),
            "RRI": float(baseline_run["metrics"]["RRI"]),
            "WEP": float(baseline_run["metrics"]["WEP"]),
            "DCE": (None if baseline_turn.get("DCE") is None
                    else float(baseline_turn["DCE"])),
            "contaminated": bool(baseline_turn.get("contaminated")),
            "retrieved_memories": [compact_memory(h) for h in baseline_turn["retrieved_memories"][:5]],
        }
        case["after"] = {
            "answer": case["freeform_answer"],
            "answer_correct": bool(case.get("answer_correct")),
            "echo_affected": bool(case.get("echo_affected")),
            "gold_preferred": bool(case.get("gold_preferred")),
            "ED": float(case["ED"]),
            "RRI": float(case["RRI"]),
            "WEP": float(case["WEP"]),
            "DCE": float(case["DCE_v3"]),
            "contaminated": bool(case.get("contaminated_v3_protocol")),
            "retrieved_memories": [compact_memory(h) for h in case["retrieved_memories"][:5]],
        }
    OUTPUT.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"enriched cases={len(cases)} A={len(a_cases)} -> {OUTPUT}")


if __name__ == "__main__":
    main()
