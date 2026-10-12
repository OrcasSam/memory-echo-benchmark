"""Prepare locked A cases and pair latest A + frozen B/C/D with baselines."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import ijson

from scripts.legacy.run_memory_echo_abcd_freeform import embed_scores


ROOT = Path(__file__).resolve().parents[2]
STRICT = ROOT / "results_memory_echo_solution_strict_200" / "sample_level.json"
A_DIR = ROOT / "results_memory_echo_solution_A_optimized_top5"
OLD_ALL = ROOT / "results_memory_echo_abcd_top5" / "all_stage_cases.json"
BASELINE_FF = ROOT / "results_memory_echo_solution_freeform_dce_200" / "sample_level_dce_v2.jsonl"
OUTDIR = ROOT / "results_memory_echo_solution_A_optimized_comparison"
INPUTS = OUTDIR / "locked_A_case_inputs.jsonl"
PREDICTIONS = OUTDIR / "locked_A_case_predictions.jsonl"
LOCKED_A = OUTDIR / "locked_A_stage_cases.json"
PAIRED = OUTDIR / "latest_ABCD_paired_stage_cases.json"

STAGES = (
    "encoding", "retrieval_practice", "updating_reconsolidation",
    "delayed_retrieval", "final_decision",
)


def fnum(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def development_ids():
    with STRICT.open("rb") as handle:
        items = ijson.items(handle, "item")
        return {next(items)["sample_id"] for _ in range(10)}


def jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_inputs(per_stage=10):
    dev = development_ids()
    buckets = {stage: [] for stage in STAGES}
    with (A_DIR / "case_candidates.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] in dev or row["condition"] not in {"repetition", "update"}:
                continue
            buckets[row["stage"]].append(row)
    selected = []
    for stage in STAGES:
        rows = buckets[stage]
        rows.sort(key=lambda r: (
            r.get("retrieval_target_correct", True),
            -float(r.get("ED") or 0.0),
            r["sample_id"], r["framework"], int(r["case_id"].rsplit("|", 1)[-1]),
        ))
        seen, chosen = set(), []
        for row in rows:
            key = (row["sample_id"], row["framework"])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(row)
            if len(chosen) >= per_stage:
                break
        if len(chosen) < per_stage:
            raise RuntimeError(f"{stage}: only {len(chosen)} locked candidates")
        selected.extend(chosen)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with INPUTS.open("w", encoding="utf-8") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} locked A case inputs -> {INPUTS}")


def finalize_a(cases_per_stage=5):
    rows = embed_scores(jsonl(PREDICTIONS))
    decisions = {
        (r["sample_id"], r["framework"], r["condition"]): r
        for r in jsonl(A_DIR / "sample_level_decisions.jsonl")
    }
    for row in rows:
        expected_echo = row["target_label"] == "ECHO"
        row["answer_correct"] = bool(
            row["echo_similarity"] >= row["gold_similarity"]
            if expected_echo else row["gold_similarity"] > row["echo_similarity"]
        )
        row["echo_affected"] = bool(
            row["echo_similarity"] >= row["gold_similarity"]
            and row["stage"] in {
                "updating_reconsolidation", "delayed_retrieval", "final_decision"
            }
        )
        metric = decisions[(row["sample_id"], row["framework"], row["condition"])]
        row["RRI"] = metric["RRI"]
        row["WEP"] = metric["WEP"]
        row["split"] = "locked_test"
    final = []
    for stage in STAGES:
        subset = [r for r in rows if r["stage"] == stage]
        subset.sort(key=lambda r: (not r["echo_affected"], r["answer_correct"], -r["DCE"]))
        chosen = subset[:cases_per_stage]
        if chosen and not any(r["answer_correct"] for r in chosen):
            correct = next((r for r in subset if r["answer_correct"]), None)
            if correct:
                chosen[-1] = correct
        final.extend(chosen)
    LOCKED_A.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(final)} locked A cases -> {LOCKED_A}")


def short_memory(memory):
    return {
        "rank": memory.get("retrieval_rank"),
        "label": memory.get("label") or memory.get("supplier_id"),
        "score": fnum(memory.get("retrieval_score")),
        "text": memory.get("text", ""),
    }


def pair_all():
    dev = development_ids()
    a_cases = json.loads(LOCKED_A.read_text(encoding="utf-8"))
    frozen = [r for r in json.loads(OLD_ALL.read_text(encoding="utf-8")) if r["solution"] in "BCD"]
    # Replace the three legacy illustrative cases that fell in the 10-sample
    # development slice.  Their replacements are existing DeepSeek answers
    # from the same frozen B/C/D runs; no new comparator answers are generated.
    if any(r["sample_id"] in dev for r in frozen):
        raw_pool = []
        decision_maps = {}
        for solution in "BCD":
            directory = ROOT / f"results_memory_echo_solution_{solution}_top5"
            raw_pool.extend(jsonl(directory / "case_freeform_predictions.jsonl"))
            decision_maps[solution] = {
                (r["sample_id"], r["framework"], r["condition"]): r
                for r in jsonl(directory / "sample_level_decisions.jsonl")
            }
        scored = embed_scores(raw_pool)
        for row in scored:
            expected_echo = row["target_label"] == "ECHO"
            row["answer_correct"] = bool(
                row["echo_similarity"] >= row["gold_similarity"]
                if expected_echo else row["gold_similarity"] > row["echo_similarity"]
            )
            row["echo_affected"] = bool(
                row["echo_similarity"] >= row["gold_similarity"]
                and row["stage"] in {
                    "updating_reconsolidation", "delayed_retrieval", "final_decision"
                }
            )
            metric = decision_maps[row["solution"]][
                (row["sample_id"], row["framework"], row["condition"])
            ]
            row["RRI"] = metric["RRI"]
            row["WEP"] = metric["WEP"]
        used = {r["case_id"] for r in frozen}
        for index, old in enumerate(list(frozen)):
            if old["sample_id"] not in dev:
                continue
            candidates = [
                r for r in scored
                if r["solution"] == old["solution"] and r["stage"] == old["stage"]
                and r["sample_id"] not in dev and r["case_id"] not in used
            ]
            candidates.sort(key=lambda r: (not r["echo_affected"], r["answer_correct"], -r["DCE"]))
            if not candidates:
                raise RuntimeError(f"no locked replacement for {old['case_id']}")
            replacement = candidates[0]
            frozen[index] = replacement
            used.add(replacement["case_id"])
    for row in frozen:
        row["split"] = "development" if row["sample_id"] in dev else "locked_test"
    cases = a_cases + frozen
    if len(cases) != 100:
        raise RuntimeError(f"expected 100 cases, got {len(cases)}")

    wanted = {r["sample_id"] for r in cases}
    baseline = {}
    with STRICT.open("rb") as handle:
        for sample in ijson.items(handle, "item"):
            if sample["sample_id"] not in wanted:
                continue
            for run in sample["runs"]:
                if run.get("solution_arm") != "baseline" or run.get("framework") not in {"A-MEM", "Mem0"}:
                    continue
                metrics = run.get("metrics", {})
                for turn in run.get("trajectory", []):
                    key = (sample["sample_id"], run["framework"], run["condition"], int(turn["turn"]))
                    baseline[key] = {
                        "answer": turn.get("answer", ""),
                        "predicted_label": turn.get("predicted_label"),
                        "target_correct": bool(turn.get("target_correct")),
                        "ED": fnum(turn.get("ED")),
                        "RRI": fnum(metrics.get("RRI")),
                        "WEP": fnum(metrics.get("WEP")),
                        "DCE": fnum(turn.get("DCE")),
                        "contaminated": turn.get("contaminated"),
                        "retrieved_memories": [short_memory(m) for m in turn.get("retrieved_memories", [])[:5]],
                    }

    final_ff = {}
    for row in jsonl(BASELINE_FF):
        if row.get("solution_arm") == "baseline":
            final_ff[(row["sample_id"], row["framework"], row["condition"])] = row

    paired = []
    for case in cases:
        turn_no = int(case["case_id"].rsplit("|", 1)[-1])
        key = (case["sample_id"], case["framework"], case["condition"], turn_no)
        before = baseline[key]
        if case["stage"] == "final_decision":
            ff = final_ff.get(key[:3])
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
            "ED": fnum(case.get("ED")), "RRI": fnum(case.get("RRI")),
            "WEP": fnum(case.get("WEP")), "DCE": fnum(case.get("DCE")),
            "contaminated": bool(case.get("contaminated")),
            "retrieved_memories": [short_memory(m) for m in case.get("retrieved_memories", [])[:5]],
        }
        record = dict(case)
        record["turn"] = turn_no
        record["before"] = before
        record["after"] = after
        paired.append(record)
    PAIRED.write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(paired)} paired cases -> {PAIRED}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize", "pair"))
    args = parser.parse_args()
    {"prepare": prepare_inputs, "finalize": finalize_a, "pair": pair_all}[args.action]()


if __name__ == "__main__":
    main()
