"""Refine A/B/C/D cases from full results, prioritizing real Echo failures."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.legacy.run_memory_echo_abcd_freeform import embed_scores, generate, jsonl


ROOT = Path(__file__).resolve().parents[2]
SOLUTIONS = "ABCD"
STAGES = ("encoding", "retrieval_practice", "updating_reconsolidation",
          "delayed_retrieval", "final_decision")


def directory(solution):
    return ROOT / f"results_memory_echo_solution_{solution}_top5"


def unique_take(rows, count, key_fields=("sample_id", "framework")):
    output, seen = [], set()
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key); output.append(row)
        if len(output) >= count:
            break
    return output


def prepare(output_root, per_stage=24):
    output_root.mkdir(parents=True, exist_ok=True)
    all_inputs = []
    for solution in SOLUTIONS:
        rows = jsonl(directory(solution) / "case_candidates.jsonl")
        for stage in STAGES[:-1]:
            subset = [r for r in rows if r["stage"] == stage and
                      r["condition"] in {"repetition", "update"}]
            # Wrong retrieval support first, then largest observed Echo density.
            subset.sort(key=lambda r: (
                bool(r["retrieval_target_correct"]),
                -float(r.get("ED") or 0.0),
                r["sample_id"], r["framework"],
            ))
            all_inputs.extend(unique_take(subset, per_stage))
    with (output_root / "case_audit_inputs.jsonl").open("w", encoding="utf-8") as out:
        for row in all_inputs:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(all_inputs)


def add_run_metrics(solution, rows):
    metrics = {}
    with (directory(solution) / "sample_level.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            run = json.loads(line)
            metrics[(run["sample_id"], run["framework"], run["condition"])] = run["metrics"]
    for row in rows:
        values = metrics.get((row["sample_id"], row["framework"], row["condition"]), {})
        row["RRI"] = values.get("RRI")
        row["WEP"] = values.get("WEP")
    return rows


def normalize(text):
    return " ".join(re.findall(r"[\w]+", str(text).lower(), flags=re.UNICODE))


def answer_matches(answer, target):
    answer_norm, target_norm = normalize(answer), normalize(target)
    if not target_norm:
        return False
    if target_norm in {"yes", "no"}:
        return answer_norm == target_norm or answer_norm.startswith(target_norm + " ")
    return len(target_norm) >= 3 and target_norm in answer_norm


def correctness(row):
    expected_echo = row["target_label"] == "ECHO"
    target = row["echo_answer"] if expected_echo else row["gold_answer"]
    competitor = row["gold_answer"] if expected_echo else row["echo_answer"]
    if answer_matches(row["freeform_answer"], target) and not answer_matches(
            row["freeform_answer"], competitor):
        return True
    if answer_matches(row["freeform_answer"], competitor) and not answer_matches(
            row["freeform_answer"], target):
        return False
    return (row["echo_similarity"] >= row["gold_similarity"] if expected_echo
            else row["gold_similarity"] > row["echo_similarity"])


def is_echo_affected(row):
    if row["stage"] not in {"updating_reconsolidation", "delayed_retrieval", "final_decision"}:
        return False
    if correctness(row):
        return False
    if answer_matches(row["freeform_answer"], row["echo_answer"]):
        return True
    margin = float(row["echo_similarity"]) - float(row["gold_similarity"])
    return bool(float(row["DCE"]) >= .70 and margin >= .08 and
                float(row["echo_similarity"]) >= .55)


def balanced_select(rows, count=5, emphasize_affected=False):
    # A case study should represent a distinct source task; the same sample
    # appearing under another framework/condition is not counted twice.
    rows = unique_take(rows, 100, key_fields=("sample_id",))
    if emphasize_affected:
        rows.sort(key=lambda r: (not r["echo_affected"], r["answer_correct"], -r["DCE"]))
    else:
        rows.sort(key=lambda r: (r["answer_correct"], -r["DCE"]))
    chosen = rows[:count]
    # Preserve a correct contrast and an error whenever both exist.
    if chosen and not any(r["answer_correct"] for r in chosen):
        replacement = next((r for r in rows if r["answer_correct"]), None)
        if replacement:
            chosen[-1] = replacement
    if chosen and all(r["answer_correct"] for r in chosen):
        replacement = next((r for r in rows if not r["answer_correct"]), None)
        if replacement:
            chosen[-1] = replacement
    return chosen


def final_cases(solution):
    decisions = jsonl(directory(solution) / "sample_level_decisions.jsonl")
    candidates = jsonl(directory(solution) / "case_candidates.jsonl")
    last_turn = {}
    for row in candidates:
        if row["stage"] != "final_decision":
            continue
        key = (row["sample_id"], row["framework"], row["condition"])
        if key not in last_turn or int(row.get("case_id", "0").rsplit("|", 1)[-1]) > int(last_turn[key].get("case_id", "0").rsplit("|", 1)[-1]):
            last_turn[key] = row
    rows = []
    for decision in decisions:
        key = (decision["sample_id"], decision["framework"], decision["condition"])
        base = dict(last_turn[key])
        base.update({
            "question": decision["question"], "snippets": decision["snippets"],
            "freeform_answer": decision["freeform_answer"],
            "echo_similarity": decision["echo_similarity"],
            "gold_similarity": decision["gold_similarity"],
            "DCE": decision["DCE"],
            "RRI": decision["RRI"], "WEP": decision["WEP"],
        })
        base["answer_correct"] = correctness(base)
        base["echo_affected"] = is_echo_affected(base)
        rows.append(base)
    rows.sort(key=lambda r: (not r["echo_affected"], r["answer_correct"], -r["DCE"]))
    return balanced_select(rows, 5, emphasize_affected=True)


def finalize(output_root):
    evaluated = embed_scores(jsonl(output_root / "case_audit_predictions.jsonl"))
    for row in evaluated:
        row["answer_correct"] = correctness(row)
        row["echo_affected"] = is_echo_affected(row)
    combined = []
    for solution in SOLUTIONS:
        solution_rows = add_run_metrics(solution, [r for r in evaluated if r["solution"] == solution])
        selected = []
        for stage in STAGES[:-1]:
            subset = [r for r in solution_rows if r["stage"] == stage]
            selected.extend(balanced_select(
                subset, 5,
                emphasize_affected=stage in {"updating_reconsolidation", "delayed_retrieval"},
            ))
        selected.extend(final_cases(solution))
        (directory(solution) / "stage_cases.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fields = ["solution", "stage", "case_id", "sample_id", "dataset", "ability",
                  "framework", "condition", "qa_type", "question", "freeform_answer",
                  "gold_answer", "echo_answer", "ED", "RRI", "WEP", "DCE",
                  "answer_correct", "echo_affected"]
        with (directory(solution) / "stage_cases.csv").open("w", encoding="utf-8-sig", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(selected)
        combined.extend(selected)
    (output_root / "all_stage_cases.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path,
                        default=ROOT / "results_memory_echo_abcd_top5")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip_generation", action="store_true")
    args = parser.parse_args()
    count = prepare(args.output_dir)
    print(f"prepared={count}", flush=True)
    if not args.skip_generation:
        generate(args.output_dir / "case_audit_inputs.jsonl",
                 args.output_dir / "case_audit_predictions.jsonl",
                 "case_id", "deepseek-chat", "https://api.deepseek.com",
                 args.workers, 4)
    combined = finalize(args.output_dir)
    print(f"final_cases={len(combined)}", flush=True)


if __name__ == "__main__":
    main()
