"""Aggregate lifecycle metrics by condition, stage, and QA type."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _rows(samples):
    for sample in samples:
        for run in sample["runs"]:
            yield sample, run


def aggregate(samples):
    run_groups, source_groups, ability_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    stage_groups, qa_groups, delay_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for sample, run in _rows(samples):
        run_groups[(run["framework"], run["condition"])].append(run)
        source_groups[(run["framework"], run["condition"], sample["source"]["dataset"])].append(run)
        ability_groups[(run["framework"], run["condition"], sample["source"]["dataset"], sample["source"]["ability"])].append(run)
        for turn in run["trajectory"]:
            stage_groups[(run["framework"], run["condition"], turn["stage"])].append(turn)
            qa_groups[(run["framework"], run["condition"], turn["qa_type"])].append(turn)
        for point in run["delay_series"]:
            delay_groups[(run["framework"], run["condition"], point["delay"])].append(point)
    condition_rows = []
    def metric_row(framework, condition, runs):
        return {
            "framework": framework, "condition": condition, "n": len(runs),
            "ED": _mean(r["metrics"]["ED"]["retrieval_practice"] for r in runs),
            "RRI": _mean(r["metrics"]["RRI"] for r in runs),
            "WEP": _mean(r["metrics"]["WEP"] for r in runs),
            "DCE": _mean(r["metrics"]["DCE"] for r in runs),
            "decision_contamination_rate": _mean(
                r["metrics"]["decision_contamination_rate"] for r in runs),
            "answer_similarity": _mean(r["metrics"]["mean_answer_similarity"] for r in runs),
        }
    for (framework, condition), runs in sorted(run_groups.items()):
        condition_rows.append(metric_row(framework, condition, runs))
    source_rows = []
    for (framework, condition, source), runs in sorted(source_groups.items()):
        source_rows.append({**metric_row(framework, condition, runs), "source_dataset": source})
    ability_rows = []
    for (framework, condition, source, ability), runs in sorted(ability_groups.items()):
        ability_rows.append({**metric_row(framework, condition, runs),
                             "source_dataset": source, "source_ability": ability})
    stage_rows = []
    for (framework, condition, stage), turns in sorted(stage_groups.items()):
        stage_rows.append({
            "framework": framework, "condition": condition, "stage": stage,
            "n": len(turns), "ED": _mean(t["ED"] for t in turns),
            "answer_similarity": _mean(t["answer_similarity"] for t in turns),
            "DCE": _mean(t["DCE"] for t in turns),
        })
    qa_rows = []
    for (framework, condition, qa_type), turns in sorted(qa_groups.items()):
        qa_rows.append({
            "framework": framework, "condition": condition, "qa_type": qa_type,
            "n": len(turns), "ED": _mean(t["ED"] for t in turns),
            "answer_similarity": _mean(t["answer_similarity"] for t in turns),
            "DCE": _mean(t["DCE"] for t in turns),
        })
    delay_rows = []
    for (framework, condition, delay), points in sorted(delay_groups.items()):
        delay_rows.append({
            "framework": framework, "condition": condition, "delay": delay,
            "n": len(points), "ED": _mean(point["ED"] for point in points),
        })
    # Paired shifts count only changes from the matched control trajectory.
    paired = []
    for sample in samples:
        by_key = {(r["framework"], r["condition"]): r for r in sample["runs"]}
        for framework in sorted({key[0] for key in by_key}):
            control = by_key.get((framework, "control"))
            if not control:
                continue
            for condition in ("repetition", "update"):
                run = by_key.get((framework, condition))
                if not run:
                    continue
                paired.append({
                    "sample_id": sample["sample_id"], "framework": framework,
                    "condition": condition,
                    "ED_effect": run["metrics"]["ED"]["retrieval_practice"] - control["metrics"]["ED"]["retrieval_practice"],
                    "RRI_effect": (run["metrics"]["RRI"] - control["metrics"]["RRI"]
                                   if run["metrics"]["RRI"] is not None and control["metrics"]["RRI"] is not None else None),
                    "WEP_effect": (run["metrics"]["WEP"] - control["metrics"]["WEP"]
                                   if run["metrics"]["WEP"] is not None and control["metrics"]["WEP"] is not None else None),
                    "DCE_effect": run["metrics"]["DCE"] - control["metrics"]["DCE"],
                })
    paired_summary = []
    grouped = defaultdict(list)
    for row in paired:
        grouped[(row["framework"], row["condition"])].append(row)
    for (framework, condition), rows in sorted(grouped.items()):
        paired_summary.append({"framework": framework, "condition": condition, "n": len(rows), **{
            metric: _mean(row[metric] for row in rows)
            for metric in ("ED_effect", "RRI_effect", "WEP_effect", "DCE_effect")
        }})
    return {"condition_summary": condition_rows,
            "per_source_summary": source_rows, "per_ability_summary": ability_rows,
            "per_stage_summary": stage_rows,
            "per_qa_type_summary": qa_rows, "per_delay_summary": delay_rows,
            "paired_effects": paired,
            "paired_effects_summary": paired_summary}


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
