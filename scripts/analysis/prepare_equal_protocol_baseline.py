"""Prepare the existing baseline contexts for a temperature-0 matched rerun."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT / "results_memory_echo_solution_freeform_dce_200" /
    "decision_inputs.jsonl"
)
OUTPUT = ROOT / "results_memory_echo_baseline_equal_protocol_top5"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("solution_arm") != "baseline":
                continue
            row["solution"] = "Baseline"
            row["ED"] = row["original_ED"]
            row["RRI"] = row["original_RRI"]
            row["WEP"] = row["original_WEP"]
            row["decision_instruction"] = None
            row["decision_context_audit"] = {
                "policy": "unfiltered_top_k",
                "retrieved_top_k": 5,
                "effective_context_size": len(row["snippets"]),
            }
            rows.append(row)
    if len(rows) != 1200:
        raise RuntimeError(f"expected 1200 baseline inputs, found {len(rows)}")
    with (OUTPUT / "decision_inputs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "arm": "Baseline",
        "samples": 200,
        "decision_observations": len(rows),
        "frameworks": ["A-MEM", "Mem0"],
        "conditions": ["control", "repetition", "update"],
        "top_k": 5,
        "model": "deepseek-chat",
        "temperature": 0.0,
        "source_contexts": str(SOURCE),
        "purpose": "equal-protocol before/after DCE comparison",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"prepared {len(rows)} baseline decisions -> {OUTPUT}")


if __name__ == "__main__":
    main()
