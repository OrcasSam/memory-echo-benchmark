"""Run the source-derived, five-stage Generic Memory Echo Benchmark."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("POSTHOG_DISABLED", "true")
os.environ.setdefault("MEM0_TELEMETRY", "false")
ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib"))

from sentence_transformers import SentenceTransformer

from memory_echo_lifecycle import CONDITIONS
from memory_echo_lifecycle.aggregator import aggregate, write_csv
from memory_echo_lifecycle.figures import write_figures
from memory_echo_lifecycle.qa_generator import generate_stage_qa
from memory_echo_lifecycle.multi_source_rewriter import (
    rewrite_multi_source_dataset, source_distribution,
)
from memory_echo_lifecycle.trajectory_builder import build_trajectory
from memory_echo_phase3.memory_adapter import AMemoryAdapter, Mem0Adapter


def _model_path():
    snapshots = (WORKSPACE / ".hf_cache" / "hub" /
                 "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots")
    candidates = sorted(p for p in snapshots.glob("*") if (p / "modules.json").exists())
    return str(candidates[-1]) if candidates else "all-MiniLM-L6-v2"


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _conclusion(aggregates):
    effects = aggregates["paired_effects_summary"]
    update = [r for r in effects if r["condition"] == "update"]
    ed = sum(r["ED_effect"] for r in update) / len(update) if update else None
    rri = sum(r["RRI_effect"] for r in update if r["RRI_effect"] is not None) / max(1, sum(r["RRI_effect"] is not None for r in update)) if update else None
    dce = sum(r["DCE_effect"] for r in update) / len(update) if update else None
    observed = bool(ed is not None and rri is not None and dce is not None and ed > 0 and rri > 0 and dce > 0)
    fmt = lambda x: "NA" if x is None else f"{x:.3f}"
    return "\n".join([
        f"Memory Echo observed: {'YES' if observed else 'NO / INCONCLUSIVE'}",
        f"Formation: update-control retrieval-practice ED effect = {fmt(ed)}.",
        f"Reconsolidation: control-adjusted RRI effect = {fmt(rri)}.",
        f"Decision transfer: update-control DCE effect = {fmt(dce)}.",
        f"Persistence: update-control WEP effect = {fmt(sum(r['WEP_effect'] for r in update if r['WEP_effect'] is not None) / max(1, sum(r['WEP_effect'] is not None for r in update)) if update else None)}.",
        "Interpretation requires all stages: occupancy alone is not evidence of downstream contamination.",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", default=str(ROOT / "data" / "locomo10.json"))
    parser.add_argument("--longmemeval", default=str(WORKSPACE / "datasets" / "LongMemEval" / "data" / "longmemeval_oracle.json"))
    parser.add_argument("--hotpotqa", default=str(WORKSPACE / "datasets" / "HotpotQA" / "hotpot_dev_distractor.parquet"))
    parser.add_argument("--beam", default=str(WORKSPACE / "datasets" / "BEAM"))
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--practice_qa", type=int, default=6)
    parser.add_argument("--final_qa", type=int, default=4)
    parser.add_argument("--delay_points", default=",".join(map(str, range(1, 31))))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dce_threshold", type=float, default=.5)
    parser.add_argument("--frameworks", default="amem")
    parser.add_argument(
        "--rewritten_dataset",
        help="Optional frozen rewritten_dataset.json. When set, no source rewriting is performed.",
    )
    parser.add_argument("--sample_start", type=int, default=0,
                        help="Inclusive index in the frozen/full rewritten dataset.")
    parser.add_argument("--sample_end", type=int,
                        help="Exclusive index in the frozen/full rewritten dataset.")
    parser.add_argument("--output_dir", default="results_memory_echo_lifecycle")
    args = parser.parse_args()
    if args.max_samples < 1:
        raise ValueError("max_samples must be positive")
    delays = tuple(int(x) for x in args.delay_points.split(",") if x.strip())
    if len(delays) < 2 or delays != tuple(range(1, delays[-1] + 1)):
        raise ValueError("delay_points must be consecutive integers beginning at 1")
    if not 0 <= args.dce_threshold <= 1:
        raise ValueError("dce_threshold must be in [0, 1]")
    frameworks = [x.strip().lower() for x in args.frameworks.split(",") if x.strip()]
    if set(frameworks) - {"amem", "mem0"}:
        raise ValueError("frameworks must contain amem and/or mem0")

    source_paths = {
        "LoCoMo": Path(args.locomo), "LongMemEval": Path(args.longmemeval),
        "HotpotQA": Path(args.hotpotqa), "BEAM": Path(args.beam),
    }
    missing_sources = [f"{name}: {path}" for name, path in source_paths.items() if not path.exists()]
    if missing_sources:
        raise FileNotFoundError("Missing source datasets: " + "; ".join(missing_sources))
    output = Path(args.output_dir)
    if not output.is_absolute(): output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    mem0_root = output / "mem0_store"
    if mem0_root.exists(): shutil.rmtree(mem0_root)
    if args.rewritten_dataset:
        frozen_path = Path(args.rewritten_dataset)
        if not frozen_path.is_absolute(): frozen_path = ROOT / frozen_path
        dataset_full = json.loads(frozen_path.read_text(encoding="utf-8"))
        if not isinstance(dataset_full, list) or not dataset_full:
            raise ValueError("rewritten_dataset must contain a non-empty JSON list")
        expected_delays = list(delays)
        if any(row.get("delay_points") != expected_delays for row in dataset_full):
            raise ValueError("Frozen dataset delay_points do not match --delay_points")
    else:
        frozen_path = None
        dataset_full = rewrite_multi_source_dataset(
            source_paths, args.max_samples, delays, args.practice_qa
        )
    if args.sample_start < 0:
        raise ValueError("sample_start must be non-negative")
    sample_end = len(dataset_full) if args.sample_end is None else args.sample_end
    if sample_end <= args.sample_start or sample_end > len(dataset_full):
        raise ValueError("sample range must be non-empty and inside the rewritten dataset")
    dataset = dataset_full[args.sample_start:sample_end]
    _json(output / "rewritten_dataset.json", dataset)
    model = SentenceTransformer(_model_path())
    samples = []
    try:
        for index, scenario in enumerate(dataset, 1):
            sample = {**scenario, "qa_plan": generate_stage_qa(scenario, args.final_qa), "runs": []}
            for framework in frameworks:
                for condition in CONDITIONS:
                    if framework == "amem":
                        adapter = AMemoryAdapter(model, None)
                    else:
                        adapter = Mem0Adapter(mem0_root, f"lifecycle_{scenario['sample_id']}_{condition}",
                                              collection_name="memory_echo_lifecycle",
                                              embedding_model=_model_path())
                    sample["runs"].append(build_trajectory(
                        scenario, sample["qa_plan"], condition, adapter, model,
                        args.top_k, args.dce_threshold,
                    ))
                    adapter.close()
            samples.append(sample)
            print(f"completed {index}/{len(dataset)}", flush=True)
    finally:
        Mem0Adapter.close_shared()

    aggregates = aggregate(samples)
    manifest = {
        "benchmark": "Source-derived Generic Memory Echo Lifecycle Benchmark",
        "source_datasets": {name: str(path) for name, path in source_paths.items()},
        "source_distribution": source_distribution(dataset),
        "samples": len(samples), "conditions": list(CONDITIONS),
        "full_rewritten_dataset_samples": len(dataset_full),
        "sample_range": [args.sample_start, sample_end],
        "frozen_rewritten_dataset": str(frozen_path) if frozen_path else None,
        "frameworks": frameworks, "stages": ["Encoding", "Retrieval Practice",
        "Updating / Reconsolidation", "Delayed Retrieval", "Final Decision"],
        "delay_points": list(delays), "top_k": args.top_k,
        "metric_definitions": {
            "ED": "rank-weighted semantic occupancy of ECHO-labelled top-k memories",
            "RRI": "mean of Echo survival after correction and residual Echo share; paired RRI subtracts control",
            "WEP": "normalized trapezoidal area under the ED curve over consecutive observed delays",
            "DCE": "scaled semantic advantage of Echo answer over correct answer; thresholded contamination also reported",
        },
    }
    _json(output / "dataset_manifest.json", manifest)
    _json(output / "sample_level.json", samples)
    _json(output / "aggregate_results.json", aggregates)
    write_csv(output / "aggregate_results.csv", aggregates["condition_summary"])
    write_csv(output / "per_stage_summary.csv", aggregates["per_stage_summary"])
    write_csv(output / "per_qa_type_summary.csv", aggregates["per_qa_type_summary"])
    write_csv(output / "per_delay_summary.csv", aggregates["per_delay_summary"])
    write_csv(output / "per_source_summary.csv", aggregates["per_source_summary"])
    write_csv(output / "per_ability_summary.csv", aggregates["per_ability_summary"])
    write_csv(output / "paired_effects.csv", aggregates["paired_effects"])
    write_figures(output, aggregates, samples)
    (output / "conclusion.txt").write_text(_conclusion(aggregates), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "samples": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
