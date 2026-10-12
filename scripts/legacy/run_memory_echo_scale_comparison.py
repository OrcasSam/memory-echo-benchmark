"""Run 30 samples x 200 QA and compare against Memory Echo Benchmark v6."""
import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT.parent / ".hf_cache" / "transformers"))
os.environ.setdefault("MEM0_DIR", str(ROOT.parent / ".mem0_echo_scale"))
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("POSTHOG_DISABLED", "true")

from sentence_transformers import SentenceTransformer
from memory_echo_phase3.cognitive_aggregator import (
    aggregate_by_phase, aggregate_by_phase_and_qa_type, aggregate_by_qa_type,
    aggregate_conditions, build_conclusion, write_csv, write_figures,
)
from memory_echo_phase3.cognitive_qa_generator import (
    PHASE_ALLOCATIONS, generate_cognitive_qa_plan,
)
from memory_echo_phase3.cognitive_trajectory_builder import (
    CONDITIONS, build_cognitive_trajectory,
)
from memory_echo_phase3.dataset_rewriter_v6 import rewrite_webshop_dataset
from memory_echo_phase3.memory_adapter import (
    AMemoryAdapter, Mem0Adapter, MODEL_NAME,
)
from memory_echo_phase3.scale_comparator import (
    compare_condition_rows, compare_phase_rows, load_prior_results,
    write_comparison_figures,
)
from memory_echo_phase3.semantic_answer_evaluator import (
    SemanticAnswerEvaluator, THRESHOLDS,
)
from memory_echo_phase3.webshop_loader import load_webshop_scenarios


def main():
    parser = argparse.ArgumentParser(description="Run scaled Memory Echo comparison")
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "webshop" / "data_10k"))
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--qa_count", type=int, default=200, choices=[100, 200])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", default="results_memory_echo_30x200")
    parser.add_argument("--prior_dir", default="results_memory_echo_v6")
    parser.add_argument("--frameworks", default="amem,mem0")
    args = parser.parse_args()
    if args.max_samples < 30:
        raise ValueError("The scaled comparison requires at least 30 independent samples")

    raw, manifest = load_webshop_scenarios(Path(args.data_dir), args.max_samples)
    scenarios = rewrite_webshop_dataset(raw)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mem0_root = output / "mem0_store"
    if mem0_root.exists():
        shutil.rmtree(mem0_root)
    frameworks = [x.strip() for x in args.frameworks.split(",")]
    model = SentenceTransformer(MODEL_NAME)
    evaluator = SemanticAnswerEvaluator(model)

    manifest.update({
        "experiment": "Memory Echo scaled comparison 30 samples x 200 QA",
        "qa_turns_per_sample": args.qa_count,
        "phase_allocation": PHASE_ALLOCATIONS[args.qa_count],
        "conditions": list(CONDITIONS),
        "frameworks": ["A-MEM", "official Mem0"],
        "semantic_thresholds": THRESHOLDS,
        "semantic_scoring": "0.70 * SBERT cosine similarity + 0.30 * canonical ASIN match",
        "prior_experiment": str(Path(args.prior_dir)),
    })
    rewritten = [
        {**s, "longitudinal_qa_plan": generate_cognitive_qa_plan(s, args.qa_count)}
        for s in scenarios
    ]
    (output / "rewritten_longitudinal_dataset.json").write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    samples = []
    try:
        for scenario in scenarios:
            sample = {**scenario, "runs": []}
            for framework in frameworks:
                for condition in CONDITIONS:
                    adapter = (
                        AMemoryAdapter(model, None) if framework == "amem"
                        else Mem0Adapter(
                            mem0_root, f"scale_s{scenario['sample_id']}_{condition}"
                        )
                    )
                    run = build_cognitive_trajectory(
                        scenario, condition, adapter, evaluator,
                        top_k=args.top_k, qa_count=args.qa_count,
                    )
                    run["framework"] = adapter.framework
                    sample["runs"].append(run)
                    adapter.close()
            samples.append(sample)
            print(f"completed scaled sample {scenario['sample_id'] + 1}/{len(scenarios)}",
                  flush=True)
    finally:
        Mem0Adapter.close_shared()

    qa_rows = aggregate_by_qa_type(samples)
    phase_qa_rows = aggregate_by_phase_and_qa_type(samples)
    phase_rows = aggregate_by_phase(samples)
    condition_rows = aggregate_conditions(samples)
    conclusion = build_conclusion(condition_rows, qa_rows, phase_rows)
    prior_samples = json.loads(
        (Path(args.prior_dir) / "sample_level.json").read_text(encoding="utf-8")
    )
    prior_ids = {s["instruction_id"] for s in prior_samples}
    shared_ids = prior_ids & {s["instruction_id"] for s in samples}
    matched_new_samples = [s for s in samples if s["instruction_id"] in shared_ids]
    matched_old_samples = [s for s in prior_samples if s["instruction_id"] in shared_ids]
    matched_condition_rows = aggregate_conditions(matched_new_samples)
    matched_old_condition_rows = aggregate_conditions(matched_old_samples)
    prior_conditions, prior_phases = load_prior_results(Path(args.prior_dir))
    comparison = compare_condition_rows(
        prior_conditions, condition_rows, matched_condition_rows,
        matched_old_condition_rows, len(shared_ids),
    )
    phase_comparison = compare_phase_rows(prior_phases, phase_rows)

    (output / "sample_level.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output / "aggregate_by_qa_type.csv", qa_rows)
    write_csv(output / "aggregate_by_phase_and_qa_type.csv", phase_qa_rows)
    write_csv(output / "aggregate_by_phase.csv", phase_rows)
    write_csv(output / "condition_summary.csv", condition_rows)
    write_csv(output / "old_vs_new_comparison.csv", comparison)
    write_csv(output / "old_vs_new_phase_comparison.csv", phase_comparison)
    aggregate = {
        "by_qa_type": qa_rows,
        "by_phase_and_qa_type": phase_qa_rows,
        "by_phase": phase_rows,
        "condition_summary": condition_rows,
        "old_vs_new": comparison,
        "old_vs_new_phase": phase_comparison,
    }
    (output / "aggregate_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "conclusion.json").write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_figures(output, qa_rows, condition_rows, samples, phase_rows)
    write_comparison_figures(output, comparison, phase_comparison)
    print(json.dumps({"condition_summary": condition_rows,
                      "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
