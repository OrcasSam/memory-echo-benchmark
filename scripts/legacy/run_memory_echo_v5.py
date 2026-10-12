"""Fifth Memory Echo experiment: cognitive, scenario-driven WebShop benchmark."""
import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT.parent / ".hf_cache" / "transformers"))
os.environ.setdefault("MEM0_DIR", str(ROOT.parent / ".mem0_echo_v5"))
os.environ.setdefault("POSTHOG_DISABLED", "true")
os.environ.setdefault("MEM0_TELEMETRY", "false")

from sentence_transformers import SentenceTransformer
from memory_echo_phase3.memory_adapter import AMemoryAdapter, Mem0Adapter, MODEL_NAME
from memory_echo_phase3.webshop_loader import load_webshop_scenarios
from memory_echo_phase3.cognitive_qa_generator import generate_cognitive_qa_plan
from memory_echo_phase3.semantic_answer_evaluator import SemanticAnswerEvaluator, THRESHOLDS
from memory_echo_phase3.cognitive_trajectory_builder import CONDITIONS, build_cognitive_trajectory
from memory_echo_phase3.cognitive_aggregator import (
    aggregate_by_qa_type, aggregate_conditions, build_conclusion,
    write_csv, write_figures,
)


def main():
    parser = argparse.ArgumentParser(description="Run cognitive Memory Echo Benchmark v5 on WebShop")
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "webshop" / "data"))
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", default="results_memory_echo_v5")
    parser.add_argument("--frameworks", default="amem,mem0")
    args = parser.parse_args()

    scenarios, manifest = load_webshop_scenarios(Path(args.data_dir), args.max_samples)
    manifest.update({
        "experiment": "Memory Echo Benchmark v5 cognitive longitudinal paradigm",
        "cognitive_phases": ["encoding", "retrieval_practice", "updating_reconsolidation",
                             "delayed_retrieval", "final_decision"],
        "qa_types": ["recommendation", "comparison", "justification", "memory_recall",
                     "update_awareness", "decision", "consistency_reevaluation"],
        "semantic_thresholds": THRESHOLDS,
        "conditions": list(CONDITIONS),
    })
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    mem0_root = output / "mem0_store"
    if mem0_root.exists(): shutil.rmtree(mem0_root)
    frameworks = [x.strip() for x in args.frameworks.split(",")]
    model = SentenceTransformer(MODEL_NAME)
    evaluator = SemanticAnswerEvaluator(model)

    rewritten = []
    for scenario in scenarios:
        rewritten.append({**scenario, "longitudinal_qa_plan": generate_cognitive_qa_plan(scenario)})
    (output / "rewritten_longitudinal_dataset.json").write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    samples = []
    try:
        for scenario in scenarios:
            sample = {**scenario, "runs": []}
            for framework in frameworks:
                for condition in CONDITIONS:
                    adapter = (AMemoryAdapter(model, None) if framework == "amem" else
                               Mem0Adapter(mem0_root, f"v5_s{scenario['sample_id']}_{condition}"))
                    run = build_cognitive_trajectory(scenario, condition, adapter, evaluator, args.top_k)
                    run["framework"] = adapter.framework
                    sample["runs"].append(run)
                    adapter.close()
            samples.append(sample)
            print(f"completed cognitive WebShop sample {scenario['sample_id'] + 1}/{len(scenarios)}", flush=True)
    finally:
        Mem0Adapter.close_shared()

    qa_rows = aggregate_by_qa_type(samples)
    condition_rows = aggregate_conditions(samples)
    conclusion = build_conclusion(condition_rows, qa_rows)
    (output / "sample_level.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "aggregate_by_qa_type.csv", qa_rows)
    write_csv(output / "condition_summary.csv", condition_rows)
    (output / "aggregate_results.json").write_text(
        json.dumps({"by_qa_type": qa_rows, "condition_summary": condition_rows}, indent=2), encoding="utf-8"
    )
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "conclusion.json").write_text(json.dumps(conclusion, indent=2), encoding="utf-8")
    (output / "conclusion.txt").write_text(
        conclusion["interpretation"] + "\nMost affected QA type: " + conclusion["most_affected_qa_type"] + "\n",
        encoding="utf-8"
    )
    write_figures(output, qa_rows, condition_rows, samples)
    print(json.dumps({"condition_summary": condition_rows, "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
