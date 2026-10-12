"""Recompute deterministic answers from saved retrievals without rerunning memory."""
import argparse
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from memory_echo_generic.generic_trajectory_builder import (
    _predicted_label,
    _run_metrics,
    agent_answer_for_label,
)
from memory_echo_phase3.semantic_answer_evaluator import (
    SemanticAnswerEvaluator,
)
from scripts.legacy.run_generic_memory_echo import (
    _local_model_path,
    _preload_semantic_texts,
)


CORE_METRICS = ("ED", "UIR", "EPS", "DBI")


def rebuild(output_dir: Path):
    sample_path = output_dir / "sample_level.json"
    checkpoint_path = output_dir / "sample_level.checkpoint.jsonl"
    samples = json.loads(sample_path.read_text(encoding="utf-8"))
    model = SentenceTransformer(str(_local_model_path()))
    evaluator = SemanticAnswerEvaluator(model)
    _preload_semantic_texts(
        evaluator, samples, samples[0]["runs"][0]["qa_count"]
    )

    changed_turns = 0
    for sample in samples:
        for run in sample["runs"]:
            before = {
                metric: run["metrics"].get(metric)
                for metric in CORE_METRICS
            }
            for turn in run["memory_trajectory"]:
                predicted_label = _predicted_label(
                    turn, turn["retrieved_memories"]
                )
                if predicted_label != turn["predicted_memory_label"]:
                    changed_turns += 1
                answer, predicted_value = agent_answer_for_label(
                    sample, turn, predicted_label
                )
                turn["predicted_memory_label"] = predicted_label
                turn["predicted_value"] = predicted_value
                turn["agent_answer"] = answer
                turn["answer_evaluation"] = evaluator.evaluate(
                    answer,
                    turn["canonical_answer"],
                    predicted_value,
                    turn["canonical_target_value"],
                )
                eligible = bool(
                    turn["phase"] == "decision_influence"
                    and turn["qa_type"] == "decision"
                    and sample["should_update"]
                )
                turn["decision_bias_eligible"] = eligible
                turn["decision_bias_flag"] = float(
                    eligible and predicted_label == "OLD"
                )
            run["metrics"] = _run_metrics(run["memory_trajectory"])
            after = {
                metric: run["metrics"].get(metric)
                for metric in CORE_METRICS
            }
            if before != after:
                raise RuntimeError(
                    "Decision-layer rebuild changed a core metric: "
                    f"sample={sample['sample_id']} "
                    f"framework={run['framework']} "
                    f"condition={run['condition']} "
                    f"before={before} after={after}"
                )
            decision_turns = [
                turn for turn in run["memory_trajectory"]
                if turn["phase"] == "decision_influence"
                and turn["qa_type"] == "decision"
            ]
            run["final_decisions"] = [
                {
                    "turn": turn["turn"],
                    "predicted_memory_label": (
                        turn["predicted_memory_label"]
                    ),
                    "predicted_value": turn["predicted_value"],
                    "canonical_target_label": (
                        turn["canonical_target_label"]
                    ),
                    "canonical_target_value": (
                        turn["canonical_target_value"]
                    ),
                    "biased_to_old": bool(turn["decision_bias_flag"]),
                }
                for turn in decision_turns
            ]

    sample_tmp = sample_path.with_suffix(".json.rebuild.tmp")
    with sample_tmp.open("w", encoding="utf-8") as handle:
        json.dump(
            samples, handle, ensure_ascii=False, separators=(",", ":")
        )
    os.replace(sample_tmp, sample_path)

    checkpoint_tmp = checkpoint_path.with_suffix(".jsonl.rebuild.tmp")
    with checkpoint_tmp.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(
                sample, ensure_ascii=False, separators=(",", ":")
            ))
            handle.write("\n")
    os.replace(checkpoint_tmp, checkpoint_path)
    return {
        "base_trajectories": len(samples),
        "decision_layer_turns_changed": changed_turns,
        "core_metrics_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="results_memory_echo_generic"
    )
    args = parser.parse_args()
    report = rebuild(Path(args.output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
