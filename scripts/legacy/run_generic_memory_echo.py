"""Run the large, domain-independent Generic Memory Echo Benchmark."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT.parent / ".hf_cache"
# The benchmark is self-contained in the workspace. Force the known local
# cache because some machine-level environments point at non-writable paths.
os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["TRANSFORMERS_CACHE"] = (
    str(CACHE_ROOT / "hub")
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MEM0_DIR", str(ROOT.parent / ".mem0_echo_generic"))
os.environ.setdefault("POSTHOG_DISABLED", "true")
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".matplotlib"))

from sentence_transformers import SentenceTransformer

from memory_echo_generic.echo_protocol_dataset_generator import (
    dataset_manifest,
    generate_protocol_dataset,
)
from memory_echo_generic.four_metric_aggregator import (
    aggregate_run_metrics,
    aggregate_turn_metrics,
    paired_condition_effects,
    paired_condition_effects_by_seed,
    write_csv,
)
from memory_echo_generic.generic_figures import write_generic_figures
from memory_echo_generic.generic_trajectory_builder import (
    CONDITIONS,
    agent_answer_for_label,
    build_generic_trajectory,
)
from memory_echo_generic.multi_type_qa_generator import generate_qa_plan
from memory_echo_generic.result_summarizer import summarize
from memory_echo_phase3.memory_adapter import (
    AMemoryAdapter,
    MODEL_NAME,
    Mem0Adapter,
)
from memory_echo_phase3.semantic_answer_evaluator import (
    SemanticAnswerEvaluator,
    THRESHOLDS,
)


def _output_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _local_model_path() -> Path:
    snapshots = (
        CACHE_ROOT / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )
    candidates = sorted(
        path for path in snapshots.glob("*")
        if (path / "modules.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Local SentenceTransformer cache is missing under {snapshots}"
        )
    return candidates[-1]


def _write_json(path: Path, value, *, compact=False):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
        )


def _preload_semantic_texts(evaluator, dataset, qa_count):
    texts = []
    for sample in dataset:
        for qa in generate_qa_plan(sample, qa_count):
            texts.append(qa["canonical_answer"])
            for label in ("OLD", "NEW", None):
                texts.append(agent_answer_for_label(sample, qa, label)[0])
    evaluator.preload(texts)


def _conclusion_text(conclusion):
    effects = conclusion["mean_paired_update_control_effects"]
    stable = (
        "稳定观察到" if conclusion["stable_memory_echo_observed"]
        else "未在全部预注册条件下稳定观察到"
    )

    def number(value):
        return "NA" if value is None else f"{value:.3f}"

    return "\n".join([
        f"总体结论：{stable}通用 Memory Echo。",
        (
            f"形成：写回组相对控制组的平均 ED 增量为 "
            f"{number(effects['ED_lift'])}。"
        ),
        (
            f"放大：写回组相对控制组的平均 UIR 下降为 "
            f"{number(effects['UIR_drop'])}；更关键的 "
            "update-minus-repetition 对比见 paired_condition_effects.csv。"
        ),
        (
            f"持续：写回组平均 EPS 为 "
            f"{number(conclusion['mean_update_condition_EPS'])}。"
        ),
        (
            f"影响：写回组相对控制组的平均 DBI 增量为 "
            f"{number(effects['DBI_lift'])}。"
        ),
        (
            f"最易形成 Echo 的协议："
            f"{conclusion['easiest_echo_formation_protocol']}；"
            f"最敏感 QA：{conclusion['most_sensitive_qa_type']}。"
        ),
        conclusion["interpretation"],
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Run the domain-independent Generic Memory Echo Benchmark"
    )
    parser.add_argument("--samples_per_protocol", type=int, default=100)
    parser.add_argument("--qa_per_trajectory", type=int, default=20)
    parser.add_argument("--seeds", default="11,29,47")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--output_dir", default="results_memory_echo_generic"
    )
    parser.add_argument("--frameworks", default="amem,mem0")
    parser.add_argument(
        "--debug_allow_small",
        action="store_true",
        help="Allow fewer than 100 trajectories per protocol; never formal.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed samples from sample_level.checkpoint.jsonl.",
    )
    args = parser.parse_args()

    seeds = tuple(
        int(value.strip()) for value in args.seeds.split(",")
        if value.strip()
    )
    frameworks = [
        value.strip().lower() for value in args.frameworks.split(",")
        if value.strip()
    ]
    unsupported = set(frameworks) - {"amem", "mem0"}
    if unsupported:
        raise ValueError(f"Unsupported frameworks: {sorted(unsupported)}")
    if not frameworks:
        raise ValueError("At least one framework is required")

    dataset = generate_protocol_dataset(
        samples_per_protocol=args.samples_per_protocol,
        seeds=seeds,
        allow_small_debug=args.debug_allow_small,
    )
    print(
        f"prepared {len(dataset)} base trajectories; loading embedding model",
        flush=True,
    )
    output = _output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    previous_manifest = {}
    previous_manifest_path = output / "dataset_manifest.json"
    if args.resume and previous_manifest_path.exists():
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
    mem0_root = output / "mem0_store"
    if mem0_root.exists() and not args.resume:
        shutil.rmtree(mem0_root)
    journal = output / "sample_level.checkpoint.jsonl"
    if journal.exists() and not args.resume:
        journal.unlink()
    resumed_samples = []
    if args.resume:
        if not journal.exists():
            raise FileNotFoundError(
                f"Cannot resume without checkpoint: {journal}"
            )
        with journal.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    resumed_samples.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid checkpoint line {line_number}"
                    ) from error
        expected_runs = len(frameworks) * len(CONDITIONS)
        if any(
            len(sample.get("runs", [])) != expected_runs
            for sample in resumed_samples
        ):
            raise ValueError(
                "Checkpoint contains an incomplete condition trajectory"
            )
        expected_ids = list(range(len(resumed_samples)))
        if [sample["sample_id"] for sample in resumed_samples] != expected_ids:
            raise ValueError(
                "Checkpoint samples are not a contiguous prefix"
            )
        print(
            f"resuming from {len(resumed_samples)}/{len(dataset)} samples",
            flush=True,
        )

    local_model_path = _local_model_path()
    model = SentenceTransformer(str(local_model_path))
    evaluator = SemanticAnswerEvaluator(model)
    semantic_started = time.monotonic()
    _preload_semantic_texts(
        evaluator, dataset, args.qa_per_trajectory
    )
    print(
        f"semantic cache ready ({len(evaluator._canonical_cache)} texts, "
        f"{time.monotonic() - semantic_started:.1f}s)",
        flush=True,
    )

    qa_dataset = [
        {
            **sample,
            "longitudinal_qa_plan": generate_qa_plan(
                sample, args.qa_per_trajectory
            ),
        }
        for sample in dataset
    ]
    _write_json(
        output / "generic_protocol_dataset.json", qa_dataset,
        compact=False,
    )
    print("protocol dataset and QA plans written", flush=True)

    manifest = dataset_manifest(
        dataset,
        args.samples_per_protocol,
        seeds,
        args.qa_per_trajectory,
    )
    manifest.update({
        "formal_run": (
            args.samples_per_protocol >= 100
            and not args.debug_allow_small
        ),
        "conditions": list(CONDITIONS),
        "frameworks": [
            "A-MEM real SimpleEmbeddingRetriever"
            if name == "amem" else
            "official mem0ai Memory.add/search/update"
            for name in frameworks
        ],
        "top_k": args.top_k,
        "condition_trajectories": (
            len(dataset) * len(CONDITIONS) * len(frameworks)
        ),
        "executed_qa_count": (
            len(dataset) * len(CONDITIONS) * len(frameworks)
            * args.qa_per_trajectory
        ),
        "experimental_unit": (
            "independent base trajectory; QA within a trajectory are "
            "correlated probes"
        ),
        "random_seed_design": (
            "matched configuration replicates across seeds; concrete values "
            "and distractor order vary by seed"
        ),
        "delay_semantics": (
            "session/noise-gap persistence; neither framework is claimed "
            "to implement wall-clock forgetting"
        ),
        "decision_policy": (
            "fixed visible replacement-scope gate followed by rank-discounted "
            "retrieval-to-answer policy, held constant across frameworks and "
            "conditions"
        ),
        "semantic_scoring": (
            "0.70 * normalized SBERT cosine similarity + "
            "0.30 * canonical value match"
        ),
        "semantic_thresholds": THRESHOLDS,
        "metric_formulas": {
            "ED": "mean count(OLD in top-k)/retrieved_count on pre-stop probe",
            "UIR": (
                "mean W_NEW/(W_OLD+W_NEW) after conflict, "
                "W(rank)=1/log2(rank+1)"
            ),
            "EPS": "ED_delayed_matched_probe / ED_pre_stop_matched_probe",
            "DBI": (
                "OLD-biased outputs / decision probes where valid evidence "
                "requires NEW"
            ),
        },
    })
    _write_json(output / "dataset_manifest.json", manifest)
    print("starting A-MEM / official Mem0 condition runs", flush=True)

    resume_count = len(resumed_samples)
    formal_resume_point = previous_manifest.get(
        "formal_execution_resume_point"
    )
    if resume_count < len(dataset):
        formal_resume_point = (
            resume_count if formal_resume_point is None
            else min(formal_resume_point, resume_count)
        )
    manifest["formal_execution_resume_point"] = formal_resume_point
    manifest["checkpoint_samples_loaded_for_this_invocation"] = resume_count
    manifest["samples_executed_this_invocation"] = (
        len(dataset) - resume_count
    )
    _write_json(output / "dataset_manifest.json", manifest)
    samples = resumed_samples
    completed_ids = {sample["sample_id"] for sample in resumed_samples}
    namespace_run_tag = (
        f"resume{resume_count}" if args.resume else "fresh"
    )
    started = time.monotonic()
    try:
        for index, scenario in enumerate(dataset, 1):
            if scenario["sample_id"] in completed_ids:
                continue
            sample = {**scenario, "runs": []}
            for framework in frameworks:
                for condition in CONDITIONS:
                    if index == resume_count + 1:
                        print(
                            f"first sample: starting {framework}/{condition}",
                            flush=True,
                        )
                    if framework == "amem":
                        adapter = AMemoryAdapter(model, None)
                    else:
                        namespace = (
                            f"generic_{namespace_run_tag}_"
                            f"seed{scenario['random_seed']}_"
                            f"{scenario['protocol']}_"
                            f"s{scenario['sample_id']}_{condition}"
                        )
                        adapter = Mem0Adapter(
                            mem0_root,
                            namespace,
                            collection_name="generic_memory_echo",
                            embedding_model=str(local_model_path),
                        )
                    run = build_generic_trajectory(
                        scenario,
                        condition,
                        adapter,
                        evaluator,
                        top_k=args.top_k,
                        qa_count=args.qa_per_trajectory,
                    )
                    run["framework"] = adapter.framework
                    sample["runs"].append(run)
                    adapter.close()
                    if index == resume_count + 1:
                        print(
                            f"first sample: finished {framework}/{condition}",
                            flush=True,
                        )
            samples.append(sample)
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    sample, ensure_ascii=False, separators=(",", ":")
                ))
                handle.write("\n")
            elapsed = time.monotonic() - started
            print(
                f"completed generic sample {index}/{len(dataset)} "
                f"({elapsed / 60:.1f} min)",
                flush=True,
            )
    finally:
        Mem0Adapter.close_shared()

    run_aggregates = aggregate_run_metrics(samples)
    turn_aggregates = aggregate_turn_metrics(samples)
    paired_rows = paired_condition_effects(samples)
    seed_paired_rows = paired_condition_effects_by_seed(samples)
    conclusion = summarize(
        run_aggregates["protocol_summary"],
        paired_rows,
        turn_aggregates["phase_summary"],
        turn_aggregates["qa_type_summary"],
        seed_paired_rows,
    )

    _write_json(output / "sample_level.json", samples, compact=True)
    aggregate_json = {
        key: value for key, value in run_aggregates.items()
        if key != "run_records"
    }
    aggregate_json.update(turn_aggregates)
    aggregate_json["paired_condition_effects"] = paired_rows
    aggregate_json["paired_condition_effects_by_seed"] = seed_paired_rows
    aggregate_json["metric_formulas"] = manifest["metric_formulas"]
    _write_json(output / "aggregate_results.json", aggregate_json)

    csv_outputs = {
        "condition_summary.csv": run_aggregates["condition_summary"],
        "protocol_summary.csv": run_aggregates["protocol_summary"],
        "phase_summary.csv": turn_aggregates["phase_summary"],
        "qa_type_summary.csv": turn_aggregates["qa_type_summary"],
        "memory_type_summary.csv": run_aggregates["memory_type_summary"],
        "seed_summary.csv": run_aggregates["seed_summary"],
        "evidence_strength_summary.csv": (
            run_aggregates["evidence_strength_summary"]
        ),
        "paired_condition_effects.csv": paired_rows,
        "paired_condition_effects_by_seed.csv": seed_paired_rows,
    }
    for filename, rows in csv_outputs.items():
        write_csv(output / filename, rows)

    _write_json(output / "conclusion.json", conclusion)
    (output / "conclusion.txt").write_text(
        _conclusion_text(conclusion), encoding="utf-8"
    )
    write_generic_figures(
        output,
        run_aggregates,
        turn_aggregates,
        paired_rows,
        samples,
    )
    print(json.dumps({
        "output_dir": str(output),
        "manifest": {
            "base_trajectories": manifest["base_trajectories"],
            "condition_trajectories": manifest["condition_trajectories"],
            "executed_qa_count": manifest["executed_qa_count"],
        },
        "conclusion": conclusion,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
