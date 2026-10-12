"""Run the strict six-level Memory Echo lifecycle governance validation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("POSTHOG_DISABLED", "true")
os.environ.setdefault("MEM0_TELEMETRY", "false")

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / ".matplotlib"))
os.environ.setdefault("MEM0_DIR", str(WORKSPACE / ".mem0"))

from sentence_transformers import SentenceTransformer

from memory_echo_lifecycle import CONDITIONS
from memory_echo_lifecycle.qa_generator import generate_stage_qa
from memory_echo_phase3.memory_adapter import AMemoryAdapter, Mem0Adapter
from memory_echo_solution import ARMS
from memory_echo_solution.aggregator import (
    _bootstrap_ci, aggregate_solution, write_csv,
)
from memory_echo_solution.cached_encoder import CachedEncoder
from memory_echo_solution.figures import write_solution_figures
from memory_echo_solution.guard import EchoGuardMemory
from memory_echo_solution.probes import (
    PROBES, run_control_probe, run_decision_gate_challenge,
)
from memory_echo_solution.trajectory import build_solution_trajectory


def _model_path():
    snapshots = (WORKSPACE / ".hf_cache" / "hub" /
                 "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots")
    candidates = sorted(p for p in snapshots.glob("*") if (p / "modules.json").exists())
    return str(candidates[-1]) if candidates else "all-MiniLM-L6-v2"


def _json(path: Path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _adapter(framework, encoder, mem0_root, namespace):
    if framework == "amem":
        return AMemoryAdapter(encoder, None)
    return Mem0Adapter(
        mem0_root, namespace, collection_name="memory_echo_solution_strict",
        embedding_model=_model_path(),
    )


def _protocol_checks(aggregates):
    rows = aggregates["condition_summary"]
    lookup = {(r["framework"], r["solution_arm"], r["condition"]): r for r in rows}
    checks = []
    for framework in sorted({r["framework"] for r in rows}):
        for condition in ("repetition", "update"):
            for metric in ("ED", "RRI", "WEP", "DCE"):
                effect = (lookup[(framework, "baseline", condition)][metric]
                          - lookup[(framework, "baseline", "control")][metric])
                checks.append({
                    "framework": framework, "check": "positive_echo_manipulation",
                    "arm": "baseline", "condition": condition, "metric": metric,
                    "observed": effect, "criterion": "> 0", "passed": effect > 0,
                })
            s1 = lookup[(framework, "s1_identity", condition)]
            for metric in ("family_pair_recall", "provenance_inheritance_accuracy"):
                checks.append({
                    "framework": framework, "check": "S1_identity_provenance",
                    "arm": "s1_identity", "condition": condition, "metric": metric,
                    "observed": s1[metric], "criterion": ">= 0.95",
                    "passed": s1[metric] >= 0.95,
                })
            s2 = lookup[(framework, "s2_retrieval_hygiene", condition)]
            checks.extend([
                {"framework": framework, "check": "S2_search_does_not_increase_confidence",
                 "arm": "s2_retrieval_hygiene", "condition": condition,
                 "metric": "max_confidence_drift_on_search",
                 "observed": s2["max_confidence_drift_on_search"], "criterion": "= 0",
                 "passed": abs(s2["max_confidence_drift_on_search"]) <= 1e-12},
                {"framework": framework, "check": "S2_retrieval_add_is_log_not_fact",
                 "arm": "s2_retrieval_hygiene", "condition": condition,
                 "metric": "blocked_derived_adds", "observed": s2["blocked_derived_adds"],
                 "criterion": "> 0", "passed": s2["blocked_derived_adds"] > 0},
            ])
        s3 = lookup[(framework, "s3_reconsolidation", "update")]
        checks.extend([
            {"framework": framework, "check": "S3_blocks_retrieval_derived_update",
             "arm": "s3_reconsolidation", "condition": "update",
             "metric": "blocked_derived_updates", "observed": s3["blocked_derived_updates"],
             "criterion": "> 0", "passed": s3["blocked_derived_updates"] > 0},
            {"framework": framework, "check": "S3_marks_verified_supersession",
             "arm": "s3_reconsolidation", "condition": "update",
             "metric": "superseded_claim_records", "observed": s3["superseded_claim_records"],
             "criterion": "> 0", "passed": s3["superseded_claim_records"] > 0},
        ])
        for condition in ("repetition", "update"):
            s4 = lookup[(framework, "s4_persistence", condition)]
            residual_excess = _mean(
                r["excess_WEP"] for r in aggregates["condition_effects"]
                if r["framework"] == framework
                and r["solution_arm"] == "s4_persistence"
                and r["condition"] == condition
            )
            checks.append({
                "framework": framework, "check": "S4_continuous_window_remediation",
                "arm": "s4_persistence", "condition": condition,
                "metric": "remediation_triggers", "observed": s4["remediation_triggers"],
                "criterion": "trigger > 0 when residual excess WEP > 0.05",
                "passed": (residual_excess <= 0.05 or s4["remediation_triggers"] > 0),
            })
        for arm in ARMS[1:]:
            for metric in ("ED", "RRI", "WEP", "DCE", "final_accuracy"):
                delta = (lookup[(framework, arm, "control")][metric]
                         - lookup[(framework, "baseline", "control")][metric])
                # State marking may lower Control RRI/WEP by design; normal task
                # utility is evaluated separately, while ED/DCE/accuracy must stay stable.
                if metric in {"ED", "DCE", "final_accuracy"}:
                    checks.append({
                        "framework": framework, "check": "control_path_invariance",
                        "arm": arm, "condition": "control", "metric": metric,
                        "observed": delta, "criterion": "abs(delta) <= 1e-9",
                        "passed": abs(delta) <= 1e-9,
                    })
    return checks


def _strict_validation(aggregates):
    effects = aggregates["condition_effects"]
    causal = aggregates["causal_estimands"]
    rows = []
    for framework in sorted({r["framework"] for r in effects}):
        for condition in ("repetition", "update"):
            subset = [r for r in effects if r["framework"] == framework
                      and r["condition"] == condition]
            for metric, key in (("ED", "ED_effect"), ("RRI", "RRI_effect"),
                                ("WEP", "excess_WEP"), ("DCE", "DCE_effect")):
                baseline = _mean(r[key] for r in subset if r["solution_arm"] == "baseline")
                full = _mean(r[key] for r in subset if r["solution_arm"] == "full_guard")
                did_metric = "excess_WEP" if metric == "WEP" else metric
                did = next(r for r in causal
                           if r["framework"] == framework and r["condition"] == condition
                           and r["treated_arm"] == "full_guard"
                           and r["reference_arm"] == "baseline"
                           and r["metric"] == did_metric)
                reduction = None if baseline is None or abs(baseline) < 1e-12 else (
                    baseline - full) / abs(baseline)
                delay30_excess = _mean(
                    r["delay30_excess"] for r in subset
                    if r["solution_arm"] == "full_guard"
                )
                if metric == "ED":
                    criterion = "extra ED reduction >= 50% and DiD 95%CI < 0"
                    passed = reduction is not None and reduction >= 0.50 and did["ci_high"] < 0
                elif metric == "RRI":
                    criterion = "RRI below Baseline, DiD 95%CI < 0, residual effect <= 0.05"
                    passed = did["ci_high"] < 0 and full <= 0.05
                elif metric == "WEP":
                    criterion = "excess WEP AUC reduction >= 60%, delay-30 excess <= 0.02"
                    passed = (reduction is not None and reduction >= 0.60
                              and did["ci_high"] < 0 and delay30_excess <= 0.02)
                else:
                    criterion = "DCE below Baseline, residual effect within 0.05 of Control"
                    passed = did["ci_high"] < 0 and abs(full) <= 0.05
                rows.append({
                    "framework": framework, "condition": condition, "metric": metric,
                    "baseline_echo_effect": baseline, "full_guard_echo_effect": full,
                    "relative_reduction": reduction, "delay30_excess": delay30_excess,
                    "ci_low_DiD": did["ci_low"], "ci_high_DiD": did["ci_high"],
                    "p_holm": did["p_holm"], "criterion": criterion,
                    "target_met": bool(passed),
                })
    return rows


def _probe_analysis(samples):
    grouped = defaultdict(list)
    for sample in samples:
        for row in sample.get("control_probes", []):
            grouped[(row["framework"], row["probe"])].append(row)
    summary = []
    for (framework, probe), rows in sorted(grouped.items()):
        summary.append({
            "framework": framework, "probe": probe, "n": len(rows),
            "ED": _mean(r["ED"] for r in rows), "DCE": _mean(r["DCE"] for r in rows),
            "contamination_rate": _mean(float(r["contaminated"]) for r in rows),
        })
    contrasts = []
    effects = []
    for framework in sorted({k[0] for k in grouped}):
        base = grouped[(framework, "search_only")]
        by_id = {r["sample_id"]: r for r in base}
        for probe in ("irrelevant_repetition", "correct_repetition", "echo_repetition"):
            for metric in ("ED", "DCE"):
                diffs = [r[metric] - by_id[r["sample_id"]][metric]
                         for r in grouped[(framework, probe)] if r["sample_id"] in by_id]
                effects.extend({
                    "sample_id": r["sample_id"], "framework": framework,
                    "probe": probe, "reference": "search_only", "metric": metric,
                    "difference": r[metric] - by_id[r["sample_id"]][metric],
                } for r in grouped[(framework, probe)] if r["sample_id"] in by_id)
                low, high = _bootstrap_ci(diffs)
                positive = probe == "echo_repetition"
                passed = (low > 0 if positive else high <= 0.05)
                contrasts.append({
                    "framework": framework, "probe": probe,
                    "reference": "search_only", "metric": metric, "n": len(diffs),
                    "mean_difference": _mean(diffs), "ci_low": low, "ci_high": high,
                    "criterion": "CI lower > 0" if positive else "CI upper <= 0.05",
                    "passed": bool(passed),
                })
    return summary, contrasts, effects


def probe_analysis_from_effects(effects):
    contrasts = []
    grouped = defaultdict(list)
    for row in effects:
        grouped[(row["framework"], row["probe"], row["metric"])].append(
            float(row["difference"])
        )
    for (framework, probe, metric), diffs in sorted(grouped.items()):
        low, high = _bootstrap_ci(diffs)
        positive = probe == "echo_repetition"
        contrasts.append({
            "framework": framework, "probe": probe, "reference": "search_only",
            "metric": metric, "n": len(diffs), "mean_difference": _mean(diffs),
            "ci_low": low, "ci_high": high,
            "criterion": "CI lower > 0" if positive else "CI upper <= 0.05",
            "passed": bool(low > 0 if positive else high <= 0.05),
        })
    return contrasts


def _decision_challenge_analysis(samples):
    grouped = defaultdict(list)
    effects = []
    for sample in samples:
        for row in sample.get("decision_gate_challenges", []):
            grouped[row["framework"]].append(row)
            effects.append({
                "sample_id": sample["sample_id"], "framework": row["framework"],
                "pre_guard_DCE": row["pre_guard_DCE"],
                "post_guard_DCE": row["post_guard_DCE"],
                "DCE_change": row["post_guard_DCE"] - row["pre_guard_DCE"],
                "decision_flip": float(row["decision_flip"]),
                "intervention_applied": float(row["intervention_applied"]),
            })
    return decision_challenge_analysis_from_effects(effects), effects


def decision_challenge_analysis_from_effects(effects):
    rows = []
    grouped = defaultdict(list)
    for row in effects:
        grouped[row["framework"]].append(row)
    for framework, values in sorted(grouped.items()):
        changes = [float(r["DCE_change"]) for r in values]
        low, high = _bootstrap_ci(changes)
        post = _mean(r["post_guard_DCE"] for r in values)
        flips = _mean(r["decision_flip"] for r in values)
        rows.append({
            "framework": framework, "n": len(values),
            "pre_guard_DCE": _mean(r["pre_guard_DCE"] for r in values),
            "post_guard_DCE": post, "DCE_change": _mean(changes),
            "ci_low_change": low, "ci_high_change": high,
            "decision_flip_rate": flips,
            "intervention_rate": _mean(r["intervention_applied"] for r in values),
            "criterion": "flip >=0.95, post DCE <=0.05, change 95%CI <0",
            "passed": bool(flips >= 0.95 and post <= 0.05 and high < 0),
        })
    return rows


def _cases(samples, count=12):
    candidates = []
    for sample in samples:
        runs = {(r["framework"], r["solution_arm"], r["condition"]): r
                for r in sample["runs"]}
        for framework in ("A-MEM", "Mem0"):
            base = runs.get((framework, "baseline", "update"))
            full = runs.get((framework, "full_guard", "update"))
            if base and full:
                candidates.append((base["metrics"]["DCE"] - full["metrics"]["DCE"], {
                    "sample_id": sample["sample_id"], "source": sample["source"],
                    "framework": framework, "echo_seed": sample["echo_seed"],
                    "verified_correction": sample["correction"],
                    "baseline": {"metrics": base["metrics"], "memory_audit": base["memory_audit"],
                                 "final_turn": base["trajectory"][-1]},
                    "full_guard": {"metrics": full["metrics"], "memory_audit": full["memory_audit"],
                                   "final_turn": full["trajectory"][-1]},
                }))
    return [row for _, row in sorted(candidates, key=lambda x: x[0], reverse=True)[:count]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewritten_dataset", default=str(
        ROOT / "results_memory_echo_amem_vs_mem0_200x30" / "rewritten_dataset.json"))
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_end", type=int)
    parser.add_argument("--frameworks", default="amem,mem0")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dce_threshold", type=float, default=.5)
    parser.add_argument("--family_threshold", type=float, default=.78)
    parser.add_argument("--activation_half_life", type=float, default=7.0)
    parser.add_argument("--persistence_trigger_windows", type=int, default=3)
    parser.add_argument("--persistence_excess_threshold", type=float, default=.05)
    parser.add_argument("--skip_probes", action="store_true")
    parser.add_argument("--output_dir", default="results_memory_echo_solution_strict")
    args = parser.parse_args()

    frameworks = [x.strip().lower() for x in args.frameworks.split(",") if x.strip()]
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if set(frameworks) - {"amem", "mem0"} or set(arms) - set(ARMS):
        raise ValueError("Unknown framework or arm")
    if arms != list(ARMS):
        raise ValueError(f"Strict validation requires all cumulative arms in order: {ARMS}")
    if conditions != list(CONDITIONS):
        raise ValueError(f"Strict validation requires conditions in order: {CONDITIONS}")

    dataset_path = Path(args.rewritten_dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    dataset_full = json.loads(dataset_path.read_text(encoding="utf-8"))[:args.max_samples]
    end = len(dataset_full) if args.sample_end is None else args.sample_end
    dataset = dataset_full[args.sample_start:end]
    if not dataset or any(row["delay_points"] != list(range(1, 31)) for row in dataset):
        raise ValueError("Strict validation needs non-empty data with delay points 1..30")

    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    mem0_root = output / "mem0_store"
    if mem0_root.exists():
        shutil.rmtree(mem0_root)

    encoder = CachedEncoder(SentenceTransformer(_model_path(), device=None))
    samples = []
    try:
        for index, scenario in enumerate(dataset, 1):
            sample = {**scenario, "qa_plan": generate_stage_qa(scenario, final_qa=4),
                      "runs": [], "control_probes": [], "decision_gate_challenges": []}
            # Official local Mem0/Qdrant calculates distances over a collection
            # before applying the namespace filter.  A distinct per-sample store
            # prevents unrelated samples from changing latency or exhausting
            # memory while retaining real Memory.add/search/update semantics.
            sample_mem0_root = mem0_root / f"sample_{scenario['sample_id']}"
            for framework in frameworks:
                for arm in arms:
                    matched_control_curve = None
                    for condition in conditions:
                        namespace = f"strict_{scenario['sample_id']}_{arm}_{condition}"
                        base = _adapter(framework, encoder, sample_mem0_root, namespace)
                        memory = EchoGuardMemory(
                            base, arm, encoder=encoder,
                            family_threshold=args.family_threshold,
                            activation_half_life=args.activation_half_life,
                            persistence_trigger_windows=args.persistence_trigger_windows,
                            persistence_excess_threshold=args.persistence_excess_threshold,
                        )
                        run = build_solution_trajectory(
                            scenario, sample["qa_plan"], condition, memory, encoder,
                            args.top_k, args.dce_threshold, matched_control_curve,
                        )
                        sample["runs"].append(run)
                        memory.close()
                        if condition == "control":
                            matched_control_curve = {
                                int(p["delay"]): float(p["ED"])
                                for p in run["delay_series"] if p["ED"] is not None
                            }
                if not args.skip_probes:
                    for probe in PROBES:
                        namespace = f"probe_{scenario['sample_id']}_{probe}"
                        base = _adapter(framework, encoder, sample_mem0_root, namespace)
                        row = run_control_probe(
                            scenario, base, encoder, probe, args.top_k, args.dce_threshold
                        )
                        row["sample_id"] = scenario["sample_id"]
                        sample["control_probes"].append(row)
                    namespace = f"decision_challenge_{scenario['sample_id']}"
                    base = _adapter(framework, encoder, sample_mem0_root, namespace)
                    sample["decision_gate_challenges"].append(
                        run_decision_gate_challenge(
                            scenario, base, encoder, args.top_k, args.dce_threshold
                        )
                    )
            # The next sample intentionally opens a new official Mem0 store.
            Mem0Adapter.close_shared()
            samples.append(sample)
            print(f"completed {index}/{len(dataset)}", flush=True)
    finally:
        Mem0Adapter.close_shared()

    aggregates = aggregate_solution(samples)
    protocol_checks = _protocol_checks(aggregates)
    validation = _strict_validation(aggregates)
    probe_summary, probe_contrasts, probe_effects = _probe_analysis(samples)
    decision_challenge_summary, decision_challenge_effects = _decision_challenge_analysis(samples)
    aggregates.update({
        "protocol_checks": protocol_checks, "validation_summary": validation,
        "probe_summary": probe_summary, "probe_contrasts": probe_contrasts,
        "probe_effects": probe_effects,
        "decision_challenge_summary": decision_challenge_summary,
        "decision_challenge_effects": decision_challenge_effects,
    })
    manifest = {
        "benchmark": "Strict Memory Echo Lifecycle Solution Validation",
        "design": "matched sample x framework x condition x cumulative S1-S4-Full",
        "samples": len(samples), "sample_range": [args.sample_start, end],
        "source_dataset": str(dataset_path), "frameworks": frameworks,
        "conditions": conditions, "solution_arms": arms,
        "delay_points": list(range(1, 31)), "top_k": args.top_k,
        "arm_protocol": {
            "baseline": "no governance",
            "s1_identity": "semantic Echo Family identity and source lineage",
            "s2_retrieval_hygiene": "S1 + exposure log, no confidence gain, no retrieval-derived fact add, family-deduplicated top-k",
            "s3_reconsolidation": "S2 + independent update gate, supersession state, correction binding",
            "s4_persistence": "S3 + confidence/activation separation, continuous decay and Control-window remediation",
            "full_guard": "S4 + two-pass counterfactual decision gate",
        },
        "pre_registered_success": {
            "ED": "extra ED relative to Control reduced >=50%, DiD 95%CI below 0",
            "RRI": "below Baseline with DiD 95%CI below 0 and residual effect <=0.05",
            "WEP": "excess AUC reduced >=60% and delay-30 excess <=0.02",
            "DCE": "below Baseline and residual effect within 0.05 of Control",
            "utility": "Control final accuracy delta lower CI > -0.02",
        },
        "parameters": {
            "family_threshold": args.family_threshold,
            "activation_half_life": args.activation_half_life,
            "persistence_trigger_windows": args.persistence_trigger_windows,
            "persistence_excess_threshold": args.persistence_excess_threshold,
        },
        "software": {
            "mem0ai": importlib.metadata.version("mem0ai") if "mem0" in frameworks else None,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        },
    }
    _json(output / "dataset_manifest.json", manifest)
    _json(output / "sample_level.json", samples)
    _json(output / "aggregate_results.json", aggregates)
    _json(output / "representative_cases.json", _cases(samples))
    for name in ("condition_summary", "per_stage_summary", "per_qa_type_summary",
                 "per_ability_summary", "per_delay_summary", "condition_effects",
                 "causal_estimands", "utility_effects", "utility_noninferiority",
                 "protocol_checks", "validation_summary", "probe_summary", "probe_contrasts",
                 "probe_effects"):
        write_csv(output / f"{name}.csv", aggregates[name])
    for name in ("decision_challenge_summary", "decision_challenge_effects"):
        write_csv(output / f"{name}.csv", aggregates[name])
    write_solution_figures(output, aggregates)
    passed = sum(r["target_met"] for r in validation)
    utility = sum(r["noninferior"] for r in aggregates["utility_noninferiority"])
    checks = sum(r["passed"] for r in protocol_checks)
    probes = sum(r["passed"] for r in probe_contrasts)
    decisions = sum(r["passed"] for r in decision_challenge_summary)
    (output / "conclusion.txt").write_text(
        "Strict Memory Echo lifecycle solution validation\n"
        f"Pre-registered metric targets: {passed}/{len(validation)}.\n"
        f"Control utility non-inferiority: {utility}/{len(aggregates['utility_noninferiority'])}.\n"
        f"Protocol checks: {checks}/{len(protocol_checks)}.\n"
        f"Positive/negative probe contrasts: {probes}/{len(probe_contrasts)}.\n"
        f"Counterfactual decision challenges: {decisions}/{len(decision_challenge_summary)}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "samples": len(samples),
                      "runs": sum(len(s["runs"]) for s in samples),
                      "probes": sum(len(s["control_probes"]) for s in samples),
                      "decision_challenges": sum(len(s["decision_gate_challenges"])
                                                  for s in samples)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
