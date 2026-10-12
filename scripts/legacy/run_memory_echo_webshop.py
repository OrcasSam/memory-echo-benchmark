import argparse, csv, json, os, shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT.parent / ".hf_cache" / "transformers"))
os.environ.setdefault("MEM0_DIR", str(ROOT.parent / ".mem0_webshop_echo"))
os.environ.setdefault("POSTHOG_DISABLED", "true")

from sentence_transformers import SentenceTransformer
from memory_echo_phase3.webshop_loader import load_webshop_scenarios
from memory_echo_phase3.memory_adapter import AMemoryAdapter, Mem0Adapter, MODEL_NAME
from memory_echo_phase3.webshop_trajectory_builder import CONDITIONS, build_webshop_100qa_trajectory


def main():
    p = argparse.ArgumentParser(description="Memory Echo benchmark on real Princeton WebShop data")
    p.add_argument("--data_dir", default=str(ROOT / "data" / "webshop" / "data"))
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--output_dir", default="results_memory_echo_webshop_100qa")
    p.add_argument("--frameworks", default="amem,mem0")
    args = p.parse_args()
    scenarios, manifest = load_webshop_scenarios(Path(args.data_dir), args.max_samples)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    mem0root = out / "mem0_store"
    if mem0root.exists(): shutil.rmtree(mem0root)
    frameworks = [x.strip() for x in args.frameworks.split(",")]
    model = SentenceTransformer(MODEL_NAME) if "amem" in frameworks else None
    samples = []
    for scenario in scenarios:
        item = {**scenario, "runs": []}
        for fw in frameworks:
            for condition in CONDITIONS:
                adapter = (AMemoryAdapter(model, None) if fw == "amem" else
                           Mem0Adapter(mem0root, f"webshop_s{scenario['sample_id']}_{condition}"))
                run = build_webshop_100qa_trajectory(scenario, condition, adapter, args.top_k)
                run["framework"] = adapter.framework
                item["runs"].append(run); adapter.close()
        samples.append(item)
        print(f"completed WebShop sample {scenario['sample_id'] + 1}/{len(scenarios)}", flush=True)
    rows = aggregate_four_dimensions(samples)
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "sample_level.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "case_studies.json").write_text(json.dumps(select_cases_four_dimensions(samples, 5), ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "aggregate_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    figures_four_dimensions(rows, out); Mem0Adapter.close_shared()
    print(json.dumps({"manifest": manifest, "aggregate": rows}, ensure_ascii=False, indent=2))


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_four_dimensions(samples):
    groups = {}
    for sample in samples:
        for run in sample["runs"]:
            groups.setdefault((run["framework"], run["condition"]), []).append(run["metrics"])
    rows = []
    for (framework, condition), metrics in groups.items():
        rows.append({
            "framework": framework, "condition": condition, "samples": len(metrics),
            "echo_density": _mean(m["echo_density"] for m in metrics),
            "echo_persistence_score": _mean(m["echo_persistence_score"] for m in metrics),
            "recommendation_error": _mean(m["recommendation_error"] for m in metrics),
            "qa_accuracy": _mean(m["qa_accuracy"] for m in metrics),
        })
    controls = {r["framework"]: r for r in rows if r["condition"] == "control"}
    for row in rows:
        control = controls[row["framework"]]
        row["echo_amplification_rate"] = row["echo_density"] - control["echo_density"]
        row["qa_accuracy_drop"] = control["qa_accuracy"] - row["qa_accuracy"]
    return rows


def figures_four_dimensions(rows, output_dir):
    import matplotlib.pyplot as plt
    specs = [
        ("echo_density", "Figure_1_Echo_Formation_ED.png", "Echo Formation: ED"),
        ("echo_amplification_rate", "Figure_2_Echo_Amplification_EAR.png", "Echo Amplification: EAR"),
        ("echo_persistence_score", "Figure_3_Echo_Persistence_EPS.png", "Echo Persistence: EPS"),
        ("qa_accuracy_drop", "Figure_4_Echo_Influence_QA_Accuracy_Drop.png", "Echo Influence: QA Accuracy Drop"),
    ]
    figdir = output_dir / "figures"; figdir.mkdir(exist_ok=True)
    for metric, filename, title in specs:
        labels = [f"{r['framework']}\n{r['condition']}" for r in rows]
        plt.figure(figsize=(11, 5)); plt.bar(labels, [r[metric] for r in rows])
        plt.title(title); plt.ylabel(metric); plt.xticks(rotation=18, ha="right"); plt.tight_layout()
        plt.savefig(figdir / filename, dpi=160); plt.close()


def select_cases_four_dimensions(samples, count=5):
    candidates = []
    for sample in samples:
        for run in sample["runs"]:
            if run["condition"] == "historical_repetition_update":
                score = run["metrics"]["echo_density"] + run["metrics"]["recommendation_error"]
                candidates.append((score, sample, run))
    cases = []
    for _, sample, run in sorted(candidates, key=lambda x: x[0], reverse=True)[:count]:
        turns = run["memory_trajectory"]
        cases.append({
            "sample_id": sample["sample_id"], "framework": run["framework"],
            "instruction": sample["user_requirement"], "ground_truth_asin": sample["ground_truth_asin"],
            "metrics": run["metrics"],
            "formation_snapshot": turns[49], "persistence_snapshot": turns[79],
            "influence_snapshot": turns[99],
        })
    return cases


if __name__ == "__main__":
    main()
