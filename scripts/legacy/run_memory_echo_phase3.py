import argparse, csv, json, os, shutil
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT.parent / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT.parent / ".hf_cache" / "transformers"))
os.environ.setdefault("MEM0_DIR", str(ROOT.parent / ".mem0"))

from sentence_transformers import SentenceTransformer
from memory_echo_phase3.scenario_generator import generate_scenarios
from memory_echo_phase3.memory_adapter import AMemoryAdapter, Mem0Adapter, MODEL_NAME
from memory_echo_phase3.trajectory_builder import CONDITIONS, build_trajectory
from memory_echo_phase3.case_analyzer import select_cases


def aggregate(samples):
    groups = defaultdict(list)
    for s in samples:
        for r in s["runs"]: groups[(r["framework"], r["condition"])].append(r["metrics"])
    rows = []
    for (framework, condition), vals in groups.items():
        row = {"framework": framework, "condition": condition, "samples": len(vals)}
        for k in ("echo_density", "supplier_dominance_score", "freshness_resistance_score", "echo_persistence_score"):
            row[k] = sum(v[k] for v in vals) / len(vals)
        rows.append(row)
    controls = {r["framework"]: r["echo_density"] for r in rows if r["condition"] == "control"}
    for r in rows: r["echo_amplification_rate"] = r["echo_density"] - controls[r["framework"]]
    return rows


def figures(rows, out):
    import matplotlib.pyplot as plt
    specs = [("echo_density", "Figure_1_Echo_Density.png", "Echo Density"),
             ("echo_amplification_rate", "Figure_2_Echo_Amplification_Rate.png", "Echo Amplification Rate"),
             ("supplier_dominance_score", "Figure_3_Supplier_Dominance_Score.png", "Supplier Dominance Score"),
             ("echo_persistence_score", "Figure_4_Echo_Persistence_Score.png", "Echo Persistence Score")]
    figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    for metric, name, title in specs:
        labels = [f"{r['framework']}\n{r['condition']}" for r in rows]
        plt.figure(figsize=(11, 5)); plt.bar(labels, [r[metric] for r in rows])
        plt.title(title); plt.ylabel(metric); plt.xticks(rotation=18, ha="right"); plt.tight_layout()
        plt.savefig(figdir / name, dpi=160); plt.close()


def main():
    p = argparse.ArgumentParser(description="Phase 3 Memory Echo benchmark: 1688 procurement assistant")
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--output_dir", default="results_memory_echo_phase3")
    p.add_argument("--frameworks", default="amem,mem0")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    mem0root = out / "mem0_store"
    if mem0root.exists(): shutil.rmtree(mem0root)
    scenarios = generate_scenarios(args.max_samples)
    frameworks = [x.strip() for x in args.frameworks.split(",")]
    model = SentenceTransformer(MODEL_NAME) if "amem" in frameworks else None
    cache = ROOT / "cached_memories_robust_deepseek_deepseek-chat_locomo10"
    samples = []
    for s in scenarios:
        item = {**s, "runs": []}
        for fw in frameworks:
            for condition in CONDITIONS:
                if fw == "amem":
                    adapter = AMemoryAdapter(model, cache / f"memory_cache_sample_{s['sample_id']}.pkl")
                else:
                    adapter = Mem0Adapter(mem0root, f"p3_s{s['sample_id']}_{condition}")
                run = build_trajectory(s, condition, adapter, args.top_k)
                run["framework"] = adapter.framework
                run["cache_source"] = getattr(adapter, "cache_source", None)
                item["runs"].append(run); adapter.close()
        samples.append(item)
        print(f"completed sample {s['sample_id'] + 1}/{len(scenarios)}", flush=True)
    rows = aggregate(samples)
    (out / "sample_level.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "aggregate_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cases = select_cases(samples, 5)
    (out / "case_studies.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    figures(rows, out)
    Mem0Adapter.close_shared()
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
