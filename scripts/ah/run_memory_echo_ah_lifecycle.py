"""Run the strict four-framework lifecycle needed by Memory Echo Tables A--C."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from memory_echo_ah.adapters import CoMAdapter, LangMemAdapter
from memory_echo_ah.evidence_catalog import protocol_fingerprint
from memory_echo_ah.protocol import CONDITIONS, run_protocol
from memory_echo_phase3.memory_adapter import AMemoryAdapter, Mem0Adapter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results_memory_echo_amem_vs_mem0_200x30" / "rewritten_dataset.json"
LOCKED = ROOT / "results_memory_echo_dce_v3_all" / "paired_sample_locked_190.csv"
FRAMEWORKS = ("A-MEM", "Mem0", "CoM", "LangMem")


def local_model_path():
    snapshots = (ROOT.parent / ".hf_cache" / "hub" /
                 "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots")
    candidates = sorted(path for path in snapshots.glob("*") if (path / "modules.json").exists())
    return str(candidates[-1]) if candidates else "all-MiniLM-L6-v2"


def mean(values):
    values = [float(value) for value in values if value is not None and value != ""]
    return sum(values) / len(values) if values else None


def load_samples(source: Path, locked_path: Path, start: int, end: int | None,
                 max_samples: int | None):
    data = json.loads(source.read_text(encoding="utf-8"))
    with locked_path.open("r", encoding="utf-8-sig", newline="") as handle:
        locked = {row["sample_id"] for row in csv.DictReader(handle)}
    selected = [sample for sample in data if sample["sample_id"] in locked]
    if len(selected) != 190:
        raise RuntimeError(f"Expected 190 locked samples, found {len(selected)}")
    selected = selected[start:end]
    return selected[:max_samples] if max_samples is not None else selected


def make_adapter(framework, model, output, namespace):
    if framework == "A-MEM":
        return AMemoryAdapter(model)
    if framework == "Mem0":
        return Mem0Adapter(
            output / "mem0_runtime", namespace,
            collection_name="memory_echo_ah_lifecycle",
            embedding_model=local_model_path(),
        )
    if framework == "CoM":
        return CoMAdapter(model)
    if framework == "LangMem":
        return LangMemAdapter(model, namespace)
    raise ValueError(framework)


def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add((row["sample_id"], row["framework"], row["condition"]))
    return completed


def aggregate(path: Path, output: Path):
    indexed = {}
    duplicate_keys = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key = (row["sample_id"], row["framework"], row["condition"])
                if key in indexed:
                    duplicate_keys.append("|".join(key))
                    continue
                indexed[key] = row
    rows = list(indexed.values())
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["framework"], row["condition"])].append(row)

    def write_csv(name, table_rows):
        if not table_rows:
            return
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)

    table_a, table_b = [], []
    for framework in FRAMEWORKS:
        for condition in CONDITIONS:
            values = grouped.get((framework, condition), [])
            if not values:
                continue
            table_a.append({
                "framework": framework,
                "history_condition": condition,
                "n": len(values),
                "pre_correction_ED": mean(row["metrics"]["pre_correction_ED"] for row in values),
                "derived_record_count": mean(row["metrics"]["derived_record_count"] for row in values),
                "post_correction_old_answer_rate": "PENDING_GENERATOR",
                "final_old_retrieval_rate_proxy": mean(
                    row["metrics"]["final_old_retrieval_wins"] for row in values
                ),
            })
            table_b.append({
                "framework": framework,
                "history_condition": condition,
                "n": len(values),
                "correction_retrieval_rate": mean(
                    row["metrics"]["correction_retrieval_rate"] for row in values
                ),
                "storage_RRI": mean(row["metrics"]["storage_RRI"] for row in values),
                "retrieval_RRI": mean(row["metrics"]["retrieval_RRI"] for row in values),
                "WEP30": mean(row["metrics"]["WEP30"] for row in values),
                "final_accuracy": "PENDING_DEEPSEEK",
            })

    table_c = []
    for framework in FRAMEWORKS:
        values = grouped.get((framework, "retrieve_writeback"), [])
        if not values:
            continue
        table_c.append({
            "framework": framework,
            "n": len(values),
            "pre_correction_ED": mean(row["metrics"]["pre_correction_ED"] for row in values),
            "post_correction_ED": mean(row["metrics"]["post_correction_ED"] for row in values),
            "ED_query_1": mean(row["metrics"]["ED_at_1"] for row in values),
            "ED_query_10": mean(row["metrics"]["ED_at_10"] for row in values),
            "ED_query_20": mean(row["metrics"]["ED_at_20"] for row in values),
            "ED_query_30": mean(row["metrics"]["ED_at_30"] for row in values),
            "final_old_answer_rate": "PENDING_GENERATOR",
            "final_old_retrieval_rate_proxy": mean(
                row["metrics"]["final_old_retrieval_wins"] for row in values
            ),
        })
    write_csv("table_A_lifecycle_partial.csv", table_a)
    write_csv("table_B_lifecycle_partial.csv", table_b)
    write_csv("table_C_lifecycle_partial.csv", table_c)
    (output / "lifecycle_integrity.json").write_text(json.dumps({
        "unique_records": len(rows),
        "duplicate_records_ignored": len(duplicate_keys),
        "duplicate_keys": duplicate_keys,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--locked", type=Path, default=LOCKED)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "results_memory_echo_ah")
    parser.add_argument("--frameworks", nargs="+", choices=FRAMEWORKS, default=list(FRAMEWORKS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_end", type=int)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--practice_queries", type=int, default=6)
    parser.add_argument("--delay_queries", type=int, default=30)
    parser.add_argument("--device")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol_version": "ah-lifecycle-v2-provenance",
        "source_dataset": str(args.source.resolve()),
        "locked_sample_file": str(args.locked.resolve()),
        "frameworks": args.frameworks,
        "conditions": args.conditions,
        "top_k": args.top_k,
        "practice_queries": args.practice_queries,
        "delay_points": list(range(1, args.delay_queries + 1)),
        "read_only_delay": True,
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
    }
    manifest_path = args.output_dir / "lifecycle_manifest.json"
    fingerprint = protocol_fingerprint(protocol)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(
                "Output directory contains a different protocol. Use a new directory; "
                "mixing results is forbidden."
            )
    else:
        manifest_path.write_text(json.dumps(
            {**protocol, "protocol_fingerprint": fingerprint}, ensure_ascii=False, indent=2
        ), encoding="utf-8")

    samples = load_samples(
        args.source, args.locked, args.sample_start, args.sample_end, args.max_samples
    )
    output_path = args.output_dir / "lifecycle_sample_level.jsonl"
    completed = load_completed(output_path)
    model = SentenceTransformer(local_model_path(), device=args.device)
    total = len(samples) * len(args.frameworks) * len(args.conditions)
    done = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            for framework in args.frameworks:
                for condition in args.conditions:
                    key = (sample["sample_id"], framework, condition)
                    if key in completed:
                        done += 1
                        print(f"reuse {done}/{total} {'|'.join(key)}", flush=True)
                        continue
                    namespace = hashlib.sha1("|".join(key).encode()).hexdigest()[:20]
                    adapter = make_adapter(framework, model, args.output_dir, namespace)
                    try:
                        row = run_protocol(
                            sample, condition, adapter, model,
                            top_k=args.top_k,
                            practice_queries=args.practice_queries,
                            delay_queries=args.delay_queries,
                        )
                    finally:
                        adapter.close()
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    completed.add(key)
                    done += 1
                    print(f"complete {done}/{total} {'|'.join(key)}", flush=True)
    Mem0Adapter.close_shared()
    aggregate(output_path, args.output_dir)
    print(f"lifecycle_complete records={len(completed)} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
