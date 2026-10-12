"""Inventory local experiment evidence before scheduling any new run.

The catalog is deliberately metadata-first: multi-gigabyte sample files are not
hashed in full.  Their byte size, line count (for JSONL), modification time and
the digest of nearby manifests provide a cheap, auditable reuse check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANIFEST_NAMES = (
    "manifest.json",
    "dataset_manifest.json",
    "integrity_report.json",
)
RESULT_PATTERNS = (
    "aggregate_results.csv",
    "aggregate_results.json",
    "sample_level.json",
    "sample_level.jsonl",
    "judged_answers.jsonl",
    "decision_inputs.jsonl",
    "all_predictions.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _jsonl_lines(path: Path) -> int | None:
    if path.suffix.lower() != ".jsonl":
        return None
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    """Return a stable identity for fields that decide experimental reuse."""
    payload = json.dumps(protocol, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_summary(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    keys = (
        "samples", "max_samples", "frameworks", "conditions", "stages",
        "delay_points", "top_k", "solution", "solutions", "a_variant",
        "generator", "generators", "model", "judge_model", "source_trajectory",
        "source_dataset", "dataset", "candidate_policy", "protocol_version",
    )
    return {key: data[key] for key in keys if key in data}


def inventory_result_dir(path: Path) -> dict[str, Any]:
    manifests = []
    protocol: dict[str, Any] = {}
    for name in MANIFEST_NAMES:
        candidate = path / name
        if candidate.is_file():
            data = _json(candidate)
            summary = _manifest_summary(data)
            protocol.update(summary)
            manifests.append({
                "name": name,
                "sha256": _sha256(candidate),
                "summary": summary,
            })

    artifacts = []
    for name in RESULT_PATTERNS:
        candidate = path / name
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        artifacts.append({
            "name": name,
            "bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "records": _jsonl_lines(candidate),
        })
    return {
        "result_dir": path.name,
        "path": str(path.resolve()),
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol) if protocol else None,
        "manifests": manifests,
        "artifacts": artifacts,
        "complete_artifact_count": len(artifacts),
    }


def build_catalog(root: Path) -> dict[str, Any]:
    result_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith("results")
    )
    entries = [inventory_result_dir(path) for path in result_dirs]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "policy": (
            "Reuse requires matching protocol fingerprint and required artifact; "
            "partial or protocol-mismatched evidence must not fill a table cell."
        ),
        "result_directories": len(entries),
        "entries": entries,
    }


def matching_entries(catalog: dict[str, Any], required: dict[str, Any],
                     required_artifacts: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Find exact local protocol matches with all requested artifacts."""
    wanted = protocol_fingerprint(required)
    required_artifacts = set(required_artifacts)
    matches = []
    for entry in catalog.get("entries", []):
        if entry.get("protocol_fingerprint") != wanted:
            continue
        present = {artifact["name"] for artifact in entry.get("artifacts", [])}
        if required_artifacts <= present:
            matches.append(entry)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "results_memory_echo_ah" / "evidence_catalog.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(args.root)
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"cataloged={catalog['result_directories']} output={output}")


if __name__ == "__main__":
    main()
