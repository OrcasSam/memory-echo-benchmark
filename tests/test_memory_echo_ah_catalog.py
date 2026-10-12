from pathlib import Path

from memory_echo_ah.evidence_catalog import (
    build_catalog, matching_entries, protocol_fingerprint,
)


def test_protocol_fingerprint_is_order_invariant():
    assert protocol_fingerprint({"a": 1, "b": [2]}) == protocol_fingerprint(
        {"b": [2], "a": 1}
    )


def test_catalog_and_exact_match(tmp_path: Path):
    result = tmp_path / "results_one"
    result.mkdir()
    (result / "manifest.json").write_text(
        '{"frameworks":["A-MEM"],"conditions":["control"],"top_k":5}',
        encoding="utf-8",
    )
    (result / "sample_level.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    catalog = build_catalog(tmp_path)
    required = {"frameworks": ["A-MEM"], "conditions": ["control"], "top_k": 5}
    matches = matching_entries(catalog, required, ["sample_level.jsonl"])
    assert len(matches) == 1
    assert matches[0]["artifacts"][0]["records"] == 2
