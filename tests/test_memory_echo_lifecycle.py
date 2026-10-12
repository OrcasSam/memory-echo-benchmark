from pathlib import Path

import pytest

from memory_echo_lifecycle import STAGES
from memory_echo_lifecycle.metrics import reconsolidation_resistance, windowed_echo_persistence
from memory_echo_lifecycle.qa_generator import generate_stage_qa
from memory_echo_lifecycle.multi_source_rewriter import rewrite_multi_source_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT_ROOT.parent
SOURCES = {
    "LoCoMo": PROJECT_ROOT / "data" / "locomo10.json",
    "LongMemEval": WORKSPACE / "datasets" / "LongMemEval" / "data" / "longmemeval_oracle.json",
    "HotpotQA": WORKSPACE / "datasets" / "HotpotQA" / "hotpot_dev_distractor.parquet",
    "BEAM": WORKSPACE / "datasets" / "BEAM",
}
requires_source_data = pytest.mark.skipif(
    not all(path.exists() for path in SOURCES.values()),
    reason="External source datasets are not distributed with this code-only repository",
)


@requires_source_data
def test_source_rewrite_is_auditable_and_abstract():
    rows = rewrite_multi_source_dataset(SOURCES, max_samples=8, delay_points=(1, 3, 7), practice_qa=4)
    assert len(rows) == 8
    assert {row["source"]["dataset"] for row in rows} == {"LoCoMo", "LongMemEval", "HotpotQA", "BEAM"}
    assert all(row["echo_value"] != row["ground_truth_value"] for row in rows)
    assert all("choice" not in row["echo_seed"]["claim"].lower() for row in rows)


@requires_source_data
def test_every_stage_has_purposeful_qa_and_multiple_delays():
    sample = rewrite_multi_source_dataset(SOURCES, 4, (1, 3, 7, 14), 6)[0]
    plan = generate_stage_qa(sample, final_qa=4)
    assert {row["stage"] for row in plan} == set(STAGES)
    assert {row["delay"] for row in plan if row["stage"] == "delayed_retrieval"} == {1, 3, 7, 14}
    assert len({row["qa_type"] for row in plan}) >= 7
    assert all(row["purpose"] for row in plan)


def test_rri_and_wep_boundaries():
    assert reconsolidation_resistance(.8, .4, .4) == pytest.approx(.5)
    assert windowed_echo_persistence([
        {"delay": 1, "ED": .8}, {"delay": 2, "ED": .4}, {"delay": 3, "ED": .2},
    ]) == pytest.approx(.45)
    assert windowed_echo_persistence([
        {"delay": 1, "ED": .8}, {"delay": 2, "ED": .8}, {"delay": 3, "ED": .8},
    ]) == pytest.approx(.8)
    with pytest.raises(ValueError):
        windowed_echo_persistence([
            {"delay": 1, "ED": .8}, {"delay": 3, "ED": .4},
        ])


@requires_source_data
def test_formal_design_has_200_balanced_samples_and_30_continuous_delays():
    rows = rewrite_multi_source_dataset(SOURCES, 200, tuple(range(1, 31)), 6)
    assert len(rows) == 200
    assert all(row["delay_points"] == list(range(1, 31)) for row in rows)
    counts = {source: sum(row["source"]["dataset"] == source for row in rows)
              for source in SOURCES}
    assert counts == {source: 50 for source in SOURCES}
