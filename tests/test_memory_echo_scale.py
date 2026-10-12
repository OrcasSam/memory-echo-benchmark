from pathlib import Path

import pytest

from memory_echo_phase3.cognitive_qa_generator import (
    PHASE_ALLOCATIONS, generate_cognitive_qa_plan,
)
from memory_echo_phase3.dataset_rewriter_v6 import rewrite_webshop_scenario
from memory_echo_phase3.scale_comparator import compare_condition_rows
from memory_echo_phase3.webshop_loader import load_webshop_scenarios


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenario():
    if not (ROOT / "data" / "webshop" / "data_10k").exists():
        pytest.skip("External WebShop data are not distributed with this repository")
    rows, _ = load_webshop_scenarios(ROOT / "data" / "webshop" / "data_10k", 30)
    return rewrite_webshop_scenario(rows[0])


def test_200_qa_allocation_and_diversity(scenario):
    plan = generate_cognitive_qa_plan(scenario, 200)
    assert len(plan) == 200
    counts = {phase: sum(q["phase"] == phase for q in plan)
              for phase in PHASE_ALLOCATIONS[200]}
    assert counts == PHASE_ALLOCATIONS[200]
    assert len({q["query"] for q in plan}) >= 100
    assert plan[99]["canonical_target_supplier"] == "A"
    assert all(q["canonical_target_supplier"] == "B" for q in plan[100:]
               if q["qa_type"] != "memory_recall")


def test_scale_comparison_delta():
    base = {
        "framework": "A-MEM", "condition": "control", "samples": 10,
        "echo_density": 0.2, "echo_persistence_score": 0.7,
        "answer_score": 0.8, "decision_score": 0.75, "ed_lift_ear": 0,
        "answer_score_drop": 0, "decision_score_drop": 0,
    }
    new = {**base, "samples": 30, "echo_density": 0.3, "answer_score": 0.7}
    rows = compare_condition_rows(
        [base], [new], [{**new, "samples": 4}], [{**base, "samples": 4}], 4
    )
    assert len(rows) == 2
    assert rows[1]["comparison_scope"] == "matched_4_shared_instructions"
    assert rows[0]["delta_echo_density"] == pytest.approx(0.1)
    assert rows[0]["delta_answer_score"] == pytest.approx(-0.1)
