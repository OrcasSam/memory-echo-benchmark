from pathlib import Path
import pytest
from memory_echo_phase3.webshop_loader import load_webshop_scenarios
from memory_echo_phase3.webshop_metrics import finalize_100qa


def test_real_webshop_loader():
    root = Path(__file__).resolve().parents[1] / "data" / "webshop" / "data"
    if not root.exists():
        pytest.skip("External WebShop data are not distributed with this repository")
    rows, manifest = load_webshop_scenarios(root, 10)
    assert len(rows) == manifest["unique_dataset_qa"] == 10
    assert all(len(x["supplier_candidates"]) == 4 for x in rows)
    assert all(x["ground_truth_asin"] == x["supplier_candidates"][1]["product_asin"] for x in rows)
    assert all(x["user_requirement"] for x in rows)
    assert manifest["qa_turns_per_sample"] == 100


def test_four_dimension_windows():
    turns = []
    for i in range(1, 101):
        ed = 0.8 if i <= 50 else 0.4
        correct = i >= 81 and i % 2 == 0
        turns.append({"turn": i, "observation": {
            "echo_density": ed, "qa_correct": correct,
            "recommended_supplier": "B" if correct else "A",
        }})
    metrics = finalize_100qa(turns, {"A": 40})
    assert metrics["echo_density"] == pytest.approx(0.8)
    assert metrics["echo_persistence_score"] == pytest.approx(0.5)
    assert metrics["recommendation_error"] == 0.5
    assert metrics["qa_accuracy"] == 0.5
