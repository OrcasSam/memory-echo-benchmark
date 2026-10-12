from scripts.ah.run_memory_echo_cross_model import _bootstrap_ci
from scripts.ah.run_raw_constraint_ablation import bootstrap_ci


def test_bootstrap_ci_is_deterministic():
    assert bootstrap_ci([0.0, 1.0, 2.0], 7, 100) == bootstrap_ci([0.0, 1.0, 2.0], 7, 100)


def test_bootstrap_ci_contains_constant():
    low, high = bootstrap_ci([0.25] * 20, 9, 100)
    assert low == high == 0.25


def test_existing_bootstrap_helper_still_works():
    low, high = _bootstrap_ci([0.0, 1.0], 3, 100)
    assert 0.0 <= low <= high <= 1.0
