"""Tests for the domain-independent Generic Memory Echo Benchmark.

These tests deliberately exercise the production dataset size, but they do not
instantiate either memory framework or run retrieval.  The metric tests use
small deterministic records so the expected values remain transparent.
"""

import json
from collections import Counter
from math import log2

import pytest

from memory_echo_generic import MEMORY_UNIT_TYPES, PROTOCOLS
from memory_echo_generic.echo_protocol_dataset_generator import (
    generate_protocol_dataset,
)
from memory_echo_generic.four_metric_aggregator import (
    aggregate_run_metrics,
    paired_condition_effects,
)
from memory_echo_generic.generic_trajectory_builder import (
    _echo_density,
    _run_metrics,
    _update_incorporation,
)
from memory_echo_generic.multi_type_qa_generator import (
    PHASES,
    PROTOCOL_PHASE_COUNTS_20,
    generate_qa_plan,
)


FORMAL_SEEDS = (11, 29, 47)
ALL_QA_TYPES = {
    "recall",
    "comparison",
    "update_awareness",
    "consistency",
    "decision",
    "justification",
}


@pytest.fixture(scope="module")
def formal_dataset():
    """The minimum valid production dataset: 100 trajectories per protocol."""
    return generate_protocol_dataset(
        samples_per_protocol=100,
        seeds=FORMAL_SEEDS,
    )


@pytest.fixture(scope="module")
def formal_qa_plans(formal_dataset):
    return {
        sample["sample_id"]: generate_qa_plan(sample, qa_count=20)
        for sample in formal_dataset
    }


def test_formal_dataset_has_four_protocols_seven_memory_types_and_many_seeds(
    formal_dataset,
):
    protocol_counts = Counter(row["protocol"] for row in formal_dataset)

    assert set(protocol_counts) == set(PROTOCOLS)
    assert all(protocol_counts[protocol] >= 100 for protocol in PROTOCOLS)
    assert len(formal_dataset) >= 100 * len(PROTOCOLS)
    assert len({row["sample_id"] for row in formal_dataset}) == len(
        formal_dataset
    )
    assert len({row["subject"] for row in formal_dataset}) == len(
        formal_dataset
    )

    for protocol in PROTOCOLS:
        protocol_rows = [
            row for row in formal_dataset if row["protocol"] == protocol
        ]
        assert {
            row["memory_unit_type"] for row in protocol_rows
        } == set(MEMORY_UNIT_TYPES)
        assert len({row["random_seed"] for row in protocol_rows}) >= 2

    assert {row["random_seed"] for row in formal_dataset} == set(FORMAL_SEEDS)


def test_random_seeds_repeat_matched_protocol_configurations(formal_dataset):
    groups = {}
    for row in formal_dataset:
        key = (row["protocol"], row["random_seed_replicate_id"])
        groups.setdefault(key, []).append(row)
    complete = [rows for rows in groups.values() if len(rows) == len(FORMAL_SEEDS)]
    assert len(complete) == len(PROTOCOLS) * (100 // len(FORMAL_SEEDS))
    matched_fields = (
        "memory_unit_type",
        "noise_level",
        "delay_sessions",
        "decision_ambiguity",
        "should_update",
    )
    for rows in complete:
        assert {row["random_seed"] for row in rows} == set(FORMAL_SEEDS)
        for field in matched_fields:
            assert len({row[field] for row in rows}) == 1
        assert len({
            row["conflicting_evidence"]["strength"] for row in rows
        }) == 1


def test_every_trajectory_has_20_qa_expected_phases_and_all_six_qa_types(
    formal_dataset,
    formal_qa_plans,
):
    for sample in formal_dataset:
        plan = formal_qa_plans[sample["sample_id"]]
        assert len(plan) == 20
        assert [qa["turn"] for qa in plan] == list(range(1, 21))
        assert set(qa["qa_type"] for qa in plan) == ALL_QA_TYPES

        phase_counts = Counter(qa["phase"] for qa in plan)
        assert set(phase_counts) == set(PHASES)
        assert tuple(phase_counts[phase] for phase in PHASES) == (
            PROTOCOL_PHASE_COUNTS_20[sample["protocol"]]
        )
        assert all(
            qa["query"]
            and qa["canonical_answer"]
            and qa["canonical_target_label"] in {"OLD", "NEW"}
            for qa in plan
        )


def test_dataset_and_qa_are_domain_independent(
    formal_dataset,
    formal_qa_plans,
):
    serialized = json.dumps(
        {
            "dataset": formal_dataset,
            "qa": formal_qa_plans,
        },
        ensure_ascii=False,
    ).casefold()
    prohibited_scene_terms = {
        # Shopping and procurement.
        "shopping",
        "shop",
        "supplier",
        "vendor",
        "seller",
        "product",
        "price",
        "retail",
        "purchase",
        "采购",
        "供应商",
        "商家",
        "商品",
        "价格",
        "购物",
        # Medical scenarios.
        "medical",
        "clinical",
        "doctor",
        "patient",
        "diagnosis",
        "disease",
        "treatment",
        "medicine",
        "医疗",
        "医生",
        "患者",
        "诊断",
        "疾病",
        "治疗",
        # Travel scenarios.
        "travel",
        "tourism",
        "hotel",
        "flight",
        "booking",
        "destination",
        "旅游",
        "酒店",
        "航班",
        "预订",
        "目的地",
    }
    assert not {
        term for term in prohibited_scene_terms if term in serialized
    }


def _hit(label, rank, score=1.0):
    return {
        "supplier_id": label,
        "retrieval_rank": rank,
        "retrieval_score": score,
    }


def test_ed_and_uir_formulas_and_boundaries():
    mixed = [
        _hit("OLD", 1),
        _hit("NEW", 2),
        _hit("NOISE", 3),
    ]
    old_rank_weight = 1.0 / log2(2.0)
    new_rank_weight = 1.0 / log2(3.0)

    assert _echo_density(mixed) == pytest.approx(1.0 / 3.0)
    assert _update_incorporation(mixed) == pytest.approx(
        new_rank_weight / (old_rank_weight + new_rank_weight)
    )

    assert _echo_density([]) is None
    assert _update_incorporation([]) is None
    assert _echo_density([_hit("OLD", 1), _hit("OLD", 2)]) == 1.0
    assert _update_incorporation(
        [_hit("OLD", 1), _hit("OLD", 2)]
    ) == 0.0
    assert _echo_density([_hit("NEW", 1), _hit("NEW", 2)]) == 0.0
    assert _update_incorporation(
        [_hit("NEW", 1), _hit("NEW", 2)]
    ) == 1.0


def _metric_turn(
    phase,
    ed,
    uir=0.0,
    *,
    qa_type="consistency",
    persistence_probe_id=None,
    decision_bias=0.0,
    decision_eligible=False,
):
    return {
        "phase": phase,
        "qa_type": qa_type,
        "persistence_probe_id": persistence_probe_id,
        "echo_density": ed,
        "update_incorporation_rate": uir,
        "relevant_retrieval_coverage": 1.0,
        "decision_bias_flag": decision_bias,
        "decision_bias_eligible": decision_eligible,
        "answer_evaluation": {"answer_score": 0.8},
        "retrieved_memories": [{}],
    }


def test_eps_and_dbi_formulas_and_zero_denominator_boundary():
    trajectory = [
        _metric_turn("echo_seeding", 0.4),
        _metric_turn("echo_seeding", 0.5),
        _metric_turn("echo_seeding", 0.6),
        _metric_turn(
            "update_adaptation", 0.5, 0.25,
            persistence_probe_id="standard-current-state",
        ),
        _metric_turn(
            "echo_retention", 0.25, 0.5,
            persistence_probe_id="standard-current-state",
        ),
        _metric_turn("echo_retention", 0.3, 0.5),
        _metric_turn(
            "decision_influence",
            0.1,
            0.75,
            qa_type="decision",
            decision_bias=1.0,
            decision_eligible=True,
        ),
        _metric_turn(
            "decision_influence",
            0.2,
            0.75,
            qa_type="decision",
            decision_bias=0.0,
            decision_eligible=True,
        ),
    ]
    metrics = _run_metrics(trajectory)

    assert metrics["ED"] == pytest.approx(0.5)
    assert metrics["UIR"] == pytest.approx(0.25)
    assert metrics["EPS"] == pytest.approx(0.5)
    assert metrics["DBI"] == pytest.approx(0.5)

    zero_baseline = [
        _metric_turn("echo_seeding", 0.0),
        _metric_turn(
            "update_adaptation", 0.0,
            persistence_probe_id="standard-current-state",
        ),
        _metric_turn(
            "echo_retention", 1.0,
            persistence_probe_id="standard-current-state",
        ),
        _metric_turn("decision_influence", 1.0),
    ]
    boundary_metrics = _run_metrics(zero_baseline)
    assert boundary_metrics["EPS"] is None
    assert boundary_metrics["DBI"] is None


def test_catch_trials_prevent_an_always_choose_new_shortcut(
    formal_dataset,
    formal_qa_plans,
):
    catch_trials = [row for row in formal_dataset if not row["should_update"]]
    update_trials = [row for row in formal_dataset if row["should_update"]]

    assert catch_trials
    assert update_trials
    assert {row["protocol"] for row in catch_trials} == set(PROTOCOLS)
    assert all(
        row["conflicting_evidence"]["replacement_status"] == "insufficient"
        and row["canonical_current_label"] == "OLD"
        for row in catch_trials
    )
    assert all(
        row["conflicting_evidence"]["replacement_status"] == "sufficient"
        and row["canonical_current_label"] == "NEW"
        for row in update_trials
    )

    for sample in catch_trials:
        post_conflict = [
            qa
            for qa in formal_qa_plans[sample["sample_id"]]
            if qa["phase"] != "echo_seeding" and qa["qa_type"] != "recall"
        ]
        assert post_conflict
        assert all(
            qa["canonical_target_label"] == "OLD" for qa in post_conflict
        )

    representative_update = next(
        row for row in update_trials
    )
    post_conflict = [
        qa
        for qa in formal_qa_plans[representative_update["sample_id"]]
        if qa["phase"] != "echo_seeding" and qa["qa_type"] != "recall"
    ]
    assert all(qa["canonical_target_label"] == "NEW" for qa in post_conflict)


def _aggregate_sample(sample_id, should_update, control, reinforced):
    return {
        "sample_id": sample_id,
        "protocol_sample_id": sample_id,
        "protocol": "update_adaptation",
        "memory_unit_type": "fact",
        "random_seed": 11,
        "conflicting_evidence": {"strength": "medium"},
        "noise_level": "low",
        "should_update": should_update,
        "runs": [
            {
                "framework": "A-MEM",
                "condition": "control",
                "metrics": {
                    **control,
                    "semantic_answer_score": 0.8,
                },
            },
            {
                "framework": "A-MEM",
                "condition": "historical_repetition_update",
                "metrics": {
                    **reinforced,
                    "semantic_answer_score": 0.7,
                },
            },
        ],
    }


def test_paired_aggregation_uses_per_sample_deltas_and_excludes_catch_trials():
    samples = [
        _aggregate_sample(
            1,
            True,
            control={"ED": 0.2, "UIR": 0.8, "EPS": 0.5, "DBI": 0.1},
            reinforced={
                "ED": 0.6,
                "UIR": 0.3,
                "EPS": 0.9,
                "DBI": 0.7,
            },
        ),
        _aggregate_sample(
            2,
            False,
            control={"ED": 0.1, "UIR": 0.9, "EPS": 0.4, "DBI": None},
            reinforced={
                "ED": 0.2,
                "UIR": 0.2,
                "EPS": 0.5,
                "DBI": None,
            },
        ),
    ]

    paired = paired_condition_effects(samples)
    overall = next(row for row in paired if row["protocol"] == "ALL_PROTOCOLS")

    assert overall["paired_trajectory_count"] == 2
    assert overall["ED_lift"] == pytest.approx(0.25)
    assert overall["EPS_lift"] == pytest.approx(0.25)
    assert overall["UIR_drop"] == pytest.approx(0.5)
    assert overall["DBI_lift"] == pytest.approx(0.6)
    assert overall["ED_lift_valid_n"] == 2
    assert overall["EPS_lift_valid_n"] == 2
    assert overall["UIR_drop_valid_n"] == 1
    assert overall["DBI_lift_valid_n"] == 1
    assert overall["UIR_drop_ci95_low"] == pytest.approx(0.5)
    assert overall["UIR_drop_ci95_high"] == pytest.approx(0.5)

    aggregated = aggregate_run_metrics(samples)["condition_summary"]
    reinforced = next(
        row
        for row in aggregated
        if row["condition"] == "historical_repetition_update"
    )
    assert reinforced["ED_valid_n"] == 2
    assert reinforced["UIR_valid_n"] == 1
    assert reinforced["DBI_valid_n"] == 1
