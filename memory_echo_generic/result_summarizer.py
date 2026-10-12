"""Summarize formation, amplification, retention, and decision influence."""
from collections import defaultdict


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _effect_rows(paired_rows, contrast, include_overall=False):
    return [
        row for row in paired_rows
        if row["contrast"] == contrast
        and (
            include_overall
            or row["protocol"] != "ALL_PROTOCOLS"
        )
    ]


def _direction_consistency(seed_rows, contrast, metric):
    rows = _effect_rows(seed_rows, contrast)
    values = [row[metric] for row in rows if row[metric] is not None]
    return {
        "consistent": bool(values) and all(value > 0 for value in values),
        "positive_strata": sum(value > 0 for value in values),
        "total_strata": len(values),
    }


def _argmax_defined(values):
    defined = {
        key: value for key, value in values.items()
        if value is not None
    }
    return max(defined, key=defined.get) if defined else None


def summarize(
    protocol_rows,
    paired_rows,
    phase_rows,
    qa_rows,
    seed_paired_rows=None,
):
    seed_paired_rows = seed_paired_rows or []
    primary = _effect_rows(
        paired_rows, "update_minus_control", include_overall=True
    )
    overall = [
        row for row in primary if row["protocol"] == "ALL_PROTOCOLS"
    ]
    mean_effects = {
        metric: _mean(row[metric] for row in overall)
        for metric in ("ED_lift", "UIR_drop", "EPS_lift", "DBI_lift")
    }

    protocol_effects = defaultdict(dict)
    for row in paired_rows:
        if row["protocol"] == "ALL_PROTOCOLS":
            continue
        protocol_effects[row["protocol"]][row["contrast"]] = {
            metric: row[metric]
            for metric in ("ED_lift", "UIR_drop", "EPS_lift", "DBI_lift")
        }
    protocol_effects = dict(protocol_effects)

    formation_by_protocol = defaultdict(list)
    amplification_by_protocol = defaultdict(list)
    influence_by_protocol = defaultdict(list)
    for row in _effect_rows(paired_rows, "update_minus_control"):
        formation_by_protocol[row["protocol"]].append(row["ED_lift"])
        influence_by_protocol[row["protocol"]].append(row["DBI_lift"])
    for row in _effect_rows(paired_rows, "update_minus_repetition"):
        amplification_by_protocol[row["protocol"]].append(row["UIR_drop"])

    formation_means = {
        protocol: _mean(values)
        for protocol, values in formation_by_protocol.items()
    }
    amplification_means = {
        protocol: _mean(values)
        for protocol, values in amplification_by_protocol.items()
    }
    influence_means = {
        protocol: _mean(values)
        for protocol, values in influence_by_protocol.items()
    }
    easiest_formation = _argmax_defined(formation_means)
    strongest_update_failure = _argmax_defined(amplification_means)
    strongest_decision_bias = _argmax_defined(influence_means)

    update_protocol_rows = [
        row for row in protocol_rows
        if row["condition"] == "historical_repetition_update"
    ]
    eps_weight = sum(
        row["EPS_valid_n"] for row in update_protocol_rows
        if row["EPS"] is not None
    )
    mean_eps = (
        sum(
            row["EPS"] * row["EPS_valid_n"]
            for row in update_protocol_rows
            if row["EPS"] is not None
        ) / eps_weight
        if eps_weight else None
    )

    phase_lookup = {
        (
            row["framework"], row["protocol"], row["condition"],
            row["phase"],
        ): row
        for row in phase_rows
    }
    phase_ed_lifts = defaultdict(list)
    for key, control in phase_lookup.items():
        framework, protocol, condition, phase = key
        if condition != "control":
            continue
        update = phase_lookup.get((
            framework, protocol, "historical_repetition_update", phase
        ))
        if update and update["ED"] is not None and control["ED"] is not None:
            phase_ed_lifts[phase].append(update["ED"] - control["ED"])
    mean_phase_ed_lifts = {
        phase: _mean(values) for phase, values in phase_ed_lifts.items()
    }
    most_echo_prone_phase = (
        max(mean_phase_ed_lifts, key=mean_phase_ed_lifts.get)
        if mean_phase_ed_lifts else None
    )

    qa_lookup = {
        (
            row["framework"], row["protocol"], row["condition"],
            row["qa_type"],
        ): row
        for row in qa_rows
    }
    qa_effects = defaultdict(list)
    for key, control in qa_lookup.items():
        framework, protocol, condition, qa_type = key
        if condition != "control":
            continue
        update = qa_lookup.get((
            framework, protocol, "historical_repetition_update", qa_type
        ))
        if update:
            qa_effects[qa_type].append(
                control["answer_score"] - update["answer_score"]
            )
    qa_drops = {
        qa_type: _mean(values) for qa_type, values in qa_effects.items()
    }
    most_sensitive_qa = _argmax_defined(qa_drops)

    formation_rows = _effect_rows(
        paired_rows, "update_minus_control"
    )
    amplification_rows = _effect_rows(
        paired_rows, "update_minus_repetition"
    )
    influence_rows = _effect_rows(
        paired_rows, "update_minus_control"
    )
    formation_pass = bool(formation_rows) and all(
        row["ED_lift"] is not None
        and row["ED_lift"] > 0.05
        and row["ED_lift_ci95_low"] is not None
        and row["ED_lift_ci95_low"] > 0
        for row in formation_rows
    )
    amplification_pass = bool(amplification_rows) and all(
        row["UIR_drop"] is not None
        and row["UIR_drop"] > 0.05
        and row["UIR_drop_ci95_low"] is not None
        and row["UIR_drop_ci95_low"] > 0
        for row in amplification_rows
    )
    retention_pass = bool(update_protocol_rows) and all(
        row["EPS"] is not None
        and row["EPS"] >= 0.80
        and row["EPS_valid_n"] > 0
        for row in update_protocol_rows
    )
    influence_pass = bool(influence_rows) and all(
        row["DBI_lift"] is not None
        and row["DBI_lift"] > 0.05
        and row["DBI_lift_ci95_low"] is not None
        and row["DBI_lift_ci95_low"] > 0
        for row in influence_rows
    )
    seed_consistency = {
        "formation": _direction_consistency(
            seed_paired_rows, "update_minus_control", "ED_lift"
        ),
        "amplification": _direction_consistency(
            seed_paired_rows, "update_minus_repetition", "UIR_drop"
        ),
        "influence": _direction_consistency(
            seed_paired_rows, "update_minus_control", "DBI_lift"
        ),
    }
    seed_pass = (
        all(item["consistent"] for item in seed_consistency.values())
        if seed_paired_rows else False
    )
    criteria = {
        "formation": formation_pass,
        "amplification": amplification_pass,
        "retention": retention_pass,
        "influence": influence_pass,
        "seed_direction_consistency": seed_pass,
    }
    stable = all(criteria.values())
    return {
        "stable_memory_echo_observed": stable,
        "predeclared_criteria": {
            "formation": (
                "for every protocol/framework: paired update-control "
                "ED lift > 0.05 and bootstrap CI lower bound > 0"
            ),
            "amplification": (
                "for every protocol/framework: paired update-repetition "
                "UIR drop > 0.05 and bootstrap CI lower bound > 0"
            ),
            "retention": (
                "for every protocol/framework: mean update-condition EPS >= 0.80"
            ),
            "influence": (
                "for every protocol/framework: paired update-control "
                "DBI lift > 0.05 and bootstrap CI lower bound > 0"
            ),
            "seed_direction_consistency": (
                "ED lift, UIR drop, and DBI lift are positive in every "
                "protocol/framework/seed stratum"
            ),
        },
        "criteria_results": criteria,
        "seed_direction_consistency": seed_consistency,
        "mean_paired_update_control_effects": mean_effects,
        "mean_update_condition_EPS": mean_eps,
        "protocol_effects": protocol_effects,
        "phase_ED_lifts": mean_phase_ed_lifts,
        "most_echo_prone_phase": most_echo_prone_phase,
        "easiest_echo_formation_protocol": easiest_formation,
        "strongest_update_failure_protocol": strongest_update_failure,
        "strongest_decision_bias_protocol": strongest_decision_bias,
        "most_sensitive_qa_type": most_sensitive_qa,
        "qa_type_answer_score_drops": qa_drops,
        "interpretation": (
            "Stable generic Memory Echo satisfied all predeclared formation, "
            "amplification, retention, influence, and seed-consistency checks."
            if stable else
            "The benchmark did not satisfy every predeclared stability check. "
            "This is a component-level or heterogeneous Echo result, not "
            "evidence of universally stable Memory Echo."
        ),
        "scope_notes": [
            (
                "Delay is implemented as persistent retrieval after a "
                "session/noise gap, not as wall-clock forgetting."
            ),
            (
                "The fixed replacement-scope gate plus rank-discounted "
                "retrieval-to-answer policy isolates memory effects; it is "
                "not an unconstrained LLM reasoning evaluation."
            ),
            (
                "Random seeds repeat matched protocol configurations while "
                "varying concrete values and distractor order."
            ),
        ],
    }
