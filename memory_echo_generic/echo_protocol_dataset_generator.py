"""Generate abstract memory-state trajectories for four Echo protocols."""
import random
from typing import Dict, Iterable, List

from . import MEMORY_UNIT_TYPES, PROTOCOLS


VALUE_WORDS = (
    "amber", "cobalt", "ivory", "jade", "lilac", "navy", "ochre",
    "pearl", "quartz", "saffron", "teal", "umber", "violet", "willow",
)

TYPE_LANGUAGE = {
    "fact": ("recorded state", "verified state"),
    "preference": ("preferred mode", "revised preferred mode"),
    "plan": ("active route", "revised route"),
    "constraint": ("applicable limit", "revised limit"),
    "hypothesis": ("leading explanation", "better-supported explanation"),
    "candidate": ("leading option", "new leading option"),
    "intermediate_conclusion": ("interim conclusion", "revised conclusion"),
}

PROTOCOL_DISTRIBUTIONS = {
    "memory_occupancy": {
        "evidence_strengths": ("weak", "medium"),
        "noise_levels": ("low", "medium"),
        "delay_sessions": (3, 7),
        "decision_ambiguities": ("medium",),
        "writeback_budget": 3,
    },
    "update_adaptation": {
        "evidence_strengths": ("weak", "medium", "strong"),
        "noise_levels": ("low", "medium"),
        "delay_sessions": (7, 14),
        "decision_ambiguities": ("low", "medium"),
        "writeback_budget": 5,
    },
    "retention_stability": {
        "evidence_strengths": ("medium", "strong"),
        "noise_levels": ("medium", "high"),
        "delay_sessions": (14, 30, 90),
        "decision_ambiguities": ("medium",),
        "writeback_budget": 4,
    },
    "decision_reliability": {
        "evidence_strengths": ("medium", "strong"),
        "noise_levels": ("low", "medium", "high"),
        "delay_sessions": (7, 30),
        "decision_ambiguities": ("low", "medium", "high"),
        "writeback_budget": 4,
    },
}


def _claim(memory_type: str, subject: str, value: str, new: bool = False,
           strength: str = "medium", should_update: bool = True) -> str:
    old_label, new_label = TYPE_LANGUAGE[memory_type]
    label = new_label if new else old_label
    if not new:
        return f"For abstract unit {subject}, the {label} is {value}."
    support = {
        "weak": "A preliminary observation suggests",
        "medium": "Two independent checks indicate",
        "strong": "A verified revision supported by three independent checks establishes",
    }[strength]
    replacement = (
        "This evidence satisfies the replacement rule."
        if should_update else
        "This evidence is out of scope for replacement and does not invalidate the stored state."
    )
    return (
        f"{support} that for abstract unit {subject}, the {label} is {value}. "
        f"{replacement}"
    )


def _distractors(rng: random.Random, memory_type: str, subject: str,
                 sample_id: int, noise_level: str) -> List[Dict]:
    count = {"low": 2, "medium": 4, "high": 6}[noise_level]
    rows = []
    for j in range(count):
        value = f"{VALUE_WORDS[(sample_id + j + 4) % len(VALUE_WORDS)]}-{(sample_id + j) % 17:02d}"
        noise_subject = subject if j % 2 == 0 else f"noise-{sample_id:04d}-{j + 1}"
        if noise_subject == subject:
            claim = (
                f"An unrelated marker for abstract unit {subject} mentions "
                f"{value}, but it is not a valid {memory_type.replace('_', ' ')} state."
            )
        else:
            claim = _claim(memory_type, noise_subject, value)
        rows.append({
            "memory_label": "NOISE",
            "subject": noise_subject,
            "value": value,
            "claim": claim,
            "hard_negative": noise_subject == subject,
        })
    rng.shuffle(rows)
    return rows


def generate_protocol_dataset(samples_per_protocol: int = 100,
                              seeds: Iterable[int] = (11, 29, 47),
                              allow_small_debug: bool = False) -> List[Dict]:
    if samples_per_protocol < 100 and not allow_small_debug:
        raise ValueError("Each generic protocol requires at least 100 independent trajectories")
    seeds = tuple(seeds)
    if len(seeds) < 2:
        raise ValueError("Use at least two random seeds")
    dataset = []
    global_id = 0
    for protocol_index, protocol in enumerate(PROTOCOLS):
        dist = PROTOCOL_DISTRIBUTIONS[protocol]
        for local_id in range(samples_per_protocol):
            seed_position = local_id % len(seeds)
            seed_replicate_id = local_id // len(seeds)
            seed = seeds[seed_position]
            rng = random.Random(
                seed * 100_000
                + protocol_index * 10_000
                + seed_replicate_id
            )
            # Experimental configuration is held constant across the seed
            # replicate; concrete values and distractor order vary by seed.
            config_rng = random.Random(
                2_027_000 + protocol_index * 10_000
                + seed_replicate_id
            )
            memory_type = MEMORY_UNIT_TYPES[
                (seed_replicate_id + protocol_index)
                % len(MEMORY_UNIT_TYPES)
            ]
            subject = (
                f"unit-{protocol_index + 1}-"
                f"{seed_replicate_id + 1:04d}-seed-{seed}"
            )
            old_index = rng.randrange(len(VALUE_WORDS))
            new_index = (old_index + rng.randrange(3, len(VALUE_WORDS) - 2)) % len(VALUE_WORDS)
            old_value = (
                f"{VALUE_WORDS[old_index]}-"
                f"{(seed_replicate_id * 3 + seed_position * 17 + 7) % 97:02d}"
            )
            new_value = (
                f"{VALUE_WORDS[new_index]}-"
                f"{(seed_replicate_id * 5 + seed_position * 19 + 11) % 97:02d}"
            )
            evidence_strength = config_rng.choice(
                dist["evidence_strengths"]
            )
            noise_level = config_rng.choice(dist["noise_levels"])
            # Catch trials prevent a trivial "always choose the new value" rule.
            # The decision protocol uses a balanced 50/50 replacement split;
            # the other protocols reserve 20% of trajectories as catch trials.
            should_update = (
                (seed_replicate_id + protocol_index) % 2 == 0
                if protocol == "decision_reliability"
                else (seed_replicate_id + protocol_index) % 5 != 0
            )
            lineage_id = (
                f"lineage-{protocol_index + 1}-"
                f"{seed_replicate_id + 1:04d}-seed-{seed}"
            )
            sample = {
                "sample_id": global_id,
                "protocol": protocol,
                "protocol_sample_id": local_id,
                "random_seed": seed,
                "random_seed_replicate_id": seed_replicate_id,
                "memory_unit_type": memory_type,
                "subject": subject,
                "initial_memory": {
                    "lineage_id": lineage_id,
                    "version": "old",
                    "memory_label": "OLD",
                    "value": old_value,
                    "claim": _claim(memory_type, subject, old_value),
                },
                "conflicting_evidence": {
                    "lineage_id": lineage_id,
                    "version": "new",
                    "memory_label": "NEW",
                    "value": new_value,
                    "strength": evidence_strength,
                    "corroboration_count": {"weak": 1, "medium": 2, "strong": 3}[evidence_strength],
                    "replacement_status": (
                        "sufficient" if should_update else "insufficient"
                    ),
                    "claim": _claim(
                        memory_type, subject, new_value, True,
                        evidence_strength, should_update,
                    ),
                },
                "delay_sessions": config_rng.choice(dist["delay_sessions"]),
                "noise_level": noise_level,
                "decision_ambiguity": config_rng.choice(
                    dist["decision_ambiguities"]
                ),
                "writeback_budget": dist["writeback_budget"],
                "distractor_memories": _distractors(
                    rng, memory_type, subject, global_id, noise_level
                ),
                "canonical_history_label": "OLD",
                "canonical_current_label": "NEW" if should_update else "OLD",
                "should_update": should_update,
            }
            dataset.append(sample)
            global_id += 1
    return dataset


def dataset_manifest(dataset: List[Dict], samples_per_protocol: int,
                     seeds: Iterable[int], qa_per_trajectory: int) -> Dict:
    seeds = tuple(seeds)
    return {
        "benchmark": "Generic Memory Echo Benchmark",
        "domain": "domain-independent abstract memory states",
        "protocols": list(PROTOCOLS),
        "memory_unit_types": list(MEMORY_UNIT_TYPES),
        "samples_per_protocol": samples_per_protocol,
        "base_trajectories": len(dataset),
        "qa_per_trajectory": qa_per_trajectory,
        "random_seeds": list(seeds),
        "complete_seed_replicates_per_protocol": (
            samples_per_protocol // len(seeds)
        ),
        "seed_remainder_trajectories_per_protocol": (
            samples_per_protocol % len(seeds)
        ),
        "debug_run": samples_per_protocol < 100,
        "protocol_distributions": PROTOCOL_DISTRIBUTIONS,
        "sample_structure": [
            "Initial Memory", "Repeated Retrieval", "Conflicting Evidence",
            "Delayed Retrieval", "Final Decision",
        ],
    }
