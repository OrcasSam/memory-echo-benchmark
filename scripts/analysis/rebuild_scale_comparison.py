"""Rebuild old-vs-new comparison artifacts without rerunning memory trajectories."""
import json
from pathlib import Path

from memory_echo_phase3.cognitive_aggregator import (
    aggregate_by_phase, aggregate_conditions, write_csv,
)
from memory_echo_phase3.scale_comparator import (
    compare_condition_rows, compare_phase_rows, load_prior_results,
    write_comparison_figures,
)


ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "results_memory_echo_30x200"
OLD = ROOT / "results_memory_echo_v6"


def main():
    new_samples = json.loads((NEW / "sample_level.json").read_text(encoding="utf-8"))
    old_samples = json.loads((OLD / "sample_level.json").read_text(encoding="utf-8"))
    new_conditions = aggregate_conditions(new_samples)
    new_phases = aggregate_by_phase(new_samples)
    old_conditions, old_phases = load_prior_results(OLD)
    shared = ({s["instruction_id"] for s in new_samples}
              & {s["instruction_id"] for s in old_samples})
    matched_new = [s for s in new_samples if s["instruction_id"] in shared]
    matched_old = [s for s in old_samples if s["instruction_id"] in shared]
    comparison = compare_condition_rows(
        old_conditions, new_conditions, aggregate_conditions(matched_new),
        aggregate_conditions(matched_old), len(shared),
    )
    phase_comparison = compare_phase_rows(old_phases, new_phases)
    write_csv(NEW / "old_vs_new_comparison.csv", comparison)
    write_csv(NEW / "old_vs_new_phase_comparison.csv", phase_comparison)
    aggregate_path = NEW / "aggregate_results.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["old_vs_new"] = comparison
    aggregate["old_vs_new_phase"] = phase_comparison
    aggregate["comparison_metadata"] = {
        "overall_comparison": "old 10x100 small subset vs new 30x200 10K subset",
        "shared_instruction_count": len(shared),
        "matched_instruction_ids": sorted(shared),
        "causal_caveat": (
            "The overall comparison changes sample count, trajectory length, and data subset. "
            "The matched comparison uses only exactly shared human instructions."
        ),
    }
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_comparison_figures(NEW, comparison, phase_comparison)
    print(json.dumps(aggregate["comparison_metadata"], indent=2))


if __name__ == "__main__":
    main()
