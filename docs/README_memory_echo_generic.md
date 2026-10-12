# Generic Memory Echo Benchmark

This benchmark tests Memory Echo without relying on shopping, travel, medical,
or another business domain. A sample is an abstract memory state with one of
seven types: fact, preference, plan, constraint, hypothesis, candidate, or
intermediate conclusion.

## Design

Every trajectory follows the same longitudinal order:

1. Initial Memory
2. Repeated Retrieval
3. Conflicting Evidence
4. Delayed Retrieval after a session/noise gap
5. Final Decision

The four protocols change the data distribution and QA emphasis:

| Protocol | Main manipulation | Main question |
|---|---|---|
| Memory Occupancy | more seeding probes, low/medium interference | does OLD occupy top-k? |
| Update Adaptation | weak/medium/strong evidence, largest write-back budget | does NEW replace OLD? |
| Retention Stability | longer gaps and medium/high noise | does Echo remain after reinforcement stops? |
| Decision Reliability | balanced valid-update/catch trials and three ambiguity levels | does OLD bias action? |

Each protocol has 100 independent trajectories. Three seeds repeat matched
protocol configurations while varying concrete values and distractor order.
Each trajectory has 20 phase-specific QA covering recall, comparison, update
awareness, consistency, decision, and justification.

The matched conditions are:

- `control`: standard evaluation probes and neutral capacity-matching writes;
- `historical_repetition`: the same probes plus unscored OLD retrieval
  exposures;
- `historical_repetition_update`: the same retrieval exposures plus real
  target `add` and `update` feedback.

The number of stored items is matched across conditions. This prevents a larger
index alone from explaining ED differences. A-MEM and official `mem0ai` both
use their real retrieval paths; Mem0 calls `Memory.add`, `Memory.search`, and
`Memory.update`.

## Metrics

For top-k retrieval result \(R_q^k\):

```text
ED  = mean(count(OLD in top-k) / retrieved_count)

w(rank) = 1 / log2(rank + 1)
UIR = mean(W_NEW / (W_OLD + W_NEW))

EPS = ED_delayed_matched_probe / ED_pre_stop_matched_probe

DBI = OLD-biased outputs / eligible decision probes
```

ED uses the standardized pre-stop occupancy probe. UIR is calculated after
conflicting evidence and is aggregated over valid-update trials. EPS uses the
same query before and after reinforcement stops; a zero denominator is
undefined, not zero. DBI includes only true `decision` QA where the visible
replacement rule requires NEW. Catch trials are excluded from DBI.

## Run and validate

```powershell
python -m scripts.legacy.run_generic_memory_echo `
  --samples_per_protocol 100 `
  --qa_per_trajectory 20 `
  --seeds 11,29,47 `
  --frameworks amem,mem0 `
  --output_dir results_memory_echo_generic

python -m scripts.analysis.validate_generic_memory_echo `
  --output_dir results_memory_echo_generic
```

Small runs require the explicit `--debug_allow_small` flag and are marked
non-formal. Interrupted formal runs can use `--resume`.

## Interpretation guide

Compare matched control, read-only repetition, and retrieval-plus-update
histories before interpreting any change in occupancy, retention, or decision
metrics. This code-only repository does not distribute measured outcomes.

Delay is a session/noise-gap test, not a claim about real wall-clock forgetting.
The answer layer applies the visible replacement-scope rule and then a fixed
rank-discounted retrieval-to-decision policy. It is held constant to isolate
memory effects; it is not an unconstrained LLM reasoning evaluation.
