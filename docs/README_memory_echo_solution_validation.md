# Strict Memory Echo lifecycle Solution validation

This experiment implements the governance specification as a cumulative,
matched validation rather than a simplified component demo.

## Cumulative arms

1. `baseline`: no governance.
2. `s1_identity`: semantic Echo Family identity, source ID, parent memory and
   provenance inheritance.
3. `s2_retrieval_hygiene`: S1 plus exposure-only search logs, zero confidence
   gain from retrieval, no retrieval-derived fact add, and family-deduplicated
   top-k retrieval.
4. `s3_reconsolidation`: S2 plus independent-evidence update gating,
   `superseded` state and mandatory correction binding.
5. `s4_persistence`: S3 plus confidence/activation separation, activation
   decay at every consecutive session 1..30, matched-Control excess monitoring
   and three-window remediation.
6. `full_guard`: S4 plus a two-pass counterfactual decision: complete Memory
   versus reliable evidence with Echo excluded.

Every sample runs all six arms, both A-MEM and official Mem0, and Control,
Repetition and Update. All causal effects are computed within the same sample.

## Dedicated controls

- Negative: search only, irrelevant repetition, and correct-answer repetition.
- Positive: explicit Echo repetition.
- Decision challenge: a pre-polluted legacy store verifies that the two-pass
  gate detects a counterfactual decision flip and removes final DCE.

## Pre-registered success criteria

- ED: extra ED versus matched Control reduced at least 50%, with the paired
  DiD 95% confidence interval below zero.
- RRI: below Baseline, DiD 95% confidence interval below zero, residual effect
  no greater than 0.05.
- WEP: excess 1..30 area reduced at least 60%, and delay-30 excess no greater
  than 0.02.
- DCE: below Baseline and within 0.05 of matched Control.
- Utility: matched Control final-accuracy loss is non-inferior at a -0.02
  margin.

Statistics use paired 10,000-resample bootstrap confidence intervals,
Wilcoxon signed-rank tests, Holm correction and paired Cohen's dz.

## Run

```powershell
python -m scripts.legacy.run_memory_echo_solution_validation `
  --max_samples 200 --frameworks amem,mem0 `
  --output_dir results_memory_echo_solution_strict_200
```

Large runs may be split into disjoint shards and merged with
`merge_memory_echo_solution_shards.py`; all paired statistics are recomputed
from the 200 sample-level effects after merging.
