# Generic Memory Echo Lifecycle Benchmark

This benchmark rewrites balanced subsets of LoCoMo, LongMemEval, HotpotQA, and
BEAM. The gold answer becomes verified correction evidence. A different real
answer from the same dataset and preferably the same ability type becomes the
Echo seed. Original question, answer, evidence, source ID, and distractor source
ID remain auditable in every sample.

Five stages are executed for every sample: Encoding, Retrieval Practice,
Updating / Reconsolidation, Delayed Retrieval at multiple session gaps, and
Final Decision. Every stage contains purpose-labelled QA.

Metrics:

- ED: rank-weighted semantic occupancy of Echo memories in top-k retrieval.
- RRI: residual Echo survival and competitive share after correction. Paired
  `RRI_effect` subtracts the matched control trajectory.
- PHL: exponential-decay half-life fitted from multiple delayed ED probes.
  `censored=true` means the fitted half-life lies outside the observed window.
- DCE: semantic affinity advantage of an Echo-driven answer over the verified
  answer. Both a continuous score and thresholded contamination are emitted.

Run a ten-sample A-MEM experiment:

```powershell
python -m scripts.legacy.run_memory_echo_lifecycle --max_samples 12 --frameworks amem
```

Run both real memory frameworks:

```powershell
python -m scripts.legacy.run_memory_echo_lifecycle --max_samples 12 --frameworks amem,mem0
```
