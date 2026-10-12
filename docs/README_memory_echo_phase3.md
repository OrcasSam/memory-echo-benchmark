# Phase 3 Memory Echo Benchmark

This benchmark models a 10-turn 1688 procurement-assistant lifecycle with four
supplier candidates, a historically recommended Supplier A, and an objectively
better Supplier B introduced at turn 6. It runs control, historical repetition,
and historical repetition plus update conditions against the repository A-MEM
retriever and official `mem0ai` `Memory.add/search/update` APIs.

## Run

```powershell
python -m scripts.legacy.run_memory_echo_phase3 --max_samples 10
python -m scripts.legacy.run_memory_echo_phase3 --max_samples 20
```

Default output is `results_memory_echo_phase3/`.

## Outputs

The runner writes local trajectory and metric files under its output directory.
Measured results are not included in this code-only repository.

