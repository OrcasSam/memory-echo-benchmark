# Contributing

Contributions should preserve the benchmark's paired and auditable design.

1. Do not replace official Mem0 with a mock or minimal implementation in benchmark runs.
2. Keep Control, Repetition, and Update conditions matched by `sample_id`.
3. Preserve consecutive delay checks from session 1 through 30 for WEP experiments.
4. Record source IDs, framework versions, thresholds, Top-k, and random seeds.
5. Add or update tests for metric, adapter, trajectory, or aggregation changes.
6. Do not commit datasets, transformed samples, predictions, retrieval traces, aggregate results, generated figures, credentials, local memory stores, or caches. Review the exact staged file list before every push.

Before opening a pull request:

```bash
python -m pytest -q tests/test_memory_echo_solution.py tests/test_dce_v3.py
```

If the four external datasets are available locally, also run:

```bash
python -m pytest -q tests/test_memory_echo_lifecycle.py
```

