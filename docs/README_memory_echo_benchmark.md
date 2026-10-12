# Memory Echo Chamber Benchmark

This is a lightweight benchmark for testing whether repeated exposure to a false memory makes that memory more likely to be retrieved and dominate a final answer.

It is intentionally small first: smoke test, local embedding retrieval, no required LLM API calls.

## Goal

The benchmark tests three conditions:

- `control`: one false memory is injected once.
- `generic_repetition`: the same false memory is injected multiple times.
- `role_based_repetition`: the same false memory is repeated by different roles.

Optional:

- `long_term_recall`: role repetition plus a consolidated summary memory.

## Systems

- `amem`: loads existing A-MEM memory cache when available.
- `mem0_minimal`: a replaceable Mem0-style local adapter. It stores one memory per LoCoMo turn and uses the same retrieval/evaluation interface. If a real Mem0 entrypoint is added later, replace this adapter without changing the benchmark metrics.

## Smoke Test

```powershell
python -m scripts.legacy.memory_echo_benchmark `
  --dataset data\locomo10.json `
  --max_samples 1 `
  --retrieve_k 10 `
  --seed 13 `
  --condition all `
  --system all `
  --output results_memory_echo_smoke
```

Expected output:

- sample-level JSON
- aggregate JSON
- aggregate CSV
- `retrieval_amplification_rate.png`
- `false_memory_dominance.png`
- a short printed conclusion about whether echo-chamber behavior appeared

## Small Benchmark

```powershell
python -m scripts.legacy.memory_echo_benchmark `
  --dataset data\locomo10.json `
  --max_samples 3 `
  --retrieve_k 10 `
  --seed 13 `
  --condition all `
  --system all `
  --output results_memory_echo_small
```

To include long-term recall:

```powershell
python -m scripts.legacy.memory_echo_benchmark `
  --dataset data\locomo10.json `
  --max_samples 3 `
  --retrieve_k 10 `
  --seed 13 `
  --condition all_with_longterm `
  --system all `
  --output results_memory_echo_small_longterm
```

## Important Parameters

| Parameter | Default | Purpose |
|---|---:|---|
| `--dataset` | `data/locomo10.json` | Dataset path |
| `--max_samples` | `1` | Smoke/small scale control: use 1/3/5/10 |
| `--retrieve_k` | `10` | Retrieval depth; supports 1/5/10/20 |
| `--seed` | `13` | Fixed random seed |
| `--condition` | `all` | One condition or all conditions |
| `--system` | `all` | `amem`, `mem0_minimal`, `mem0`, or `all` |
| `--output` | `results_memory_echo` | Output directory |
| `--cache_dir` | existing A-MEM cache | A-MEM cache directory |
| `--probe_rounds` | `3` | Repeated retrieval probes before final answer |

## Metrics

The benchmark outputs sample-level and aggregate-level metrics:

- Retrieval Amplification Rate: rank delta of the target false memory versus the control condition.
- False Memory Dominance: whether repeated false memories occupy enough top-k space to dominate the final rule-based answer.
- Memory Echo Index: repeated false-memory citations, unique rounds, role reuse, and cross-agent reuse rate.
- Auxiliary metrics: top-k hit rate, average target rank, false hits in top-k, answer correctness, contradiction rate.

## Notes

- The benchmark does not mutate the existing A-MEM cache.
- Changing `--retrieve_k` does not rebuild memory cache.
- The default benchmark does not call DeepSeek/OpenAI APIs.
- `mem0_minimal` is a lightweight adapter, not a full Mem0 implementation.
