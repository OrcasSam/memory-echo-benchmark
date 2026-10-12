# Memory Echo Chamber v2: Retrieval-only Benchmark

This benchmark validates retrieval-side echo chamber behavior only.

It does **not**:

- generate final answers
- call any LLM for answering
- call any LLM judge
- use rule-based final answer correctness as a main conclusion

It evaluates whether repeated exposure to the same false claim changes memory retrieval behavior and top-k context composition.

## Systems

- `amem`: uses existing A-MEM cache under `cached_memories_robust_deepseek_deepseek-chat_locomo10`.
- `mem0`: uses official `mem0ai` with `Memory.add(..., infer=False)` and `Memory.search(...)`.

`infer=False` is important: it stores raw memories directly and avoids LLM extraction.

## Smoke

```powershell
python -m scripts.legacy.memory_echo_retrieval_v2 `
  --dataset data\locomo10.json `
  --max_samples 1 `
  --retrieve_k 5 `
  --condition all `
  --system all `
  --output results_memory_echo_v2_smoke `
  --probe_rounds 2
```

## Small-scale Validation

```powershell
python -m scripts.legacy.memory_echo_retrieval_v2 `
  --dataset data\locomo10.json `
  --max_samples 3 `
  --retrieve_k 5 `
  --condition all `
  --system all `
  --output results_memory_echo_v2_small `
  --probe_rounds 2
```

## K Sweep

```powershell
python -m scripts.legacy.memory_echo_retrieval_v2 `
  --dataset data\locomo10.json `
  --max_samples 3 `
  --retrieve_k 1,5,10,20 `
  --condition all `
  --system all `
  --output results_memory_echo_v2_sweep `
  --probe_rounds 2
```

The script builds each backend once per `system x sample x condition`, then evaluates all requested `retrieve_k` values from the same retrieval result. Changing k does not rebuild A-MEM cache or re-add Mem0 memories for each k.

## Conditions

- `control`: one injected false claim.
- `generic_repetition`: same false claim repeated several times without roles.
- `role_based_repetition`: same false claim repeated by multiple roles.
- `long_term_recall`: delayed/consolidated recall version. Use `--condition all_with_longterm`.

## Metrics

Retrieval-level:

- `target_topk_hit_rate`
- `average_target_false_rank`
- `rank_delta_vs_control`
- `retrieval_amplification_rate_claim_level`

Echo-level:

- `echo_density`: top-k share occupied by the target false claim cluster.
- `cross_source_echo_index`: diversity of source/role/round among retrieved false-claim copies.
- `echo_growth_rate`: ED growth across retrieval probe rounds.

Context-level:

- `false_claim_occupancy`
- `false_claim_copy_count`
- `target_claim_roles_topk`
- `target_claim_sources_topk`
- `target_claim_rounds_topk`

## Outputs

Each run writes:

- sample-level JSON
- aggregate JSON
- aggregate CSV
- ED plot
- CSEI plot
- conclusion markdown

## Results policy

The smoke and k-sweep runs write outputs locally. This repository publishes the
runner, not measured results or sample-level records.
