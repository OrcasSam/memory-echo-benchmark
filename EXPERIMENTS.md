# Experiment code map

This is a code map, not a results archive. A–H share a frozen source-question cohort and paired memory histories. All dataset copies, cohort records, generated answers, retrieved-memory traces, tables, and figures stay local and are excluded from Git.

## Inputs and stages

| Stage | Purpose | Entry point(s) | Local prerequisites |
|---|---|---|---|
| A | Echo formation across one-time, read-only, and write-back histories | `scripts/ah/run_memory_echo_ah_lifecycle.py`; `scripts/ah/run_memory_echo_ah_answers.py` | Rewritten source questions and a locked cohort list |
| B | Correction effects in stored and served memory | `memory_echo_ah/protocol.py`; `scripts/ah/run_memory_echo_ah_lifecycle.py` | Same paired trajectories as A |
| C | Persistence over 30 read-only delay probes | `scripts/ah/run_memory_echo_ah_lifecycle.py` | Same paired trajectories as A |
| D | Baseline, budget, CRAG-M, PairLR, and FPA serving comparison | `scripts/ah/prepare_com_lang_method_inputs.py`; `scripts/ah/run_com_lang_method_models.py`; `scripts/ah/run_remaining_egh.py` | Lifecycle traces, model credentials for generation and judging |
| E | FPA component ablation | `scripts/ah/prepare_remaining_egh.py`; `scripts/ah/run_remaining_egh.py`; `scripts/ah/run_raw_constraint_ablation.py` | Lifecycle traces and generated answer records |
| F | DeepSeek/Kimi answer-model comparison | `scripts/ah/run_memory_echo_cross_model.py`; `scripts/ah/run_com_lang_method_models.py`; `scripts/ah/run_remaining_egh.py` | Matched decision contexts and provider credentials |
| G | Cross-task checks | `scripts/ah/prepare_remaining_egh.py`; `scripts/ah/run_remaining_egh.py` | Task-specific local inputs |
| H | Incomplete/noisy source conditions | `scripts/ah/prepare_remaining_egh.py`; `scripts/ah/run_remaining_egh.py` | Lifecycle traces and source-condition inputs |

`memory_echo_ah/` contains the four-framework adapters and strict read/write protocol. `memory_echo_multimodel/` contains answer-generation and judging helpers. Run entry points from the repository root as modules, for example `python -m scripts.ah.run_memory_echo_ah_lifecycle --help`. The staged scripts retain their local-input defaults for the original experiment layout; review each script's arguments and defaults before execution (some preparation scripts run immediately and do not implement `--help`). `scripts/ah/prepare_remaining_egh.py` expects its task-transfer input under `data/scenario_sample_level.jsonl` unless its local path is adapted.

## Earlier code: keep or omit?

The earlier lifecycle, generic-scene, WebShop, phase-3, and FPA/A–D modules remain in the repository because current A–H scripts import or build on their adapters, metrics, or serving logic. Historical one-off resumption scripts, report renderers, local manuscript builds, and vendored copies of upstream frameworks are not needed for the code release. Their outputs are never release inputs.

## External components

- A-MEM code is retained with its original license.
- Mem0 uses the official `mem0ai` package.
- CoM is loaded from a separate checkout at `third_party/CoM`.
- LangMem uses `langmem`, `langgraph`, and `langchain-openai` packages.
- Source datasets must be acquired from their official distributors; see [DATASETS.md](DATASETS.md).

Do not commit the downloaded source records, transformed examples, individual predictions, retrieval logs, cached embeddings, model weights, or aggregate result files. The repository's `.gitignore` covers common locations and extensions, but the commit file list must still be reviewed before each push.

## Source-only verification

The synthetic tests check protocol separation, adapters, input catalogues, prompt construction, bootstrap helpers, and request deduplication without publishing experiment records:

```bash
python -m pytest -q tests/test_memory_echo_ah_protocol.py tests/test_memory_echo_ah_adapters.py tests/test_memory_echo_ah_catalog.py tests/test_memory_echo_cross_model.py tests/test_raw_constraint_ablation.py tests/test_deduplicated_judge.py
```

The full `python -m pytest -q` suite includes older experiment checks. Data-dependent tests skip if their external datasets are absent; embedding-model checks run only when `RUN_MODEL_TESTS=1` is set. The CoM adapter unit tests use a synthetic filter stub and do not validate the official CoM implementation itself.

Model-backed execution and exact numeric reproduction require external datasets, framework versions, provider access, and the original frozen cohort. These are not bundled in this repository.
