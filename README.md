# Memory Echo Benchmark

Code for studying how repeated retrieval and write-back can turn one outdated memory source into apparently independent evidence, and for evaluating a family–provenance–authority (FPA) serving rule. The current study compares A-MEM, Mem0, CoM, and LangMem across a five-stage memory lifecycle.

> **Code-only repository.** No source datasets, transformed questions, model answers, retrieval traces, aggregate result tables, or generated plots are distributed here. The experiments read locally supplied inputs and write their outputs to ignored directories. This repository documents the pipeline.

## Start here

| Guide | Purpose |
|---|---|
| [Experiment map](EXPERIMENTS.md) | A–H stages, entry points, inputs, and checks |
| [Dataset setup](DATASETS.md) | Official source links and local input conventions |
| [Environment template](.env.example) | Optional model-provider settings, with no credentials |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Upstream frameworks and attribution |
| [Earlier experiment guides](docs/) | Historical benchmark families |

## The lifecycle

```text
Source question + historical seed
              ↓
Encoding → Retrieval practice → Verified correction
              ↓
Thirty read-only delay probes → Final answer
```

The paired histories distinguish one-time encoding, repeated read-only access, and retrieval followed by write-back. The benchmark measures retrieval occupancy (ED), resistance after correction (RRI), persistence across delayed probes (WEP), and final-answer contamination (DCE). The FPA method groups related records by source family, evaluates provenance and validity, and serves a verified answer when one is available.

## Current A–H experiments

| Experiment | Question | Main code |
|---|---|---|
| A | Does write-back multiply evidence from one source? | `scripts/ah/run_memory_echo_ah_lifecycle.py`, `scripts/ah/run_memory_echo_ah_answers.py` |
| B | Where does the old claim survive correction? | `memory_echo_ah/`, `scripts/ah/run_memory_echo_ah_lifecycle.py` |
| C | Does it persist through thirty read-only queries? | `scripts/ah/run_memory_echo_ah_lifecycle.py` |
| D | How do serving methods compare? | `scripts/ah/prepare_com_lang_method_inputs.py`, `scripts/ah/run_com_lang_method_models.py`, `scripts/ah/run_remaining_egh.py` |
| E | Which FPA components matter? | `scripts/ah/prepare_remaining_egh.py`, `scripts/ah/run_remaining_egh.py`, `scripts/ah/run_raw_constraint_ablation.py` |
| F | Does the serving effect transfer across answer models? | `scripts/ah/run_memory_echo_cross_model.py`, `scripts/ah/run_com_lang_method_models.py` |
| G | Does the method transfer across tasks? | `scripts/ah/prepare_remaining_egh.py`, `scripts/ah/run_remaining_egh.py` |
| H | What happens when source information is incomplete or noisy? | `scripts/ah/prepare_remaining_egh.py`, `scripts/ah/run_remaining_egh.py` |

The scripts are modular and reuse intermediate outputs from earlier stages. See [EXPERIMENTS.md](EXPERIMENTS.md) before running a stage independently.
Run scripts from the repository root with `python -m`, for example `python -m scripts.ah.run_memory_echo_ah_lifecycle --help`; this keeps imports and local paths consistent after the folder reorganization.

## Repository map

```text
memory_echo_*/            Reusable protocol, adapter, metric, and serving modules
scripts/ah/               Current A–H experiment entry points and preparation
scripts/legacy/           Earlier benchmark and method runners
scripts/analysis/         Aggregation, validation, and inspection utilities
scripts/ops/              Shell and PowerShell orchestration
tests/                    Synthetic unit tests
docs/                     Detailed guides for earlier experiment families
```

Earlier benchmark and method runners remain as implementation history; the A–H entry points above are the current experiment map. We publish first-party code and synthetic tests, not every intermediate run, resumption script, report renderer, or local cache.

## Environment

Python 3.10+ is recommended for the base project. The A–H adapters may require newer versions of upstream packages; use a fresh environment and install the [A–H dependencies](requirements-ah.txt) in addition to the base requirements.

```bash
python -m venv .venv
python -m pip install -r requirements-ah.txt
```

CoM is not vendored. Obtain its official source separately and place it at `third_party/CoM` if using the CoM adapter. The LangMem adapter uses the upstream `langmem`, `langgraph`, and `langchain-openai` packages. The Mem0 adapter uses `mem0ai`. See [third-party notices](THIRD_PARTY_NOTICES.md) for source links. Copy `.env.example` to `.env` only if a model-backed phase is needed, then provide your own credentials locally; `.env` is ignored.

## Checks

The A–H unit tests use synthetic fixtures and do not need benchmark data or API keys:

```bash
python -m pytest -q tests/test_memory_echo_ah_protocol.py tests/test_memory_echo_ah_adapters.py tests/test_memory_echo_ah_catalog.py tests/test_memory_echo_cross_model.py tests/test_raw_constraint_ablation.py tests/test_deduplicated_judge.py
```

Running `python -m pytest -q` across the entire repository also checks the earlier code: tests needing unpublished source datasets or an external embedding model skip when those prerequisites are absent. Set `RUN_MODEL_TESTS=1` only after arranging model access locally. The CoM unit tests use a synthetic filter stand-in to check our adapter and lifecycle wiring; an official CoM integration run still requires a separate upstream checkout.

Full experiments require their source datasets, frozen cohort files, framework installations, and model access. Those inputs and all outputs are deliberately outside this repository.

## Attribution and license

This project builds on [A-MEM](https://github.com/WujiangXu/AgenticMemory) and retains its [MIT license](LICENSE). Third-party frameworks and datasets keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
