# Dataset setup

Raw third-party datasets are intentionally not committed. Download each dataset from its official source, follow its license, and pass the local path to the runner.

## LoCoMo

- Official repository: https://github.com/snap-research/locomo
- Expected file: `data/locomo10.json`
- CLI argument: `--locomo`

## LongMemEval

- Official repository: https://github.com/xiaowu0162/LongMemEval
- Expected file used by this project: `data/longmemeval_oracle.json`
- CLI argument: `--longmemeval`

The current loader targets the original LongMemEval release, not LongMemEval-V2.

## HotpotQA

- Official website: https://hotpotqa.github.io/
- Expected file: `hotpot_dev_distractor.parquet`
- CLI argument: `--hotpotqa`
- The official site states that HotpotQA is distributed under CC BY-SA 4.0.

## BEAM

- Official repository: https://github.com/mohammadtavakoli78/BEAM
- Expected input: the extracted BEAM repository or dataset directory
- CLI argument: `--beam`

## Example

```bash
python -m scripts.legacy.run_memory_echo_lifecycle \
  --locomo C:/datasets/locomo/data/locomo10.json \
  --longmemeval C:/datasets/LongMemEval/data/longmemeval_oracle.json \
  --hotpotqa C:/datasets/HotpotQA/hotpot_dev_distractor.parquet \
  --beam C:/datasets/BEAM \
  --max_samples 200 \
  --frameworks amem,mem0
```

This repository is code-only. Do not commit source records or transformed sample-level records here, regardless of whether a dataset's license would otherwise permit redistribution.

