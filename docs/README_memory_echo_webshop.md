# Memory Echo Benchmark on WebShop

This experiment replaces the generated procurement scenarios with real data from
the Princeton WebShop dataset. Obtain the dataset separately from an authorized
source; no product catalog or derived trajectory is included here.

## Scale

- 10 real crowd-sourced WebShop instructions (10 independent samples)
- 100 longitudinal QA turns per sample
- A-MEM and official `mem0ai`
- control, repetition, and repetition plus update
- 60 trajectories and 6,000 retrieval interactions in total

The WebShop goal ASIN is Candidate B and the ground truth. Candidate A and the
two distractors are selected deterministically from the same real 1,000-product
catalog using TF-IDF similarity; no product, title, price, rating, attribute, or
shopping requirement is synthesized.

## Run

```powershell
python -m scripts.legacy.run_memory_echo_webshop `
  --data_dir data/webshop/data `
  --max_samples 10 `
  --output_dir results_memory_echo_webshop_100qa
```

## Outputs

The runner writes local retrieval, persistence, and decision metrics. Measured
results are not included in this code-only repository.
