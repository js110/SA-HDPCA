# SA-HDPCA

Code release for **SA-HDPCA** (Sensitivity-Aware Differentially Private Clustering).

This repository contains the source code, configs, and experiment scripts used for the strict-DP revision of the method.
Large result files, datasets, and manuscript files are intentionally excluded.

## Repository Structure

- `src/`: core implementation (DP clustering, scheduling, preprocessing, metrics, runner)
- `baselines/`: baseline method implementations
- `configs/`: experiment configurations
- `scripts/`: helper scripts for running comparisons and generating figures/tables
- `requirements.txt`: Python dependencies

## Environment

- Python 3.10+ (tested with Python 3.12)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

Default paths are defined in `configs/default.yaml`:

- HAR: `./UCI HAR Dataset`
- GAS: `./gas`

Expected HAR structure:

```text
UCI HAR Dataset/
  train/X_train.txt
  train/y_train.txt
  test/X_test.txt
  test/y_test.txt
```

Expected GAS structure:

```text
gas/
  batch1.dat
  batch2.dat
  ...
```

## Quick Start

### 1) Smoke test

```bash
python -m src.runner --config configs/default.yaml --out outputs/smoke --smoke
```

### 2) Main experiments (example: e1)

```bash
python -m src.runner --config configs/default.yaml --out outputs/main_e1 --exp e1
```

### 3) Strict Tier A tuning and frozen evaluation

```bash
python scripts/run_tier_a_strict.py \
  --config configs/default.yaml \
  --out outputs/tier_a_strict
```

This runner performs a fair strict-DP tuning stage on the core budgets and then freezes the best configuration for the final Tier A evaluation.

### 4) Recent baseline comparison

```bash
python scripts/run_recent_baselines.py \
  --datasets HAR GAS \
  --eps 0.5 0.8 1.0 1.5 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --out_dir outputs/compare_recent_2024_2025 \
  --use_full_method \
  --config configs/default.yaml
```

Plot recent baseline figures:

```bash
python scripts/plot_recent_baselines.py \
  --input outputs/compare_recent_2024_2025/raw_results_new.csv \
  --outdir outputs/compare_recent_2024_2025/figures
```

### 5) Revision extension experiments

```bash
python scripts/run_revision_exps.py --config configs/revision.yaml --out outputs/revision --only all
python scripts/make_revision_figs.py
```

## Notes

- Runner outputs are written under your `--out` folder (CSV summary, per-run history, figures).
- Use fixed seeds in configs for reproducible runs.
- This public repo is code-only by design.
