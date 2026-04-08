# Evaluation Analysis

This folder contains lightweight analysis utilities for the report-facing
evaluation outputs produced by `./build/evaluation`.

Run:

```bash
python3 analysis/plot_evaluation.py
```

The script reads CSVs from `outputs/evaluation/` and writes:

- PNG plots to `outputs/analysis/plots/`
- a short Markdown summary to `outputs/analysis/summary.md`

It expects `pandas`, `matplotlib`, and `seaborn` to be available.
