# Rejected hypothesis: 83 rich features + feature selection

**Status: NOT adopted. Kept for the record, not deleted.**
The official benchmark model is still `whh_bench/` (6 curated signals,
causal personalization, 40.5% ± 2.55%). This experiment is documented here
so the claim on the dashboard ("we tried this and it didn't help") is
reproducible, not just asserted.

## What this tested

An independently-built prediction pipeline (Mamoune's design) expanded
each wearable signal into 8 summary statistics per file (mean, median,
std, min, max, p10, p90, count) — 83 candidate features total — then
compared 4 feature-selection strategies, all evaluated with
`StratifiedGroupKFold` + `balanced_accuracy` (stricter than the plain
accuracy used in `whh_bench`).

## Results (see script output for full log)

| Strategy | Balanced accuracy | Features kept |
|---|---|---|
| No selection (83 raw) | 33.9% ± 2.08% | 83 / 83 |
| Mutual-info, top 20 | 28.7% ± 1.95% | 20 / 83 |
| L1-logistic | 34.1% ± 0.62% | 81 / 83 |
| RF-based (median threshold) | 34.0% ± 0.90% | 42 / 83 |

Majority baseline: ~34.2%. None of the four beat it meaningfully, and all
sit below the official model's 40.5%.

## Why it didn't work (our best explanation)

Mean/median/min/max/p10/p90 of the same underlying daily series are highly
redundant. Feature selection can only remove duplicates — it can't
manufacture signal that isn't there. The official model's smaller, hand-
picked feature set (daily mean + real intra-day HRV variability) already
captured the useful signal without the redundancy.

## What's still worth reusing from this

RF-based selection cut the feature count in half (83 \u2192 42) with **no**
accuracy loss and lower fold-to-fold variance. Not an accuracy win, but
worth revisiting for model interpretability in a future round.

## How to reproduce

```bash
python rejected_rich_features_experiment.py
```
Requires the same `data/raw/` mcPHASES files as the main pipeline
(see the top-level `data/README.md` for access instructions).
