# Data — not included in this repository

The raw mcPHASES CSVs are **not, and will never be, committed to this repo.**
mcPHASES is distributed under PhysioNet's Restricted Health Data License —
a Data Use Agreement (DUA), not a fully open download. Redistributing the
raw files here would violate that license.

## How to get the data yourself

1. Create a PhysioNet account: https://physionet.org
2. Complete the required training/credentialing for restricted health data
3. Request access to mcPHASES: https://physionet.org/content/mcphases/1.0.0/
4. Once approved, download these files into `data/raw/` (gitignored):
   - hormones_and_selfreport.csv
   - subject-info.csv
   - resting_heart_rate.csv
   - computed_temperature.csv
   - heart_rate_variability_details.csv
   - respiratory_rate_summary.csv
   - active_minutes.csv
   - sleep_score.csv
   - glucose.csv

## Regenerating results.json from your own credentialed copy

```bash
pip install -e .
python -m whh_bench.export_results --data-dir data/raw --out results/latest_run.json
cp results/latest_run.json public/results/latest_run.json
```

This is the only file that leaves your machine — the derived, aggregate
numbers, never the raw records.
