# HormoneSignal Bench

A benchmark for predicting menstrual cycle phase from wearable signals —
real code, real (credentialed) data, and an honest account of what worked
and what didn't.

## Structure

```
whh_bench/            Python modeling package (preprocessing, personalization,
                       label-confidence checking, cross-validation, metrics)
results/latest_run.json   Precomputed, derived results — the ONLY artifact
                       generated from real data that leaves a researcher's
                       machine. No raw mcPHASES rows are ever committed.
public/results/        Same file, served to the deployed frontend
src/                   React dashboard (frontend)
data/                  No data here — see data/README.md for how to get your
                       own credentialed copy from PhysioNet
```

## Run the frontend locally
```bash
npm install
npm run dev
```

## Run the real modeling pipeline (requires your own PhysioNet-credentialed mcPHASES download — see data/README.md)
```bash
pip install -e .
python -m whh_bench.baseline --data-dir data/raw
```

## Regenerate the numbers the dashboard displays
```bash
python -m whh_bench.export_results --data-dir data/raw --out results/latest_run.json
cp results/latest_run.json public/results/latest_run.json
git add results/latest_run.json public/results/latest_run.json
git commit -m "Update results.json from latest pipeline run"
```
The dashboard fetches `./results/latest_run.json` at page load. If the file
is missing (e.g. someone previews the site without ever running the
pipeline), it falls back to the last-known bundled numbers and says so in
the UI — it never silently pretends a fallback number is live.

## Deploy so the team can view it

### Option A — Vercel (easiest, free, auto-redeploys on every push)
1. Push this folder to a GitHub repo.
2. vercel.com → New Project → import the repo. Framework preset: Vite.
3. Deploy. Auto-updates on every push to `main`.

### Option B — Netlify
netlify.com → Import project → build command `npm run build`, publish dir `dist`.

### Option C — GitHub Pages
```bash
npm install --save-dev gh-pages
# add to package.json scripts: "deploy": "vite build && gh-pages -d dist"
npm run deploy
```

## Push to GitHub
```bash
git init
git add .
git commit -m "HormoneSignal Bench — frontend + modeling package + real results"
git branch -M main
git remote add origin https://github.com/<your-org>/<repo-name>.git
git push -u origin main
```

## Team workflow note
Each contributor's feature branch should only touch new files/columns
(see the parallel-work plan). Re-run `export_results.py` and commit the
refreshed `results.json` as part of the weekly shared sync — never hand-edit
that file.
