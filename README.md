# HormoneSignal Bench — Dashboard Deployment

## Run locally
```
npm install
npm run dev
```
Opens at http://localhost:5173

## Deploy so the team can view it (pick one)

### Option A — Vercel (easiest, free, auto-redeploys on every push)
1. Push this folder to a new GitHub repo (see below).
2. Go to vercel.com → "New Project" → import that repo.
3. Framework preset: Vite. Leave build command (`npm run build`) and output dir (`dist`) as default.
4. Deploy. You get a shareable URL (e.g., `hormonesignal-bench.vercel.app`) immediately, and it auto-updates every time someone pushes to `main`.

### Option B — Netlify (same idea as Vercel)
1. netlify.com → "Add new site" → "Import an existing project" → connect the GitHub repo.
2. Build command: `npm run build`, publish directory: `dist`.

### Option C — GitHub Pages (fully free, no third-party account needed)
```
npm install --save-dev gh-pages
```
Add to package.json scripts: `"deploy": "vite build && gh-pages -d dist"`
Then: `npm run deploy`
Site appears at `https://<your-username>.github.io/<repo-name>/`

## Push to GitHub
```
git init
git add .
git commit -m "HormoneSignal Bench dashboard v1"
git branch -M main
git remote add origin https://github.com/<your-org>/<repo-name>.git
git push -u origin main
```

## Team workflow note
Each contributor's feature work (personalization, ground-truth, timeseries/behavioral schema — see the role-split plan) should live in **separate branches** and only touch **new files/columns**, per the parallel-work plan. Merge into `main` only after the weekly shared re-scoring run. Vercel/Netlify will auto-preview every pull request with its own temporary URL before it's merged — useful for reviewing each person's work in isolation before it hits the shared `main` deploy.
