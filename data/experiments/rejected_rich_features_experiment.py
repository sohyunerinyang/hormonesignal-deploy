from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_parser = argparse.ArgumentParser()
_parser.add_argument("--data-dir", default="data/raw", help="Folder containing the raw mcPHASES CSVs")
_args, _ = _parser.parse_known_args()
from functools import partial

DATA_FOLDER = Path(_args.data_dir)
ID_COLUMN, INTERVAL_COLUMN, DAY_COLUMN, TARGET_COLUMN = "id", "study_interval", "day_in_study", "phase"
RANDOM_STATE = 42

# ---------- CONFLICT #1 RESOLVED: keep Erin's real label text as-is ----------
def normalize_phase(value):
    if pd.isna(value): return np.nan
    text = str(value).strip()
    mapping = {"menstrual":"Menstrual","follicular":"Follicular","fertility":"Fertility","luteal":"Luteal"}
    return mapping.get(text.lower(), np.nan)

# ---------- CONFLICT #2 RESOLVED: Mamoune's rich per-file statistic extraction ----------
FILE_SPECS = {
    "resting_heart_rate": {"file":"resting_heart_rate.csv","day_candidates":["day_in_study"],"value_columns":["value"]},
    "heart_rate_variability": {"file":"heart_rate_variability_details.csv","day_candidates":["day_in_study"],"value_columns":["rmssd"]},
    "computed_temperature": {"file":"computed_temperature.csv","day_candidates":["sleep_start_day_in_study","day_in_study"],"value_columns":["nightly_temperature"]},
    "sleep_score": {"file":"sleep_score.csv","day_candidates":["day_in_study"],"value_columns":["overall_score"]},
    "active_minutes": {"file":"active_minutes.csv","day_candidates":["day_in_study"],"value_columns":["sedentary","lightly","moderately","very"]},
    "respiratory_rate": {"file":"respiratory_rate_summary.csv","day_candidates":["day_in_study"],"value_columns":["full_sleep_breathing_rate"]},
    "stress_score": {"file":"stress_score.csv","day_candidates":["day_in_study"],"value_columns":["stress_score"]},
    "glucose": {"file":"glucose.csv","day_candidates":["day_in_study"],"value_columns":["glucose_value"]},
}
MIN_FEATURE_COVERAGE = 0.05

def choose_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def aggregate_passive_file(modality, spec):
    fp = DATA_FOLDER / spec["file"]
    if not fp.exists():
        print(f"Skipping missing file: {spec['file']}"); return None
    df = pd.read_csv(fp, low_memory=False)
    if ID_COLUMN not in df.columns or INTERVAL_COLUMN not in df.columns:
        print(f"Skipping {spec['file']}: missing id/study_interval"); return None
    day_col = choose_existing_column(df, spec["day_candidates"])
    if day_col is None:
        print(f"Skipping {spec['file']}: no day column"); return None
    available = [c for c in spec["value_columns"] if c in df.columns]
    if not available:
        print(f"Skipping {spec['file']}: no value columns"); return None
    work = df[[ID_COLUMN, INTERVAL_COLUMN, day_col, *available]].copy()
    for c in available:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    aggs = {}
    for c in available:
        aggs[f"{modality}__{c}__mean"] = pd.NamedAgg(column=c, aggfunc="mean")
        aggs[f"{modality}__{c}__median"] = pd.NamedAgg(column=c, aggfunc="median")
        aggs[f"{modality}__{c}__std"] = pd.NamedAgg(column=c, aggfunc="std")
        aggs[f"{modality}__{c}__min"] = pd.NamedAgg(column=c, aggfunc="min")
        aggs[f"{modality}__{c}__max"] = pd.NamedAgg(column=c, aggfunc="max")
        aggs[f"{modality}__{c}__p10"] = pd.NamedAgg(column=c, aggfunc=lambda v: v.quantile(0.10))
        aggs[f"{modality}__{c}__p90"] = pd.NamedAgg(column=c, aggfunc=lambda v: v.quantile(0.90))
        aggs[f"{modality}__{c}__count"] = pd.NamedAgg(column=c, aggfunc="count")
    daily = work.groupby([ID_COLUMN, INTERVAL_COLUMN, day_col], dropna=False).agg(**aggs).reset_index()
    daily = daily.rename(columns={day_col: DAY_COLUMN})
    print(f"Loaded {spec['file']}: {len(available)} raw cols -> {len(daily.columns)-3} daily features")
    return daily

def load_targets():
    df = pd.read_csv(DATA_FOLDER / "hormones_and_selfreport.csv", low_memory=False)
    req = [ID_COLUMN, INTERVAL_COLUMN, DAY_COLUMN, TARGET_COLUMN]
    t = df[req].copy()
    t[TARGET_COLUMN] = t[TARGET_COLUMN].apply(normalize_phase)
    t = t.dropna(subset=req)
    return t.drop_duplicates(subset=[ID_COLUMN, INTERVAL_COLUMN, DAY_COLUMN], keep="last")

def build_daily_dataset():
    daily = load_targets()
    for modality, spec in FILE_SPECS.items():
        ft = aggregate_passive_file(modality, spec)
        if ft is None: continue
        daily = daily.merge(ft, on=[ID_COLUMN, INTERVAL_COLUMN, DAY_COLUMN], how="left")
    feat_cols = [c for c in daily.columns if "__" in c]
    coverage = daily[feat_cols].notna().mean()
    feat_cols = coverage[coverage >= MIN_FEATURE_COVERAGE].index.tolist()
    feat_cols = [c for c in feat_cols if daily[c].nunique(dropna=True) > 1]
    return daily, feat_cols

print("="*70); print("BUILDING DAILY DATASET (Mamoune-style rich features)"); print("="*70)
daily, feature_columns = build_daily_dataset()
print(f"\nLabeled participant-days: {len(daily)} | Participants: {daily[ID_COLUMN].nunique()} | Candidate features: {len(feature_columns)}")
print(daily[TARGET_COLUMN].value_counts())

# ---------- CONFLICT #5 RESOLVED: Erin's independent LH-surge label confidence check ----------
def add_label_confidence(daily):
    horm = pd.read_csv(DATA_FOLDER / "hormones_and_selfreport.csv", low_memory=False)
    horm = horm[[ID_COLUMN, DAY_COLUMN, "lh"]]
    d = daily.merge(horm, on=[ID_COLUMN, DAY_COLUMN], how="left")
    d = d.sort_values([ID_COLUMN, INTERVAL_COLUMN, DAY_COLUMN])
    d["lh_baseline"] = d.groupby(ID_COLUMN)["lh"].transform(lambda x: x.rolling(5, min_periods=3, center=True).median())
    d["our_fertility_call"] = (d["lh"] > 1.4 * d["lh_baseline"]).astype(int)
    mira_fertility = (d[TARGET_COLUMN] == "Fertility").astype(int)
    has_data = d["lh_baseline"].notna() & d["lh"].notna()
    agree = d["our_fertility_call"] == mira_fertility
    d["label_confidence"] = np.select([~has_data, agree], ["unknown","high"], default="low")
    return d

daily = add_label_confidence(daily)
conf_report = daily["label_confidence"].value_counts()
print("\n" + "="*70); print("LABEL CONFIDENCE (Erin's independent LH-surge check)"); print("="*70)
print(conf_report)
valid = daily["label_confidence"] != "unknown"
print(f"Agreement rate: {(daily.loc[valid,'label_confidence']=='high').mean():.3f}")

# ---------- CONFLICT #3 RESOLVED: Erin's causal rolling personalization, generalized to all 83 features ----------
WINDOW_DAYS, MIN_PERIODS = 14, 5
def add_causal_personalized_features(df, feature_cols):
    df = df.sort_values([ID_COLUMN, INTERVAL_COLUMN, DAY_COLUMN]).copy()
    for f in feature_cols:
        roll_mean = df.groupby(ID_COLUMN)[f].transform(lambda x: x.shift(1).rolling(WINDOW_DAYS, min_periods=MIN_PERIODS).mean())
        roll_std = df.groupby(ID_COLUMN)[f].transform(lambda x: x.shift(1).rolling(WINDOW_DAYS, min_periods=MIN_PERIODS).std())
        df[f + "_cz"] = (df[f] - roll_mean) / (roll_std + 1e-6)
    return df

daily_p = add_causal_personalized_features(daily, feature_columns)
personalized_cols = [f + "_cz" for f in feature_columns]
print(f"\nRows after 14-day rolling warmup: {daily_p[personalized_cols].notna().any(axis=1).sum()} (of {len(daily_p)})")

# ---------- CONFLICT #4 RESOLVED: Mamoune's StratifiedGroupKFold ----------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
    ])

def run_stratified_group_cv(df, feature_cols, label_confidence_filter=None, n_splits=5):
    sub = df.dropna(subset=[TARGET_COLUMN])
    if label_confidence_filter:
        sub = sub[sub["label_confidence"].isin(label_confidence_filter)]
    sub = sub.dropna(subset=feature_cols, how="all")  # need at least some signal
    X = sub[feature_cols]
    y = sub[TARGET_COLUMN].astype(str)
    groups = sub[ID_COLUMN]
    groups_per_class = sub.groupby(TARGET_COLUMN)[ID_COLUMN].nunique()
    n_folds = min(n_splits, int(groups_per_class.min()))
    if n_folds < 2:
        return None
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in sgkf.split(X, y, groups=groups):
        pipe = make_pipeline()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        accs.append(balanced_accuracy_score(y.iloc[te], pred))
        f1s.append(f1_score(y.iloc[te], pred, average="macro", zero_division=0))
    return {"n_folds": n_folds, "n_rows": len(sub), "n_participants": sub[ID_COLUMN].nunique(),
            "balanced_acc_mean": float(np.mean(accs)), "balanced_acc_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "majority_baseline": float(sub[TARGET_COLUMN].value_counts(normalize=True).max())}

print("\n" + "="*70); print("FINAL COMPARISON — StratifiedGroupKFold, balanced accuracy"); print("="*70)

r_raw = run_stratified_group_cv(daily, feature_columns)
print(f"\n[Mamoune features, RAW, all confidence]      bal_acc={r_raw['balanced_acc_mean']*100:.1f}%\u00b1{r_raw['balanced_acc_std']*100:.2f}%  macroF1={r_raw['macro_f1_mean']:.3f}  n={r_raw['n_rows']} maj={r_raw['majority_baseline']*100:.1f}%")

r_pers = run_stratified_group_cv(daily_p, personalized_cols)
print(f"[Mamoune features, PERSONALIZED (Erin), all conf] bal_acc={r_pers['balanced_acc_mean']*100:.1f}%\u00b1{r_pers['balanced_acc_std']*100:.2f}%  macroF1={r_pers['macro_f1_mean']:.3f}  n={r_pers['n_rows']} maj={r_pers['majority_baseline']*100:.1f}%")

r_pers_highconf = run_stratified_group_cv(daily_p, personalized_cols, label_confidence_filter=["high"])
print(f"[Mamoune features, PERSONALIZED, HIGH-CONF only]  bal_acc={r_pers_highconf['balanced_acc_mean']*100:.1f}%\u00b1{r_pers_highconf['balanced_acc_std']*100:.2f}%  macroF1={r_pers_highconf['macro_f1_mean']:.3f}  n={r_pers_highconf['n_rows']} maj={r_pers_highconf['majority_baseline']*100:.1f}%")

# ---------- Does Mamoune's feature-selection step recover value from the 83 features? ----------
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from functools import partial

def make_pipeline_with_selection(k=20):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=partial(mutual_info_classif, random_state=RANDOM_STATE), k=k)),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
    ])

def run_cv_with_selection(df, feature_cols, k=20, n_splits=5):
    sub = df.dropna(subset=[TARGET_COLUMN]).dropna(subset=feature_cols, how="all")
    X = sub[feature_cols]; y = sub[TARGET_COLUMN].astype(str); groups = sub[ID_COLUMN]
    groups_per_class = sub.groupby(TARGET_COLUMN)[ID_COLUMN].nunique()
    n_folds = min(n_splits, int(groups_per_class.min()))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    for tr, te in sgkf.split(X, y, groups=groups):
        pipe = make_pipeline_with_selection(k)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        accs.append(balanced_accuracy_score(y.iloc[te], pipe.predict(X.iloc[te])))
    return np.mean(accs), np.std(accs), len(sub)

acc, sd, n = run_cv_with_selection(daily_p, personalized_cols, k=20)
print(f"\n[Mamoune features, PERSONALIZED + top-20 mutual-info selection] bal_acc={acc*100:.1f}%\u00b1{sd*100:.2f}%  n={n}")

# ---------- Isolate variables: does StratifiedGroupKFold alone (on the ORIGINAL curated 6+1 feature set) change anything? ----------
core_raw = ["resting_heart_rate__value__mean", "computed_temperature__nightly_temperature__mean",
            "heart_rate_variability__rmssd__mean", "respiratory_rate__full_sleep_breathing_rate__mean"]
core_available = [c for c in core_raw if c in feature_columns]
core_personalized = [c + "_cz" for c in core_available] + ["heart_rate_variability__rmssd__std"]
acc, sd, n = run_cv_with_selection(daily_p, core_personalized, k=min(5, len(core_personalized)))
r_core = run_stratified_group_cv(daily_p, core_personalized)
print(f"\n[ORIGINAL curated core signals, personalized, StratifiedGroupKFold] bal_acc={r_core['balanced_acc_mean']*100:.1f}%\u00b1{r_core['balanced_acc_std']*100:.2f}%  n={r_core['n_rows']} maj={r_core['majority_baseline']*100:.1f}%")

# ---------- Check: is the gap mostly the accuracy vs balanced_accuracy metric choice? ----------
from sklearn.metrics import accuracy_score
def run_both_metrics(df, feature_cols, n_splits=5):
    sub = df.dropna(subset=[TARGET_COLUMN]).dropna(subset=feature_cols, how="all")
    X = sub[feature_cols]; y = sub[TARGET_COLUMN].astype(str); groups = sub[ID_COLUMN]
    groups_per_class = sub.groupby(TARGET_COLUMN)[ID_COLUMN].nunique()
    n_folds = min(n_splits, int(groups_per_class.min()))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    plain_accs, bal_accs = [], []
    for tr, te in sgkf.split(X, y, groups=groups):
        pipe = make_pipeline()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        plain_accs.append(accuracy_score(y.iloc[te], pred))
        bal_accs.append(balanced_accuracy_score(y.iloc[te], pred))
    return np.mean(plain_accs), np.mean(bal_accs)

plain, bal = run_both_metrics(daily_p, core_personalized)
print(f"\n[Same model/data] plain accuracy={plain*100:.1f}%  vs  balanced accuracy={bal*100:.1f}%  <- metric choice alone explains this much of the gap")

# ---------- Remaining Mamoune selection strategies: L1-logistic and RF-based ----------
from sklearn.linear_model import LogisticRegression

def make_pipeline_l1(C=0.5):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(LogisticRegression(solver="liblinear", penalty="l1", C=C,
                                                          class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
                                       threshold=1e-6)),
        ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
    ])

def make_pipeline_rfselect():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                       class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1), threshold="median")),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
    ])

def run_cv_generic(df, feature_cols, pipeline_factory, n_splits=5):
    sub = df.dropna(subset=[TARGET_COLUMN]).dropna(subset=feature_cols, how="all")
    X = sub[feature_cols]; y = sub[TARGET_COLUMN].astype(str); groups = sub[ID_COLUMN]
    groups_per_class = sub.groupby(TARGET_COLUMN)[ID_COLUMN].nunique()
    n_folds = min(n_splits, int(groups_per_class.min()))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    bal_accs, plain_accs, n_selected = [], [], []
    for tr, te in sgkf.split(X, y, groups=groups):
        pipe = pipeline_factory()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        bal_accs.append(balanced_accuracy_score(y.iloc[te], pred))
        plain_accs.append(accuracy_score(y.iloc[te], pred))
        if "selector" in pipe.named_steps:
            try: n_selected.append(int(pipe.named_steps["selector"].get_support().sum()))
            except Exception: pass
    return {"bal_mean": np.mean(bal_accs), "bal_sd": np.std(bal_accs),
            "plain_mean": np.mean(plain_accs), "n_rows": len(sub),
            "avg_n_selected": np.mean(n_selected) if n_selected else len(feature_cols)}

print("\n" + "="*70); print("FULL SELECTION-STRATEGY GRID — personalized 83-feature set"); print("="*70)

strategies = {
    "no_selection": lambda: make_pipeline(),
    "mutual_info_top20": lambda: make_pipeline_with_selection(20),
    "l1_logistic": make_pipeline_l1,
    "rf_selection_median": make_pipeline_rfselect,
}
results_grid = {}
for name, factory in strategies.items():
    r = run_cv_generic(daily_p, personalized_cols, factory)
    results_grid[name] = r
    print(f"{name:22s} balanced_acc={r['bal_mean']*100:.1f}%\u00b1{r['bal_sd']*100:.2f}%  plain_acc={r['plain_mean']*100:.1f}%  avg_features_kept={r['avg_n_selected']:.0f}/{len(personalized_cols)}  n={r['n_rows']}")

maj = daily_p.dropna(subset=personalized_cols, how="all")[TARGET_COLUMN].value_counts(normalize=True).max()
print(f"\nmajority baseline (balanced sense n/a, plain): {maj*100:.1f}%")
