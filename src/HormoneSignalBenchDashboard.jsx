import React, { useState, useMemo, useRef } from "react";
import Papa from "papaparse";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, CartesianGrid, Legend,
} from "recharts";
import {
  FileText, Database, ShieldCheck, Trophy, Lock, ListChecks, TrendingUp, Activity,
  CheckCircle2, XCircle, ChevronRight, Info, AlertTriangle, Link2, UploadCloud, X,
} from "lucide-react";

// ============================================================
// Design tokens — navy/blue healthcare palette
// ============================================================
const T = {
  navy: "#123A5C",
  navyDark: "#0B2740",
  navyLight: "#EAF1F6",
  steel: "#2E6F95",
  steelLight: "#DCE9EF",
  teal: "#2F9E8F",
  tealLight: "#DFF1EE",
  amber: "#A9762A",
  amberLight: "#F3E7D3",
  bg: "#F5F8FA",
  card: "#FFFFFF",
  textDark: "#16232E",
  textMuted: "#5C6B78",
  hairline: "#DCE4EA",
};

// cycle-progression color: darker = earlier in the cycle, lighter = later
const CYCLE_COLOR = { Menstrual: "#0B2740", Follicular: "#1F547A", Fertility: "#3E7FA6", Luteal: "#93BCD3" };

// ============================================================
// deterministic seeded hash / rng
// ============================================================
function hashSeed(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
}

// ============================================================
// illustrative synthetic layer records (for leaderboard weighting demo)
// ============================================================
const LAYER_WEIGHT = { 1: 0.3, 2: 0.6, 3: 0.65, 4: 1.0 };
const LAYER_LABEL = {
  1: "Layer 1 \u00b7 Self-report",
  2: "Layer 2 \u00b7 Wearable (sleep/HR/temp/activity)",
  3: "Layer 3 \u00b7 Continuous glucose monitor",
  4: "Layer 4 \u00b7 Hormone assay (LH/E3G/PdG)",
};
const LAYER_SHARE = { 1: 0.30, 2: 0.35, 3: 0.15, 4: 0.20 };

const MODEL_LAYER_ACC = {
  "Logistic Regression (baseline)": { 1: 0.52, 2: 0.74, 3: 0.68, 4: 0.90 },
  "Gradient Boosting": { 1: 0.60, 2: 0.80, 3: 0.75, 4: 0.93 },
  "Random baseline": { 1: 0.30, 2: 0.29, 3: 0.31, 4: 0.28 },
};

const N_RECORDS = 120;
const baseRng = hashSeed("seed-v0.1-layers-hsb");
const RECORDS = Array.from({ length: N_RECORDS }, (_, i) => {
  const r = baseRng();
  let layer = 1, cum = 0;
  for (const l of [1, 2, 3, 4]) { cum += LAYER_SHARE[l]; if (r <= cum) { layer = l; break; } }
  return { id: i, layer };
});

function correctnessFor(modelName) {
  return RECORDS.map((rec) => {
    const rng = hashSeed(modelName + ":" + rec.id);
    return rng() < MODEL_LAYER_ACC[modelName][rec.layer];
  });
}

function scoreFor(modelName, includedLayers) {
  const correctness = correctnessFor(modelName);
  let numW = 0, denW = 0, numU = 0, denU = 0;
  RECORDS.forEach((rec, i) => {
    if (!includedLayers[rec.layer]) return;
    const w = LAYER_WEIGHT[rec.layer];
    denW += w; denU += 1;
    if (correctness[i]) { numW += w; numU += 1; }
  });
  return { weighted: denW ? numW / denW : 0, unweighted: denU ? numU / denU : 0, n: denU };
}

function scoreForUrl(url) {
  const rng = hashSeed("url:" + url);
  const unweighted = 0.5 + rng() * 0.32;
  const weighted = Math.min(0.97, unweighted + rng() * 0.12);
  return { weighted, unweighted, n: 80 + Math.floor(rng() * 200) };
}

// ---------- real CSV parsing helpers (Method B — actual client-side parsing, no network) ----------
const PHASE_COLOR_FALLBACKS = [T.navy, T.steel, T.teal, T.amber, "#7A5AA6", "#B0524A"];

function detectPhaseColumn(fields, rows) {
  const nameHit = fields.find((f) => /phase|stage|label|cycle_?type/i.test(f));
  const candidates = nameHit ? [nameHit] : fields;
  for (const f of candidates) {
    const vals = rows.map((r) => r[f]).filter((v) => v !== null && v !== undefined && v !== "");
    const uniq = Array.from(new Set(vals.map(String)));
    if (uniq.length >= 2 && uniq.length <= 8 && vals.length > 0) {
      return f;
    }
  }
  return null;
}

function computeValueCounts(rows, field) {
  const counts = {};
  rows.forEach((r) => {
    const v = r[field];
    if (v === null || v === undefined || v === "") return;
    const key = String(v);
    counts[key] = (counts[key] || 0) + 1;
  });
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count], i) => ({ name, count, fill: CYCLE_COLOR[name] || PHASE_COLOR_FALLBACKS[i % PHASE_COLOR_FALLBACKS.length] }));
}

function parseCsvFile(file, onDone) {
  Papa.parse(file, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: (results) => {
      const fields = results.meta.fields || [];
      const rows = results.data || [];
      const phaseField = detectPhaseColumn(fields, rows);
      const phaseCounts = phaseField ? computeValueCounts(rows, phaseField) : null;
      onDone({
        id: Math.random().toString(36).slice(2),
        name: file.name,
        tag: "real",
        rowCount: rows.length,
        columns: fields,
        preview: rows.slice(0, 5),
        phaseField,
        phaseCounts,
      });
    },
    error: () => {
      onDone({ id: Math.random().toString(36).slice(2), name: file.name, tag: "real", error: true });
    },
  });
}

// ---------- REAL confusion matrix — RandomForest trained this session on real mcPHASES data ----------
// Features: resting_hr, nightly_temperature, rmssd, respiratory_rate, active_minutes, sleep_score
// — all PERSONALIZED (per-participant z-score), wearable-only, no hormone leakage
// Split: GroupShuffleSplit by participant id (25% held out) — the methodologically correct way to test generalization to a NEW person
const CM_LABELS = ["Fertility", "Follicular", "Luteal", "Menstrual"];
const CM = [
  [33, 78, 122, 12],
  [38, 117, 96, 16],
  [24, 68, 206, 14],
  [19, 91, 101, 12],
];
const CM_TOTAL = CM.flat().reduce((a, b) => a + b, 0);
const CM_CORRECT = CM.reduce((s, row, i) => s + row[i], 0);
const UNWEIGHTED_ACC = CM_CORRECT / CM_TOTAL;

const REAL_SPLIT_COMPARISON = [
  { name: "Majority-class guess", value: 33.4, note: "Always predict \u201cLuteal\u201d (the most common label)" },
  { name: "Naive random split (leaky)", value: 37.7, note: "Days from the same person leak into both train and test \u2014 don't trust this number" },
  { name: "Rigorous split, raw features", value: 27.0, note: "Participant-held-out, but no per-person normalization" },
  { name: "Rigorous split, personalized features", value: 35.1, note: "Same split, features z-scored within each person \u2014 our best honest number" },
];

const REAL_SIGNAL_COMPARISON = [
  { name: "Sleep score only", value: 27.3 },
  { name: "Activity only", value: 27.8 },
  { name: "Resting HR only", value: 28.9 },
  { name: "Temperature only", value: 29.0 },
  { name: "HRV only", value: 30.3 },
  { name: "Glucose only (Round 1 only, smaller n)", value: 30.2 },
  { name: "Respiratory rate only", value: 31.2 },
  { name: "All combined (6 signals)", value: 35.1 },
  { name: "All combined + glucose (smaller n)", value: 37.1 },
];

const PHASE_COUNTS = [
  { name: "Menstrual", count: 1079, fill: CYCLE_COLOR.Menstrual },
  { name: "Follicular", count: 1386, fill: CYCLE_COLOR.Follicular },
  { name: "Fertility", count: 1281, fill: CYCLE_COLOR.Fertility },
  { name: "Luteal", count: 1912, fill: CYCLE_COLOR.Luteal },
];

// per-phase means, computed by joining hormone/phase labels with real wearable signals on (id, day_in_study)
const REAL_PHASE_SIGNALS = [
  { phase: "Menstrual", n: 767, restingHr: 59.36, temp: 33.81, rmssd: 57.61, estrogen: 93.63 },
  { phase: "Follicular", n: 1037, restingHr: 61.32, temp: 33.60, rmssd: 55.71, estrogen: 99.28 },
  { phase: "Fertility", n: 919, restingHr: 61.15, temp: 33.74, rmssd: 53.68, estrogen: 170.60 },
  { phase: "Luteal", n: 1364, restingHr: 61.05, temp: 33.93, rmssd: 51.91, estrogen: 138.46 },
];

const SCHEMA_FIELDS = [
  ["patient_id", "string", "Anonymized record identifier"],
  ["cycle_day", "integer", "Day count since last menses onset (Day 0)"],
  ["phase_label", "Menstrual|Follicular|Fertility|Luteal", "Current cycle phase (Mira algorithm-derived)"],
  ["source_layer", "1 | 2 | 3 | 4", "Which data layer this record came from"],
  ["provenance.population", "string", "Sample population this threshold was derived from"],
  ["provenance.year", "integer", "Year the reference data was collected"],
  ["provenance.n", "integer", "Sample size behind the threshold"],
  ["phase_definition_version", "string", "e.g. \u201cWHO-ESHRE+PACTS-2025\u201d"],
  ["data_snapshot_version", "string", "e.g. \u201c2026-07-hackathon\u201d"],
  ["features", "object", "sleep, heart_rate, temperature, glucose, hormone_LH, hormone_E3G, hormone_PdG"],
];

const SAMPLE_RECORDS = [
  { label: "Sample A \u2014 complete record", record: {
    patient_id: "p_0031", cycle_day: 14, phase_label: ["Fertility"], source_layer: 4,
    provenance: { population: "mcPHASES cohort", year: 2024, n: 42 },
    phase_definition_version: "Mira-algorithm+PACTS-2025", data_snapshot_version: "2026-07-hackathon",
  }},
  { label: "Sample B \u2014 missing provenance", record: {
    patient_id: "p_0032", cycle_day: 3, phase_label: ["Menstrual"], source_layer: 2,
    phase_definition_version: "Mira-algorithm+PACTS-2025", data_snapshot_version: "2026-07-hackathon",
  }},
];

function validateRecord(record) {
  const errors = [];
  if (!record.patient_id) errors.push("patient_id missing");
  if (!record.cycle_day && record.cycle_day !== 0) errors.push("cycle_day missing");
  if (!record.phase_label || !record.phase_label.length) errors.push("phase_label missing");
  if (!record.source_layer) errors.push("source_layer missing");
  if (!record.provenance) { errors.push("provenance object missing"); }
  else {
    if (!record.provenance.population) errors.push("provenance.population missing");
    if (!record.provenance.year) errors.push("provenance.year missing");
    if (!record.provenance.n) errors.push("provenance.n missing");
  }
  if (!record.phase_definition_version) errors.push("phase_definition_version missing");
  if (!record.data_snapshot_version) errors.push("data_snapshot_version missing");
  return { valid: errors.length === 0, errors };
}

const PHASE_GUIDE = [
  { p: "Menstrual", hormone: "Estrogen & progesterone both low", note: "Real data: lowest average resting heart rate of all four phases (59.4 bpm vs ~61 elsewhere)." },
  { p: "Follicular", hormone: "Estrogen rising", note: "Real data: lowest average nightly temperature (33.60\u00b0C) \u2014 matches the classic pre-ovulatory temperature dip." },
  { p: "Fertility", hormone: "LH surge, estrogen peak", note: "Real data: highest average estrogen (170.6) of any phase \u2014 matches the expected pre-ovulatory estrogen surge." },
  { p: "Luteal", hormone: "Progesterone rises then falls", note: "Real data: highest average nightly temperature (33.93\u00b0C) \u2014 matches progesterone's known thermogenic effect." },
];

// ---------- cross-dataset validation: mcPHASES (phase-label derived) vs Marquette NFP (independent, n=159) ----------
const CYCLE_TIMING_COMPARISON = [
  { metric: "Total cycle length", mcphases: "29.8 days (SD 4.5)", marquette: "29.3 days (SD 3.9)", n: "128 vs 1,665 cycles", verdict: "match", note: "Both count the same thing the same way \u2014 days from one period's start to the next. Safe to pool." },
  { metric: "Luteal phase length", mcphases: "10.0 days (SD 3.0)", marquette: "13.3 days (SD 2.7)", n: "126 vs 1,665 cycles", verdict: "mismatch", note: "A >3-day gap \u2014 too large to be measurement noise. See note below." },
];

const ONBOARDING_CHECKLIST = [
  { step: "1. Map columns to the shared schema", detail: "Write a crosswalk table: raw column name \u2192 schema field. (e.g., Marquette's LengthofCycle \u2192 our cycle_length_days.)" },
  { step: "2. Declare the measurement method, not just the value", detail: "Every phase- or timing-derived field must state HOW it was determined \u2014 hormone assay, wearable inference, or self-observed fertility signs \u2014 in phase_definition_version." },
  { step: "3. Check field-by-field: same method, or different?", detail: "Two fields with the same name are only mergeable if their underlying measurement method matches. If not, keep them side by side, both visible, neither pooled." },
  { step: "4. Let the schema validator enforce it, not memory", detail: "Any pipeline step that averages or pools a phase-derived field must first check that all contributing rows share one phase_definition_version. Reject silently-mixed pools." },
  { step: "5. Publish the comparison either way", detail: "A field that fails to merge is still a finding \u2014 document the mismatch and the likely reason, the way we did for luteal length below. Don't just drop it quietly." },
];

// ---------- verified real statistics, computed directly from the 4 uploaded mcPHASES CSVs ----------
const REAL_MCPHASES_TABLES = [
  { table: "hormones_and_selfreport.csv", rows: 5659, ids2022: 42, ids2024: "\u2014", note: "Real phase label + LH/estrogen/PdG (PdG only 33% populated)" },
  { table: "subject-info.csv", rows: 42, ids2022: 42, ids2024: "\u2014", note: "Demographics: ethnicity, menarche age, birth year" },
  { table: "resting_heart_rate.csv", rows: 13737, ids2022: 42, ids2024: "\u2014", note: "Daily resting HR (mean 63.8 bpm)" },
  { table: "computed_temperature.csv", rows: 5575, ids2022: 42, ids2024: "\u2014", note: "Nightly wrist skin temperature (mean 33.7\u00b0C)" },
  { table: "respiratory_rate_summary.csv", rows: 6301, ids2022: 40, ids2024: "\u2014", note: "Sleep breathing rate" },
  { table: "heart_rate_variability_details.csv", rows: 436262, ids2022: 40, ids2024: "\u2014", note: "5-min interval RMSSD (HRV)" },
  { table: "sleep.csv", rows: 14765, ids2022: 42, ids2024: 20, note: "Sleep stage/session records" },
  { table: "exercise.csv", rows: 7282, ids2022: 24, ids2024: 20, note: "Logged workouts, mostly walking" },
  { table: "stress_score.csv", rows: 7932, ids2022: 31, ids2024: 20, note: "6,945 valid (987 rows marked NO_DATA)" },
  { table: "sleep_score.csv", rows: 5308, ids2022: 42, ids2024: 20, note: "Nightly composite sleep score" },
];

const REAL_SLEEP_SCORE_BUCKETS = [
  { name: "30\u201339", count: 7 }, { name: "40\u201349", count: 20 }, { name: "50\u201359", count: 216 },
  { name: "60\u201369", count: 780 }, { name: "70\u201379", count: 1959 }, { name: "80\u201389", count: 2214 }, { name: "90\u201399", count: 112 },
].map((d) => ({ ...d, fill: T.steel }));

const REAL_STRESS_SCORE_BUCKETS = [
  { name: "0\u20139", count: 24 }, { name: "50\u201359", count: 42 }, { name: "60\u201369", count: 964 },
  { name: "70\u201379", count: 3546 }, { name: "80\u201389", count: 2216 }, { name: "90\u201399", count: 153 },
].map((d) => ({ ...d, fill: T.amber }));

const REAL_EXERCISE_TYPES = [
  { name: "Walk", count: 6200 }, { name: "Aerobic", count: 404 }, { name: "Sport", count: 257 },
  { name: "Outdoor Bike", count: 140 }, { name: "Run", count: 121 }, { name: "Weights", count: 49 },
].map((d) => ({ ...d, fill: T.navy }));

const LIFE_STAGES = [
  { stage: "Adolescence", years: "~10\u201319", focus: "Cycles are often irregular for 2\u20133 years after menarche while the hypothalamic-pituitary-ovarian axis matures. Real data point: our 42 mcPHASES participants report a mean age at first menarche of 11.9 (SD 1.16, range 10\u201315) \u2014 consistent with published population norms.", value: "Continuous tracking distinguishes normal post-menarche irregularity from early signs of PCOS or thyroid dysfunction \u2014 years earlier than a single clinic visit would." },
  { stage: "Reproductive years", years: "~20s\u201330s", focus: "Regular follicular \u2192 fertility \u2192 luteal cycling; the phase most wearable studies (including ours) are built around. Our real cohort (mean birth year 2001) sits squarely in this stage.", value: "Enables personalized baselines, fertility-window estimation, and detection of anovulatory cycles tied to stress, weight change, or PCOS." },
  { stage: "Perimenopause", years: "~40s\u201350s", focus: "Cycle length and hormone patterns become erratic; single blood tests are unreliable because hormone levels swing widely day to day.", value: "Continuous signals can catch the transition starting years before STRAW+10's survey-based staging would, opening a window for preventive bone and cardiovascular care." },
  { stage: "Postmenopause", years: "50s+", focus: "Cycling stops entirely; hormone levels stabilize at a new, lower baseline.", value: "The same signal pipeline (sleep, HR, temperature) can pivot to monitoring cardiometabolic and bone-health risk markers instead of cycle phase." },
];

const BEHAVIORAL_SIGNALS = [
  {
    signal: "Symptom severity (DRSP instrument)",
    method: "Adopt the Daily Record of Severity of Problems (DRSP) \u2014 a validated 21-item clinical scale, not an ad hoc 1\u20135 rating",
    citation: "Endicott, Nee & Harrison (2006), Archives of Women's Mental Health \u2014 high test-retest reliability and internal consistency across two validation studies",
    ties_to: "Cross-checks the phase label itself: a Luteal day with no symptom-score rise is a second, independent low-confidence flag alongside the Mira-vs-derived-algorithm disagreement",
  },
  {
    signal: "Contraceptive method + adherence",
    method: "Standardized categorical field: method type, start date, adherence log \u2014 a separate schema branch, not a missing-data case",
    citation: "Scoping review (PMC12889786): combined-pill users show a flattened, permanently elevated temperature/HR curve instead of the biphasic pattern; a 450+-person wearable cohort found +2 bpm resting HR through the follicular phase versus naturally cycling women",
    ties_to: "Without this field, a contraceptive user's personalization baseline gets computed from an already-distorted signal \u2014 the same category of error that made luteal-length unmergeable with Marquette",
  },
  {
    signal: "Training load / perceived exertion (RPE)",
    method: "Simple daily self-rated exertion score logged alongside exercise data already collected",
    citation: "Documented accuracy gap: wearables have been shown to misclassify luteal-phase-driven fatigue as overtraining, because the algorithm detects a symptom it wasn't built to interpret",
    ties_to: "HRV and respiratory rate are our two best single signals \u2014 both are exertion-sensitive, so an unlogged hard workout is a plausible hidden confound behind today's 30.3% / 31.2% numbers",
  },
];

const FRICTION_PROTOCOLS = [
  { method: "Use data the wearable is already collecting", friction: "Near zero", note: "mcPHASES-style: sleep, HR, temperature, activity that Fitbit/Oura already log \u2014 no new behavior required." },
  { method: "Attach logging to an existing habit", friction: "Very low", note: "10-second voice note while brushing teeth, e.g. \u201chabit stacking\u201d \u2014 rides on a routine that already happens daily." },
  { method: "Passive lock-screen widget", friction: "Low", note: "One tap from the lock screen; no need to open an app." },
  { method: "CGM patch (for those already wearing one)", friction: "Low, opt-in only", note: "Same principle as mcPHASES's continuous glucose monitor arm \u2014 piggybacks on a device some people wear anyway." },
  { method: "Active daily survey", friction: "High", note: "Last resort only; highest drop-off risk in longitudinal studies." },
];

const TABS = [
  { id: "schema", label: "Schema", icon: FileText },
  { id: "dataset", label: "Dataset", icon: Database },
  { id: "lifestage", label: "Life Stage", icon: TrendingUp },
  { id: "behavior", label: "Behavioral Signals", icon: Activity },
  { id: "plan", label: "Data Plan", icon: ListChecks },
  { id: "baseline", label: "Baseline & Eval", icon: ShieldCheck },
  { id: "leaderboard", label: "Leaderboard", icon: Trophy },
  { id: "privacy", label: "Privacy", icon: Lock },
];

// ============================================================
// small reusable pieces
// ============================================================
function Card({ children, style }) {
  return <div style={{ background: T.card, border: `1px solid ${T.hairline}`, borderRadius: 10, padding: 22, ...style }}>{children}</div>;
}
function Tag({ children, color = T.steel, bg }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 9px", borderRadius: 5,
      fontSize: 11, fontWeight: 600, color, background: bg || color + "16",
      border: `1px solid ${color}40`, fontFamily: "'IBM Plex Mono', monospace",
    }}>{children}</span>
  );
}
function Illustrative() {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4, marginBottom: 10,
      fontSize: 10.5, fontWeight: 600, color: T.amber, background: T.amberLight,
      padding: "3px 8px", borderRadius: 5, border: `1px solid ${T.amber}40`,
    }}><AlertTriangle size={11} /> Illustrative, not yet real data</span>
  );
}
function VerifiedReal({ children }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4, marginBottom: 10,
      fontSize: 10.5, fontWeight: 600, color: T.teal, background: T.tealLight,
      padding: "3px 8px", borderRadius: 5, border: `1px solid ${T.teal}40`,
    }}><CheckCircle2 size={11} /> {children || "Verified real (computed this session)"}</span>
  );
}
function Insight({ children }) {
  return (
    <div style={{ display: "flex", gap: 10, marginTop: 14, padding: "12px 14px", borderRadius: 8, background: T.navyLight, border: `1px solid ${T.steel}30` }}>
      <Info size={16} color={T.steel} style={{ flexShrink: 0, marginTop: 1 }} />
      <div style={{ fontSize: 12.5, color: T.textDark, lineHeight: 1.5 }}>{children}</div>
    </div>
  );
}
function SectionTitle({ children, sub }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontWeight: 700, fontSize: 15, color: T.textDark }}>{children}</div>
      {sub && <div style={{ fontSize: 12, color: T.textMuted, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function UploadZone({ onFiles }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault(); setDragOver(false);
        const files = Array.from(e.dataTransfer.files).filter((f) => f.name.endsWith(".csv"));
        if (files.length) onFiles(files);
      }}
      onClick={() => inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragOver ? T.teal : T.hairline}`, borderRadius: 10, padding: "26px 16px",
        textAlign: "center", cursor: "pointer", background: dragOver ? T.tealLight : T.navyLight, transition: "all .15s",
      }}
    >
      <input ref={inputRef} type="file" accept=".csv" multiple style={{ display: "none" }}
        onChange={(e) => { const files = Array.from(e.target.files || []); if (files.length) onFiles(files); e.target.value = ""; }} />
      <UploadCloud size={26} color={dragOver ? T.teal : T.steel} style={{ marginBottom: 8 }} />
      <div style={{ fontSize: 13, fontWeight: 600, color: T.textDark }}>Drag a CSV here, or click to browse</div>
      <div style={{ fontSize: 11.5, color: T.textMuted, marginTop: 3 }}>Parsed entirely in your browser \u2014 no upload to a server, no network request</div>
    </div>
  );
}

// ============================================================
// main component
// ============================================================
export default function HormoneSignalBenchDashboard() {
  const [activeTab, setActiveTab] = useState("schema");
  const [validated, setValidated] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  function handleFiles(files) {
    files.forEach((file) => {
      parseCsvFile(file, (parsed) => setUploadedFiles((prev) => [...prev, parsed]));
    });
  }
  function removeFile(id) {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== id));
  }
  function setTag(id, tag) {
    setUploadedFiles((prev) => prev.map((f) => (f.id === id ? { ...f, tag } : f)));
  }

  const hasLiveData = uploadedFiles.some((f) => !f.error);
  const liveRealCount = uploadedFiles.filter((f) => !f.error && f.tag === "real").reduce((s, f) => s + f.rowCount, 0);
  const liveSyntheticCount = uploadedFiles.filter((f) => !f.error && f.tag === "synthetic").reduce((s, f) => s + f.rowCount, 0);
  const liveRealSynth = [
    { name: `Real (uploaded, n=${liveRealCount})`, value: liveRealCount || 0.0001, fill: T.teal },
    { name: `Synthetic (uploaded, n=${liveSyntheticCount})`, value: liveSyntheticCount || 0.0001, fill: T.amber },
  ];
  const filesWithPhase = uploadedFiles.filter((f) => f.phaseCounts && f.phaseCounts.length);
  const livePhaseCounts = filesWithPhase.length
    ? Object.values(
        filesWithPhase.flatMap((f) => f.phaseCounts).reduce((acc, row) => {
          acc[row.name] = acc[row.name] || { name: row.name, count: 0, fill: row.fill };
          acc[row.name].count += row.count;
          return acc;
        }, {})
      )
    : PHASE_COUNTS;

  const REAL_SIGNAL_OPTIONS = [
    { key: "sleep", label: "Sleep score only (personalized)", acc: 27.3, n: 1082 },
    { key: "activity", label: "Activity only (personalized)", acc: 27.8, n: 1069 },
    { key: "resting_hr", label: "Resting heart rate only (personalized)", acc: 28.9, n: 1107 },
    { key: "temperature", label: "Wrist temperature only (personalized)", acc: 29.0, n: 1107 },
    { key: "hrv", label: "HRV only (personalized)", acc: 30.3, n: 1107 },
    { key: "glucose", label: "Glucose only (personalized, Round 1 only)", acc: 30.2, n: 500 },
    { key: "respiratory", label: "Respiratory rate only (personalized)", acc: 31.2, n: 1094 },
    { key: "all6", label: "All 6 signals combined (personalized)", acc: 35.1, n: 1047 },
    { key: "all7", label: "All 6 + glucose (personalized, smaller n)", acc: 37.1, n: 490 },
  ];
  const [selectedSignal, setSelectedSignal] = useState("all6");
  const [submissions, setSubmissions] = useState([
    { name: "Majority-class guess (always \u201cLuteal\u201d)", acc: 33.4, n: 4035, pinned: true },
    { name: "Naive random split (leaky \u2014 don't trust)", acc: 37.7, n: 1009, pinned: true, flagged: true },
    { name: "RandomForest \u2014 raw features, participant-held-out", acc: 27.0, n: 1138, pinned: true },
    { name: "RandomForest \u2014 personalized features, participant-held-out", acc: 35.1, n: 1047, pinned: true },
  ]);

  const [datasetUrl, setDatasetUrl] = useState("");
  const [pipelineStep, setPipelineStep] = useState(0);
  const [urlResult, setUrlResult] = useState(null);
  const timers = useRef([]);

  const [epsilon, setEpsilon] = useState(1.0);
  const trueMean = 33.74; // real mean nightly wrist temperature, computed from computed_temperature.csv
  const noise = useMemo(() => {
    const u = Math.random() - 0.5;
    const scale = 1 / Math.max(epsilon, 0.05);
    return -Math.sign(u) * scale * Math.log(1 - 2 * Math.abs(u)) * 0.1;
  }, [epsilon]);
  const noisedMean = Math.max(30, Math.min(37, trueMean + noise));
  const utilityCurve = useMemo(() => {
    const pts = [];
    for (let e = 0.1; e <= 5; e += 0.1) pts.push({ epsilon: Number(e.toFixed(1)), loss: Number((1 / e).toFixed(2)) });
    return pts;
  }, []);

  function submitRun() {
    const opt = REAL_SIGNAL_OPTIONS.find((o) => o.key === selectedSignal);
    setSubmissions((prev) => [...prev, { name: `RandomForest \u2014 ${opt.label}`, acc: opt.acc, n: opt.n, pinned: false }].sort((a, b) => b.acc - a.acc));
  }

  function runUrlPipeline() {
    if (!datasetUrl.trim()) return;
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setUrlResult(null);
    setPipelineStep(0);
    [1, 2, 3, 4, 5].forEach((step, i) => {
      const t = setTimeout(() => setPipelineStep(step), (i + 1) * 480);
      timers.current.push(t);
    });
    const t = setTimeout(() => {
      const s = scoreForUrl(datasetUrl.trim());
      setUrlResult(s);
      let host = datasetUrl.trim();
      try { host = new URL(datasetUrl.trim()).hostname; } catch (e) {}
      setSubmissions((prev) => [...prev, { name: `External (simulated): ${host}`, acc: Number((s.weighted * 100).toFixed(1)), n: s.n, pinned: false, external: true }].sort((a, b) => b.acc - a.acc));
    }, 5 * 480 + 200);
    timers.current.push(t);
  }

  const pipelineSteps = [
    "Fetch data from the link",
    "Check it against the schema (Tab 1)",
    "Align cycles to Day 0 (PACTS-style phase alignment)",
    "Score it with the baseline formula (Tab 5)",
    "Publish the result to the leaderboard",
  ];

  return (
    <div style={{ fontFamily: "'IBM Plex Sans', sans-serif", color: T.textDark, background: T.bg }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
        .pbd-tab { transition: background .15s, color .15s; cursor: pointer; }
        .pbd-tab:hover { background: rgba(255,255,255,0.08); }
        .pbd-row { transition: background .15s; }
        .pbd-row:hover { background: ${T.navyLight}; }
        input[type=range].pbd-slider { accent-color: ${T.steel}; }
        button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid ${T.steel}; outline-offset: 2px; }
      `}</style>

      {/* header */}
      <div style={{ background: T.navy, padding: "20px 26px 16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ fontSize: 11.5, letterSpacing: 1, color: "#9FC3DA", fontWeight: 600, marginBottom: 3 }}>WOMEN'S HORMONAL HEALTH BENCHMARK</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#fff" }}>HormoneSignal Bench v0.1</div>
            <div style={{ fontSize: 11.5, color: "#9FC3DA", marginTop: 2 }}>Which passive signals best reveal hormonal cycle phase?</div>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Tag color="#BFE0FF" bg="rgba(255,255,255,0.08)">phase_def: WHO-ESHRE+PACTS-2025</Tag>
            <Tag color="#BFE0FF" bg="rgba(255,255,255,0.08)">data_snapshot: 2026-07-hackathon</Tag>
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, marginTop: 16, flexWrap: "wrap" }}>
          {TABS.map((t) => {
            const Icon = t.icon;
            const active = activeTab === t.id;
            return (
              <div key={t.id} className="pbd-tab" onClick={() => setActiveTab(t.id)}
                style={{ display: "flex", alignItems: "center", gap: 6, padding: "8px 14px", borderRadius: 6, background: active ? "#fff" : "transparent", color: active ? T.navy : "#CBDCE8", fontWeight: 600, fontSize: 13 }}>
                <Icon size={15} /> {t.label}
              </div>
            );
          })}
        </div>
      </div>

      {/* honesty banner */}
      <div style={{ padding: "16px 26px 0" }}>
        <div style={{ display: "flex", gap: 12, padding: "13px 16px", borderRadius: 8, background: T.tealLight, border: `1px solid ${T.teal}40` }}>
          <CheckCircle2 size={18} color={T.teal} style={{ flexShrink: 0, marginTop: 1 }} />
          <div style={{ fontSize: 12.5, color: T.textDark, lineHeight: 1.55 }}>
            {hasLiveData ? (
              <><b>Live data loaded:</b> the Dataset tab's phase distribution and real/synthetic split below are computed in your browser from {uploadedFiles.filter((f) => !f.error).length} freshly-uploaded file(s), on top of the verified real mcPHASES files already baked in.</>
            ) : (
              <><b>Data status, plainly stated:</b> the phase distribution, confusion matrix, and single-signal comparisons are now <b>real, computed from uploaded mcPHASES files</b> (42 real participants, all 7 of Option 1's signal categories now covered). What's still illustrative: the Privacy tab's numbers, which remain conceptual until wired to real aggregate release logic. Real sources in use: <Tag color={T.teal}>mcPHASES (real, PhysioNet)</Tag> <Tag color={T.steel}>Marquette NFP (real, cycle-length metric only)</Tag> \u2014 a synthetic Kaggle dataset was evaluated and excluded entirely. See the Dataset tab's Full Provenance Ledger for exactly which number came from which source.</>
            )}
          </div>
        </div>
      </div>

      <div style={{ padding: "20px 26px 32px" }}>
        {/* ================= SCHEMA ================= */}
        {activeTab === "schema" && (
          <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 18 }}>
            <Card>
              <SectionTitle sub="These fields are real \u2014 every record, real or synthetic, must satisfy this schema.">schema_v0.1.json \u2014 fields</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr 1.6fr", padding: "8px 0", borderBottom: `2px solid ${T.hairline}`, fontSize: 11.5, fontWeight: 700, color: T.textMuted }}>
                <div>FIELD</div><div>TYPE</div><div>DESCRIPTION</div>
              </div>
              {SCHEMA_FIELDS.map(([f, ty, d], i) => (
                <div key={i} className="pbd-row" style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr 1.6fr", padding: "9px 4px", borderBottom: `1px solid ${T.hairline}`, fontSize: 12.5 }}>
                  <div style={{ fontFamily: "'IBM Plex Mono', monospace", color: T.navy, fontWeight: 600 }}>{f}</div>
                  <div style={{ color: T.textMuted, fontFamily: "'IBM Plex Mono', monospace" }}>{ty}</div>
                  <div>{d}</div>
                </div>
              ))}
              <Insight>
                <code>cycle_day</code> is what makes different people's cycles comparable at all \u2014 a 24-day cycle and a 32-day cycle can't be lined up by calendar date, only by "days since last period started." This is the same idea behind the PACTS alignment method we're adopting rather than inventing our own.
              </Insight>
            </Card>

            <Card>
              <SectionTitle sub="Click to see the schema reject an incomplete record in real time.">Validate a sample record</SectionTitle>
              {SAMPLE_RECORDS.map((s, i) => {
                const result = validated === i ? validateRecord(s.record) : null;
                return (
                  <div key={i} style={{ marginBottom: 16, padding: 14, borderRadius: 8, background: T.navyLight, border: `1px solid ${T.hairline}` }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>{s.label}</div>
                    <pre style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10.5, background: T.navyDark, color: "#CFE3EF", padding: 10, borderRadius: 6, overflowX: "auto", margin: 0 }}>
{JSON.stringify(s.record, null, 2)}
                    </pre>
                    <button onClick={() => setValidated(i)} style={{ marginTop: 10, padding: "7px 14px", borderRadius: 6, border: "none", background: T.navy, color: "#fff", fontWeight: 600, fontSize: 12.5, cursor: "pointer" }}>Run validation</button>
                    {result && (
                      <div style={{ marginTop: 10, display: "flex", alignItems: "flex-start", gap: 8 }}>
                        {result.valid ? <CheckCircle2 size={18} color={T.teal} /> : <XCircle size={18} color="#B14A3A" />}
                        <div style={{ fontSize: 12.5 }}>
                          {result.valid ? <span style={{ color: T.teal, fontWeight: 600 }}>Schema valid \u2014 all required fields present</span> : (
                            <div><span style={{ color: "#B14A3A", fontWeight: 600 }}>Schema invalid:</span>
                              <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>{result.errors.map((e, j) => <li key={j}>{e}</li>)}</ul>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </Card>
          </div>
        )}

        {/* ================= DATASET ================= */}
        {activeTab === "dataset" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 18 }}>
            <Card>
              <SectionTitle sub="Computed directly from the 4 Fitbit-derived CSVs you uploaded \u2014 these numbers are genuinely real, verified against the published paper">Verified real signals (mcPHASES)</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 4, padding: "8px 0", borderBottom: `2px solid ${T.hairline}`, fontSize: 11, fontWeight: 700, color: T.textMuted }}>
                <div>TABLE</div><div>ROWS</div><div>PARTICIPANTS (2022 / 2024)</div><div>NOTE</div>
              </div>
              {REAL_MCPHASES_TABLES.map((t, i) => (
                <div key={i} className="pbd-row" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", padding: "9px 0", borderBottom: `1px solid ${T.hairline}`, fontSize: 12 }}>
                  <div style={{ fontFamily: "'IBM Plex Mono', monospace", color: T.navy, fontWeight: 600 }}>{t.table}</div>
                  <div>{t.rows.toLocaleString()}</div>
                  <div>{t.ids2022} / {t.ids2024}</div>
                  <div style={{ color: T.textMuted }}>{t.note}</div>
                </div>
              ))}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 14 }}>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Sleep score distribution (n=5,308 nights)</div>
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={REAL_SLEEP_SCORE_BUCKETS}>
                      <XAxis dataKey="name" tick={{ fontSize: 9.5, fill: T.textMuted }} axisLine={{ stroke: T.hairline }} tickLine={false} interval={0} />
                      <YAxis tick={{ fontSize: 9.5, fill: T.textMuted }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ borderRadius: 6, border: `1px solid ${T.hairline}`, fontSize: 11.5 }} />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>{REAL_SLEEP_SCORE_BUCKETS.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Stress score distribution (n=6,945 valid days)</div>
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={REAL_STRESS_SCORE_BUCKETS}>
                      <XAxis dataKey="name" tick={{ fontSize: 9.5, fill: T.textMuted }} axisLine={{ stroke: T.hairline }} tickLine={false} interval={0} />
                      <YAxis tick={{ fontSize: 9.5, fill: T.textMuted }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ borderRadius: 6, border: `1px solid ${T.hairline}`, fontSize: 11.5 }} />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>{REAL_STRESS_SCORE_BUCKETS.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <Insight>
                42 participants in 2022, 20 returning in 2024 across every table \u2014 matches the published paper's recruitment numbers exactly. And now, with <code>hormones_and_selfreport.csv</code> joined in, we have the real phase label too.
              </Insight>

              <div style={{ marginTop: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Real signal averages by phase (joined on id + day_in_study, n=4,087 matched days)</div>
                <div style={{ display: "grid", gridTemplateColumns: "0.9fr repeat(4, 1fr)", fontSize: 11.5 }}>
                  <div style={{ fontWeight: 700, color: T.textMuted, paddingBottom: 6, borderBottom: `2px solid ${T.hairline}` }}>PHASE</div>
                  <div style={{ fontWeight: 700, color: T.textMuted, paddingBottom: 6, borderBottom: `2px solid ${T.hairline}` }}>N</div>
                  <div style={{ fontWeight: 700, color: T.textMuted, paddingBottom: 6, borderBottom: `2px solid ${T.hairline}` }}>RESTING HR</div>
                  <div style={{ fontWeight: 700, color: T.textMuted, paddingBottom: 6, borderBottom: `2px solid ${T.hairline}` }}>TEMP (\u00b0C)</div>
                  <div style={{ fontWeight: 700, color: T.textMuted, paddingBottom: 6, borderBottom: `2px solid ${T.hairline}` }}>ESTROGEN</div>
                  {REAL_PHASE_SIGNALS.map((r) => (
                    <React.Fragment key={r.phase}>
                      <div style={{ padding: "7px 0", borderBottom: `1px solid ${T.hairline}` }}>
                        <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: 5, background: CYCLE_COLOR[r.phase], color: "#fff", fontWeight: 700, fontSize: 11 }}>{r.phase}</span>
                      </div>
                      <div style={{ padding: "7px 0", borderBottom: `1px solid ${T.hairline}` }}>{r.n}</div>
                      <div style={{ padding: "7px 0", borderBottom: `1px solid ${T.hairline}` }}>{r.restingHr.toFixed(1)}</div>
                      <div style={{ padding: "7px 0", borderBottom: `1px solid ${T.hairline}` }}>{r.temp.toFixed(2)}</div>
                      <div style={{ padding: "7px 0", borderBottom: `1px solid ${T.hairline}` }}>{r.estrogen.toFixed(1)}</div>
                    </React.Fragment>
                  ))}
                </div>
                <Insight>
                  These averages independently reproduce known menstrual physiology from real wearable + hormone data: temperature is lowest in Follicular and highest in Luteal (progesterone's thermogenic effect), and estrogen peaks in Fertility (the pre-ovulatory surge). That's a genuine validation that these signals carry real information \u2014 <b>at the population level.</b> Whether that's enough to predict one new person's phase on one new day is a separate, harder question, answered in the Baseline & Eval tab.
                </Insight>
              </div>
            </Card>

            <Card>
              <SectionTitle sub="Real client-side CSV parsing — drop mcPHASES, Kaggle, or any other cycle-data CSV here">Upload data</SectionTitle>
              <UploadZone onFiles={handleFiles} />
              {uploadedFiles.length > 0 && (
                <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
                  {uploadedFiles.map((f) => (
                    <div key={f.id} style={{ padding: 12, borderRadius: 8, background: T.navyLight, border: `1px solid ${T.hairline}` }}>
                      {f.error ? (
                        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5 }}>
                          <XCircle size={16} color="#B14A3A" /> Couldn't parse <b>{f.name}</b> as CSV.
                          <button onClick={() => removeFile(f.id)} style={{ marginLeft: "auto", border: "none", background: "none", cursor: "pointer" }}><X size={14} /></button>
                        </div>
                      ) : (
                        <>
                          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                            <FileText size={15} color={T.navy} />
                            <b style={{ fontSize: 12.5 }}>{f.name}</b>
                            <span style={{ fontSize: 11.5, color: T.textMuted }}>{f.rowCount} rows \u00b7 {f.columns.length} columns{f.phaseField ? ` \u00b7 detected phase-like column: "${f.phaseField}"` : " \u00b7 no phase-like column auto-detected"}</span>
                            <div style={{ marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" }}>
                              <span style={{ fontSize: 11, color: T.textMuted }}>Tag as:</span>
                              {["real", "synthetic"].map((tg) => (
                                <button key={tg} onClick={() => setTag(f.id, tg)} style={{
                                  fontSize: 11, padding: "4px 10px", borderRadius: 5, cursor: "pointer",
                                  border: `1px solid ${f.tag === tg ? (tg === "real" ? T.teal : T.amber) : T.hairline}`,
                                  background: f.tag === tg ? (tg === "real" ? T.tealLight : T.amberLight) : "#fff",
                                  color: f.tag === tg ? (tg === "real" ? T.teal : T.amber) : T.textMuted, fontWeight: 600,
                                }}>{tg}</button>
                              ))}
                              <button onClick={() => removeFile(f.id)} style={{ border: "none", background: "none", cursor: "pointer" }}><X size={14} color={T.textMuted} /></button>
                            </div>
                          </div>
                          {f.preview && f.preview.length > 0 && (
                            <div style={{ marginTop: 8, overflowX: "auto" }}>
                              <table style={{ borderCollapse: "collapse", fontSize: 10.5, width: "100%" }}>
                                <thead><tr>{f.columns.slice(0, 6).map((c) => <th key={c} style={{ textAlign: "left", padding: "4px 8px", color: T.textMuted, borderBottom: `1px solid ${T.hairline}` }}>{c}</th>)}</tr></thead>
                                <tbody>{f.preview.map((row, ri) => (
                                  <tr key={ri}>{f.columns.slice(0, 6).map((c) => <td key={c} style={{ padding: "4px 8px", borderBottom: `1px solid ${T.hairline}` }}>{String(row[c] ?? "")}</td>)}</tr>
                                ))}</tbody>
                              </table>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
              <Insight>
                This is real parsing, not a simulation \u2014 <code>FileReader</code> + PapaParse run entirely in your browser, so it works even though this sandbox can't reach outside network links (the limitation you hit earlier with the Leaderboard URL box doesn't apply to local files). Tag each file real or synthetic yourself; we don't try to guess.
              </Insight>
            </Card>

            <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 18 }}>
              <Card style={{ position: "relative" }}>
                {hasLiveData ? <VerifiedReal>Live from your uploaded file(s)</VerifiedReal> : <VerifiedReal>Real (hormones_and_selfreport.csv, n=5,659)</VerifiedReal>}
                <SectionTitle sub={hasLiveData ? "Computed live from your uploaded file(s)" : "Real phase labels, derived by Mira's algorithm from daily hormone assays"}>Cycle-phase distribution</SectionTitle>
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={livePhaseCounts} margin={{ left: 8, right: 8, top: 8, bottom: 4 }}>
                    <CartesianGrid stroke={T.hairline} vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 11.5, fill: T.textMuted }} axisLine={{ stroke: T.hairline }} tickLine={false} interval={0} />
                    <YAxis tick={{ fontSize: 12, fill: T.textMuted }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ borderRadius: 6, border: `1px solid ${T.hairline}`, fontSize: 12.5 }} />
                    <Bar dataKey="count" radius={[5, 5, 0, 0]}>{livePhaseCounts.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
                <Insight>
                  {filesWithPhase.length
                    ? `Built from the column${filesWithPhase.length > 1 ? "s" : ""} auto-detected across your uploaded file(s). If this looks wrong, the auto-detector may have picked the wrong column — rename it to include "phase" or "label" and re-upload.`
                    : "1,912 Luteal / 1,386 Follicular / 1,281 Fertility / 1,079 Menstrual, out of 5,659 total logged days across 42 real participants. Luteal is the largest bucket \u2014 consistent with it being the longest phase of a typical cycle."}
                </Insight>
              </Card>

              <Card style={{ position: "relative" }}>
                <VerifiedReal>{hasLiveData ? "Live from the tags you assigned" : "See ledger below"}</VerifiedReal>
                <SectionTitle sub={hasLiveData ? "Computed live from the tags you assigned above" : "This card only shows live data once you upload files above"}>Your uploaded files: real vs. synthetic</SectionTitle>
                {hasLiveData ? (
                  <>
                    <ResponsiveContainer width="100%" height={230}>
                      <PieChart>
                        <Pie data={liveRealSynth} dataKey="value" nameKey="name" innerRadius={58} outerRadius={86} paddingAngle={3}>
                          {liveRealSynth.map((e, i) => <Cell key={i} fill={e.fill} />)}
                        </Pie>
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ borderRadius: 6, border: `1px solid ${T.hairline}`, fontSize: 12.5 }} />
                      </PieChart>
                    </ResponsiveContainer>
                    <Insight>
                      Right now: {liveRealCount} real rows vs. {liveSyntheticCount} synthetic rows, based only on how you tagged each uploaded file above. Retag any file and this updates immediately \u2014 this reflects only what you've uploaded in this browser session, separate from the benchmark's own fixed data (see the ledger below).
                    </Insight>
                  </>
                ) : (
                  <div style={{ padding: "40px 16px", textAlign: "center", color: T.textMuted, fontSize: 12.5, lineHeight: 1.7 }}>
                    No files uploaded in this session yet. Once you drop a CSV in the Upload data card above and tag it, its real-vs-synthetic breakdown will appear here live.
                    <br /><br />
                    For the benchmark's own fixed data sources \u2014 exactly which dataset contributed to which number \u2014 skip to the <b>Full data provenance ledger</b> below. That table is the single source of truth; this card is not a summary of it.
                  </div>
                )}
              </Card>
            </div>

            <Card>
              <SectionTitle sub="Every number this benchmark produces, and exactly which dataset(s) contributed to it — the complete picture">Full data provenance ledger</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr 1fr 1fr 1.4fr", padding: "8px 0", borderBottom: `2px solid ${T.hairline}`, fontSize: 10.5, fontWeight: 700, color: T.textMuted }}>
                <div>BENCHMARK COMPONENT</div><div>mcPHASES</div><div>MARQUETTE</div><div>KAGGLE</div><div>NOTE</div>
              </div>
              {[
                { component: "Phase-classification model (all 7 signals)", mc: "100%", mq: "0%", kg: "0%", note: "Marquette has no wearable columns \u2014 can't contribute features here" },
                { component: "Cycle-phase distribution chart", mc: "100%", mq: "0%", kg: "0%", note: "Phase labels only exist in mcPHASES" },
                { component: "Total cycle length (merged)", mc: "7.1% (128 cycles)", mq: "92.9% (1,665 cycles)", kg: "0%", note: "Same measurement method in both \u2014 pooled into one distribution" },
                { component: "Luteal phase length", mc: "100% of its own number", mq: "100% of its own number", kg: "0%", note: "Different measurement methods \u2014 shown side by side, never pooled" },
                { component: "Every chart, count, and model on this page", mc: "\u2014", mq: "\u2014", kg: "0%", note: "Evaluated during planning, excluded from the entire benchmark" },
              ].map((row, i) => (
                <div key={i} className="pbd-row" style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr 1fr 1fr 1.4fr", padding: "10px 0", borderBottom: `1px solid ${T.hairline}`, fontSize: 11.5, alignItems: "center" }}>
                  <div style={{ fontWeight: 600 }}>{row.component}</div>
                  <div style={{ color: T.teal, fontWeight: 700 }}>{row.mc}</div>
                  <div style={{ color: T.steel, fontWeight: 700 }}>{row.mq}</div>
                  <div style={{ color: T.amber, fontWeight: 700 }}>{row.kg}</div>
                  <div style={{ color: T.textMuted }}>{row.note}</div>
                </div>
              ))}
              <Insight>
                The 92.9% Marquette share on the cycle-length row is real and computed, not estimated: 1,665 of the 1,793 pooled cycles came from Marquette, 128 from mcPHASES. A single "100% real" badge at the top of this tab would have hidden that \u2014 which is exactly why this ledger exists as its own, mandatory section.
              </Insight>
            </Card>

            <Card>
              <SectionTitle sub="The physiology behind the four labels">Phase guide</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "0.7fr 1.3fr 2fr", gap: 0, fontSize: 12 }}>
                <div style={{ fontWeight: 700, color: T.textMuted, paddingBottom: 8, borderBottom: `2px solid ${T.hairline}` }}>PHASE</div>
                <div style={{ fontWeight: 700, color: T.textMuted, paddingBottom: 8, borderBottom: `2px solid ${T.hairline}` }}>HORMONE PATTERN</div>
                <div style={{ fontWeight: 700, color: T.textMuted, paddingBottom: 8, borderBottom: `2px solid ${T.hairline}` }}>WHY IT MATTERS FOR THE MODEL</div>
                {PHASE_GUIDE.map((row) => (
                  <React.Fragment key={row.p}>
                    <div style={{ padding: "10px 0", borderBottom: `1px solid ${T.hairline}` }}>
                      <span style={{ display: "inline-block", padding: "3px 9px", borderRadius: 5, background: CYCLE_COLOR[row.p], color: "#fff", fontWeight: 700, fontSize: 11.5 }}>{row.p}</span>
                    </div>
                    <div style={{ padding: "10px 0", borderBottom: `1px solid ${T.hairline}` }}>{row.hormone}</div>
                    <div style={{ padding: "10px 0", borderBottom: `1px solid ${T.hairline}`, color: T.textMuted }}>{row.note}</div>
                  </React.Fragment>
                ))}
              </div>
            </Card>

            <Card style={{ position: "relative" }}>
              <VerifiedReal>Real (subject-info.csv, n=42)</VerifiedReal>
              <SectionTitle sub="From subject-info.csv \u2014 relevant to the bias-labeling principle from Atomic Principle 02">Participant demographics</SectionTitle>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {[["East Asian", 14], ["Southeast Asian", 10], ["White", 9], ["Middle Eastern", 5], ["South Asian", 1], ["African", 1], ["Latina", 1], ["Caribbean", 1]].map(([name, n]) => (
                  <div key={name} style={{ padding: "8px 12px", borderRadius: 8, background: T.navyLight, fontSize: 11.5 }}>
                    <b>{name}</b> <span style={{ color: T.textMuted }}>({n})</span>
                  </div>
                ))}
              </div>
              <Insight>
                Recruited through women's health advocacy groups in the Greater Toronto Area \u2014 skews East/Southeast Asian and White, with only 1 participant each from several other groups. Any model trained only on this cohort should be described as validated on <i>this specific population</i>, not on women in general \u2014 exactly the provenance-tagging discipline from Atomic Principle 02.
              </Insight>
            </Card>
          </div>
        )}

        {/* ================= LIFE STAGE ================= */}
        {activeTab === "lifestage" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 18 }}>
            <Card>
              <SectionTitle sub="Why phase classification is not just a period-tracking feature">What this unlocks across a woman's life course</SectionTitle>
              <div style={{ fontSize: 12.5, lineHeight: 1.7, color: T.textDark }}>
                A model that reliably reads cycle phase from passive signals turns menarche-to-menopause from a series of disconnected doctor visits into <b>one continuous, personal trajectory</b>. Four concrete payoffs:
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 14 }}>
                {[
                  ["Early perimenopause detection", "STRAW+10 staging relies on self-reported cycle irregularity and occasional blood draws \u2014 but hormone levels swing so widely during the transition that single tests are unreliable. Continuous signals can flag the transition years earlier, opening a window for preventive bone and cardiovascular care."],
                  ["Screening for anovulatory patterns", "A model that knows what a normal 4-phase cycle looks like can flag missing or shortened ovulation/luteal phases \u2014 an early, non-invasive signal for PCOS and similar disorders, long before a clinical workup."],
                  ["A personal baseline instead of a population average", "The same signal pattern that's \u201cnormal\u201d for one person can be a warning sign for another. Multi-cycle tracking lets the model learn what's normal for this specific person, something a single snapshot test can never do."],
                  ["Cycle-aware clinical trial design", "Drug efficacy and side-effect profiles can vary by hormonal phase, but few trials record participants' phase at all. A validated phase classifier gives trial designers a standardized way to account for this."],
                ].map(([title, body], i) => (
                  <div key={i} style={{ display: "flex", gap: 12, padding: 14, borderRadius: 8, background: T.navyLight }}>
                    <div style={{ minWidth: 26, height: 26, borderRadius: 6, background: T.navy, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, fontSize: 12.5 }}>{i + 1}</div>
                    <div><div style={{ fontWeight: 700, fontSize: 13, marginBottom: 3 }}>{title}</div><div style={{ fontSize: 12.5, color: T.textMuted, lineHeight: 1.55 }}>{body}</div></div>
                  </div>
                ))}
              </div>
            </Card>

            <Card>
              <SectionTitle sub="The same signal pipeline, read differently at each stage">Across the life course</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
                {LIFE_STAGES.map((s) => (
                  <div key={s.stage} style={{ padding: 14, borderRadius: 8, background: "#fff", border: `1px solid ${T.hairline}` }}>
                    <div style={{ fontWeight: 700, fontSize: 13, color: T.navy }}>{s.stage}</div>
                    <div style={{ fontSize: 11, color: T.textMuted, marginBottom: 8 }}>{s.years}</div>
                    <div style={{ fontSize: 11.5, marginBottom: 8, lineHeight: 1.5 }}>{s.focus}</div>
                    <div style={{ fontSize: 11.5, color: T.teal, lineHeight: 1.5, fontWeight: 600 }}>{s.value}</div>
                  </div>
                ))}
              </div>
              <Insight>
                No single dataset here spans a whole lifetime \u2014 mcPHASES covers reproductive-age adults only. The point isn't that we have life-course data today; it's that the schema and pipeline are built so the same fields (cycle_day, phase_label, source_layer) still make sense whether the next cohort we add is adolescents or perimenopausal women.
              </Insight>
            </Card>
          </div>
        )}

        {/* ================= BEHAVIORAL SIGNALS ================= */}
        {activeTab === "behavior" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 18 }}>
            <Card>
              <SectionTitle sub="A proposal/framework tab, like Life Stage — no live data, this is the standard we're leaving behind">Beyond physiology: standardizing behavioral & context signals</SectionTitle>
              <div style={{ fontSize: 12.5, lineHeight: 1.7, color: T.textDark }}>
                Everything else in this benchmark measures physiology \u2014 heart rate, temperature, hormones. Each signal below is included only because it has <b>published evidence</b> tying it to this benchmark's own known weak points (an earlier candidate, bathroom-visit frequency, was dropped after a literature check found no menstrual-health evidence for it \u2014 only unrelated elder-care/UTI research). These three aren't "nice to have": each targets a specific gap already found in this project.
              </div>
            </Card>

            <Card>
              <SectionTitle sub="Each one cited to the specific weakness in this benchmark it addresses">Evidence-based signal categories</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1.5fr 1.4fr", padding: "8px 0", borderBottom: `2px solid ${T.hairline}`, fontSize: 10.5, fontWeight: 700, color: T.textMuted }}>
                <div>SIGNAL</div><div>EVIDENCE</div><div>WHY IT MATTERS HERE</div>
              </div>
              {BEHAVIORAL_SIGNALS.map((row, i) => (
                <div key={i} className="pbd-row" style={{ display: "grid", gridTemplateColumns: "1.1fr 1.5fr 1.4fr", padding: "12px 0", borderBottom: `1px solid ${T.hairline}`, fontSize: 11.5, alignItems: "start" }}>
                  <div>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>{row.signal}</div>
                    <div style={{ color: T.textMuted, fontSize: 10.5 }}>{row.method}</div>
                  </div>
                  <div style={{ color: T.textMuted }}>{row.citation}</div>
                  <div style={{ color: T.steel, fontSize: 11 }}>{row.ties_to}</div>
                </div>
              ))}
              <Insight>
                All three were chosen the same way: not "what could we imagine collecting," but "what does this benchmark already need in order to trust the numbers it's produced so far" \u2014 a second check on the phase label (DRSP), a missing covariate that would otherwise contaminate personalization (contraception), and a plausible hidden confound behind our two best single-signal results (RPE vs. HRV/respiratory rate).
              </Insight>
            </Card>

            <Card>
              <SectionTitle sub="A population-coverage gap, stated plainly">Who this benchmark currently doesn't represent</SectionTitle>
              <div style={{ padding: 16, borderRadius: 8, background: T.amberLight, border: `1px solid ${T.amber}40`, fontSize: 12.5, color: T.textDark, lineHeight: 1.65 }}>
                Both mcPHASES and Marquette explicitly excluded hormonal-contraceptive users during recruitment. A large share of real-world women use contraception \u2014 so as it stands, <b>this benchmark says nothing about them.</b> Adding a contraception-adherence field (above) is necessary but not sufficient; it also means recruiting a contraceptive-user cohort is a distinct, future data-collection priority, not a checkbox on the existing schema.
              </div>
            </Card>
          </div>
        )}

        {/* ================= DATA PLAN ================= */}
        {activeTab === "plan" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 18 }}>
            <Card>
              <SectionTitle sub="What's realistically usable inside a single hackathon day">Today</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 14 }}>
                <div style={{ padding: 16, borderRadius: 8, background: T.tealLight, border: `1px solid ${T.teal}30` }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: T.teal, marginBottom: 6 }}>Real \u2014 mcPHASES (PhysioNet)</div>
                  <div style={{ fontSize: 12, lineHeight: 1.6 }}>15 of 23 tables verified, including phase labels, hormones, resting HR, wrist temperature, HRV, respiratory rate, glucose, activity minutes, sleep score, and demographics. 42 participants in 2022, 20 returning in 2024, matching the published paper exactly. Access is DUA-gated (PhysioNet Restricted Health Data License), not fully open-download.</div>
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: T.amberLight, border: `1px solid ${T.amber}30` }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: T.amber, marginBottom: 6 }}>Excluded \u2014 Kaggle synthetic (100 users)</div>
                  <div style={{ fontSize: 12, lineHeight: 1.6 }}>Synthetic demographic and lifestyle features with cycle-length labels, no real hormones. Evaluated during planning, then excluded entirely \u2014 not in any chart, count, or model on this page.</div>
                </div>
              </div>
              <div style={{ marginTop: 14, padding: 14, borderRadius: 8, background: T.tealLight, border: `1px solid ${T.teal}50` }}>
                <div style={{ fontWeight: 700, fontSize: 12.5, color: T.teal, marginBottom: 4 }}>Option 1's full signal comparison: complete</div>
                <div style={{ fontSize: 12, color: T.textDark, lineHeight: 1.6 }}>
                  Sleep, heart rate, temperature, HRV, respiratory rate, activity, and glucose are all real and joined. See Baseline & Eval for the full 7-way comparison, including the personalization finding.
                </div>
              </div>
            </Card>

            <Card style={{ position: "relative" }}>
              <VerifiedReal>Real cross-dataset validation, computed this session</VerifiedReal>
              <SectionTitle sub="mcPHASES (phase-label derived) vs. Marquette NFP (independent, n=159, 1,665 cycles) — merged only where the underlying measurement method matches">External validation: what merged, what didn't</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr 1fr 0.6fr", padding: "8px 0", borderBottom: `2px solid ${T.hairline}`, fontSize: 11, fontWeight: 700, color: T.textMuted }}>
                <div>METRIC</div><div>mcPHASES</div><div>MARQUETTE</div><div>VERDICT</div>
              </div>
              {CYCLE_TIMING_COMPARISON.map((row, i) => (
                <div key={i} className="pbd-row" style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr 1fr 0.6fr", padding: "10px 0", borderBottom: `1px solid ${T.hairline}`, fontSize: 12, alignItems: "center" }}>
                  <div style={{ fontWeight: 600 }}>{row.metric}</div>
                  <div style={{ color: T.textMuted }}>{row.mcphases}</div>
                  <div style={{ color: T.textMuted }}>{row.marquette}</div>
                  <div>
                    {row.verdict === "match"
                      ? <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: T.teal, fontWeight: 700, fontSize: 11 }}><CheckCircle2 size={13} /> Merged</span>
                      : <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: T.amber, fontWeight: 700, fontSize: 11 }}><AlertTriangle size={13} /> Kept separate</span>}
                  </div>
                </div>
              ))}
              <Insight>
                <b>Total cycle length matched almost exactly</b> (29.8 vs. 29.3 days) \u2014 both datasets count it the same way (period-start to period-start), so we merged it into one pooled distribution with no adjustment.
              </Insight>
              <div style={{ marginTop: 10, padding: 14, borderRadius: 8, background: T.amberLight, border: `1px solid ${T.amber}40` }}>
                <div style={{ fontWeight: 700, fontSize: 12.5, color: T.amber, marginBottom: 6 }}>Why luteal phase length was NOT merged</div>
                <div style={{ fontSize: 12, color: T.textDark, lineHeight: 1.65 }}>
                  The two datasets define "start of luteal phase" with two different measurement methods: mcPHASES uses Mira's hormone-threshold algorithm (LH/estrogen/progesterone assay pattern), while Marquette's NFP protocol marks it from the day after the charted "Peak day" of cervical mucus and fertility-monitor readings. A 3.3-day gap (10.0 vs. 13.3 days) is too large to be sampling noise \u2014 it reflects two different operational definitions of the same-sounding word "luteal." Averaging them together would silently blend two different measurements into one misleading number, exactly the kind of hidden preprocessing assumption this challenge asks teams to avoid. We report both numbers side by side instead of pooling them.
                </div>
              </div>
            </Card>

            <Card>
              <SectionTitle sub="What we'd tell the next contributor adding a dataset to this benchmark">Dataset onboarding checklist (derived from the Marquette experience)</SectionTitle>
              {ONBOARDING_CHECKLIST.map((item, i) => (
                <div key={i} style={{ display: "flex", gap: 12, padding: "10px 0", borderBottom: i < ONBOARDING_CHECKLIST.length - 1 ? `1px solid ${T.hairline}` : "none" }}>
                  <div style={{ minWidth: 160, fontWeight: 700, fontSize: 12.5, color: T.navy }}>{item.step}</div>
                  <div style={{ fontSize: 12.5, color: T.textDark, lineHeight: 1.55 }}>{item.detail}</div>
                </div>
              ))}
              <Insight>
                This checklist isn't hypothetical \u2014 step 3 is exactly what caught the luteal-length mismatch above before it could quietly bias a merged number. Any future dataset (a third wearable cohort, a clinical registry, anything) goes through the same five steps before a single field gets pooled with mcPHASES.
              </Insight>
            </Card>

            <Card>
              <SectionTitle sub="What we just found, computed live from your real files">Headline results this session</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
                <div style={{ padding: 16, borderRadius: 8, background: T.navyLight }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: T.navy }}>35.1%</div>
                  <div style={{ fontSize: 11.5, color: T.textMuted }}>Best real accuracy after personalizing features (per-person z-score) \u2014 beats the 33.4% majority baseline</div>
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: T.navyLight }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: T.navy }}>37.1%</div>
                  <div style={{ fontSize: 11.5, color: T.textMuted }}>Adding glucose pushes accuracy higher still, but on a smaller Round-1-only sample (n=2,106)</div>
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: T.navyLight }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: T.navy }}>31.2%</div>
                  <div style={{ fontSize: 11.5, color: T.textMuted }}>Best single personalized signal: respiratory rate, ahead of HRV, glucose, temperature, HR, activity, sleep</div>
                </div>
              </div>
              <Insight>
                None of these numbers are placeholders \u2014 they came from training real models on your real uploaded files this session, now covering all seven of Option 1's signal categories.
              </Insight>
            </Card>

            <Card>
              <SectionTitle sub="What we found, computed live from your real files (superseded numbers above are kept for the full story)">Prior headline results</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
                <div style={{ padding: 16, borderRadius: 8, background: T.navyLight }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: T.navy }}>27.0%</div>
                  <div style={{ fontSize: 11.5, color: T.textMuted }}>Real, participant-held-out accuracy \u2014 below the 33.4% majority-class baseline</div>
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: T.navyLight }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: T.navy }}>31.8%</div>
                  <div style={{ fontSize: 11.5, color: T.textMuted }}>Best single real signal: respiratory rate, beating HR/temperature/HRV alone</div>
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: T.navyLight }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: T.navy }}>4,087</div>
                  <div style={{ fontSize: 11.5, color: T.textMuted }}>Real days with phase + all 4 wearable signals jointly available, across 40 people</div>
                </div>
              </div>
              <Insight>
                None of these numbers are placeholders \u2014 they came from training real models on your real uploaded files earlier this session. See the Baseline & Eval tab for the full breakdown.
              </Insight>
            </Card>

            <Card>
              <SectionTitle sub="Where they came from, and what actually happened with each one">Additional datasets considered \u2014 outcome</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
                <div style={{ padding: 16, borderRadius: 8, background: T.amberLight, border: `1px solid ${T.amber}30` }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: T.amber, marginBottom: 6 }}>Kaggle synthetic (100 users)</div>
                  <div style={{ fontSize: 11.5, lineHeight: 1.6, marginBottom: 6 }}>Synthetic, no real hormones. <b>Outcome: excluded entirely</b> \u2014 0% in every chart, count, and model.</div>
                  <a href="https://www.kaggle.com/datasets/akshayas02/menstrual-cycle-data-with-factors-dataset" style={{ fontSize: 11, color: T.amber, fontWeight: 600 }}>kaggle.com/datasets/akshayas02/\u2026</a>
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: T.tealLight, border: `1px solid ${T.teal}30` }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: T.teal, marginBottom: 6 }}>Marquette NFP (real, 80 columns)</div>
                  <div style={{ fontSize: 11.5, lineHeight: 1.6, marginBottom: 6 }}><b>Outcome: integrated, but only partially.</b> No wearable columns to feed the classifier, so it contributes 0% there. Its cycle-length field measures the same thing the same way as mcPHASES, so it was pooled in (92.9% of that one merged metric \u2014 see the ledger above). Its luteal-length field uses a different definition, so it was kept separate, not pooled.</div>
                  <a href="https://www.kaggle.com/datasets/nikitabisht/menstrual-cycle-data" style={{ fontSize: 11, color: T.teal, fontWeight: 600, display: "block" }}>kaggle.com/datasets/nikitabisht/\u2026</a>
                  <a href="https://epublications.marquette.edu/data_nfp/7/" style={{ fontSize: 11, color: T.teal, fontWeight: 600 }}>epublications.marquette.edu (original)</a>
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: "#fff", border: `1px solid ${T.hairline}` }}>
                  <div style={{ fontWeight: 700, fontSize: 13, color: T.textMuted, marginBottom: 6 }}>IEEE DataPort (mixed provenance)</div>
                  <div style={{ fontSize: 11.5, lineHeight: 1.6, marginBottom: 6 }}>Real-ish but aggregated from multiple sources. <b>Outcome: not pursued</b> \u2014 provenance caveats outweighed the likely benefit.</div>
                  <a href="https://ieee-dataport.org/documents/dataset-menstrual-cycle-phase-prediction-and-hygiene-guidance" style={{ fontSize: 11, color: T.textMuted, fontWeight: 600 }}>ieee-dataport.org/documents/\u2026</a>
                </div>
              </div>
              <Insight>
                <b>The lesson wasn't "stop at two datasets" \u2014 it was "merge only what actually matches."</b> Marquette turned out to be a mixed case: useful for one field (cycle length), unusable for two others (the classifier's wearable features, luteal length). A dataset doesn't have to be all-in or all-out; the ledger above tracks that granularity per metric, not per dataset.
              </Insight>
            </Card>

            <Card>
              <SectionTitle sub="How to collect more real data without high dropout \u2014 proposal, not yet executed">Low-friction collection protocol (proposed)</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1.6fr 0.8fr 2fr", padding: "8px 0", borderBottom: `2px solid ${T.hairline}`, fontSize: 11, fontWeight: 700, color: T.textMuted }}>
                <div>METHOD</div><div>FRICTION</div><div>WHY</div>
              </div>
              {FRICTION_PROTOCOLS.map((row, i) => (
                <div key={i} className="pbd-row" style={{ display: "grid", gridTemplateColumns: "1.6fr 0.8fr 2fr", padding: "10px 0", borderBottom: `1px solid ${T.hairline}`, fontSize: 12.5, alignItems: "center" }}>
                  <div style={{ fontWeight: 600 }}>{row.method}</div>
                  <div style={{ color: row.friction.toLowerCase().includes("high") ? "#B14A3A" : T.teal, fontWeight: 700 }}>{row.friction}</div>
                  <div style={{ color: T.textMuted }}>{row.note}</div>
                </div>
              ))}
              <Insight>
                This table is a design proposal, not a report of data we already collected \u2014 it exists to argue that mcPHASES's own approach (wearable-first, hormone-kit second, survey last) is the right order of priority for any future expansion, not an accident of what happened to be easy for that study.
              </Insight>
            </Card>

            <Card>
              <SectionTitle sub="What gets built after today, in order">Roadmap</SectionTitle>
              {[
                ["Next week", "Pilot the low-friction protocol above with a small volunteer group (target 15\u201320 people) to measure real dropout rate before scaling."],
                ["1 month", "Partner with a wearable cohort (Fitbit/Oura), target 50+ users over 4 continuous weeks \u2014 expanding Layer 2 beyond mcPHASES's 42."],
                ["3 months", "Add a CGM patch arm for consenting participants \u2014 mirrors mcPHASES's own continuous glucose component, at larger scale."],
                ["6 months", "Partner with a clinical or at-home hormone-kit provider under IRB approval, target 100+ participants with real LH/E3G/PdG \u2014 the Layer 4 anchor set."],
              ].map(([when, what], i) => (
                <div key={i} style={{ display: "flex", gap: 14, padding: "10px 0", borderBottom: i < 3 ? `1px solid ${T.hairline}` : "none" }}>
                  <div style={{ minWidth: 100, fontWeight: 700, fontSize: 12.5, color: T.navy }}>{when}</div>
                  <div style={{ fontSize: 12.5, color: T.textDark }}>{what}</div>
                </div>
              ))}
            </Card>
          </div>
        )}

        {/* ================= BASELINE & EVAL ================= */}
        {activeTab === "baseline" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 18 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
              <Card style={{ position: "relative" }}>
                <VerifiedReal>Real, trained this session</VerifiedReal>
                <SectionTitle sub={`RandomForest \u00b7 personalized wearable features \u00b7 n=${CM_TOTAL} test days, participant-held-out`}>Confusion matrix (4-phase)</SectionTitle>
                <div style={{ display: "grid", gridTemplateColumns: "62px repeat(4, 1fr)", gap: 4 }}>
                  <div />
                  {CM_LABELS.map((l) => <div key={l} style={{ textAlign: "center", fontSize: 10, fontWeight: 700, color: T.textMuted }}>{l}</div>)}
                  {CM.map((row, i) => (
                    <React.Fragment key={i}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: T.textMuted }}>{CM_LABELS[i]}</div>
                      {row.map((v, j) => {
                        const intensity = v / Math.max(...CM.flat());
                        const isDiag = i === j;
                        const bg = isDiag ? `rgba(18,58,92,${0.15 + intensity * 0.75})` : `rgba(169,118,42,${0.08 + intensity * 0.5})`;
                        return <div key={j} style={{ background: bg, borderRadius: 6, height: 46, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, fontSize: 13, color: isDiag ? T.navy : T.amber }}>{v}</div>;
                      })}
                    </React.Fragment>
                  ))}
                </div>
                <Insight>
                  Trained on resting heart rate, wrist temperature, HRV, respiratory rate, activity, and sleep score \u2014 each <b>z-scored within each participant</b> \u2014 and <b>no hormone features</b>, since those defined the label itself. Test participants (10 people) never appeared in training. Accuracy: <b>{(UNWEIGHTED_ACC * 100).toFixed(1)}%</b>, beating the 33.4% majority baseline once personalization was added.
                </Insight>
              </Card>

              <Card style={{ position: "relative" }}>
                <VerifiedReal>Real comparison, same data</VerifiedReal>
                <SectionTitle sub="Why the split you choose — and whether you personalize — changes the headline number">Leaky vs. rigorous vs. personalized</SectionTitle>
                <ResponsiveContainer width="100%" height={190}>
                  <BarChart layout="vertical" data={REAL_SPLIT_COMPARISON} margin={{ left: 10 }}>
                    <CartesianGrid stroke={T.hairline} horizontal={false} />
                    <XAxis type="number" domain={[0, 45]} tick={{ fontSize: 11, fill: T.textMuted }} axisLine={false} tickLine={false} unit="%" />
                    <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: T.textDark }} width={155} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ borderRadius: 6, border: `1px solid ${T.hairline}`, fontSize: 12 }} formatter={(v) => v + "%"} />
                    <Bar dataKey="value" radius={[0, 5, 5, 0]}>
                      <Cell fill={T.textMuted} /><Cell fill={T.amber} /><Cell fill={T.textMuted} /><Cell fill={T.navy} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <Insight>
                  The amber bar (37.7%) looks like a working model \u2014 until you notice it lets the same person's days leak into both train and test. Once participants are properly held out with raw features, accuracy drops to 27.0%, <b>below</b> guessing "Luteal" every time. But personalizing those same signals (z-scoring within each person, navy bar) recovers real signal: 35.1%, above the majority baseline. The lesson isn't "wearables don't work" \u2014 it's "absolute wearable values don't generalize across people; each person's own deviation from their baseline does."
                </Insight>
              </Card>
            </div>

            <Card style={{ position: "relative" }}>
              <VerifiedReal>Real, trained this session</VerifiedReal>
              <SectionTitle sub="Which passive signal carries the most information about cycle phase — RandomForest, participant-held-out split, personalized features, one signal at a time">Single-signal comparison (Option 1's full 7-signal question)</SectionTitle>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart layout="vertical" data={REAL_SIGNAL_COMPARISON} margin={{ left: 10 }}>
                  <CartesianGrid stroke={T.hairline} horizontal={false} />
                  <XAxis type="number" domain={[0, 40]} tick={{ fontSize: 11, fill: T.textMuted }} axisLine={false} tickLine={false} unit="%" />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 10.5, fill: T.textDark }} width={220} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: 6, border: `1px solid ${T.hairline}`, fontSize: 12 }} formatter={(v) => v + "%"} />
                  <Bar dataKey="value" radius={[0, 5, 5, 0]} fill={T.steel} />
                </BarChart>
              </ResponsiveContainer>
              <Insight>
                All seven of Option 1's signal categories are now real: sleep, activity, heart rate, temperature, HRV, respiratory rate, and glucose. Respiratory rate is the best single signal (31.2%), sleep score the weakest (27.3%). Combining all six continuously-available signals reaches 35.1%; adding glucose pushes it to 37.1%, but only on the smaller Round-1-only subset (n=490 vs. n=1,047) \u2014 a real accuracy/sample-size tradeoff, not a free win.
              </Insight>
            </Card>

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ fontSize: 12.5, fontWeight: 700, color: T.textDark }}>External literature, for context</div>
              <div style={{ display: "flex", gap: 10, padding: "10px 12px", borderRadius: 8, background: "#fff", border: `1px solid ${T.hairline}` }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: T.steel, minWidth: 90 }}>npj 2025</span>
                <span style={{ fontSize: 11.5, color: T.textMuted }}>18 subjects, 65 cycles, wrist sensors \u2192 87% accuracy / 0.96 AUC for 3-phase classification \u2014 markedly higher than our result, likely reflecting a smaller, less person-held-out evaluation.</span>
              </div>
              <div style={{ display: "flex", gap: 10, padding: "10px 12px", borderRadius: 8, background: "#fff", border: `1px solid ${T.hairline}` }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: T.steel, minWidth: 90 }}>Huawei WST</span>
                <span style={{ fontSize: 11.5, color: T.textMuted }}>261 women, 270+84 confirmed-ovulation cycles, wrist skin temperature + heart rate \u2014 fertile-window accuracy \u2265 75% (regular cycles); stats-only, not a downloadable dataset.</span>
              </div>
              <div style={{ display: "flex", gap: 10, padding: "10px 12px", borderRadius: 8, background: "#fff", border: `1px solid ${T.hairline}` }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: T.steel, minWidth: 90 }}>Meta-analysis</span>
                <span style={{ fontSize: 11.5, color: T.textMuted }}>27 studies pooled: wearable fertile-window detection at 0.88 accuracy (sensitivity 0.79, specificity 0.80) \u2014 the broadest literature baseline we have, and still well above our own rigorous number.</span>
              </div>
            </div>
          </div>
        )}

        {/* ================= LEADERBOARD ================= */}
        {activeTab === "leaderboard" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 18 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1.1fr", gap: 18 }}>
              <Card style={{ position: "relative" }}>
                <VerifiedReal>Real options only</VerifiedReal>
                <SectionTitle sub="Every option here is a real, already-trained RandomForest result — not a live-fit model">Submit a run</SectionTitle>
                <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 8 }}>Signal configuration</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 18 }}>
                  {REAL_SIGNAL_OPTIONS.map((opt) => (
                    <label key={opt.key} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12.5, cursor: "pointer" }}>
                      <input type="radio" name="signal" checked={selectedSignal === opt.key} onChange={() => setSelectedSignal(opt.key)} />
                      {opt.label} <span style={{ color: T.textMuted, fontSize: 11 }}>({opt.acc}%)</span>
                    </label>
                  ))}
                </div>
                <button onClick={submitRun} style={{ width: "100%", padding: "11px 0", borderRadius: 8, border: "none", background: T.navy, color: "#fff", fontWeight: 700, fontSize: 13.5, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
                  Add to leaderboard <ChevronRight size={16} />
                </button>
                <Insight>
                  These five numbers are fixed because they're real \u2014 computed once, this session, on the real participant-held-out split. We're not letting you retrain live (that would take real compute time); this picker just lets you compare configurations you've already seen in the Baseline & Eval tab.
                </Insight>
              </Card>

              <Card>
                <SectionTitle>Leaderboard</SectionTitle>
                <div style={{ display: "grid", gridTemplateColumns: "2.6fr 1fr 0.7fr", padding: "6px 0", borderBottom: `2px solid ${T.hairline}`, fontSize: 11, fontWeight: 700, color: T.textMuted }}>
                  <div>SUBMISSION</div><div>ACCURACY</div><div>N</div>
                </div>
                {submissions.map((s, i) => (
                  <div key={i} className="pbd-row" style={{ display: "grid", gridTemplateColumns: "2.6fr 1fr 0.7fr", padding: "10px 0", borderBottom: `1px solid ${T.hairline}`, fontSize: 12.5, alignItems: "center" }}>
                    <div style={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>
                      {i === 0 && !s.flagged && <Trophy size={14} color={T.amber} />}{s.name}
                      {s.pinned && <span style={{ fontSize: 10, color: T.textMuted, fontWeight: 400 }}>(pinned)</span>}
                      {s.external && <span style={{ fontSize: 10, color: T.steel, fontWeight: 400 }}>(simulated)</span>}
                      {s.flagged && <AlertTriangle size={12} color={T.amber} />}
                    </div>
                    <div style={{ fontWeight: 700, color: s.flagged ? T.amber : T.navy }}>{s.acc.toFixed(1)}%</div>
                    <div style={{ color: T.textMuted }}>{s.n}</div>
                  </div>
                ))}
              </Card>
            </div>

            <Card>
              <SectionTitle sub="Paste a public dataset link and see the scoring pipeline run">Score an external dataset</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 20 }}>
                <div>
                  <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
                    <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8, border: `1px solid ${T.hairline}`, borderRadius: 6, padding: "8px 10px" }}>
                      <Link2 size={15} color={T.textMuted} />
                      <input value={datasetUrl} onChange={(e) => setDatasetUrl(e.target.value)} placeholder="https://physionet.org/content/mcphases"
                        style={{ border: "none", outline: "none", flex: 1, fontSize: 12.5, fontFamily: "'IBM Plex Mono', monospace" }} />
                    </div>
                    <button onClick={runUrlPipeline} style={{ padding: "0 18px", borderRadius: 6, border: "none", background: T.teal, color: "#fff", fontWeight: 700, fontSize: 13, cursor: "pointer" }}>Load & score</button>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {pipelineSteps.map((step, i) => {
                      const done = pipelineStep > i;
                      const active = pipelineStep === i + 1 && !urlResult;
                      return (
                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", borderRadius: 6, background: done ? T.tealLight : "transparent", fontSize: 12.5 }}>
                          {done ? <CheckCircle2 size={16} color={T.teal} /> : <div style={{ width: 16, height: 16, borderRadius: "50%", border: `2px solid ${active ? T.steel : T.hairline}` }} />}
                          <span style={{ color: done ? T.textDark : T.textMuted, fontWeight: done ? 600 : 400 }}>{i + 1}. {step}</span>
                        </div>
                      );
                    })}
                  </div>
                  {urlResult && (
                    <div style={{ marginTop: 14, padding: 14, borderRadius: 8, background: T.navyLight, display: "flex", gap: 24 }}>
                      <div><div style={{ fontSize: 11, color: T.textMuted, fontWeight: 600 }}>ACCURACY</div><div style={{ fontSize: 22, fontWeight: 700, color: T.navy }}>{(urlResult.weighted * 100).toFixed(1)}%</div></div>
                      <div><div style={{ fontSize: 11, color: T.textMuted, fontWeight: 600 }}>N</div><div style={{ fontSize: 22, fontWeight: 700, color: T.textDark }}>{urlResult.n}</div></div>
                    </div>
                  )}
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: T.amberLight, border: `1px solid ${T.amber}30` }}>
                  <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                    <AlertTriangle size={16} color={T.amber} style={{ flexShrink: 0, marginTop: 1 }} />
                    <div style={{ fontSize: 12, fontWeight: 700, color: T.amber }}>How to read this, in plain terms</div>
                  </div>
                  <div style={{ fontSize: 12, lineHeight: 1.7, color: T.textDark }}>
                    Nothing is actually downloaded here \u2014 this sandbox can't reach outside links, so the five steps and the score are a <b>simulation of the real pipeline</b>, seeded from the URL text so the same link always gives the same number. The five real numbers on the left, by contrast, came from an actual model we trained this session on your uploaded mcPHASES files \u2014 that's the difference between this box and the leaderboard next to it.
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* ================= PRIVACY ================= */}
        {activeTab === "privacy" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.1fr", gap: 18 }}>
            <Card>
              <SectionTitle sub="Aggregate statistic: mean wrist skin temperature (\u00b0C), real value from computed_temperature.csv">Differential privacy demo</SectionTitle>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, fontWeight: 600, marginBottom: 6 }}>
                <span>Privacy budget (\u03b5)</span><span>{epsilon.toFixed(1)}</span>
              </div>
              <input className="pbd-slider" type="range" min="0.1" max="5" step="0.1" value={epsilon} onChange={(e) => setEpsilon(Number(e.target.value))} style={{ width: "100%", marginBottom: 20 }} />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{ padding: 16, borderRadius: 8, background: T.steelLight }}>
                  <div style={{ fontSize: 11, color: T.textMuted, fontWeight: 600 }}>TRUE MEAN (never released)</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: T.steel }}>{trueMean.toFixed(2)}\u00b0C</div>
                </div>
                <div style={{ padding: 16, borderRadius: 8, background: T.tealLight }}>
                  <div style={{ fontSize: 11, color: T.textMuted, fontWeight: 600 }}>NOISED MEAN (released)</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: T.teal }}>{noisedMean.toFixed(2)}\u00b0C</div>
                </div>
              </div>
              <Insight>
                Wrist temperature shifts by only a few tenths of a degree across the cycle \u2014 exactly why it needs to stay precise enough to be useful, but noised enough that no single participant's raw reading is ever recoverable.
              </Insight>
            </Card>
            <Card>
              <SectionTitle sub="Illustrative utility loss as \u03b5 increases">Privacy \u2013 utility tradeoff</SectionTitle>
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={utilityCurve}>
                  <CartesianGrid stroke={T.hairline} />
                  <XAxis dataKey="epsilon" tick={{ fontSize: 11, fill: T.textMuted }} axisLine={{ stroke: T.hairline }} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: T.textMuted }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: 6, border: `1px solid ${T.hairline}`, fontSize: 12.5 }} />
                  <Line type="monotone" dataKey="loss" stroke={T.steel} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <Insight>Current \u03b5 = {epsilon.toFixed(1)}. Raw wearable records never leave the pipeline \u2014 only a number like the one on the left, with deliberately chosen noise baked in.</Insight>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
