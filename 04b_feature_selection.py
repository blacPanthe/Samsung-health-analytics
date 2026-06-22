"""
====================================================================
  SCRIPT 4b — FEATURE SELECTION
  Samsung Health Personal Analytics Project
  Author: Atharva

  PURPOSE:
  Script 04 built 108 candidate features. Not all of them are useful —
  some are mostly empty, some just duplicate information already
  captured by another column. This script trims that down to a lean,
  high-signal feature set before EDA/ML consume it.

  STEPS:
  ──────────────────────────────────────────────────────────────────
  1. Drop high-null features      (> 50% missing → unreliable)
  2. Drop redundant features      (|correlation| > 0.90 → duplicate
                                    signal, keep the one with fewer
                                    nulls)
  3. Drop target/leakage columns  (e.g. weight itself when predicting
                                    weight change — these would let
                                    the model "cheat")
  4. Rank survivors by Random Forest importance, keep top N
  5. Retrain the fat-loss classifier on the trimmed set and compare
     accuracy against the full feature set (same train/test split
     used in 06_ml_models.py: train on 2019–2024, test on 2025)

  OUTPUT:
  ──────────────────────────────────────────────────────────────────
  master_selected.csv          ← date + targets + top N features
  ml_output/feature_selection_report.txt
====================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(SCRIPT_DIR, "master_featured.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "master_selected.csv")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "ml_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N            = 25
NULL_THRESHOLD   = 0.50   # drop columns missing more than this
CORR_THRESHOLD   = 0.90   # drop one of a pair correlated above this

# Columns that are identifiers, dates, or targets — never candidates
# for selection, but kept in the output file as-is.
ALWAYS_KEEP = ['date', 'week_label', 'weight_change_7d', 'weight_is_real']

# Columns that would leak the answer if used to predict weight change
# (they ARE the weight, just lagged/smoothed/derived).
LEAKAGE_COLS = [
    'weight', 'bmi', 'weight_7d_avg', 'weight_14d_avg', 'weight_30d_avg',
    'weight_lag_7d', 'weight_lag_14d', 'weight_lag_30d', 'weight_momentum',
    'weight_change_14d', 'weight_change_30d', 'week_weight_change',
    'distance_to_goal', 'weight_rate_per_week', 'est_weeks_to_goal',
    'theoretical_weight_loss_kg', 'cumulative_calorie_deficit',
]

# Non-numeric / pure identifier columns — excluded from the numeric
# candidate pool (categorical buckets are redundant with the numeric
# columns they were derived from, e.g. bmi_category vs bmi).
NON_CANDIDATE_OBJECT_COLS = [
    'date', 'month_name', 'day_name', 'season', 'year_month', 'week_start',
    'bmi_category', 'weight_zone', 'sleep_quality_category',
    'activity_level', 'step_zone', 'energy_balance_category',
    'overtraining_risk', 'week_label',
]


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════
section("LOADING FEATURED DATA")
df = pd.read_csv(INPUT_FILE, parse_dates=['date'])
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
candidates = [c for c in numeric_cols if c not in ALWAYS_KEEP and c not in LEAKAGE_COLS]
print(f"  Starting numeric candidates: {len(candidates)}")
print(f"  Excluded as leakage: {len(LEAKAGE_COLS)}  →  {', '.join(LEAKAGE_COLS[:5])}...")


# ══════════════════════════════════════════════════════════════════
#  STEP 1 — DROP HIGH-NULL FEATURES
# ══════════════════════════════════════════════════════════════════
section("STEP 1 — DROPPING HIGH-NULL FEATURES")

null_pct = df[candidates].isnull().mean()
dropped_null = null_pct[null_pct > NULL_THRESHOLD].sort_values(ascending=False)
candidates = [c for c in candidates if c not in dropped_null.index]

print(f"  Threshold: > {NULL_THRESHOLD*100:.0f}% missing")
print(f"  Dropped {len(dropped_null)} features:")
for feat, pct in dropped_null.items():
    print(f"    ✗ {feat:30s}  {pct*100:5.1f}% missing")
print(f"\n  Remaining candidates: {len(candidates)}")


# ══════════════════════════════════════════════════════════════════
#  STEP 2 — DROP REDUNDANT (HIGHLY CORRELATED) FEATURES
# ══════════════════════════════════════════════════════════════════
section("STEP 2 — DROPPING REDUNDANT (HIGHLY CORRELATED) FEATURES")

corr_matrix = df[candidates].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

null_pct_remaining = df[candidates].isnull().mean()
dropped_corr = {}
for col in upper.columns:
    correlated_with = upper.index[upper[col] > CORR_THRESHOLD].tolist()
    for other in correlated_with:
        if col in dropped_corr or other in dropped_corr:
            continue
        # Keep whichever of the pair has fewer missing values
        loser = col if null_pct_remaining[col] >= null_pct_remaining[other] else other
        keeper = other if loser == col else col
        dropped_corr[loser] = (keeper, upper.loc[other, col] if other in upper.index else upper.loc[col, other])

print(f"  Threshold: |correlation| > {CORR_THRESHOLD}")
print(f"  Dropped {len(dropped_corr)} redundant features:")
for loser, (keeper, corr_val) in dropped_corr.items():
    print(f"    ✗ {loser:28s}  (corr={corr_val:.3f} with {keeper})")

candidates = [c for c in candidates if c not in dropped_corr]
print(f"\n  Remaining candidates: {len(candidates)}")


# ══════════════════════════════════════════════════════════════════
#  STEP 3 — RANK BY RANDOM FOREST IMPORTANCE, KEEP TOP N
# ══════════════════════════════════════════════════════════════════
section("STEP 3 — RANKING BY IMPORTANCE")

train_df = df[df['week_label'].isin(['fat_loss', 'maintenance', 'weight_gain'])].copy()
train_df = train_df.dropna(subset=candidates + ['week_label'])

X_all = train_df[candidates].fillna(0)
y_all = train_df['week_label']

rf_rank = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf_rank.fit(X_all, y_all)

importance = pd.Series(rf_rank.feature_importances_, index=candidates).sort_values(ascending=False)
selected_features = importance.head(TOP_N).index.tolist()

print(f"  Trained Random Forest on {len(candidates)} surviving candidates")
print(f"  Keeping top {TOP_N} by importance:\n")
for i, (feat, imp) in enumerate(importance.head(TOP_N).items(), 1):
    bar = '█' * int(imp * 300)
    print(f"    {i:2d}. {feat:30s}  {imp:.4f}  {bar}")

dropped_low_importance = importance.iloc[TOP_N:]
print(f"\n  Dropped {len(dropped_low_importance)} lower-importance features "
      f"(combined importance: {dropped_low_importance.sum():.3f})")


# ══════════════════════════════════════════════════════════════════
#  STEP 4 — BEFORE / AFTER ACCURACY COMPARISON
# ══════════════════════════════════════════════════════════════════
section("STEP 4 — BEFORE / AFTER MODEL COMPARISON")
print("  Same split as 06_ml_models.py: train on 2019–2024, test on 2025\n")

train_mask = train_df['date'].dt.year <= 2024
test_mask  = train_df['date'].dt.year == 2025

def evaluate(feature_list, label):
    X = train_df[feature_list].fillna(0)
    y = train_df['week_label']
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    if len(X_test) == 0:
        print(f"  {label}: no 2025 test rows available, skipping")
        return None
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  {label:35s} → accuracy = {acc*100:.1f}%  ({len(feature_list)} features)")
    return acc

acc_full = evaluate(candidates, "Full surviving feature set")
acc_selected = evaluate(selected_features, f"Top {TOP_N} selected features")

if acc_full is not None and acc_selected is not None:
    delta = (acc_selected - acc_full) * 100
    verdict = "improved" if delta > 0 else ("unchanged" if delta == 0 else "slightly lower, but far simpler")
    print(f"\n  Accuracy change: {delta:+.1f} percentage points ({verdict})")


# ══════════════════════════════════════════════════════════════════
#  SAVE SELECTED DATASET
# ══════════════════════════════════════════════════════════════════
section("SAVING SELECTED DATASET")

output_cols = ALWAYS_KEEP + selected_features
df_selected = df[output_cols].copy()
df_selected.to_csv(OUTPUT_FILE, index=False)
print(f"  ✅  Saved: {OUTPUT_FILE}")
print(f"  Shape: {df_selected.shape[0]:,} rows × {df_selected.shape[1]} columns "
      f"({len(ALWAYS_KEEP)} kept + {len(selected_features)} selected)")


# ══════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════
report = f"""
================================================================================
  FEATURE SELECTION REPORT — SAMSUNG HEALTH PROJECT
  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

STARTING POINT
  master_featured.csv: 108 columns total
  Numeric candidates considered for selection: {len(numeric_cols) - len(ALWAYS_KEEP) - len(LEAKAGE_COLS)}
  Excluded upfront as leakage (would let the model "cheat"): {len(LEAKAGE_COLS)}

STEP 1 — High-null drop (> {NULL_THRESHOLD*100:.0f}% missing)
  Dropped: {len(dropped_null)}
  {', '.join(dropped_null.index.tolist()) if len(dropped_null) else '(none)'}

STEP 2 — Redundant feature drop (|corr| > {CORR_THRESHOLD})
  Dropped: {len(dropped_corr)}
  {', '.join(dropped_corr.keys()) if dropped_corr else '(none)'}

STEP 3 — Kept top {TOP_N} by Random Forest importance
{chr(10).join(f'  {i+1}. {feat}  ({imp:.4f})' for i, (feat, imp) in enumerate(importance.head(TOP_N).items()))}

STEP 4 — Model accuracy: full set vs selected set (test year: 2025)
  Full feature set ({len(candidates)} features):     {f'{acc_full*100:.1f}%' if acc_full is not None else 'N/A'}
  Selected feature set ({TOP_N} features):            {f'{acc_selected*100:.1f}%' if acc_selected is not None else 'N/A'}

RESULT
  master_selected.csv → {df_selected.shape[0]:,} rows × {df_selected.shape[1]} columns
  Reduced from 108 → {len(output_cols)} columns ({len(ALWAYS_KEEP)} identifiers/targets + {TOP_N} features)

================================================================================
"""
print(report)
with open(os.path.join(OUTPUT_DIR, 'feature_selection_report.txt'), 'w') as f:
    f.write(report)

print("=" * 70)
print("  ✅  FEATURE SELECTION COMPLETE")
print("=" * 70)
print(f"\n  Next step: point 05_eda.py / 06_ml_models.py at master_selected.csv")
print(f"  to confirm cleaner inputs produce equal-or-better results.")
