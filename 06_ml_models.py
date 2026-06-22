"""
====================================================================
  SCRIPT 6 — MACHINE LEARNING MODELS
  Samsung Health Personal Analytics Project
  Author: Atharva
  Phase 5: Machine Learning

  MODELS BUILT:
  ─────────────────────────────────────────────────────────────────
  1. Week Classifier (Random Forest)
     → Labels each week as: fat_loss / maintenance / weight_gain
     → Shows which behaviors separate fat loss weeks from others

  2. Feature Importance Analysis
     → Which inputs (sleep, steps, calories, workouts) matter most
     → Separate importance for classification vs weight prediction

  3. Goal Achievement Simulation
     → Projects weight to 85 kg under 4 lifestyle scenarios
     → Calculates exact weeks needed under each strategy

  OUTPUT FILES:
  ─────────────────────────────────────────────────────────────────
  ml_charts/
    01_confusion_matrix.png
    02_feature_importance_classifier.png
    03_feature_importance_weight.png
    04_goal_simulation.png
    05_scenario_comparison.png
  ml_results/
    week_predictions.csv
    simulation_results.csv
    model_summary.txt
====================================================================

  DEPENDENCIES:
  pip install scikit-learn
====================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Dependency check ────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  mean_absolute_error, mean_squared_error, r2_score)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score
except ModuleNotFoundError:
    print("\n❌  scikit-learn not found.")
    print("    Install it with:  pip install scikit-learn")
    print("    Then re-run this script.\n")
    sys.exit(1)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(SCRIPT_DIR, "master_featured.csv")
CHART_DIR    = os.path.join(SCRIPT_DIR, "ml_charts")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "ml_results")

os.makedirs(CHART_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Personal constants
GOAL_WEIGHT_KG   = 85.0
HEIGHT_CM        = 182.9
KCAL_PER_KG_FAT  = 7700   # physiological constant

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi']      = 150
plt.rcParams['font.size']       = 11
plt.rcParams['axes.titlesize']  = 14
plt.rcParams['axes.labelsize']  = 12

COLORS = {
    'blue':    '#2563EB',
    'orange':  '#F59E0B',
    'red':     '#EF4444',
    'green':   '#10B981',
    'purple':  '#8B5CF6',
    'gray':    '#6B7280',
}

chart_count = 0
def save_chart(fig, name):
    global chart_count
    chart_count += 1
    path = os.path.join(CHART_DIR, f"{chart_count:02d}_{name}.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      📊 Saved: {path}")
    return path

def section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
section("LOADING DATA")

df = pd.read_csv(INPUT_FILE, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Current weight: {df['weight'].iloc[-1]:.1f} kg")
print(f"  Goal weight:    {GOAL_WEIGHT_KG:.1f} kg")
print(f"  To lose:        {df['weight'].iloc[-1] - GOAL_WEIGHT_KG:.1f} kg")

# ══════════════════════════════════════════════════════════════════
#  PART 1 — WEEK CLASSIFIER
#  Goal: predict fat_loss / maintenance / weight_gain from behaviors
# ══════════════════════════════════════════════════════════════════
section("PART 1 — WEEK CLASSIFIER (Random Forest)")

# ── 1a. Feature selection (no leakage) ──────────────────────────
# We use BEHAVIORAL inputs only — no current-week weight change
# because that IS the label. We look at habits from the same week
# using lagged weight to anchor context.

CLASSIFIER_FEATURES = [
    # Activity
    'step_count',
    'exercise_duration',
    'workout_intensity_score',
    'weekly_workout_count',
    'weekly_training_volume',
    'consecutive_workout_days',
    'rest_days_since_workout',
    'floor_count',
    # Sleep
    'sleep_hours',
    'sleep_debt_7d',
    'sleep_consistency_7d',
    'prev_night_sleep',
    # Heart rate
    'avg_heart_rate',
    # Energy & calories
    'total_calories_burned',
    'net_calorie_balance',
    'calorie_balance_7d_avg',
    'water_ml',
    # Lagged weight (context, not leakage)
    'weight_lag_7d',
    'weight_lag_14d',
    # Previous week averages
    'steps_avg_prev_week',
    'calories_avg_prev_week',
    'sleep_avg_prev_week',
    'workouts_prev_14d',
    # Time
    'month',
    'is_weekend',
    'days_since_start',
]

# ── 1b. Prepare data ────────────────────────────────────────────
# Drop 'unknown' labels (only 10 rows at edges of date range)
clf_df = df[df['week_label'].isin(['fat_loss', 'maintenance', 'weight_gain'])].copy()
clf_df = clf_df.dropna(subset=CLASSIFIER_FEATURES + ['week_label'])

X_clf = clf_df[CLASSIFIER_FEATURES].fillna(0)
y_clf = clf_df['week_label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_clf)
label_names = le.classes_   # ['fat_loss', 'maintenance', 'weight_gain']

print(f"\n  Training samples: {len(X_clf):,}")
print(f"  Label distribution:")
for lbl, cnt in y_clf.value_counts().items():
    pct = cnt / len(y_clf) * 100
    print(f"    {lbl:15s}: {cnt:5d} ({pct:.1f}%)")

# ── 1c. Time-based train/test split ─────────────────────────────
# Train on 2019-2024, test on 2025 → prevents future data leakage
split_date = pd.Timestamp('2025-01-01')
train_mask = clf_df['date'] < split_date
test_mask  = clf_df['date'] >= split_date

X_train = X_clf[train_mask]
X_test  = X_clf[test_mask]
y_train = y_encoded[train_mask]
y_test  = y_encoded[test_mask]

print(f"\n  Train set: {len(X_train):,} rows  (2019 – 2024)")
print(f"  Test set:  {len(X_test):,} rows  (2025)")

# ── 1d. Train Random Forest Classifier ──────────────────────────
print("\n  Training Random Forest Classifier...")
rfc = RandomForestClassifier(
    n_estimators   = 300,
    max_depth      = 8,
    min_samples_leaf = 20,
    class_weight   = 'balanced',   # handles imbalanced fat_loss class
    random_state   = 42,
    n_jobs         = -1,
)
rfc.fit(X_train, y_train)

# Cross-val on training set
cv_scores = cross_val_score(rfc, X_train, y_train, cv=5, scoring='accuracy')
print(f"  Cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Test set evaluation
y_pred = rfc.predict(X_test)
test_acc = (y_pred == y_test).mean()
print(f"  Test accuracy:      {test_acc:.3f}")

print(f"\n  Classification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=label_names))

# Add predictions to full dataframe
clf_df_copy = clf_df.copy()
clf_df_copy['predicted_label'] = le.inverse_transform(rfc.predict(X_clf.fillna(0)))
clf_df_copy[['date', 'week_label', 'predicted_label']].to_csv(
    os.path.join(RESULTS_DIR, 'week_predictions.csv'), index=False
)
print(f"  Saved predictions → ml_results/week_predictions.csv")

# ── 1e. Confusion Matrix Chart ──────────────────────────────────
print("\n  Generating: Confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_names,
    yticklabels=label_names,
    ax=ax,
    linewidths=0.5,
    annot_kws={'size': 14, 'weight': 'bold'},
)
ax.set_title('Week Classifier — Confusion Matrix (2025 Test Set)', fontsize=14, pad=15)
ax.set_ylabel('Actual Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
save_chart(fig, 'confusion_matrix')


# ══════════════════════════════════════════════════════════════════
#  PART 2 — WEIGHT PREDICTOR (for simulation calibration)
#  Predicts weight 7 days ahead using lagged behavioral features
# ══════════════════════════════════════════════════════════════════
section("PART 2 — WEIGHT PREDICTOR (7-Day Forecast)")

# ── 2a. Create target: weight 7 days from now ───────────────────
reg_df = df.copy()
reg_df['target_weight_7d'] = reg_df['weight'].shift(-7)   # look ahead 7 days

REGRESSOR_FEATURES = [
    # Current weight context
    'weight',
    'weight_7d_avg',
    'weight_14d_avg',
    'weight_30d_avg',
    'bmi',
    # Behavioral inputs
    'step_count',
    'steps_7d_avg',
    'sleep_hours',
    'sleep_7d_avg',
    'sleep_debt_7d',
    'avg_heart_rate',
    'hr_7d_avg',
    'net_calorie_balance',
    'calorie_balance_7d_avg',
    'exercise_duration',
    'workout_intensity_score',
    'weekly_workout_count',
    # Lag features
    'weight_lag_7d',
    'weight_lag_14d',
    'steps_lag_7d',
    'calories_avg_prev_week',
    # Time
    'days_since_start',
    'month',
    'year',
    'is_weekend',
]

reg_df = reg_df.dropna(subset=REGRESSOR_FEATURES + ['target_weight_7d'])
X_reg = reg_df[REGRESSOR_FEATURES].fillna(0)
y_reg = reg_df['target_weight_7d']

# Time-based split
train_mask_r = reg_df['date'] < pd.Timestamp('2025-01-01')
test_mask_r  = reg_df['date'] >= pd.Timestamp('2025-01-01')

X_train_r = X_reg[train_mask_r]
X_test_r  = X_reg[test_mask_r]
y_train_r = y_reg[train_mask_r]
y_test_r  = y_reg[test_mask_r]

print(f"\n  Training Random Forest Regressor...")
rfr = RandomForestRegressor(
    n_estimators    = 300,
    max_depth       = 10,
    min_samples_leaf = 15,
    random_state    = 42,
    n_jobs          = -1,
)
rfr.fit(X_train_r, y_train_r)

y_pred_r    = rfr.predict(X_test_r)
mae         = mean_absolute_error(y_test_r, y_pred_r)
rmse        = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2          = r2_score(y_test_r, y_pred_r)

print(f"  MAE  (mean absolute error): {mae:.3f} kg")
print(f"  RMSE (root mean sq error):  {rmse:.3f} kg")
print(f"  R²   (explained variance):  {r2:.3f}")
print(f"\n  Interpretation:")
print(f"    On average, the model predicts your weight in 7 days")
print(f"    within ±{mae:.2f} kg  (R² = {r2:.3f})")


# ══════════════════════════════════════════════════════════════════
#  PART 3 — FEATURE IMPORTANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════
section("PART 3 — FEATURE IMPORTANCE ANALYSIS")

# ── 3a. Classifier feature importance ───────────────────────────
print("\n  Top features for WEEK CLASSIFICATION:")
feat_imp_clf = pd.Series(rfc.feature_importances_, index=CLASSIFIER_FEATURES)
feat_imp_clf = feat_imp_clf.sort_values(ascending=False)

for i, (feat, imp) in enumerate(feat_imp_clf.head(10).items()):
    bar = '█' * int(imp * 200)
    print(f"    {i+1:2d}. {feat:35s}  {imp:.4f}  {bar}")

# Chart — Classifier importance
fig, ax = plt.subplots(figsize=(10, 8))
top15_clf = feat_imp_clf.head(15)
colors_bar = [COLORS['blue'] if i < 5 else COLORS['gray'] for i in range(len(top15_clf))]
bars = ax.barh(
    top15_clf.index[::-1],
    top15_clf.values[::-1],
    color=colors_bar[::-1],
    edgecolor='white',
    linewidth=0.5,
)
for bar_item, val in zip(bars, top15_clf.values[::-1]):
    ax.text(bar_item.get_width() + 0.001, bar_item.get_y() + bar_item.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, color='#374151')
ax.set_title('Feature Importance — Week Classifier\n(What behaviors predict fat loss vs maintenance vs gain weeks?)',
             fontsize=13, pad=15)
ax.set_xlabel('Importance Score (higher = more predictive)', fontsize=11)
ax.axvline(x=feat_imp_clf.values[4], color='red', linestyle='--', alpha=0.4, linewidth=1)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_chart(fig, 'feature_importance_classifier')

# ── 3b. Regressor feature importance ────────────────────────────
print("\n  Top features for WEIGHT PREDICTION (7-day):")
feat_imp_reg = pd.Series(rfr.feature_importances_, index=REGRESSOR_FEATURES)
feat_imp_reg = feat_imp_reg.sort_values(ascending=False)

for i, (feat, imp) in enumerate(feat_imp_reg.head(10).items()):
    bar = '█' * int(imp * 200)
    print(f"    {i+1:2d}. {feat:35s}  {imp:.4f}  {bar}")

# Chart — Regressor importance
fig, ax = plt.subplots(figsize=(10, 8))
top15_reg = feat_imp_reg.head(15)
colors_bar2 = [COLORS['green'] if i < 5 else COLORS['gray'] for i in range(len(top15_reg))]
bars2 = ax.barh(
    top15_reg.index[::-1],
    top15_reg.values[::-1],
    color=colors_bar2[::-1],
    edgecolor='white',
    linewidth=0.5,
)
for bar_item, val in zip(bars2, top15_reg.values[::-1]):
    ax.text(bar_item.get_width() + 0.001, bar_item.get_y() + bar_item.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, color='#374151')
ax.set_title('Feature Importance — Weight Predictor (7-Day)\n(What factors most influence your weight next week?)',
             fontsize=13, pad=15)
ax.set_xlabel('Importance Score (higher = more influential)', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_chart(fig, 'feature_importance_weight')


# ══════════════════════════════════════════════════════════════════
#  PART 4 — GOAL ACHIEVEMENT SIMULATION
#  Physiological model: 7,700 kcal deficit = 1 kg lost
#  Models 4 lifestyle scenarios forward in time
# ══════════════════════════════════════════════════════════════════
section("PART 4 — GOAL ACHIEVEMENT SIMULATION")

# ── 4a. Current baseline metrics ────────────────────────────────
current_weight   = float(df['weight'].iloc[-1])
current_date     = pd.Timestamp(df['date'].iloc[-1])
avg_deficit_real = float(df['net_calorie_balance'].tail(90).mean())   # last 90 days

print(f"\n  Starting point:")
print(f"    Current weight:        {current_weight:.1f} kg")
print(f"    Goal weight:           {GOAL_WEIGHT_KG:.1f} kg")
print(f"    Need to lose:          {current_weight - GOAL_WEIGHT_KG:.1f} kg")
print(f"    Avg daily balance      {avg_deficit_real:+.0f} kcal/day  (last 90 days)")
print(f"\n  Physics: 1 kg fat = {KCAL_PER_KG_FAT:,} kcal deficit")

# ── 4b. Define 4 scenarios ──────────────────────────────────────
# net_deficit > 0 means surplus (gain), < 0 means deficit (loss)
SCENARIOS = {
    'Current Routine':    avg_deficit_real,           # ~+4 kcal/day
    'Better Diet\n(-300 kcal intake)':   -300,        # reduce food by 300 kcal
    'More Cardio\n(+3K steps daily)':    -175,        # +3000 steps ≈ +175 kcal burned
    'Optimal\n(Diet + Cardio)':          -500,        # combined: best realistic scenario
}

SCENARIO_COLORS = [COLORS['gray'], COLORS['orange'], COLORS['blue'], COLORS['green']]
SCENARIO_STYLES = ['--', '-.', ':', '-']

# ── 4c. Simulate weight trajectories ────────────────────────────
np.random.seed(42)
SIM_DAYS     = 600     # simulate 600 days forward (~1.6 years)
results      = {}
summary_rows = []

print(f"\n  Simulating {SIM_DAYS} days forward:")
print(f"  {'Scenario':<30s} {'Daily deficit':>14s} {'Weeks to goal':>14s} {'Date reached':>14s}")
print(f"  {'-'*76}")

for i, (name, daily_deficit) in enumerate(SCENARIOS.items()):
    weights      = [current_weight]
    goal_day     = None
    goal_date    = None

    for day in range(1, SIM_DAYS + 1):
        # Physiological daily weight change
        daily_kg_change   = -daily_deficit / KCAL_PER_KG_FAT   # negative deficit = loss
        # Add realistic daily noise (water retention, digestion, etc.)
        noise             = np.random.normal(0, 0.15)
        new_weight        = weights[-1] + daily_kg_change + noise
        # Hard floor — won't go below 60 kg in simulation
        new_weight        = max(new_weight, 60.0)
        weights.append(new_weight)

        if goal_day is None and new_weight <= GOAL_WEIGHT_KG:
            goal_day  = day
            goal_date = current_date + pd.Timedelta(days=day)

    weeks_to_goal = goal_day / 7 if goal_day else None
    results[name] = weights

    clean_name = name.replace('\n', ' ')
    weeks_str  = f"{weeks_to_goal:.1f} wks" if weeks_to_goal else ">600 days"
    date_str   = goal_date.strftime('%b %Y') if goal_date else "N/A"
    print(f"  {clean_name:<30s} {daily_deficit:>+12.0f} kcal {weeks_str:>14s} {date_str:>14s}")

    summary_rows.append({
        'scenario':          clean_name,
        'daily_deficit_kcal': daily_deficit,
        'weeks_to_goal':     round(weeks_to_goal, 1) if weeks_to_goal else None,
        'estimated_date':    str(goal_date.date()) if goal_date else 'N/A',
    })

# Save summary
sim_summary = pd.DataFrame(summary_rows)
sim_summary.to_csv(os.path.join(RESULTS_DIR, 'simulation_results.csv'), index=False)
print(f"\n  Saved simulation → ml_results/simulation_results.csv")

# ── 4d. Plot: Weight trajectory for all 4 scenarios ─────────────
print("\n  Generating: Goal simulation chart...")
fig, ax = plt.subplots(figsize=(16, 7))

sim_dates = [current_date + pd.Timedelta(days=d) for d in range(SIM_DAYS + 1)]

for (name, weights), color, style in zip(results.items(), SCENARIO_COLORS, SCENARIO_STYLES):
    # Smooth the noisy simulation with rolling average for cleaner viz
    w_series = pd.Series(weights).rolling(14, min_periods=1).mean()
    clean_name = name.replace('\n', ' ')
    ax.plot(sim_dates, w_series, label=clean_name, color=color,
            linestyle=style, linewidth=2.5, alpha=0.9)

# Goal line
ax.axhline(y=GOAL_WEIGHT_KG, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Goal: {GOAL_WEIGHT_KG} kg')

# Current date marker
ax.axvline(x=current_date, color='black', linestyle=':', linewidth=1, alpha=0.5)
ax.text(current_date, current_weight + 0.5, ' Now', fontsize=10, color='black', va='bottom')

# Shaded goal zone
ax.axhspan(GOAL_WEIGHT_KG - 1, GOAL_WEIGHT_KG, alpha=0.07, color='green', label='Goal zone (±1 kg)')

ax.set_title('Goal Achievement Simulation — 4 Lifestyle Scenarios\nProjected weight trajectory to 85 kg goal',
             fontsize=14, pad=15)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Weight (kg)', fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.set_ylim(80, current_weight + 3)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_chart(fig, 'goal_simulation')

# ── 4e. Scenario comparison bar chart ───────────────────────────
print("  Generating: Scenario comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: weeks to goal
clean_names_short = [name.replace('\n', '\n') for name in SCENARIOS.keys()]
weeks_list  = [r['weeks_to_goal'] for r in summary_rows]
colors_list = SCENARIO_COLORS

ax1 = axes[0]
bars_w = ax1.barh(
    range(len(SCENARIOS)),
    [w if w else SIM_DAYS / 7 for w in weeks_list],
    color=colors_list,
    edgecolor='white',
    linewidth=0.5,
)
ax1.set_yticks(range(len(SCENARIOS)))
ax1.set_yticklabels(clean_names_short, fontsize=10)
ax1.set_xlabel('Weeks to reach 85 kg', fontsize=11)
ax1.set_title('Weeks to Goal Weight\n(85 kg)', fontsize=13, pad=12)
for bar_item, r in zip(bars_w, summary_rows):
    label = f"{r['weeks_to_goal']} wks" if r['weeks_to_goal'] else ">86 wks"
    ax1.text(bar_item.get_width() + 1, bar_item.get_y() + bar_item.get_height()/2,
             label, va='center', fontsize=10, fontweight='bold')
ax1.spines[['top', 'right']].set_visible(False)
ax1.set_xlim(0, max(w for w in weeks_list if w) * 1.35 if any(weeks_list) else 200)

# Right: daily deficit required
ax2 = axes[1]
deficits = list(SCENARIOS.values())
bar_colors2 = [COLORS['red'] if d > 0 else COLORS['green'] for d in deficits]
bars_d = ax2.barh(
    range(len(SCENARIOS)),
    deficits,
    color=bar_colors2,
    edgecolor='white',
)
ax2.set_yticks(range(len(SCENARIOS)))
ax2.set_yticklabels(clean_names_short, fontsize=10)
ax2.set_xlabel('Daily Calorie Balance (kcal)', fontsize=11)
ax2.set_title('Daily Calorie Strategy\n(negative = deficit = fat loss)', fontsize=13, pad=12)
ax2.axvline(0, color='black', linewidth=0.8, alpha=0.5)
for bar_item, val in zip(bars_d, deficits):
    offset = -15 if val < 0 else 5
    ax2.text(val + offset, bar_item.get_y() + bar_item.get_height()/2,
             f'{val:+.0f}', va='center', fontsize=10, fontweight='bold', color='white')
ax2.spines[['top', 'right']].set_visible(False)

plt.suptitle('Scenario Comparison — What Will It Take to Reach 85 kg?',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_chart(fig, 'scenario_comparison')


# ══════════════════════════════════════════════════════════════════
#  PART 5 — MODEL SUMMARY & INSIGHTS
# ══════════════════════════════════════════════════════════════════
section("ML SUMMARY & KEY INSIGHTS")

# Compute weekly fat loss rate for optimal scenario
optimal_deficit = -500
weekly_kg_loss  = (optimal_deficit * -7) / KCAL_PER_KG_FAT
optimal_weeks   = (current_weight - GOAL_WEIGHT_KG) / weekly_kg_loss

summary_text = f"""
================================================================================
  ML MODEL RESULTS — SAMSUNG HEALTH ANALYTICS PROJECT
  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

DATA SUMMARY:
  Total days analyzed:    {len(df):,}
  Date range:             {df['date'].min().date()} → {df['date'].max().date()}
  Current weight:         {current_weight:.1f} kg
  Goal weight:            {GOAL_WEIGHT_KG:.1f} kg
  Weight to lose:         {current_weight - GOAL_WEIGHT_KG:.1f} kg

MODEL 1 — WEEK CLASSIFIER (Random Forest):
  Task:     Predict fat_loss / maintenance / weight_gain from daily habits
  Accuracy: {test_acc:.1%}  (tested on 2025 data unseen during training)
  Training: 2019 – 2024  |  Test: 2025

  Top 5 behavioral predictors of fat loss weeks:
{chr(10).join(f'    {i+1}. {feat}  ({imp:.3f})' for i, (feat, imp) in enumerate(feat_imp_clf.head(5).items()))}

MODEL 2 — WEIGHT PREDICTOR (Random Forest Regressor):
  Task:     Predict weight 7 days into the future
  MAE:      {mae:.3f} kg  (average prediction error)
  RMSE:     {rmse:.3f} kg
  R²:       {r2:.3f}  (variance explained)

  Top 5 drivers of next week's weight:
{chr(10).join(f'    {i+1}. {feat}  ({imp:.3f})' for i, (feat, imp) in enumerate(feat_imp_reg.head(5).items()))}

SIMULATION RESULTS:
"""
for r in summary_rows:
    wks = f"{r['weeks_to_goal']} weeks" if r['weeks_to_goal'] else 'Never (>600 days)'
    summary_text += f"  {r['scenario']:<30s}  {r['daily_deficit_kcal']:>+5.0f} kcal/day  →  {wks}\n"

summary_text += f"""
KEY INSIGHTS:
  1. Your average daily calorie balance (last 90 days) is {avg_deficit_real:+.0f} kcal
     → This is essentially MAINTENANCE — no fat loss will happen at this rate.

  2. To reach 85 kg, you need a sustained calorie DEFICIT.
     → Optimal scenario (-500 kcal/day): ~{optimal_weeks:.0f} weeks = {optimal_weeks/4.3:.1f} months

  3. Top behavioral predictors of your fat loss weeks:
     → {feat_imp_clf.index[0]} and {feat_imp_clf.index[1]} are your strongest levers.

  4. Current plateau days: {df['is_plateau'].sum():,} out of {len(df):,} ({df['is_plateau'].mean()*100:.0f}%)
     → More than one-third of your journey has been stalled.

RECOMMENDATIONS (data-driven):
  → Create a 300–500 kcal daily deficit (track food accurately)
  → Add 3,000+ steps/day on top of current average ({df['step_count'].mean():.0f} steps)
  → Aim for 2+ structured workouts per week (current: {df['weekly_workout_count'].mean():.1f}/wk)
  → Prioritize 7.5 hrs sleep (current avg: {df['sleep_hours'].mean():.1f} hrs)

================================================================================
"""

print(summary_text)

# Save to file
with open(os.path.join(RESULTS_DIR, 'model_summary.txt'), 'w') as f:
    f.write(summary_text)
print(f"  Saved summary → ml_results/model_summary.txt")


# ── Final tally ─────────────────────────────────────────────────
print("\n" + "="*70)
print("  ✅  PHASE 5 COMPLETE — ML MODELS")
print("="*70)
print(f"\n  Charts saved to: ml_charts/")
for i in range(1, chart_count + 1):
    files = [f for f in os.listdir(CHART_DIR) if f.startswith(f"{i:02d}_")]
    if files:
        print(f"    📊  {files[0]}")

print(f"\n  Results saved to: ml_results/")
print(f"    📄  week_predictions.csv")
print(f"    📄  simulation_results.csv")
print(f"    📄  model_summary.txt")

print(f"""
  NEXT STEPS:
    → Phase 6: Import master_featured.csv into Power BI
    → Phase 7: AI Simulation layer (scenario optimizer)
""")
