"""
====================================================================
  SCRIPT 4 — ADVANCED FEATURE ENGINEERING
  Samsung Health Personal Analytics Project
  Author: Atharva
  Description: Reads master_daily_augmented.csv and adds 30+ advanced
               features for ML models and Power BI dashboards.
               Outputs: master_featured.csv
====================================================================

  FEATURES ADDED:
  ─────────────────────────────────────────────────────────────────
  BODY         → BMI, weight change (7d/14d/30d), weight momentum,
                 weight zone, distance from goal
  WORKOUT      → intensity score, consecutive days, rest gap,
                 weekly count, monthly count, training volume
  SLEEP        → quality category, sleep debt, prev night sleep,
                 sleep consistency, sleep-to-step ratio
  ACTIVITY     → activity level, calories per step, step zone,
                 active minutes category
  ENERGY       → net calorie balance, TDEE estimate, surplus/deficit
  LAG FEATURES → weight/steps/sleep/HR lagged 7, 14, 30 days
  TREND        → 30-day rolling avgs, weight momentum, plateau flag
  TIME         → days since start, month progress, season
====================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
INPUT_FILE  = "master_daily_augmented.csv"
OUTPUT_FILE = "master_featured.csv"

# ─────────────────────────────────────────────
#  YOUR PERSONAL STATS (edit these if needed)
# ─────────────────────────────────────────────
HEIGHT_CM       = 182.9      # your height in cm
HEIGHT_M        = HEIGHT_CM / 100
AGE_APPROX      = 22         # approximate age
GOAL_WEIGHT_KG  = 85.0       # your target weight
DAILY_INTAKE_EST = 2400      # estimated daily calorie intake (kcal)
SLEEP_TARGET_HRS = 7.5       # recommended sleep target

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
print("="*65)
print("  LOADING AUGMENTED MASTER TABLE")
print("="*65)

df = pd.read_csv(INPUT_FILE, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

original_cols = len(df.columns)
print(f"  Rows: {len(df):,} | Columns: {original_cols}")
print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")


# ══════════════════════════════════════════════════════════════════
#  1. BODY COMPOSITION FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n[1/8] Adding: Body Composition features...")

# BMI = weight (kg) / height (m)²
df['bmi'] = (df['weight'] / (HEIGHT_M ** 2)).round(2)

# BMI category
def bmi_category(bmi):
    if pd.isna(bmi):
        return 'unknown'
    elif bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'

df['bmi_category'] = df['bmi'].apply(bmi_category)

# Weight change over different periods
df['weight_change_7d']  = (df['weight'] - df['weight'].shift(7)).round(2)
df['weight_change_14d'] = (df['weight'] - df['weight'].shift(14)).round(2)
df['weight_change_30d'] = (df['weight'] - df['weight'].shift(30)).round(2)

# Distance from goal weight
df['distance_to_goal'] = (df['weight'] - GOAL_WEIGHT_KG).round(2)

# Estimated weeks to reach goal (based on 30-day rate of change)
df['weight_rate_per_week'] = (df['weight_change_30d'] / 4.3).round(3)  # 30 days ≈ 4.3 weeks
df['est_weeks_to_goal'] = np.where(
    df['weight_rate_per_week'] < -0.1,    # only if actively losing weight
    (df['distance_to_goal'] / df['weight_rate_per_week']).round(1),
    np.nan                                 # no estimate if not in deficit
)
# Cap at reasonable range
df['est_weeks_to_goal'] = df['est_weeks_to_goal'].clip(lower=0, upper=200)

# Weight zone — where are you relative to your range?
weight_min = df['weight'].min()
weight_max = df['weight'].max()
weight_range = weight_max - weight_min
df['weight_zone'] = np.where(
    df['weight'] < weight_min + weight_range * 0.33, 'low',
    np.where(df['weight'] < weight_min + weight_range * 0.66, 'mid', 'high')
)

# 30-day rolling weight average (for smoother trend)
df['weight_30d_avg'] = df['weight'].rolling(30, min_periods=7).mean().round(2)

print(f"      BMI range: {df['bmi'].min():.1f} – {df['bmi'].max():.1f}")
print(f"      Current distance to goal ({GOAL_WEIGHT_KG} kg): {df['distance_to_goal'].iloc[-1]:.1f} kg")


# ══════════════════════════════════════════════════════════════════
#  2. WORKOUT INTELLIGENCE FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n[2/8] Adding: Workout Intelligence features...")

# Workout intensity score (0–100)
# Combines duration, calories, and heart rate into a single metric
if 'exercise_duration' in df.columns and 'exercise_calories' in df.columns:
    # Normalize each component to 0–1 range
    dur_max = df.loc[df['exercise_duration'] > 0, 'exercise_duration'].quantile(0.95) if (df['exercise_duration'] > 0).any() else 60
    cal_max = df.loc[df['exercise_calories'] > 0, 'exercise_calories'].quantile(0.95) if (df['exercise_calories'] > 0).any() else 500

    dur_norm = (df['exercise_duration'] / dur_max).clip(0, 1)
    cal_norm = (df['exercise_calories'] / cal_max).clip(0, 1)

    # Heart rate component (use exercise HR if available, else avg HR)
    hr_col = 'avg_heart_rate_ex' if 'avg_heart_rate_ex' in df.columns else 'avg_heart_rate'
    hr_max = df[hr_col].quantile(0.95) if df[hr_col].notna().any() else 160
    hr_min = df[hr_col].quantile(0.05) if df[hr_col].notna().any() else 60
    hr_norm = ((df[hr_col].fillna(0) - hr_min) / (hr_max - hr_min)).clip(0, 1)

    # Weighted score: 40% duration, 35% calories, 25% heart rate
    df['workout_intensity_score'] = (
        (dur_norm * 40 + cal_norm * 35 + hr_norm * 25) * df['workout_day']
    ).round(1)
else:
    df['workout_intensity_score'] = 0

# Consecutive workout days (streak)
df['consecutive_workout_days'] = 0
streak = 0
for i in range(len(df)):
    if df.loc[i, 'workout_day'] == 1:
        streak += 1
    else:
        streak = 0
    df.loc[i, 'consecutive_workout_days'] = streak

# Rest days since last workout
df['rest_days_since_workout'] = 0
last_workout = -999
for i in range(len(df)):
    if df.loc[i, 'workout_day'] == 1:
        last_workout = i
        df.loc[i, 'rest_days_since_workout'] = 0
    else:
        df.loc[i, 'rest_days_since_workout'] = i - last_workout if last_workout >= 0 else 0

# Rolling weekly workout count (last 7 days)
df['weekly_workout_count'] = df['workout_day'].rolling(7, min_periods=1).sum().astype(int)

# Rolling monthly workout count (last 30 days)
df['monthly_workout_count'] = df['workout_day'].rolling(30, min_periods=1).sum().astype(int)

# Weekly training volume (sum of intensity scores in past 7 days)
df['weekly_training_volume'] = df['workout_intensity_score'].rolling(7, min_periods=1).sum().round(1)

# Overtraining risk flag (>5 consecutive days OR >6 workouts in 7 days)
df['overtraining_risk'] = (
    (df['consecutive_workout_days'] > 5) | (df['weekly_workout_count'] > 6)
).astype(int)

print(f"      Avg weekly workouts: {df['weekly_workout_count'].mean():.1f}")
print(f"      Max workout streak: {df['consecutive_workout_days'].max()} days")


# ══════════════════════════════════════════════════════════════════
#  3. SLEEP INTELLIGENCE FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n[3/8] Adding: Sleep Intelligence features...")

# Sleep quality category
def sleep_quality(hours, score=None):
    if pd.isna(hours):
        return 'unknown'
    elif hours < 5:
        return 'poor'
    elif hours < 6.5:
        return 'fair'
    elif hours < 8:
        return 'good'
    else:
        return 'excellent'

df['sleep_quality_category'] = df['sleep_hours'].apply(sleep_quality)

# Sleep debt — cumulative deficit from target over past 7 days
df['daily_sleep_deficit'] = df['sleep_hours'] - SLEEP_TARGET_HRS
df['sleep_debt_7d'] = df['daily_sleep_deficit'].rolling(7, min_periods=1).sum().round(2)

# Previous night's sleep (lag feature — very important for ML)
df['prev_night_sleep'] = df['sleep_hours'].shift(1).round(2)
df['prev_2_night_avg_sleep'] = df['sleep_hours'].rolling(2).mean().shift(1).round(2)

# Sleep consistency (std dev of sleep hours over past 7 days)
# Lower = more consistent sleep schedule = better
df['sleep_consistency_7d'] = df['sleep_hours'].rolling(7, min_periods=3).std().round(2)

# Is this a recovery night? (>8 hrs sleep after a bad night)
df['is_recovery_sleep'] = (
    (df['sleep_hours'] > 8) & (df['prev_night_sleep'] < 6)
).astype(int)

# Sleep efficiency trend (7-day rolling avg)
if 'efficiency' in df.columns:
    df['efficiency_7d_avg'] = df['efficiency'].rolling(7, min_periods=1).mean().round(1)

print(f"      Avg sleep debt (7d): {df['sleep_debt_7d'].mean():.1f} hrs")
print(f"      Sleep quality distribution: {df['sleep_quality_category'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════
#  4. ACTIVITY & ENERGY FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n[4/8] Adding: Activity & Energy features...")

# Activity level categories (based on daily steps)
def activity_level(steps):
    if pd.isna(steps):
        return 'unknown'
    elif steps < 3000:
        return 'sedentary'
    elif steps < 5000:
        return 'light'
    elif steps < 7500:
        return 'moderate'
    elif steps < 10000:
        return 'active'
    else:
        return 'very_active'

df['activity_level'] = df['step_count'].apply(activity_level)

# Calories per step (energy efficiency)
df['calories_per_step'] = np.where(
    df['step_count'] > 100,
    (df['total_calories_burned'] / df['step_count']).round(3),
    np.nan
)

# Active minutes category
if 'active_time_min' in df.columns:
    df['active_minutes_category'] = pd.cut(
        df['active_time_min'].fillna(0),
        bins=[-1, 15, 30, 60, 120, 9999],
        labels=['inactive', 'low', 'moderate', 'high', 'very_high']
    )

# Step zone (quick category for Power BI charts)
df['step_zone'] = pd.cut(
    df['step_count'],
    bins=[-1, 2000, 5000, 7500, 10000, 50000],
    labels=['very_low', 'low', 'moderate', 'high', 'very_high']
)

# Steps 30-day rolling average
df['steps_30d_avg'] = df['step_count'].rolling(30, min_periods=7).mean().round(0)

print(f"      Activity level distribution: {df['activity_level'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════
#  5. CALORIE & ENERGY BALANCE FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n[5/8] Adding: Calorie & Energy Balance features...")

# TDEE estimate from actual data (Total Daily Energy Expenditure)
if 'rest_calorie' in df.columns and 'active_calorie' in df.columns:
    df['tdee'] = (df['rest_calorie'].fillna(0) + df['active_calorie'].fillna(0)).round(0)
else:
    df['tdee'] = df['total_calories_burned']

# Net calorie balance = burned - intake estimate
df['net_calorie_balance'] = (df['total_calories_burned'] - DAILY_INTAKE_EST).round(0)

# Cumulative calorie deficit (running total from start)
df['cumulative_calorie_deficit'] = df['net_calorie_balance'].cumsum().round(0)

# Theoretical weight loss from calorie deficit (1 kg ≈ 7,700 kcal deficit)
df['theoretical_weight_loss_kg'] = (df['cumulative_calorie_deficit'] / 7700).round(2)

# 7-day average calorie balance (smoothed)
df['calorie_balance_7d_avg'] = df['net_calorie_balance'].rolling(7, min_periods=1).mean().round(0)

# Energy balance category
def energy_category(balance):
    if pd.isna(balance):
        return 'unknown'
    elif balance < -300:
        return 'deficit'
    elif balance < 300:
        return 'maintenance'
    else:
        return 'surplus'

df['energy_balance_category'] = df['calorie_balance_7d_avg'].apply(energy_category)

print(f"      Avg net calorie balance: {df['net_calorie_balance'].mean():.0f} kcal/day")
print(f"      Energy balance: {df['energy_balance_category'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════
#  6. LAG FEATURES (critical for ML time-series prediction)
# ══════════════════════════════════════════════════════════════════
print("\n[6/8] Adding: Lag features for ML...")

# Weight lags
df['weight_lag_7d']  = df['weight'].shift(7).round(2)
df['weight_lag_14d'] = df['weight'].shift(14).round(2)
df['weight_lag_30d'] = df['weight'].shift(30).round(2)

# Steps lags
df['steps_lag_7d']   = df['step_count'].shift(7)
df['steps_avg_prev_week']  = df['step_count'].rolling(7).mean().shift(1).round(0)
df['steps_avg_prev_month'] = df['step_count'].rolling(30).mean().shift(1).round(0)

# Sleep lags
df['sleep_lag_7d']   = df['sleep_hours'].shift(7).round(2)
df['sleep_avg_prev_week']  = df['sleep_hours'].rolling(7).mean().shift(1).round(2)

# Heart rate lags
df['hr_lag_7d']      = df['avg_heart_rate'].shift(7).round(1)
df['hr_avg_prev_week'] = df['avg_heart_rate'].rolling(7).mean().shift(1).round(1)

# Calories lags
df['calories_avg_prev_week']  = df['total_calories_burned'].rolling(7).mean().shift(1).round(0)
df['calories_avg_prev_month'] = df['total_calories_burned'].rolling(30).mean().shift(1).round(0)

# Workout frequency lag (how many workouts in last 14 days)
df['workouts_prev_14d'] = df['workout_day'].rolling(14).sum().shift(1)

print(f"      Added 14 lag features (7d, 14d, 30d lookbacks)")


# ══════════════════════════════════════════════════════════════════
#  7. TREND & MOMENTUM FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n[7/8] Adding: Trend & Momentum features...")

# Weight momentum — rate of change acceleration
# positive = weight change is accelerating (bad if gaining)
# negative = weight change is decelerating (good if cutting)
df['weight_momentum'] = (df['weight_change_7d'] - df['weight_change_7d'].shift(7)).round(3)

# Is weight in a plateau? (less than ±0.3 kg change over 14 days)
df['is_plateau'] = (df['weight_change_14d'].abs() < 0.3).astype(int)

# Plateau streak (consecutive days of plateau)
df['plateau_streak'] = 0
p_streak = 0
for i in range(len(df)):
    if df.loc[i, 'is_plateau'] == 1:
        p_streak += 1
    else:
        p_streak = 0
    df.loc[i, 'plateau_streak'] = p_streak

# Steps trend direction (is step count increasing or decreasing?)
df['steps_trend'] = np.where(
    df['steps_7d_avg'] > df['steps_14d_avg'] * 1.05, 'increasing',
    np.where(df['steps_7d_avg'] < df['steps_14d_avg'] * 0.95, 'decreasing', 'stable')
)

# Sleep trend direction
df['sleep_trend'] = np.where(
    df['sleep_7d_avg'] > df['sleep_7d_avg'].shift(7) + 0.3, 'improving',
    np.where(df['sleep_7d_avg'] < df['sleep_7d_avg'].shift(7) - 0.3, 'worsening', 'stable')
)

# HR trend (lower resting HR = improving fitness)
df['hr_trend'] = np.where(
    df['hr_7d_avg'] < df['hr_7d_avg'].shift(7) - 1, 'improving',
    np.where(df['hr_7d_avg'] > df['hr_7d_avg'].shift(7) + 1, 'worsening', 'stable')
)

print(f"      Plateau days detected: {df['is_plateau'].sum()} ({df['is_plateau'].mean()*100:.1f}%)")
print(f"      Max plateau streak: {df['plateau_streak'].max()} days")


# ══════════════════════════════════════════════════════════════════
#  8. TIME-BASED FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n[8/8] Adding: Time-based features...")

# Days since start of tracking
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Month progress (0.0 = start of month, 1.0 = end of month)
df['month_progress'] = (df['date'].dt.day / df['date'].dt.days_in_month).round(3)

# Season
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

df['season'] = df['month'].apply(get_season)

# Year-month string (useful for Power BI grouping)
df['year_month'] = df['date'].dt.strftime('%Y-%m')

# Week start date (Monday of each week — useful for weekly aggregations)
df['week_start'] = df['date'] - pd.to_timedelta(df['day_of_week'], unit='D')

print(f"      Total tracking period: {df['days_since_start'].max():,} days")


# ══════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ══════════════════════════════════════════════════════════════════
new_cols = len(df.columns) - original_cols

print("\n" + "="*65)
print("  FEATURE ENGINEERING COMPLETE")
print("="*65)
print(f"  Original columns:  {original_cols}")
print(f"  New features added: {new_cols}")
print(f"  Total columns:     {len(df.columns)}")
print(f"  Total rows:        {len(df):,}")

print(f"\n  New features by category:")
categories = {
    "Body Composition":   ['bmi','bmi_category','weight_change_7d','weight_change_14d',
                           'weight_change_30d','distance_to_goal','est_weeks_to_goal',
                           'weight_zone','weight_30d_avg','weight_rate_per_week'],
    "Workout Intelligence": ['workout_intensity_score','consecutive_workout_days',
                             'rest_days_since_workout','weekly_workout_count',
                             'monthly_workout_count','weekly_training_volume','overtraining_risk'],
    "Sleep Intelligence": ['sleep_quality_category','daily_sleep_deficit','sleep_debt_7d',
                           'prev_night_sleep','prev_2_night_avg_sleep','sleep_consistency_7d',
                           'is_recovery_sleep','efficiency_7d_avg'],
    "Activity & Energy":  ['activity_level','calories_per_step','step_zone','steps_30d_avg',
                           'tdee','net_calorie_balance','cumulative_calorie_deficit',
                           'theoretical_weight_loss_kg','calorie_balance_7d_avg',
                           'energy_balance_category'],
    "ML Lag Features":    ['weight_lag_7d','weight_lag_14d','weight_lag_30d','steps_lag_7d',
                           'steps_avg_prev_week','steps_avg_prev_month','sleep_lag_7d',
                           'sleep_avg_prev_week','hr_lag_7d','hr_avg_prev_week',
                           'calories_avg_prev_week','calories_avg_prev_month','workouts_prev_14d'],
    "Trends & Momentum":  ['weight_momentum','is_plateau','plateau_streak','steps_trend',
                           'sleep_trend','hr_trend'],
    "Time-Based":         ['days_since_start','month_progress','season','year_month','week_start'],
}

for cat, cols in categories.items():
    present = [c for c in cols if c in df.columns]
    print(f"    {cat:<25} → {len(present)} features")


# ══════════════════════════════════════════════════════════════════
#  SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*65}")
print(f"  ✅  FEATURED MASTER TABLE SAVED")
print(f"  File: {os.path.abspath(OUTPUT_FILE)}")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n  This file is ready for:")
print(f"    → Power BI / Tableau  (import master_featured.csv)")
print(f"    → ML Models           (weight prediction, classification)")
print(f"    → Scenario Simulation (calorie/weight projection)")
print(f"{'='*65}")
