"""
====================================================================
  SCRIPT 2 — BUILD MASTER DAILY TABLE
  Samsung Health Personal Analytics Project
  Author: Atharva
  Description: Merges all cleaned CSVs into one unified daily table.
               One row per date. This is the foundation for
               Power BI, Tableau, and all ML models.
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
CLEAN_DIR  = "cleaned"      # output of 01_clean_data.py
OUTPUT_DIR = "."            # where master CSV will be saved

# ─────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────
def load(filename):
    path = os.path.join(CLEAN_DIR, filename)
    if not os.path.exists(path):
        print(f"  ⚠  Skipping {filename} (not found)")
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    print(f"  ✓  Loaded {filename:<40} {len(df):>5} rows")
    return df


# ══════════════════════════════════════════════════════════════════
#  LOAD ALL CLEAN FILES
# ══════════════════════════════════════════════════════════════════
print("="*60)
print("  LOADING CLEAN FILES")
print("="*60)

df_weight   = load("weight_clean.csv")
df_sleep    = load("sleep_clean.csv")
df_exercise = load("exercise_clean.csv")
df_steps    = load("steps_clean.csv")
df_calories = load("calories_clean.csv")
df_water    = load("water_clean.csv")
df_hr       = load("heartrate_clean.csv")
df_stages   = load("sleep_stage_clean.csv")
df_floors   = load("floors_clean.csv")


# ══════════════════════════════════════════════════════════════════
#  BUILD FULL DATE SPINE
#  One row for every calendar date in the full range
# ══════════════════════════════════════════════════════════════════
all_dates = []
for df in [df_weight, df_sleep, df_exercise, df_steps,
           df_calories, df_water, df_hr, df_stages, df_floors]:
    if df is not None and 'date' in df.columns:
        all_dates.extend(df['date'].dropna().tolist())

min_date = pd.Timestamp('2019-06-12')   # your earliest Samsung data
max_date = max(all_dates) if all_dates else pd.Timestamp('2025-12-31')

date_spine = pd.DataFrame({
    'date': pd.date_range(start=min_date, end=max_date, freq='D')
})

print(f"\n  Full date range: {min_date.date()} → {max_date.date()}")
print(f"  Total days in spine: {len(date_spine):,}")


# ══════════════════════════════════════════════════════════════════
#  MERGE ALL DATASETS — LEFT JOIN on date spine
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MERGING DATASETS")
print("="*60)

master = date_spine.copy()

# -- Steps (most complete; use as primary activity reference)
if df_steps is not None:
    cols = ['date', 'step_count', 'distance_km', 'calorie', 'active_time_min',
            'run_step_count', 'walk_step_count']
    cols = [c for c in cols if c in df_steps.columns]
    master = master.merge(df_steps[cols], on='date', how='left')
    master.rename(columns={'calorie': 'step_calorie'}, inplace=True)
    print(f"  ✓ Steps merged")

# -- Calories Burned
if df_calories is not None:
    cols = ['date', 'rest_calorie', 'active_calorie', 'total_calories_burned']
    cols = [c for c in cols if c in df_calories.columns]
    master = master.merge(df_calories[cols], on='date', how='left')
    print(f"  ✓ Calories merged")

# -- Exercise
if df_exercise is not None:
    cols = ['date', 'exercise_sessions', 'exercise_calories',
            'exercise_duration', 'avg_heart_rate_ex']
    cols = [c for c in cols if c in df_exercise.columns]
    master = master.merge(df_exercise[cols], on='date', how='left')
    # Days with no exercise = 0 sessions (not missing)
    master['exercise_sessions']  = master['exercise_sessions'].fillna(0)
    master['exercise_calories']  = master['exercise_calories'].fillna(0)
    master['exercise_duration']  = master['exercise_duration'].fillna(0)
    master['workout_day']        = (master['exercise_sessions'] > 0).astype(int)
    print(f"  ✓ Exercise merged")

# -- Sleep
if df_sleep is not None:
    cols = ['date', 'sleep_hours', 'sleep_score', 'efficiency',
            'quality', 'physical_recovery', 'mental_recovery']
    cols = [c for c in cols if c in df_sleep.columns]
    master = master.merge(df_sleep[cols], on='date', how='left')
    print(f"  ✓ Sleep merged")

# -- Sleep Stage
if df_stages is not None:
    stage_cols = ['date'] + [c for c in df_stages.columns
                              if c not in ['date'] and 'sleep' in c.lower()]
    if len(stage_cols) > 1:
        master = master.merge(df_stages[stage_cols], on='date', how='left')
    print(f"  ✓ Sleep Stage merged")

# -- Heart Rate
if df_hr is not None:
    cols = ['date', 'avg_heart_rate', 'min_heart_rate',
            'max_heart_rate', 'hr_readings']
    cols = [c for c in cols if c in df_hr.columns]
    master = master.merge(df_hr[cols], on='date', how='left')
    print(f"  ✓ Heart Rate merged")

# -- Water Intake
if df_water is not None:
    master = master.merge(df_water[['date', 'water_ml']], on='date', how='left')
    print(f"  ✓ Water Intake merged")

# -- Floor Summary
if df_floors is not None:
    cols = ['date', 'floor_count']
    cols = [c for c in cols if c in df_floors.columns]
    master = master.merge(df_floors[cols], on='date', how='left')
    master['floor_count'] = master['floor_count'].fillna(0)
    print(f"  ✓ Floors merged")

# -- Weight (forward-fill between weigh-in dates for continuity)
if df_weight is not None:
    cols = ['date', 'weight']
    if 'body_fat' in df_weight.columns:
        cols.append('body_fat')
    master = master.merge(df_weight[cols], on='date', how='left')

    # FIX 3: Mark ONLY actual weigh-in dates as real BEFORE forward-filling
    # This flag tells Script 3 which of the 17 dates have genuine measurements
    master['weight_is_real'] = master['weight'].notna().astype(int)

    # Now forward fill weight (last known value carries forward to empty days)
    master['weight'] = master['weight'].ffill()
    print(f"  ✓ Weight merged (forward-filled)")
    print(f"     Real weigh-in days flagged: {master['weight_is_real'].sum()}")


# ══════════════════════════════════════════════════════════════════
#  ADD CALENDAR FEATURES (useful for Power BI and ML)
# ══════════════════════════════════════════════════════════════════
print("\n  Adding calendar features...")

master['year']         = master['date'].dt.year
master['month']        = master['date'].dt.month
master['month_name']   = master['date'].dt.strftime('%b')
master['week_of_year'] = master['date'].dt.isocalendar().week.astype(int)
master['day_of_week']  = master['date'].dt.dayofweek          # 0=Mon, 6=Sun
master['day_name']     = master['date'].dt.strftime('%A')
master['is_weekend']   = (master['day_of_week'] >= 5).astype(int)
master['quarter']      = master['date'].dt.quarter


# ══════════════════════════════════════════════════════════════════
#  ADD DERIVED FEATURES
# ══════════════════════════════════════════════════════════════════
print("  Adding derived features...")

# 7-day rolling averages
if 'step_count' in master.columns:
    master['steps_7d_avg']   = master['step_count'].rolling(7, min_periods=1).mean().round(0)
if 'sleep_hours' in master.columns:
    master['sleep_7d_avg']   = master['sleep_hours'].rolling(7, min_periods=1).mean().round(2)
if 'weight' in master.columns:
    master['weight_7d_avg']  = master['weight'].rolling(7, min_periods=1).mean().round(2)

# 14-day rolling averages
if 'step_count' in master.columns:
    master['steps_14d_avg']  = master['step_count'].rolling(14, min_periods=1).mean().round(0)
if 'weight' in master.columns:
    master['weight_14d_avg'] = master['weight'].rolling(14, min_periods=1).mean().round(2)

# Week number (sequential from start of data)
master['week_num'] = ((master['date'] - master['date'].min()).dt.days // 7).astype(int)

# Calorie deficit estimate (if rest_calorie exists)
# Assumes average food intake of 2400 kcal (can be refined)
ASSUMED_INTAKE = 2400
if 'total_calories_burned' in master.columns:
    master['calorie_deficit_est'] = master['total_calories_burned'] - ASSUMED_INTAKE


# ══════════════════════════════════════════════════════════════════
#  COMPLETENESS REPORT
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MASTER TABLE COMPLETENESS REPORT")
print("="*60)
print(f"  Total rows (days): {len(master):,}")
print(f"  Total columns:     {len(master.columns)}")
print(f"\n  Column fill rates:")

key_cols = ['weight', 'step_count', 'total_calories_burned',
            'sleep_hours', 'sleep_score', 'avg_heart_rate',
            'water_ml', 'exercise_duration', 'floor_count']
for col in key_cols:
    if col in master.columns:
        fill = master[col].notna().mean() * 100
        bar  = '█' * int(fill / 5) + '░' * (20 - int(fill / 5))
        status = "✅" if fill > 80 else ("🟡" if fill > 40 else "🔴")
        print(f"  {status} {col:<30} {bar}  {fill:>5.1f}%")


# ══════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════
out_path = os.path.join(OUTPUT_DIR, "master_daily.csv")
master.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print(f"  ✅  MASTER TABLE SAVED")
print(f"  File: {os.path.abspath(out_path)}")
print(f"  Shape: {master.shape[0]:,} rows × {master.shape[1]} columns")
print(f"  Date range: {master['date'].min().date()} → {master['date'].max().date()}")
print(f"\n  Next step: run  03_fabricate_data.py")
print(f"{'='*60}")
