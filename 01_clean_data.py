"""
====================================================================
  SCRIPT 1 — DATA CLEANING
  Samsung Health Personal Analytics Project
  Author: Atharva
  Description: Reads all raw Samsung Health CSV files, fixes column
               alignment, parses dates, renames columns, drops junk,
               and saves clean CSVs ready for merging.
====================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PATHS  — change INPUT_DIR if needed
# ─────────────────────────────────────────────
INPUT_DIR  = "."          # folder with raw Samsung CSVs
OUTPUT_DIR = "cleaned"    # folder where clean CSVs will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  HELPER — read a Samsung Health CSV correctly
# ─────────────────────────────────────────────
def read_samsung(filename):
    """
    Samsung Health CSVs have:
      Line 0 → metadata  (skip)
      Line 1 → column headers
      Line 2+ → actual data
    Also uses index_col=False to prevent column shift bug
    caused by extra comma in data rows.
    """
    path = os.path.join(INPUT_DIR, filename)
    df = pd.read_csv(path, header=1, index_col=False,
                     low_memory=False, encoding='utf-8-sig')
    return df


def strip_prefix(df, prefix):
    """Remove Samsung's long column prefix from all column names."""
    df.columns = [c.replace(prefix, '') for c in df.columns]
    return df


def parse_ms_timestamp(series):
    """Convert Unix millisecond timestamps to datetime."""
    return pd.to_datetime(series, unit='ms', errors='coerce')


def parse_str_timestamp(series):
    """Parse string datetime like '2021-02-22 07:00:00.000' to datetime."""
    return pd.to_datetime(series, errors='coerce')


def to_date_only(dt_series):
    """Strip time component, keep only date."""
    return pd.to_datetime(dt_series).dt.normalize()


# ══════════════════════════════════════════════════════════════════
#  1. BODY WEIGHT
# ══════════════════════════════════════════════════════════════════
print("Cleaning: Body Weight...")

df_weight = read_samsung("Body Weight.csv")

# Keep relevant columns
keep = ['start_time', 'weight', 'height', 'body_fat',
        'muscle_mass', 'fat_free_mass', 'basal_metabolic_rate']
df_weight = df_weight[[c for c in keep if c in df_weight.columns]].copy()

# Parse date
df_weight['date'] = to_date_only(parse_str_timestamp(df_weight['start_time']))
df_weight.drop(columns=['start_time'], inplace=True)

# Convert numeric columns
for col in ['weight', 'height', 'body_fat', 'muscle_mass',
            'fat_free_mass', 'basal_metabolic_rate']:
    if col in df_weight.columns:
        df_weight[col] = pd.to_numeric(df_weight[col], errors='coerce')

# Drop rows with no date or no weight
df_weight.dropna(subset=['date', 'weight'], inplace=True)

# Sort
df_weight.sort_values('date', inplace=True)
df_weight.reset_index(drop=True, inplace=True)

print(f"  → {len(df_weight)} rows | {df_weight['date'].min().date()} to {df_weight['date'].max().date()}")
print(f"     Weight range: {df_weight['weight'].min()} – {df_weight['weight'].max()} kg")
df_weight.to_csv(os.path.join(OUTPUT_DIR, "weight_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  2. SLEEP
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Sleep...")

df_sleep = read_samsung("Sleep.csv")
df_sleep = strip_prefix(df_sleep, 'com.samsung.health.sleep.')

# Keep relevant columns
keep = ['start_time', 'end_time', 'sleep_score', 'sleep_duration',
        'efficiency', 'quality', 'physical_recovery', 'mental_recovery',
        'total_rem_duration', 'total_light_duration', 'movement_awakening']
df_sleep = df_sleep[[c for c in keep if c in df_sleep.columns]].copy()

# Parse dates
df_sleep['start_time'] = parse_str_timestamp(df_sleep['start_time'])
df_sleep['end_time']   = parse_str_timestamp(df_sleep['end_time'])

# Derive sleep date = the day you WENT TO SLEEP
df_sleep['date'] = to_date_only(df_sleep['start_time'])

# Compute sleep hours from start/end times (most reliable method)
# Note: sleep_duration in Samsung Health is stored in MINUTES, not ms
df_sleep['sleep_hours'] = (
    (df_sleep['end_time'] - df_sleep['start_time'])
    .dt.total_seconds() / 3600
)

# Fallback: if start/end parse failed, try sleep_duration column (in minutes)
missing_mask = df_sleep['sleep_hours'].isna() | (df_sleep['sleep_hours'] <= 0)
if missing_mask.any() and 'sleep_duration' in df_sleep.columns:
    df_sleep.loc[missing_mask, 'sleep_hours'] = (
        pd.to_numeric(df_sleep.loc[missing_mask, 'sleep_duration'], errors='coerce') / 60
    )

# Convert numeric columns
for col in ['sleep_score', 'efficiency', 'quality',
            'physical_recovery', 'mental_recovery',
            'total_rem_duration', 'total_light_duration']:
    if col in df_sleep.columns:
        df_sleep[col] = pd.to_numeric(df_sleep[col], errors='coerce')

# Convert rem/light duration from ms to hours
for col in ['total_rem_duration', 'total_light_duration']:
    if col in df_sleep.columns:
        df_sleep[col] = df_sleep[col] / 3600000

# Drop rows with no date
df_sleep.dropna(subset=['date'], inplace=True)

# FIX 1: Remove nap records — Samsung sometimes logs short naps as "sleep"
# Only keep genuine overnight sleep sessions (>= 2.5 hours)
naps_removed = (df_sleep['sleep_hours'] < 2.5).sum()
df_sleep = df_sleep[df_sleep['sleep_hours'] >= 2.5].copy()
if naps_removed > 0:
    print(f"     Removed {naps_removed} nap records (< 2.5 hrs) — not real sleep sessions")

# Keep only 1 record per day (longest sleep if multiple)
df_sleep = (df_sleep.sort_values('sleep_hours', ascending=False)
                    .drop_duplicates(subset='date', keep='first'))

df_sleep.sort_values('date', inplace=True)
df_sleep.reset_index(drop=True, inplace=True)

# Final columns
final_cols = ['date', 'sleep_hours', 'sleep_score', 'efficiency',
              'quality', 'physical_recovery', 'mental_recovery',
              'total_rem_duration', 'total_light_duration']
df_sleep = df_sleep[[c for c in final_cols if c in df_sleep.columns]]

print(f"  → {len(df_sleep)} rows | {df_sleep['date'].min().date()} to {df_sleep['date'].max().date()}")
print(f"     Avg sleep: {df_sleep['sleep_hours'].mean():.1f} hrs")
df_sleep.to_csv(os.path.join(OUTPUT_DIR, "sleep_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  3. EXERCISE
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Exercise...")

df_ex = read_samsung("Exercise.csv")
df_ex = strip_prefix(df_ex, 'com.samsung.health.exercise.')

# Keep relevant columns
keep = ['start_time', 'end_time', 'duration', 'calorie',
        'distance', 'mean_heart_rate', 'max_heart_rate',
        'min_heart_rate', 'mean_speed']
df_ex = df_ex[[c for c in keep if c in df_ex.columns]].copy()

# Parse date
df_ex['start_time'] = parse_str_timestamp(df_ex['start_time'])
df_ex['date'] = to_date_only(df_ex['start_time'])
df_ex.drop(columns=['start_time', 'end_time'], errors='ignore', inplace=True)

# Convert numeric
for col in ['duration', 'calorie', 'distance', 'mean_heart_rate',
            'max_heart_rate', 'min_heart_rate', 'mean_speed']:
    if col in df_ex.columns:
        df_ex[col] = pd.to_numeric(df_ex[col], errors='coerce')

# Duration: milliseconds → minutes
df_ex['duration_min'] = df_ex['duration'] / 60000
df_ex.drop(columns=['duration'], inplace=True)

# FIX 2: Cap absurd exercise durations — max realistic single session = 300 min (5 hrs)
# One Samsung record showed 1,199 min (20 hrs!) — clearly a data corruption
outliers = (df_ex['duration_min'] > 300).sum()
if outliers > 0:
    print(f"     Capped {outliers} exercise session(s) with duration > 300 min (Samsung data corruption)")
df_ex['duration_min'] = df_ex['duration_min'].clip(upper=300)

# Distance: meters → km
if 'distance' in df_ex.columns:
    df_ex['distance_km'] = df_ex['distance'] / 1000
    df_ex.drop(columns=['distance'], inplace=True)

# Drop rows missing date or calorie
df_ex.dropna(subset=['date', 'calorie'], inplace=True)

# If multiple workouts in one day, aggregate
df_ex = df_ex.groupby('date').agg(
    exercise_sessions  = ('calorie',        'count'),
    exercise_calories  = ('calorie',        'sum'),
    exercise_duration  = ('duration_min',   'sum'),
    distance_km        = ('distance_km',    'sum') if 'distance_km' in df_ex.columns else ('calorie', 'count'),
    avg_heart_rate_ex  = ('mean_heart_rate','mean'),
).reset_index()

df_ex.sort_values('date', inplace=True)
df_ex.reset_index(drop=True, inplace=True)

print(f"  → {len(df_ex)} rows | {df_ex['date'].min().date()} to {df_ex['date'].max().date()}")
print(f"     Total sessions: {df_ex['exercise_sessions'].sum():.0f}")
df_ex.to_csv(os.path.join(OUTPUT_DIR, "exercise_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  4. PEDOMETER / STEPS
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Pedometer Day Summary...")

df_ped = read_samsung("Pedometer Day summary.csv")

# day_time is Unix ms timestamp
df_ped['date'] = to_date_only(parse_ms_timestamp(df_ped['day_time']))

# Keep relevant columns
keep = ['date', 'step_count', 'distance', 'calorie',
        'active_time', 'run_step_count', 'walk_step_count']
df_ped = df_ped[[c for c in keep if c in df_ped.columns]].copy()

# Convert numeric
for col in ['step_count', 'distance', 'calorie', 'active_time',
            'run_step_count', 'walk_step_count']:
    if col in df_ped.columns:
        df_ped[col] = pd.to_numeric(df_ped[col], errors='coerce')

# active_time in ms → minutes
df_ped['active_time_min'] = df_ped['active_time'] / 60000
df_ped.drop(columns=['active_time'], inplace=True)

# distance: meters → km
df_ped['distance_km'] = df_ped['distance'] / 1000
df_ped.drop(columns=['distance'], inplace=True)

df_ped.dropna(subset=['date', 'step_count'], inplace=True)
df_ped.drop_duplicates(subset='date', keep='last', inplace=True)
df_ped.sort_values('date', inplace=True)
df_ped.reset_index(drop=True, inplace=True)

print(f"  → {len(df_ped)} rows | {df_ped['date'].min().date()} to {df_ped['date'].max().date()}")
print(f"     Avg steps/day: {df_ped['step_count'].mean():.0f}")
df_ped.to_csv(os.path.join(OUTPUT_DIR, "steps_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  5. CALORIES BURNED
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Calories Burned...")

df_cal = read_samsung("Calories Burned details.csv")
df_cal = strip_prefix(df_cal, 'com.samsung.shealth.calories_burned.')

# day_time is Unix ms
df_cal['date'] = to_date_only(parse_ms_timestamp(df_cal['day_time']))

# Keep relevant columns
keep = ['date', 'rest_calorie', 'active_calorie']
df_cal = df_cal[[c for c in keep if c in df_cal.columns]].copy()

# Convert numeric
for col in ['rest_calorie', 'active_calorie']:
    if col in df_cal.columns:
        df_cal[col] = pd.to_numeric(df_cal[col], errors='coerce')

# Total calories
df_cal['total_calories_burned'] = df_cal.get('rest_calorie', 0) + df_cal.get('active_calorie', 0)

df_cal.dropna(subset=['date', 'total_calories_burned'], inplace=True)
df_cal.drop_duplicates(subset='date', keep='last', inplace=True)
df_cal.sort_values('date', inplace=True)
df_cal.reset_index(drop=True, inplace=True)

print(f"  → {len(df_cal)} rows | {df_cal['date'].min().date()} to {df_cal['date'].max().date()}")
print(f"     Avg total calories burned/day: {df_cal['total_calories_burned'].mean():.0f}")
df_cal.to_csv(os.path.join(OUTPUT_DIR, "calories_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  6. WATER INTAKE
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Water Intake...")

df_water = read_samsung("Water Intake.csv")

df_water['date'] = to_date_only(parse_str_timestamp(df_water['start_time']))
df_water['amount'] = pd.to_numeric(df_water['amount'], errors='coerce')

df_water = df_water[['date', 'amount']].dropna()

# Sum all entries per day (multiple glasses per day)
df_water = df_water.groupby('date')['amount'].sum().reset_index()
df_water.rename(columns={'amount': 'water_ml'}, inplace=True)

df_water.sort_values('date', inplace=True)
df_water.reset_index(drop=True, inplace=True)

print(f"  → {len(df_water)} rows | {df_water['date'].min().date()} to {df_water['date'].max().date()}")
print(f"     Avg water/day: {df_water['water_ml'].mean():.0f} ml")
df_water.to_csv(os.path.join(OUTPUT_DIR, "water_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  7. HEART RATE
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Heart Rate...")

df_hr = read_samsung("Heart Rate.csv")
df_hr = strip_prefix(df_hr, 'com.samsung.health.heart_rate.')

df_hr['start_time'] = parse_str_timestamp(df_hr['start_time'])
df_hr['date'] = to_date_only(df_hr['start_time'])
df_hr['heart_rate'] = pd.to_numeric(df_hr['heart_rate'], errors='coerce')

df_hr.dropna(subset=['date', 'heart_rate'], inplace=True)

# Aggregate to daily stats
df_hr_daily = df_hr.groupby('date').agg(
    avg_heart_rate = ('heart_rate', 'mean'),
    min_heart_rate = ('heart_rate', 'min'),
    max_heart_rate = ('heart_rate', 'max'),
    hr_readings    = ('heart_rate', 'count'),
).reset_index()

df_hr_daily['avg_heart_rate'] = df_hr_daily['avg_heart_rate'].round(1)

df_hr_daily.sort_values('date', inplace=True)
df_hr_daily.reset_index(drop=True, inplace=True)

print(f"  → {len(df_hr_daily)} rows | {df_hr_daily['date'].min().date()} to {df_hr_daily['date'].max().date()}")
print(f"     Avg resting HR: {df_hr_daily['avg_heart_rate'].mean():.0f} bpm")
df_hr_daily.to_csv(os.path.join(OUTPUT_DIR, "heartrate_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  8. SLEEP STAGE
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Sleep Stage...")

df_stage = read_samsung("Sleep Stage.csv")
df_stage = strip_prefix(df_stage, 'com.samsung.shealth.sleep.')

# Find the start_time column
st_col = [c for c in df_stage.columns if 'start_time' in c.lower()]
if st_col:
    df_stage['start_time'] = parse_str_timestamp(df_stage[st_col[0]])
    df_stage['date'] = to_date_only(df_stage['start_time'])

# Stage column: 1=awake, 2=REM, 3=light, 4=deep
if 'stage' in df_stage.columns:
    df_stage['stage'] = pd.to_numeric(df_stage['stage'], errors='coerce')
    stage_map = {1: 'awake', 2: 'rem', 3: 'light', 4: 'deep'}
    df_stage['stage_name'] = df_stage['stage'].map(stage_map)

    # Parse start/end times for duration
    if 'start_time' in df_stage.columns and 'end_time' in df_stage.columns:
        df_stage['end_time'] = parse_str_timestamp(df_stage['end_time'])
        df_stage['duration_min'] = (
            (df_stage['end_time'] - df_stage['start_time'])
            .dt.total_seconds() / 60
        )

df_stage.dropna(subset=['date'], inplace=True)
df_stage.sort_values(['date', 'start_time'], inplace=True)
df_stage.reset_index(drop=True, inplace=True)

# Aggregate: minutes per stage per night
if 'stage_name' in df_stage.columns and 'duration_min' in df_stage.columns:
    df_stage_agg = df_stage.pivot_table(
        index='date', columns='stage_name',
        values='duration_min', aggfunc='sum'
    ).reset_index()
    df_stage_agg.columns.name = None
    df_stage_agg.rename(columns={
        'deep':  'deep_sleep_min',
        'light': 'light_sleep_min',
        'rem':   'rem_sleep_min',
        'awake': 'awake_min'
    }, inplace=True)
    df_stage_agg.to_csv(os.path.join(OUTPUT_DIR, "sleep_stage_clean.csv"), index=False)
    print(f"  → {len(df_stage_agg)} rows (aggregated by night)")
else:
    df_stage.to_csv(os.path.join(OUTPUT_DIR, "sleep_stage_clean.csv"), index=False)
    print(f"  → {len(df_stage)} rows (raw stage records)")


# ══════════════════════════════════════════════════════════════════
#  9. STEP DAILY TREND
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Step Daily Trend...")

df_trend = read_samsung("Step Daily Trend.csv")

df_trend['date'] = to_date_only(parse_ms_timestamp(df_trend['day_time']))

for col in ['count', 'distance', 'speed', 'calorie']:
    if col in df_trend.columns:
        df_trend[col] = pd.to_numeric(df_trend[col], errors='coerce')

df_trend = df_trend[['date', 'count', 'distance', 'speed', 'calorie']].copy()
df_trend.rename(columns={
    'count':    'steps',
    'distance': 'distance_m',
    'calorie':  'step_calorie'
}, inplace=True)

df_trend['distance_km'] = df_trend['distance_m'] / 1000

df_trend.dropna(subset=['date', 'steps'], inplace=True)
df_trend.drop_duplicates(subset='date', keep='last', inplace=True)
df_trend.sort_values('date', inplace=True)
df_trend.reset_index(drop=True, inplace=True)

print(f"  → {len(df_trend)} rows | {df_trend['date'].min().date()} to {df_trend['date'].max().date()}")
df_trend.to_csv(os.path.join(OUTPUT_DIR, "step_trend_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  10. FLOOR DAY SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\nCleaning: Floor Day Summary...")

df_floor = read_samsung("Floor day Summary.csv")

df_floor['date'] = to_date_only(parse_ms_timestamp(df_floor['day_time']))
df_floor['floor_count'] = pd.to_numeric(df_floor['floor_count'], errors='coerce')

for col in ['calorie', 'distance']:
    if col in df_floor.columns:
        df_floor[col] = pd.to_numeric(df_floor[col], errors='coerce')

keep = ['date', 'floor_count', 'calorie', 'distance']
df_floor = df_floor[[c for c in keep if c in df_floor.columns]].copy()
df_floor.rename(columns={'calorie': 'floor_calorie'}, inplace=True)

df_floor.dropna(subset=['date', 'floor_count'], inplace=True)
df_floor.drop_duplicates(subset='date', keep='last', inplace=True)
df_floor.sort_values('date', inplace=True)
df_floor.reset_index(drop=True, inplace=True)

print(f"  → {len(df_floor)} rows | {df_floor['date'].min().date()} to {df_floor['date'].max().date()}")
df_floor.to_csv(os.path.join(OUTPUT_DIR, "floors_clean.csv"), index=False)


# ══════════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  ✅  ALL FILES CLEANED SUCCESSFULLY")
print(f"  Output folder: {os.path.abspath(OUTPUT_DIR)}/")
print("="*60)
print("\n  Files created:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size  = os.path.getsize(fpath)
    rows  = sum(1 for _ in open(fpath)) - 1
    print(f"    {f:<35} {rows:>5} rows  ({size:,} bytes)")
print("\n  Next step: run  02_build_master.py")
