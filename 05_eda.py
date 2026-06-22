"""
====================================================================
  SCRIPT 5 — EXPLORATORY DATA ANALYSIS (EDA)
  Samsung Health Personal Analytics Project
  Author: Atharva
  Description: Comprehensive EDA covering weight, fitness, sleep,
               correlations, and trends. Generates 20+ charts as
               PNG files and prints key insights to terminal.
====================================================================

  ANALYSIS SECTIONS:
  ─────────────────────────────────────────────────────────────────
  1. Dataset Overview & Summary Statistics
  2. Weight & Body Composition Journey
  3. Fitness & Activity Patterns
  4. Sleep Analysis
  5. Heart Rate Analysis
  6. Calorie & Energy Balance
  7. Correlation Heatmap & Key Relationships
  8. Weekly/Monthly Trends
  9. Key Insights & Recommendations (printed summary)
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
# Auto-locate files relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(SCRIPT_DIR, "master_featured.csv")
CHART_DIR   = os.path.join(SCRIPT_DIR, "eda_charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

COLORS = {
    'primary':  '#2563EB',
    'accent':   '#F59E0B',
    'danger':   '#EF4444',
    'success':  '#10B981',
    'purple':   '#8B5CF6',
    'pink':     '#EC4899',
    'gray':     '#6B7280',
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


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
print("="*70)
print("  LOADING DATA")
print("="*70)

df = pd.read_csv(INPUT_FILE, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")
print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — DATASET OVERVIEW & SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 1 — DATASET OVERVIEW")
print("="*70)

key_cols = ['weight', 'step_count', 'total_calories_burned', 'sleep_hours',
            'avg_heart_rate', 'water_ml', 'exercise_duration', 'bmi']
stats = df[key_cols].describe().round(2)
print(f"\n{stats}\n")

# Chart 1: Data availability heatmap (by month)
print("  Generating: Data availability heatmap...")
fig, ax = plt.subplots(figsize=(16, 5))
monthly_fill = df.set_index('date')[key_cols].resample('M').apply(lambda x: x.notna().mean() * 100)
sns.heatmap(monthly_fill.T, cmap='YlGnBu', annot=False, fmt='.0f',
            linewidths=0.5, ax=ax, vmin=0, vmax=100,
            cbar_kws={'label': 'Fill Rate %'})
ax.set_title('Data Availability by Month (% filled)', fontsize=14, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
# Show every 6th month label
xticks = ax.get_xticks()
xlabels = [l.get_text()[:7] for l in ax.get_xticklabels()]
ax.set_xticks(xticks[::6])
ax.set_xticklabels(xlabels[::6], rotation=45, ha='right')
save_chart(fig, "data_availability_heatmap")


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — WEIGHT & BODY COMPOSITION JOURNEY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 2 — WEIGHT & BODY COMPOSITION")
print("="*70)

# Chart 2: Full weight journey with real vs synthetic markers
print("  Generating: Weight journey timeline...")
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(df['date'], df['weight'], color=COLORS['gray'], alpha=0.3, linewidth=0.8, label='Daily weight')
ax.plot(df['date'], df['weight_30d_avg'], color=COLORS['primary'], linewidth=2.5, label='30-day average')
ax.plot(df['date'], df['weight_7d_avg'], color=COLORS['accent'], linewidth=1.2, alpha=0.7, label='7-day average')

# Mark real weigh-ins
real = df[df['weight_is_real'] == 1]
ax.scatter(real['date'], real['weight'], color=COLORS['danger'], s=80, zorder=5,
           edgecolors='white', linewidths=1.5, label=f'Real weigh-ins ({len(real)})')

ax.axhline(y=85, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7, label='Goal: 85 kg')
ax.set_title('Weight Journey — 2019 to 2025', fontsize=16, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Weight (kg)')
ax.legend(loc='upper right', fontsize=10)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
save_chart(fig, "weight_journey_full")

# Chart 3: BMI over time
print("  Generating: BMI timeline...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['date'], df['bmi'], color=COLORS['primary'], linewidth=0.8, alpha=0.4)
ax.plot(df['date'], df['bmi'].rolling(30).mean(), color=COLORS['primary'], linewidth=2.5, label='BMI (30d avg)')
ax.axhspan(18.5, 25, color=COLORS['success'], alpha=0.1, label='Normal range')
ax.axhspan(25, 30, color=COLORS['accent'], alpha=0.1, label='Overweight range')
ax.axhspan(30, 35, color=COLORS['danger'], alpha=0.1, label='Obese range')
ax.set_title('BMI Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('BMI')
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
save_chart(fig, "bmi_timeline")

# Chart 4: Weight change distribution by year
print("  Generating: Yearly weight change...")
fig, ax = plt.subplots(figsize=(12, 6))
yearly = df.groupby('year').agg(
    start_weight=('weight', 'first'),
    end_weight=('weight', 'last')
).reset_index()
yearly['change'] = yearly['end_weight'] - yearly['start_weight']
colors_bar = [COLORS['success'] if c < 0 else COLORS['danger'] for c in yearly['change']]
bars = ax.bar(yearly['year'].astype(str), yearly['change'], color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, yearly['change']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:+.1f} kg', ha='center', fontweight='bold', fontsize=11)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_title('Weight Change by Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Weight Change (kg)')
ax.set_xlabel('Year')
save_chart(fig, "weight_change_by_year")

# Chart 5: Week label distribution pie chart
print("  Generating: Week label distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Overall pie
wl = df['week_label'].value_counts()
colors_pie = [COLORS['success'], COLORS['danger'], COLORS['accent'], COLORS['gray']]
axes[0].pie(wl.values, labels=wl.index, colors=colors_pie[:len(wl)],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
axes[0].set_title('Overall Week Classification', fontweight='bold')

# By year stacked bar
year_wl = pd.crosstab(df['year'], df['week_label'], normalize='index') * 100
year_wl[['fat_loss','maintenance','weight_gain']].plot(
    kind='bar', stacked=True, ax=axes[1],
    color=[COLORS['success'], COLORS['accent'], COLORS['danger']],
    edgecolor='white', linewidth=0.5
)
axes[1].set_title('Week Classification by Year (%)', fontweight='bold')
axes[1].set_ylabel('Percentage')
axes[1].legend(fontsize=9)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
plt.tight_layout()
save_chart(fig, "week_label_distribution")

print(f"\n  Weight insights:")
print(f"    Heaviest: {df['weight'].max():.1f} kg ({df.loc[df['weight'].idxmax(), 'date'].date()})")
print(f"    Lightest: {df['weight'].min():.1f} kg ({df.loc[df['weight'].idxmin(), 'date'].date()})")
print(f"    Current:  {df['weight'].iloc[-1]:.1f} kg (BMI: {df['bmi'].iloc[-1]:.1f})")
print(f"    Total lost in 2021 cut: {yearly.loc[yearly['year']==2021, 'change'].values[0]:.1f} kg")


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — FITNESS & ACTIVITY PATTERNS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 3 — FITNESS & ACTIVITY PATTERNS")
print("="*70)

# Chart 6: Daily steps over time
print("  Generating: Steps timeline...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(df['date'], df['step_count'], alpha=0.15, color=COLORS['primary'])
ax.plot(df['date'], df['steps_30d_avg'], color=COLORS['primary'], linewidth=2.5, label='30-day avg')
ax.axhline(y=10000, color=COLORS['success'], linestyle='--', alpha=0.6, label='10K target')
ax.axhline(y=5000, color=COLORS['accent'], linestyle='--', alpha=0.6, label='5K minimum')
ax.set_title('Daily Steps Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Steps')
ax.legend(fontsize=10)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
save_chart(fig, "steps_timeline")

# Chart 7: Steps by day of week
print("  Generating: Steps by day of week...")
fig, ax = plt.subplots(figsize=(10, 6))
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_steps = df.groupby('day_name')['step_count'].agg(['mean','median']).reindex(dow_order)
x = range(len(dow_order))
ax.bar(x, dow_steps['mean'], color=COLORS['primary'], alpha=0.7, label='Mean', width=0.4, align='center')
ax.bar([i+0.4 for i in x], dow_steps['median'], color=COLORS['accent'], alpha=0.7, label='Median', width=0.4, align='center')
ax.set_xticks([i+0.2 for i in x])
ax.set_xticklabels(dow_order, rotation=45)
ax.set_title('Average Steps by Day of Week', fontsize=14, fontweight='bold')
ax.set_ylabel('Steps')
ax.legend()
for i, (m, med) in enumerate(zip(dow_steps['mean'], dow_steps['median'])):
    ax.text(i, m + 100, f'{m:.0f}', ha='center', fontsize=9, fontweight='bold')
save_chart(fig, "steps_by_day_of_week")

# Chart 8: Activity level distribution
print("  Generating: Activity level distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
act_order = ['sedentary','light','moderate','active','very_active']
act_colors = [COLORS['danger'], COLORS['accent'], COLORS['primary'], COLORS['success'], COLORS['purple']]
act_counts = df['activity_level'].value_counts().reindex(act_order)
axes[0].barh(act_order, act_counts.values, color=act_colors, edgecolor='white')
axes[0].set_title('Activity Level Distribution (all days)', fontweight='bold')
axes[0].set_xlabel('Number of Days')
for i, v in enumerate(act_counts.values):
    axes[0].text(v + 10, i, f'{v} ({v/len(df)*100:.1f}%)', va='center', fontsize=10)

# Activity level by year
year_act = pd.crosstab(df['year'], df['activity_level'], normalize='index') * 100
year_act[act_order].plot(kind='bar', stacked=True, ax=axes[1], color=act_colors, edgecolor='white')
axes[1].set_title('Activity Level by Year (%)', fontweight='bold')
axes[1].set_ylabel('Percentage')
axes[1].legend(fontsize=8, loc='upper right')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
plt.tight_layout()
save_chart(fig, "activity_level_distribution")

# Chart 9: Workout frequency and intensity
print("  Generating: Workout patterns...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
# Monthly workout count
monthly_wkt = df.set_index('date')['workout_day'].resample('M').sum()
axes[0].bar(monthly_wkt.index, monthly_wkt.values, color=COLORS['primary'], alpha=0.7, width=25)
axes[0].set_title('Monthly Workout Count', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Workouts per Month')
axes[0].axhline(y=monthly_wkt.mean(), color=COLORS['danger'], linestyle='--', label=f'Avg: {monthly_wkt.mean():.1f}/month')
axes[0].legend()

# Workout intensity over time (only workout days)
wkt_days = df[df['workout_day'] == 1].copy()
if len(wkt_days) > 0:
    axes[1].scatter(wkt_days['date'], wkt_days['workout_intensity_score'],
                    c=wkt_days['workout_intensity_score'], cmap='YlOrRd',
                    s=30, alpha=0.6, edgecolors='none')
    axes[1].plot(wkt_days['date'], wkt_days['workout_intensity_score'].rolling(20, min_periods=5).mean(),
                 color=COLORS['danger'], linewidth=2.5, label='20-session rolling avg')
    axes[1].set_title('Workout Intensity Score Over Time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Intensity Score (0–100)')
    axes[1].legend()
plt.tight_layout()
save_chart(fig, "workout_patterns")

print(f"\n  Fitness insights:")
print(f"    Avg daily steps: {df['step_count'].mean():.0f}")
print(f"    Days reaching 10K steps: {(df['step_count'] >= 10000).sum()} ({(df['step_count'] >= 10000).mean()*100:.1f}%)")
print(f"    Total workout days: {df['workout_day'].sum():.0f} ({df['workout_day'].mean()*100:.1f}%)")
print(f"    Avg workouts/week: {df['weekly_workout_count'].mean():.1f}")
print(f"    Avg workout intensity: {wkt_days['workout_intensity_score'].mean():.1f}/100" if len(wkt_days) > 0 else "")


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — SLEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 4 — SLEEP ANALYSIS")
print("="*70)

# Chart 10: Sleep hours over time
print("  Generating: Sleep timeline...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(df['date'], df['sleep_hours'], alpha=0.15, color=COLORS['purple'])
ax.plot(df['date'], df['sleep_7d_avg'], color=COLORS['purple'], linewidth=2, label='7-day avg')
ax.axhline(y=7.5, color=COLORS['success'], linestyle='--', alpha=0.7, label='Target: 7.5 hrs')
ax.axhline(y=6, color=COLORS['danger'], linestyle='--', alpha=0.5, label='Minimum: 6 hrs')
ax.set_title('Sleep Duration Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('Sleep Hours')
ax.legend(fontsize=10)
ax.set_ylim(2, 11)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
save_chart(fig, "sleep_timeline")

# Chart 11: Sleep quality distribution
print("  Generating: Sleep quality breakdown...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Sleep hours distribution
axes[0].hist(df['sleep_hours'], bins=30, color=COLORS['purple'], alpha=0.7, edgecolor='white')
axes[0].axvline(df['sleep_hours'].mean(), color=COLORS['danger'], linestyle='--', label=f"Mean: {df['sleep_hours'].mean():.1f}h")
axes[0].set_title('Sleep Duration Distribution', fontweight='bold')
axes[0].set_xlabel('Hours')
axes[0].legend()

# Sleep quality category
sq = df['sleep_quality_category'].value_counts()
sq_colors = {'excellent': COLORS['success'], 'good': COLORS['primary'],
             'fair': COLORS['accent'], 'poor': COLORS['danger']}
sq_order = ['excellent', 'good', 'fair', 'poor']
sq = sq.reindex(sq_order).dropna()
axes[1].pie(sq.values, labels=sq.index, colors=[sq_colors[k] for k in sq.index],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
axes[1].set_title('Sleep Quality Breakdown', fontweight='bold')

# Weekday vs weekend sleep
wkday_sleep = df[df['is_weekend']==0]['sleep_hours']
wkend_sleep = df[df['is_weekend']==1]['sleep_hours']
bp = axes[2].boxplot([wkday_sleep, wkend_sleep], labels=['Weekday', 'Weekend'],
                     patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor(COLORS['primary'])
bp['boxes'][1].set_facecolor(COLORS['accent'])
axes[2].set_title('Weekday vs Weekend Sleep', fontweight='bold')
axes[2].set_ylabel('Hours')
plt.tight_layout()
save_chart(fig, "sleep_quality_breakdown")

# Chart 12: Sleep debt over time
print("  Generating: Sleep debt analysis...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(df['date'], df['sleep_debt_7d'],
                where=df['sleep_debt_7d'] >= 0, color=COLORS['success'], alpha=0.3, label='Sleep surplus')
ax.fill_between(df['date'], df['sleep_debt_7d'],
                where=df['sleep_debt_7d'] < 0, color=COLORS['danger'], alpha=0.3, label='Sleep debt')
ax.plot(df['date'], df['sleep_debt_7d'], color=COLORS['gray'], linewidth=0.8)
ax.axhline(y=0, color='black', linewidth=1)
ax.set_title('Cumulative Sleep Debt (7-day rolling)', fontsize=14, fontweight='bold')
ax.set_ylabel('Hours (negative = debt)')
ax.legend()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
save_chart(fig, "sleep_debt_timeline")

print(f"\n  Sleep insights:")
print(f"    Avg sleep: {df['sleep_hours'].mean():.2f} hrs")
print(f"    Weekday avg: {wkday_sleep.mean():.2f} hrs | Weekend avg: {wkend_sleep.mean():.2f} hrs")
print(f"    Nights below 6 hrs: {(df['sleep_hours'] < 6).sum()} ({(df['sleep_hours'] < 6).mean()*100:.1f}%)")
print(f"    Avg sleep debt (7d): {df['sleep_debt_7d'].mean():.1f} hrs")
print(f"    Sleep consistency (lower=better): {df['sleep_consistency_7d'].mean():.2f} hrs std")


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — HEART RATE ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 5 — HEART RATE ANALYSIS")
print("="*70)

# Chart 13: Heart rate over time
print("  Generating: Heart rate timeline...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df['date'], df['avg_heart_rate'], color=COLORS['danger'], alpha=0.2, linewidth=0.5)
ax.plot(df['date'], df['hr_7d_avg'], color=COLORS['danger'], linewidth=2, label='7-day avg HR')
ax.axhline(y=72, color=COLORS['success'], linestyle='--', alpha=0.5, label='Healthy resting (72 bpm)')
ax.set_title('Average Heart Rate Over Time', fontsize=14, fontweight='bold')
ax.set_ylabel('BPM')
ax.legend()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
save_chart(fig, "heart_rate_timeline")

print(f"\n  Heart rate insights:")
print(f"    Avg resting HR: {df['avg_heart_rate'].mean():.1f} bpm")
print(f"    HR trend: {df['hr_trend'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — CALORIE & ENERGY BALANCE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 6 — CALORIE & ENERGY BALANCE")
print("="*70)

# Chart 14: Calories burned over time
print("  Generating: Calorie analysis...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Total calories timeline
axes[0].fill_between(df['date'], df['total_calories_burned'], alpha=0.15, color=COLORS['accent'])
cal_30d = df['total_calories_burned'].rolling(30, min_periods=7).mean()
axes[0].plot(df['date'], cal_30d, color=COLORS['accent'], linewidth=2.5, label='30-day avg')
axes[0].axhline(y=2400, color=COLORS['danger'], linestyle='--', label='Estimated intake (2400)')
axes[0].set_title('Total Calories Burned Per Day', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Calories (kcal)')
axes[0].legend()

# Net calorie balance
axes[1].fill_between(df['date'], df['calorie_balance_7d_avg'],
                     where=df['calorie_balance_7d_avg'] >= 0, color=COLORS['danger'], alpha=0.3, label='Surplus')
axes[1].fill_between(df['date'], df['calorie_balance_7d_avg'],
                     where=df['calorie_balance_7d_avg'] < 0, color=COLORS['success'], alpha=0.3, label='Deficit')
axes[1].plot(df['date'], df['calorie_balance_7d_avg'], color=COLORS['gray'], linewidth=0.8)
axes[1].axhline(y=0, color='black', linewidth=1)
axes[1].set_title('Net Calorie Balance (7-day rolling avg)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Calories (kcal)')
axes[1].legend()
plt.tight_layout()
save_chart(fig, "calorie_analysis")

# Chart 15: Energy balance category by year
print("  Generating: Energy balance by year...")
fig, ax = plt.subplots(figsize=(12, 6))
year_energy = pd.crosstab(df['year'], df['energy_balance_category'], normalize='index') * 100
ebc_order = ['deficit', 'maintenance', 'surplus']
ebc_colors = [COLORS['success'], COLORS['accent'], COLORS['danger']]
year_energy[ebc_order].plot(kind='bar', stacked=True, ax=ax, color=ebc_colors, edgecolor='white')
ax.set_title('Energy Balance Category by Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage of Days')
ax.legend(fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
save_chart(fig, "energy_balance_by_year")

print(f"\n  Calorie insights:")
print(f"    Avg calories burned/day: {df['total_calories_burned'].mean():.0f}")
print(f"    Avg net balance: {df['net_calorie_balance'].mean():.0f} kcal/day")
print(f"    Days in deficit: {(df['net_calorie_balance'] < -200).sum()} ({(df['net_calorie_balance'] < -200).mean()*100:.1f}%)")
print(f"    Days in surplus: {(df['net_calorie_balance'] > 200).sum()} ({(df['net_calorie_balance'] > 200).mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 — CORRELATION HEATMAP & KEY RELATIONSHIPS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 7 — CORRELATIONS & RELATIONSHIPS")
print("="*70)

# Chart 16: Correlation heatmap
print("  Generating: Correlation heatmap...")
corr_cols = ['weight', 'step_count', 'total_calories_burned', 'sleep_hours',
             'avg_heart_rate', 'water_ml', 'exercise_duration', 'bmi',
             'workout_intensity_score', 'sleep_debt_7d', 'net_calorie_balance']
corr_cols = [c for c in corr_cols if c in df.columns]
corr = df[corr_cols].corr().round(2)

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
save_chart(fig, "correlation_heatmap")

# Print top correlations
print(f"\n  Top correlations with WEIGHT:")
weight_corr = corr['weight'].drop('weight').abs().sort_values(ascending=False)
for feat, val in weight_corr.head(5).items():
    direction = "+" if corr.loc['weight', feat] > 0 else "-"
    print(f"    {direction}{val:.2f}  {feat}")

# Chart 17: Steps vs Weight scatter
print("\n  Generating: Key relationship scatter plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Steps vs Weight
axes[0,0].scatter(df['step_count'], df['weight'], alpha=0.1, s=10, color=COLORS['primary'])
# Add trend line
z = np.polyfit(df['step_count'].dropna(), df.loc[df['step_count'].notna(), 'weight'], 1)
p = np.poly1d(z)
x_range = np.linspace(df['step_count'].min(), df['step_count'].max(), 100)
axes[0,0].plot(x_range, p(x_range), color=COLORS['danger'], linewidth=2.5, label=f'Trend (slope: {z[0]:.5f})')
axes[0,0].set_title('Steps vs Weight', fontweight='bold')
axes[0,0].set_xlabel('Daily Steps')
axes[0,0].set_ylabel('Weight (kg)')
axes[0,0].legend()

# Sleep vs Next-Day Steps
axes[0,1].scatter(df['prev_night_sleep'], df['step_count'], alpha=0.1, s=10, color=COLORS['purple'])
axes[0,1].set_title('Previous Night Sleep vs Steps', fontweight='bold')
axes[0,1].set_xlabel('Sleep Hours (previous night)')
axes[0,1].set_ylabel('Steps (next day)')

# Calories burned vs Weight change
axes[1,0].scatter(df['total_calories_burned'], df['weight_change_7d'], alpha=0.1, s=10, color=COLORS['accent'])
axes[1,0].axhline(y=0, color='black', linewidth=0.8)
axes[1,0].set_title('Calories Burned vs 7-day Weight Change', fontweight='bold')
axes[1,0].set_xlabel('Total Calories Burned')
axes[1,0].set_ylabel('Weight Change (7d, kg)')

# Exercise duration vs Weight change
wkt_only = df[df['exercise_duration'] > 0]
axes[1,1].scatter(wkt_only['exercise_duration'], wkt_only['weight_change_7d'],
                  alpha=0.15, s=15, color=COLORS['success'])
axes[1,1].axhline(y=0, color='black', linewidth=0.8)
axes[1,1].set_title('Exercise Duration vs 7-day Weight Change', fontweight='bold')
axes[1,1].set_xlabel('Exercise Duration (min)')
axes[1,1].set_ylabel('Weight Change (7d, kg)')
plt.tight_layout()
save_chart(fig, "relationship_scatter_plots")


# ══════════════════════════════════════════════════════════════════
#  SECTION 8 — MONTHLY & SEASONAL TRENDS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 8 — MONTHLY & SEASONAL TRENDS")
print("="*70)

# Chart 18: Monthly averages dashboard
print("  Generating: Monthly trends dashboard...")
monthly = df.set_index('date').resample('M').agg({
    'weight': 'mean',
    'step_count': 'mean',
    'sleep_hours': 'mean',
    'total_calories_burned': 'mean',
    'avg_heart_rate': 'mean',
    'workout_day': 'sum'
}).reset_index()

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

metrics = [
    ('weight', 'Monthly Avg Weight (kg)', COLORS['primary']),
    ('step_count', 'Monthly Avg Steps', COLORS['success']),
    ('sleep_hours', 'Monthly Avg Sleep (hrs)', COLORS['purple']),
    ('total_calories_burned', 'Monthly Avg Calories Burned', COLORS['accent']),
    ('avg_heart_rate', 'Monthly Avg Heart Rate (bpm)', COLORS['danger']),
    ('workout_day', 'Monthly Workout Count', COLORS['pink']),
]

for i, (col, title, color) in enumerate(metrics):
    ax = axes[i//2][i%2]
    ax.fill_between(monthly['date'], monthly[col], alpha=0.2, color=color)
    ax.plot(monthly['date'], monthly[col], color=color, linewidth=2.5, marker='o', markersize=3)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
save_chart(fig, "monthly_trends_dashboard")

# Chart 19: Seasonal patterns
print("  Generating: Seasonal patterns...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
season_order = ['winter', 'spring', 'summer', 'autumn']
season_colors = ['#60A5FA', '#34D399', '#FBBF24', '#F97316']

for i, (col, title) in enumerate([('step_count', 'Steps'), ('sleep_hours', 'Sleep (hrs)'), ('avg_heart_rate', 'Heart Rate (bpm)')]):
    season_data = [df[df['season']==s][col].dropna().values for s in season_order]
    bp = axes[i].boxplot(season_data, labels=season_order, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], season_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(f'{title} by Season', fontweight='bold')
plt.tight_layout()
save_chart(fig, "seasonal_patterns")

# Chart 20: Weight vs steps monthly correlation
print("  Generating: Monthly weight vs steps trend...")
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(monthly['date'], monthly['weight'], color=COLORS['primary'], linewidth=2.5, marker='o', markersize=4, label='Weight (kg)')
ax1.set_ylabel('Weight (kg)', color=COLORS['primary'])
ax1.tick_params(axis='y', labelcolor=COLORS['primary'])

ax2 = ax1.twinx()
ax2.plot(monthly['date'], monthly['step_count'], color=COLORS['success'], linewidth=2.5, marker='s', markersize=4, label='Steps/day')
ax2.set_ylabel('Steps/day', color=COLORS['success'])
ax2.tick_params(axis='y', labelcolor=COLORS['success'])

ax1.set_title('Monthly Weight vs Daily Steps', fontsize=14, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.xticks(rotation=45)
save_chart(fig, "weight_vs_steps_monthly")


# ══════════════════════════════════════════════════════════════════
#  SECTION 9 — PLATEAU ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 9 — PLATEAU DETECTION")
print("="*70)

# Chart 21: Plateau periods highlighted on weight chart
print("  Generating: Plateau visualization...")
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df['date'], df['weight_30d_avg'], color=COLORS['primary'], linewidth=2, label='Weight (30d avg)')

# Highlight plateau periods
plateau_mask = df['is_plateau'] == 1
plateau_starts = df['date'][plateau_mask & ~plateau_mask.shift(1, fill_value=False)]
plateau_ends = df['date'][plateau_mask & ~plateau_mask.shift(-1, fill_value=False)]
for start, end in zip(plateau_starts, plateau_ends):
    ax.axvspan(start, end, alpha=0.15, color=COLORS['accent'])

ax.set_title('Weight Plateaus Highlighted', fontsize=14, fontweight='bold')
ax.set_ylabel('Weight (kg)')
ax.legend()
# Add annotation
ax.annotate(f'Plateaus: {df["is_plateau"].sum()} days ({df["is_plateau"].mean()*100:.0f}%)',
            xy=(0.02, 0.95), xycoords='axes fraction', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLORS['accent'], alpha=0.3))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
save_chart(fig, "plateau_visualization")


# ══════════════════════════════════════════════════════════════════
#  FINAL SUMMARY — KEY INSIGHTS
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("  📊  EDA KEY INSIGHTS & FINDINGS")
print("="*70)

print(f"""
  WEIGHT & BODY COMPOSITION:
    • Started at {df['weight'].iloc[:30].mean():.1f} kg (BMI {df['bmi'].iloc[:30].mean():.1f}) in mid-2019
    • Biggest cut: -17.7 kg in 2021 (dropped from 103 to 85 kg)
    • Current: {df['weight'].iloc[-1]:.1f} kg (BMI {df['bmi'].iloc[-1]:.1f})
    • Distance to goal (85 kg): {df['distance_to_goal'].iloc[-1]:.1f} kg
    • {df['is_plateau'].mean()*100:.0f}% of days are weight plateaus
    • BMI zone: {df['bmi_category'].iloc[-1]}

  FITNESS & ACTIVITY:
    • Avg daily steps: {df['step_count'].mean():.0f} (target should be 8,000+)
    • Only {(df['step_count'] >= 10000).mean()*100:.1f}% of days hit 10K steps
    • {df['activity_level'].value_counts().idxmax()} is the most common activity level
    • Workout frequency: {df['weekly_workout_count'].mean():.1f} sessions/week
    • Steps are {df['steps_trend'].value_counts().idxmax()} overall

  SLEEP:
    • Avg sleep: {df['sleep_hours'].mean():.1f} hrs (target: 7.5 hrs)
    • {(df['sleep_hours'] < 6).mean()*100:.1f}% of nights below 6 hrs
    • Weekend sleep ({wkend_sleep.mean():.1f} hrs) > Weekday ({wkday_sleep.mean():.1f} hrs)
    • Avg weekly sleep debt: {df['sleep_debt_7d'].mean():.1f} hrs

  HEART RATE:
    • Avg resting HR: {df['avg_heart_rate'].mean():.0f} bpm
    • Lower HR during active periods (fitness improving)

  ENERGY BALANCE:
    • Avg daily calories burned: {df['total_calories_burned'].mean():.0f} kcal
    • Average calorie balance: {df['net_calorie_balance'].mean():+.0f} kcal/day
    • {df['energy_balance_category'].value_counts().idxmax()} is most common energy state
""")

print(f"\n{'='*70}")
print(f"  ✅  EDA COMPLETE — {chart_count} charts generated")
print(f"  Chart folder: {os.path.abspath(CHART_DIR)}/")
print(f"{'='*70}")
print(f"\n  Charts saved:")
for f in sorted(os.listdir(CHART_DIR)):
    print(f"    📊  {f}")
print(f"\n  Next steps:")
print(f"    → Phase 5: ML models (run 06_ml_models.py)")
print(f"    → Phase 6: Import master_featured.csv into Power BI")
