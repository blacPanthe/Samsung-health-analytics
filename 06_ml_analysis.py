"""
====================================================================
  SCRIPT 6 — ML ANALYSIS ON EXISTING DATA
  Samsung Health Personal Analytics Project
  Author: Atharva

  PURPOSE:
  Analyze patterns and insights IN your existing health data using ML.
  No future prediction — only understanding what already happened.

  ──────────────────────────────────────────────────────────────────
  PART 1 — CLUSTERING
    Groups your days into behavioral profiles automatically.
    Reveals which "type of day" produced the best outcomes.

  PART 2 — PHASE DETECTION
    Detects distinct life phases in your journey (e.g. "2021 cut",
    "post-cut maintenance", "sedentary plateau") based on behaviors.

  PART 3 — FEATURE IMPORTANCE
    Ranks which habits (sleep, steps, calories, workouts) most
    strongly correlated with your fat loss weeks in real history.

  OUTPUT FILES:
  ──────────────────────────────────────────────────────────────────
  ml_charts/
    01_cluster_profiles.png
    02_cluster_timeline.png
    03_cluster_weight_outcomes.png
    04_cluster_pca.png
    05_phase_timeline.png
    06_phase_profiles.png
    07_feature_importance_fatloss.png
    08_feature_importance_weightchange.png
    09_correlation_heatmap_fatloss.png

  ml_output/
    data_with_clusters.csv      ← your full data + cluster label
    data_with_phases.csv        ← your full data + phase label
    feature_importance.csv      ← ranked feature importance table
    analysis_summary.txt        ← written summary of findings
====================================================================

  REQUIREMENT: pip install scikit-learn
====================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Dependency check ─────────────────────────────────────────────
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import silhouette_score
    from sklearn.inspection import permutation_importance
except ModuleNotFoundError:
    print("\n❌  scikit-learn not found.")
    print("    Run:  pip install scikit-learn")
    sys.exit(1)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(SCRIPT_DIR, "master_featured.csv")
CHART_DIR   = os.path.join(SCRIPT_DIR, "ml_charts")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "ml_output")
os.makedirs(CHART_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi']     = 150
plt.rcParams['font.size']      = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette for clusters / phases
CLUSTER_COLORS = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
PHASE_COLORS   = ['#1D4ED8', '#047857', '#B45309', '#9D174D', '#4C1D95', '#065F46']

chart_count = 0
def save_chart(fig, name):
    global chart_count
    chart_count += 1
    path = os.path.join(CHART_DIR, f"{chart_count:02d}_{name}.png")
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      📊  Saved: {path}")
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


# ══════════════════════════════════════════════════════════════════
#  PART 1 — CLUSTERING
#  "What type of day were you having, and which type led to fat loss?"
#
#  Method: K-Means on 7 behavioral features (normalized).
#  Each day gets a cluster label. We then check which cluster
#  overlaps most with your fat loss weeks.
# ══════════════════════════════════════════════════════════════════
section("PART 1 — BEHAVIORAL CLUSTERING")

# ── Features for clustering (behavioral signals only) ───────────
CLUSTER_FEATURES = [
    'step_count',              # daily movement
    'sleep_hours',             # rest quality
    'workout_intensity_score', # how hard you trained
    'net_calorie_balance',     # calorie deficit or surplus
    'avg_heart_rate',          # cardiovascular load
    'exercise_duration',       # time spent working out
    'sleep_debt_7d',           # accumulated sleep deficit
]

cluster_df = df[CLUSTER_FEATURES + ['date', 'weight', 'week_label']].dropna()
print(f"\n  Clustering on {len(cluster_df):,} days using {len(CLUSTER_FEATURES)} behavioral features")
print(f"  Features: {', '.join(CLUSTER_FEATURES)}")

# ── Normalize features ──────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df[CLUSTER_FEATURES])

# ── Find optimal K using elbow method + silhouette score ─────────
print("\n  Finding optimal number of clusters...")
inertias    = []
sil_scores  = []
K_range     = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"    k={k}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.3f}")

# Pick k with best silhouette score (balances compactness + separation)
best_k = list(K_range)[np.argmax(sil_scores)]
print(f"\n  ✅  Optimal k = {best_k}  (silhouette = {max(sil_scores):.3f})")

# ── Final clustering with best_k ────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=15)
cluster_df = cluster_df.copy()
cluster_df['cluster'] = km_final.fit_predict(X_scaled)

# ── Name clusters based on their behavioral profile ─────────────
# Compute mean of each feature per cluster, then assign human-readable names
centroids_df = pd.DataFrame(
    scaler.inverse_transform(km_final.cluster_centers_),
    columns=CLUSTER_FEATURES
)

print("\n  Cluster Profiles (raw feature means):")
print(centroids_df.round(1).to_string())

# Auto-name clusters: rank by step_count × workout_intensity − sleep_debt
centroids_df['activity_score'] = (
    centroids_df['step_count'] / centroids_df['step_count'].max() +
    centroids_df['workout_intensity_score'] / (centroids_df['workout_intensity_score'].max() + 1e-6)
)
centroids_df['deficit_score'] = -centroids_df['net_calorie_balance'] / (abs(centroids_df['net_calorie_balance']).max() + 1e-6)
centroids_df['sleep_score']   = centroids_df['sleep_hours'] / centroids_df['sleep_hours'].max()
centroids_df['composite']     = (
    centroids_df['activity_score'] * 0.35 +
    centroids_df['deficit_score']  * 0.40 +
    centroids_df['sleep_score']    * 0.25
)

# Sort clusters by composite score (best = most active, most deficit, best sleep)
rank = centroids_df['composite'].rank(ascending=False).astype(int)

# Assign labels based on quartile ranking
LABEL_MAP = {
    1: '🔥 Fat Loss Mode',
    2: '💪 Active Maintenance',
    3: '😴 Recovery / Low Activity',
    4: '📉 Sedentary / Surplus',
    5: '⚖️  Balanced Plateau',
    6: '🛌 Rest Day',
}
cluster_name_map = {idx: LABEL_MAP.get(r, f'Cluster {r}') for idx, r in rank.items()}
cluster_df['cluster_name'] = cluster_df['cluster'].map(cluster_name_map)

print("\n  Cluster Labels Assigned:")
for cid, name in cluster_name_map.items():
    count = (cluster_df['cluster'] == cid).sum()
    pct   = count / len(cluster_df) * 100
    print(f"    Cluster {cid} → {name:30s}  {count:5d} days ({pct:.1f}%)")

# ── Chart 1: Cluster Profiles (radar / bar comparison) ──────────
print("\n  Generating: Cluster profiles chart...")
fig, axes = plt.subplots(2, 4, figsize=(18, 9)) if best_k <= 4 else plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

features_display = {
    'step_count':              'Steps/day',
    'sleep_hours':             'Sleep (hrs)',
    'workout_intensity_score': 'Workout Intensity',
    'net_calorie_balance':     'Calorie Balance',
    'avg_heart_rate':          'Avg Heart Rate',
    'exercise_duration':       'Exercise (min)',
    'sleep_debt_7d':           'Sleep Debt (7d)',
}

for cid in range(best_k):
    ax  = axes[cid]
    sub = cluster_df[cluster_df['cluster'] == cid]
    vals = [sub[f].mean() for f in CLUSTER_FEATURES]
    bars = ax.bar(range(len(CLUSTER_FEATURES)), vals,
                  color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(CLUSTER_FEATURES)))
    ax.set_xticklabels([features_display[f] for f in CLUSTER_FEATURES],
                       rotation=35, ha='right', fontsize=8)
    ax.set_title(cluster_name_map[cid], fontsize=11, fontweight='bold',
                 color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
    ax.set_ylabel('Mean Value', fontsize=9)
    n = len(sub)
    pct = n / len(cluster_df) * 100
    ax.text(0.97, 0.97, f'n={n} ({pct:.0f}%)', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='gray')
    ax.spines[['top', 'right']].set_visible(False)

# Hide unused axes
for i in range(best_k, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Behavioral Cluster Profiles\n(Each cluster = a distinct type of day you regularly had)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'cluster_profiles')

# ── Chart 2: Cluster timeline ────────────────────────────────────
print("  Generating: Cluster timeline...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

for cid in range(best_k):
    mask = cluster_df['cluster'] == cid
    ax1.scatter(cluster_df.loc[mask, 'date'],
                cluster_df.loc[mask, 'weight'],
                c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                s=8, alpha=0.6, label=cluster_name_map[cid], zorder=3)

# Smooth weight overlay
weight_smooth = df.set_index('date')['weight'].rolling('30D').mean()
ax1.plot(weight_smooth.index, weight_smooth.values,
         color='black', linewidth=1.5, alpha=0.4, label='30d avg weight', zorder=4)
ax1.set_ylabel('Weight (kg)', fontsize=11)
ax1.set_title('Your Weight Journey Colored by Behavioral Cluster', fontsize=13)
ax1.legend(loc='upper right', fontsize=8, markerscale=2)
ax1.spines[['top', 'right']].set_visible(False)

# Bottom: cluster label over time as colored blocks
for cid in range(best_k):
    mask = cluster_df['cluster'] == cid
    ax2.scatter(cluster_df.loc[mask, 'date'],
                [cid] * mask.sum(),
                c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
                s=12, alpha=0.7, marker='s')

ax2.set_yticks(range(best_k))
ax2.set_yticklabels([cluster_name_map[i] for i in range(best_k)], fontsize=9)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_title('Cluster Assignment Over Time', fontsize=13)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_chart(fig, 'cluster_timeline')

# ── Chart 3: Cluster × Week Label (which cluster had most fat loss) ──
print("  Generating: Cluster vs weight outcomes...")
valid_labels = cluster_df[cluster_df['week_label'].isin(['fat_loss', 'maintenance', 'weight_gain'])]
cross = pd.crosstab(
    valid_labels['cluster_name'],
    valid_labels['week_label'],
    normalize='index'
) * 100

fig, ax = plt.subplots(figsize=(12, 5))
cross[['fat_loss', 'maintenance', 'weight_gain']].plot(
    kind='bar', ax=ax,
    color=['#10B981', '#6B7280', '#EF4444'],
    edgecolor='white', linewidth=0.5
)
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('% of Days in Each Week Type', fontsize=11)
ax.set_title('Which Behavioral Cluster Led to Fat Loss vs Weight Gain?\n(% breakdown per cluster)',
             fontsize=13, pad=12)
ax.legend(['Fat Loss', 'Maintenance', 'Weight Gain'], fontsize=10)
ax.tick_params(axis='x', rotation=20)
ax.spines[['top', 'right']].set_visible(False)
for bar in ax.patches:
    if bar.get_height() > 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
save_chart(fig, 'cluster_weight_outcomes')

# ── Chart 4: PCA 2D scatter (visualize clusters in 2D) ──────────
print("  Generating: PCA cluster visualization...")
pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100

fig, ax = plt.subplots(figsize=(10, 8))
for cid in range(best_k):
    mask = cluster_df['cluster'] == cid
    ax.scatter(
        X_2d[mask.values, 0],
        X_2d[mask.values, 1],
        c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
        label=cluster_name_map[cid],
        s=15, alpha=0.5
    )
    # Cluster center
    cx = X_2d[mask.values, 0].mean()
    cy = X_2d[mask.values, 1].mean()
    ax.scatter(cx, cy, c=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
               s=200, marker='*', edgecolors='black', linewidths=0.8, zorder=5)
    ax.annotate(cluster_name_map[cid].split()[1] if len(cluster_name_map[cid].split()) > 1 else cluster_name_map[cid],
                (cx, cy), textcoords='offset points', xytext=(8, 5), fontsize=9, fontweight='bold')

ax.set_xlabel(f'Component 1 ({var1:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'Component 2 ({var2:.1f}% variance)', fontsize=11)
ax.set_title('Behavioral Clusters — PCA 2D View\n(Each dot = one day; stars = cluster centers)',
             fontsize=13, pad=12)
ax.legend(fontsize=9, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_chart(fig, 'cluster_pca')

# Save data with clusters
df_with_clusters = df.merge(
    cluster_df[['date', 'cluster', 'cluster_name']],
    on='date', how='left'
)
df_with_clusters.to_csv(os.path.join(OUTPUT_DIR, 'data_with_clusters.csv'), index=False)
print(f"\n  ✅  Clustering complete. Output: ml_output/data_with_clusters.csv")


# ══════════════════════════════════════════════════════════════════
#  PART 2 — PHASE DETECTION
#  "What distinct life phases existed in your health journey?"
#
#  Method: KMeans on 30-day rolling behavioral averages.
#  Clusters contiguous blocks of time into phases based on how
#  your habits shifted — not just weight.
# ══════════════════════════════════════════════════════════════════
section("PART 2 — PHASE DETECTION")

# ── Rolling 30-day behavioral averages ──────────────────────────
PHASE_FEATURES = [
    'weight',
    'step_count',
    'sleep_hours',
    'avg_heart_rate',
    'workout_intensity_score',
    'net_calorie_balance',
    'exercise_duration',
]

phase_df = df[['date'] + PHASE_FEATURES].copy()
for col in PHASE_FEATURES:
    phase_df[f'{col}_30d'] = phase_df[col].rolling(30, min_periods=15).mean()

rolling_cols = [f'{c}_30d' for c in PHASE_FEATURES]
phase_df = phase_df.dropna(subset=rolling_cols)

print(f"\n  Detecting phases on {len(phase_df):,} days using 30-day rolling averages")
print(f"  Features: {', '.join(PHASE_FEATURES)}")

# ── Normalize & cluster ─────────────────────────────────────────
scaler_p = StandardScaler()
X_phase  = scaler_p.fit_transform(phase_df[rolling_cols])

# Find best k for phases (we expect ~4-6 distinct life phases)
sil_p = []
for k in range(3, 8):
    km_p  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl   = km_p.fit_predict(X_phase)
    sil_p.append(silhouette_score(X_phase, lbl))
    print(f"  k={k}  silhouette={sil_p[-1]:.3f}")

best_k_p = list(range(3, 8))[np.argmax(sil_p)]
print(f"\n  ✅  Optimal phases = {best_k_p}")

km_phase = KMeans(n_clusters=best_k_p, random_state=42, n_init=15)
phase_df = phase_df.copy()
phase_df['phase_raw'] = km_phase.fit_predict(X_phase)

# ── Name phases by their behavioral profile ─────────────────────
phase_centers = pd.DataFrame(
    scaler_p.inverse_transform(km_phase.cluster_centers_),
    columns=rolling_cols
)

# Rank phases: primary sort = weight (descending = heaviest first in time typically)
# Secondary = net calorie balance (most negative = cutting)
phase_centers['weight_rank']   = phase_centers['weight_30d'].rank(ascending=False)
phase_centers['deficit_rank']  = phase_centers['net_calorie_balance_30d'].rank(ascending=True)
phase_centers['activity_rank'] = phase_centers['step_count_30d'].rank(ascending=False)
phase_centers['combined_rank'] = (
    phase_centers['weight_rank']   * 0.4 +
    phase_centers['deficit_rank']  * 0.3 +
    phase_centers['activity_rank'] * 0.3
)

PHASE_LABEL_MAP = {
    1: 'Heavy / High Weight Phase',
    2: 'Active Cutting Phase',
    3: 'Post-Cut Rebound',
    4: 'Maintenance Plateau',
    5: 'Low Activity Phase',
    6: 'Recovery Phase',
}

phase_rank = phase_centers['combined_rank'].rank().astype(int)
phase_name_map = {idx: PHASE_LABEL_MAP.get(r, f'Phase {r}') for idx, r in phase_rank.items()}
phase_df['phase_name'] = phase_df['phase_raw'].map(phase_name_map)

print("\n  Phase Profiles (30-day rolling averages):")
for pid in range(best_k_p):
    mask   = phase_df['phase_raw'] == pid
    days   = mask.sum()
    pct    = days / len(phase_df) * 100
    avg_wt = phase_df.loc[mask, 'weight_30d'].mean()
    avg_st = phase_df.loc[mask, 'step_count_30d'].mean()
    avg_sl = phase_df.loc[mask, 'sleep_hours_30d'].mean()
    avg_ca = phase_df.loc[mask, 'net_calorie_balance_30d'].mean()
    print(f"\n  Phase {pid}: {phase_name_map[pid]}")
    print(f"    Days: {days} ({pct:.0f}%)  |  Avg Weight: {avg_wt:.1f} kg  "
          f"|  Steps: {avg_st:.0f}  |  Sleep: {avg_sl:.1f}h  |  Balance: {avg_ca:+.0f} kcal")

# ── Smooth phase labels (remove noise — short gaps flip phase) ───
# Apply a rolling mode to smooth out 1-2 day flips between phases
def rolling_mode(series, window=14):
    result = series.copy()
    for i in range(len(series)):
        lo = max(0, i - window // 2)
        hi = min(len(series), i + window // 2)
        result.iloc[i] = series.iloc[lo:hi].mode().iloc[0]
    return result

phase_df['phase_smooth'] = rolling_mode(phase_df['phase_raw'], window=21)
phase_df['phase_name_smooth'] = phase_df['phase_smooth'].map(phase_name_map)

# ── Chart 5: Phase timeline ──────────────────────────────────────
print("\n  Generating: Phase timeline chart...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

# Top: weight with phase-colored background
prev_phase = None
prev_date  = None

for i, row in phase_df.iterrows():
    if row['phase_smooth'] != prev_phase:
        if prev_phase is not None:
            ax1.axvspan(prev_date, row['date'],
                        alpha=0.15,
                        color=PHASE_COLORS[int(prev_phase) % len(PHASE_COLORS)])
        prev_phase = row['phase_smooth']
        prev_date  = row['date']

# Final segment
if prev_date is not None:
    ax1.axvspan(prev_date, phase_df['date'].iloc[-1],
                alpha=0.15,
                color=PHASE_COLORS[int(prev_phase) % len(PHASE_COLORS)])

# Weight line
weight_s = df.set_index('date')['weight'].rolling('30D').mean()
ax1.plot(weight_s.index, weight_s.values, color='#1E293B', linewidth=2)
ax1.set_ylabel('Weight (kg)', fontsize=11)
ax1.set_title('Health Journey — Life Phases Detected by ML\n(Background color = detected life phase)',
              fontsize=13, pad=12)
ax1.spines[['top', 'right']].set_visible(False)

# Legend patches
patches = [mpatches.Patch(color=PHASE_COLORS[pid % len(PHASE_COLORS)],
                           label=phase_name_map[pid], alpha=0.7)
           for pid in range(best_k_p)]
ax1.legend(handles=patches, fontsize=9, loc='upper right', ncol=2)

# Bottom: phase label bars
phase_df['phase_y'] = 1
for pid in range(best_k_p):
    mask = phase_df['phase_smooth'] == pid
    ax2.scatter(phase_df.loc[mask, 'date'],
                [pid] * mask.sum(),
                c=PHASE_COLORS[pid % len(PHASE_COLORS)],
                s=14, marker='s', alpha=0.8)

ax2.set_yticks(range(best_k_p))
ax2.set_yticklabels([phase_name_map[i] for i in range(best_k_p)], fontsize=9)
ax2.set_xlabel('Date', fontsize=11)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_chart(fig, 'phase_timeline')

# ── Chart 6: Phase profile comparison ───────────────────────────
print("  Generating: Phase profiles chart...")
profile_features = ['weight_30d', 'step_count_30d', 'sleep_hours_30d',
                    'net_calorie_balance_30d', 'workout_intensity_score_30d']
profile_labels   = ['Weight (kg)', 'Steps/day', 'Sleep (hrs)',
                    'Calorie Balance', 'Workout Intensity']

fig, axes = plt.subplots(1, len(profile_features), figsize=(18, 6))

for ax, feat, lbl in zip(axes, profile_features, profile_labels):
    phase_means = [phase_df[phase_df['phase_smooth'] == pid][feat].mean()
                   for pid in range(best_k_p)]
    bars = ax.bar(range(best_k_p), phase_means,
                  color=[PHASE_COLORS[pid % len(PHASE_COLORS)] for pid in range(best_k_p)],
                  edgecolor='white')
    ax.set_xticks(range(best_k_p))
    ax.set_xticklabels([f'P{i}' for i in range(best_k_p)], fontsize=9)
    ax.set_title(lbl, fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    for bar_item, val in zip(bars, phase_means):
        ax.text(bar_item.get_x() + bar_item.get_width()/2,
                bar_item.get_height() * 1.01,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Phase Behavior Profiles — Average Metrics per Detected Life Phase',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_chart(fig, 'phase_profiles')

# Save data with phases
df_with_phases = df.merge(
    phase_df[['date', 'phase_raw', 'phase_smooth', 'phase_name_smooth']].rename(
        columns={'phase_name_smooth': 'phase_name'}),
    on='date', how='left'
)
df_with_phases.to_csv(os.path.join(OUTPUT_DIR, 'data_with_phases.csv'), index=False)
print(f"\n  ✅  Phase detection complete. Output: ml_output/data_with_phases.csv")


# ══════════════════════════════════════════════════════════════════
#  PART 3 — FEATURE IMPORTANCE ANALYSIS
#  "Which habits actually correlated with fat loss in your data?"
#
#  Two lenses:
#  A) Classification: what separates fat_loss weeks from others?
#  B) Regression: what correlates with weight_change_7d magnitude?
#  Both are on EXISTING historical data — no future prediction.
# ══════════════════════════════════════════════════════════════════
section("PART 3 — FEATURE IMPORTANCE ANALYSIS")

# ── Features for importance analysis ────────────────────────────
IMPORTANCE_FEATURES = [
    'step_count',
    'sleep_hours',
    'exercise_duration',
    'workout_intensity_score',
    'weekly_workout_count',
    'avg_heart_rate',
    'net_calorie_balance',
    'total_calories_burned',
    'water_ml',
    'sleep_debt_7d',
    'sleep_consistency_7d',
    'floor_count',
    'calorie_balance_7d_avg',
    'steps_avg_prev_week',
    'sleep_avg_prev_week',
    'workouts_prev_14d',
    'consecutive_workout_days',
    'rest_days_since_workout',
    'is_weekend',
    'month',
]

# ── 3A: Classification — fat_loss vs rest ───────────────────────
print("\n  [3A] Training: Fat Loss Week Classifier (existing data)...")

imp_df = df[IMPORTANCE_FEATURES + ['week_label', 'weight_change_7d', 'date']].copy()
imp_df = imp_df[imp_df['week_label'].isin(['fat_loss', 'maintenance', 'weight_gain'])]
imp_df = imp_df.dropna(subset=IMPORTANCE_FEATURES + ['week_label'])

X_imp = imp_df[IMPORTANCE_FEATURES].fillna(0)

# Binary: fat_loss (1) vs everything else (0)
y_binary = (imp_df['week_label'] == 'fat_loss').astype(int)

rfc_imp = RandomForestClassifier(
    n_estimators=500, max_depth=8, min_samples_leaf=20,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rfc_imp.fit(X_imp, y_binary)

fi_clf = pd.Series(rfc_imp.feature_importances_, index=IMPORTANCE_FEATURES)
fi_clf = fi_clf.sort_values(ascending=False)

print(f"\n  Top 10 behaviors that characterize your FAT LOSS weeks:")
for i, (feat, imp) in enumerate(fi_clf.head(10).items()):
    bar = '█' * int(imp * 300)
    print(f"    {i+1:2d}. {feat:35s}  {imp:.4f}  {bar}")

# ── Chart 7: Feature importance for fat loss classification ─────
print("\n  Generating: Feature importance — fat loss classifier...")
fig, ax = plt.subplots(figsize=(11, 8))
top20_clf = fi_clf.head(20)
bar_colors = ['#10B981' if i < 5 else '#6B7280' for i in range(len(top20_clf))]
bars = ax.barh(top20_clf.index[::-1], top20_clf.values[::-1],
               color=bar_colors[::-1], edgecolor='white', linewidth=0.5)
for b, v in zip(bars, top20_clf.values[::-1]):
    ax.text(b.get_width() + 0.001, b.get_y() + b.get_height()/2,
            f'{v:.4f}', va='center', fontsize=9, color='#374151')
ax.set_title('Feature Importance — What Behaviors Characterized Your Fat Loss Weeks?\n'
             '(Random Forest trained on your historical data — top = most important)',
             fontsize=12, pad=15)
ax.set_xlabel('Importance Score', fontsize=11)
ax.axvline(x=fi_clf.values[4], color='red', linestyle='--', alpha=0.3, linewidth=1)
ax.text(fi_clf.values[4] + 0.001, 0.5, 'top 5 cutoff', color='red', fontsize=8, alpha=0.6,
        transform=ax.get_xaxis_transform())
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_chart(fig, 'feature_importance_fatloss')

# ── 3B: Regression — what correlates with weight change? ────────
print("\n  [3B] Training: Weight Change Regressor (historical correlation)...")

reg_df2 = df[IMPORTANCE_FEATURES + ['weight_change_7d']].dropna()
X_reg2  = reg_df2[IMPORTANCE_FEATURES].fillna(0)
y_reg2  = reg_df2['weight_change_7d']   # negative = lost weight

rfr_imp = RandomForestRegressor(
    n_estimators=500, max_depth=8, min_samples_leaf=20,
    random_state=42, n_jobs=-1
)
rfr_imp.fit(X_reg2, y_reg2)

fi_reg = pd.Series(rfr_imp.feature_importances_, index=IMPORTANCE_FEATURES)
fi_reg = fi_reg.sort_values(ascending=False)

print(f"\n  Top 10 features correlated with weight change:")
for i, (feat, imp) in enumerate(fi_reg.head(10).items()):
    bar = '█' * int(imp * 300)
    print(f"    {i+1:2d}. {feat:35s}  {imp:.4f}  {bar}")

# ── Chart 8: Feature importance for weight change ───────────────
print("\n  Generating: Feature importance — weight change regressor...")
fig, ax = plt.subplots(figsize=(11, 8))
top20_reg = fi_reg.head(20)
bar_colors2 = ['#2563EB' if i < 5 else '#6B7280' for i in range(len(top20_reg))]
bars2 = ax.barh(top20_reg.index[::-1], top20_reg.values[::-1],
                color=bar_colors2[::-1], edgecolor='white', linewidth=0.5)
for b, v in zip(bars2, top20_reg.values[::-1]):
    ax.text(b.get_width() + 0.001, b.get_y() + b.get_height()/2,
            f'{v:.4f}', va='center', fontsize=9, color='#374151')
ax.set_title('Feature Importance — What Behaviors Correlated with Weight Change?\n'
             '(Magnitude of change, positive or negative — trained on your history)',
             fontsize=12, pad=15)
ax.set_xlabel('Importance Score', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
save_chart(fig, 'feature_importance_weightchange')

# ── Chart 9: Correlation heatmap — fat loss weeks only ──────────
print("  Generating: Correlation heatmap (fat loss weeks)...")
fat_loss_days = df[df['week_label'] == 'fat_loss'][IMPORTANCE_FEATURES[:12]].dropna()
corr_matrix   = fat_loss_days.corr()

fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap — Your FAT LOSS Days Only\n'
             '(How do your habits interact on days when you were losing weight?)',
             fontsize=12, pad=15)
plt.tight_layout()
save_chart(fig, 'correlation_heatmap_fatloss')

# ── Save feature importance table ───────────────────────────────
fi_combined = pd.DataFrame({
    'feature':               IMPORTANCE_FEATURES,
    'importance_fatloss_classifier': [fi_clf.get(f, 0) for f in IMPORTANCE_FEATURES],
    'importance_weightchange_regressor': [fi_reg.get(f, 0) for f in IMPORTANCE_FEATURES],
})
fi_combined['avg_importance'] = (
    fi_combined['importance_fatloss_classifier'] +
    fi_combined['importance_weightchange_regressor']
) / 2
fi_combined = fi_combined.sort_values('avg_importance', ascending=False).reset_index(drop=True)
fi_combined.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
print(f"\n  ✅  Feature importance saved → ml_output/feature_importance.csv")


# ══════════════════════════════════════════════════════════════════
#  SUMMARY & INSIGHTS
# ══════════════════════════════════════════════════════════════════
section("SUMMARY & KEY INSIGHTS")

# Best cluster for fat loss
best_cluster_for_fatloss = cross['fat_loss'].idxmax() if 'fat_loss' in cross.columns else 'N/A'
top_feature_fatloss      = fi_clf.index[0]
top_feature_weightchange = fi_reg.index[0]

summary = f"""
================================================================================
  ML ANALYSIS SUMMARY — SAMSUNG HEALTH PROJECT
  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

PART 1 — CLUSTERING ({best_k} behavioral clusters found)
{chr(10).join(f'  Cluster {i}: {cluster_name_map[i]}  ({(cluster_df["cluster"]==i).sum()} days)' for i in range(best_k))}

  Best cluster for fat loss: {best_cluster_for_fatloss}
  → This is the behavioral profile most associated with your weight loss weeks.

PART 2 — PHASE DETECTION ({best_k_p} life phases detected)
{chr(10).join(f'  Phase {i}: {phase_name_map[i]}' for i in range(best_k_p))}

  → These phases were automatically detected from changes in your 30-day
    rolling behavior patterns — no manual labeling needed.

PART 3 — FEATURE IMPORTANCE

  A) Fat Loss Week Classifier — top 5 features:
{chr(10).join(f'     {i+1}. {feat}  ({imp:.4f})' for i, (feat, imp) in enumerate(fi_clf.head(5).items()))}

  B) Weight Change Regressor — top 5 features:
{chr(10).join(f'     {i+1}. {feat}  ({imp:.4f})' for i, (feat, imp) in enumerate(fi_reg.head(5).items()))}

  KEY INSIGHT:
  → {top_feature_fatloss} is the #1 behavioral predictor of your fat loss weeks.
  → {top_feature_weightchange} is most correlated with actual weight change magnitude.
  → Focus on these two levers above everything else.

OUTPUT FILES:
  ml_charts/    → 9 charts
  ml_output/    → data_with_clusters.csv, data_with_phases.csv,
                   feature_importance.csv, analysis_summary.txt

================================================================================
"""

print(summary)

with open(os.path.join(OUTPUT_DIR, 'analysis_summary.txt'), 'w') as f:
    f.write(summary)

# ── Final tally ──────────────────────────────────────────────────
print("="*70)
print("  ✅  ML ANALYSIS COMPLETE")
print("="*70)
print(f"\n  {chart_count} charts saved to: ml_charts/")
print(f"  4 data files saved to:         ml_output/")
print(f"""
  NEXT STEPS:
    → Phase 6: Import data_with_clusters.csv + data_with_phases.csv into Power BI
    → Use cluster labels and phase labels as slicers in your dashboard
    → Phase 7: AI simulation and scenario optimization
""")
