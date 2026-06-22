# Samsung Health Personal Analytics

A personal data science project analyzing 6+ years of Samsung Health data to uncover behavioral patterns, life phases, and the key habits that drove fat loss.

## Project Overview

Raw Samsung Health exports → cleaned data → feature engineering → EDA → ML analysis.

**Data:** ~2,400 days of personal health metrics (steps, sleep, weight, heart rate, workouts, calories, water intake)

## Pipeline

| Script | Description |
|--------|-------------|
| `01_clean_data.py` | Clean and standardize raw Samsung Health CSVs |
| `02_build_master.py` | Merge all sources into a single daily master dataset |
| `03_fabricate_data.py` | Augment sparse periods with statistically consistent data |
| `04_feature_engineering.py` | Build lag features, rolling averages, week labels |
| `04b_feature_selection.py` | Trim 108 engineered features down to the top 25 by importance |
| `05_eda.py` | Exploratory data analysis — 21 charts |
| `06_ml_analysis.py` | ML analysis — clustering, phase detection, feature importance |
| `06_ml_models.py` | Fat-loss week classifier, weight predictor, goal-achievement simulation |

## ML Models Used

- **K-Means Clustering** — grouped days into behavioral profiles (Fat Loss Mode, Active Maintenance, etc.)
- **K-Means on rolling averages** — detected distinct life phases automatically
- **Random Forest Classifier** — identified habits that characterized fat loss weeks
- **Random Forest Regressor** — ranked habits by correlation with weight change, and predicts weight 7 days out
- **PCA** — 2D visualization of behavioral clusters

## Feature Selection (Bonus Step)

`04_feature_engineering.py` generates 108 candidate features — more than any model needs, and some are mostly empty or redundant. `04b_feature_selection.py` trims this down:

1. **Drop high-null features** (>50% missing) — 6 dropped, e.g. `body_fat` (99.9% missing)
2. **Drop redundant features** (|correlation| > 0.90 with another feature) — 19 dropped, e.g. `steps_avg_prev_month` was 99.6% correlated with `steps_30d_avg`
3. **Rank survivors by Random Forest importance**, keep top 25
4. **Validate the cut** — retrained the fat-loss classifier on the trimmed set vs. the full survivor set (same 2019–2024 train / 2025 test split): **20.1% → 22.0% accuracy**, with 22 fewer features

Result: `master_selected.csv` (108 → 29 columns). This is kept as a standalone, documented step — the main EDA/ML pipeline (05/06) still runs on the full `master_featured.csv`; this demonstrates the feature-reduction methodology and its measured effect rather than replacing the primary pipeline. See `ml_output/feature_selection_report.txt` for full output.

## Key Findings

- `month` is the #1 predictor of fat loss weeks — seasonal patterns dominate
- `calorie_balance_7d_avg` and `rest_days_since_workout` are the two most actionable levers
- 6 distinct life phases detected automatically from behavioral shifts, no manual labeling needed
- At current habits (+201 kcal/day average), no fat loss is happening — model projects ~32 weeks to goal weight under an optimal -500 kcal/day deficit scenario

## Output

- `eda_charts/` — 21 exploratory charts
- `ml_charts/` — 14 ML visualization charts (clustering, phases, feature importance, goal simulation)
- `ml_output/` — clustered data, phase labels, feature importance rankings, feature selection report
- `ml_results/` — week predictions, scenario simulation results, model summary
- `master_selected.csv` — lean 29-column feature set (see Feature Selection above)

## Stack

Python · pandas · scikit-learn · matplotlib · seaborn

---

*Built with assistance from [Claude](https://claude.ai) (Anthropic)*
