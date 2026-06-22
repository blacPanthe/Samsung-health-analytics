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
| `05_eda.py` | Exploratory data analysis — 21 charts |
| `06_ml_analysis.py` | ML analysis — clustering, phase detection, feature importance |

## ML Models Used

- **K-Means Clustering** — grouped days into behavioral profiles (Fat Loss Mode, Active Maintenance, etc.)
- **K-Means on rolling averages** — detected distinct life phases automatically
- **Random Forest Classifier** — identified habits that characterized fat loss weeks
- **Random Forest Regressor** — ranked habits by correlation with weight change
- **PCA** — 2D visualization of behavioral clusters

## Key Findings

- `month` is the #1 predictor of fat loss weeks — seasonal patterns dominate
- `calorie_balance_7d_avg` and `rest_days_since_workout` are the two most actionable levers
- 6 distinct life phases detected automatically from behavioral shifts, no manual labeling needed

## Output

- `eda_charts/` — 21 exploratory charts
- `ml_charts/` — 9 ML visualization charts
- `ml_output/` — clustered data, phase labels, feature importance rankings

## Stack

Python · pandas · scikit-learn · matplotlib · seaborn

---

*Built with assistance from [Claude](https://claude.ai) (Anthropic)*
