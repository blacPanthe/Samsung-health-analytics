# Samsung Health Analytics — 1-Page Project Summary
*For verbal explanation / interview prep*

## The 30-second pitch
"I built an end-to-end data science pipeline on 6 years of my own Samsung Health data — 2,400 days of steps, sleep, weight, heart rate, and workouts. I cleaned it, engineered 108 features, ran clustering and classification models, and ended up with a working tool that simulates how long it'd take me to hit a goal weight under different lifestyle scenarios."

## The pipeline (say this if asked "walk me through it")
1. **Cleaning** — raw Samsung exports are messy (shifted columns, junk rows, corrupted sessions). Standardized and fixed those.
2. **Master build** — merged 9 separate data sources (steps, sleep, calories, heart rate, water, weight, etc.) into one daily table by date.
3. **Augmentation** — many days had missing data (e.g. sleep was only ~4% real). I filled gaps using statistical models (day-of-week patterns, Harris-Benedict formula for calories) rather than dropping rows — this keeps the time series continuous, which matters for rolling/lag features.
4. **Feature engineering** — built 60 new features: 7-day sleep debt, rest-days-since-workout, calorie balance rolling averages, plateau detection, lag features (weight/steps 7/14/30 days ago).
5. **EDA** — 21 charts: weight trajectory, correlations, seasonal patterns.
6. **Modeling** — K-Means clustering (behavioral profiles), K-Means on rolling averages (life phases), Random Forest classifier (fat-loss weeks) and regressor (weight prediction + goal simulation).

## Behavioral drivers found (the actual insight)
- **`calorie_balance_7d_avg`** and **`rest_days_since_workout`** were consistently the top 2 actionable predictors across both the classifier and regressor — i.e., the smoothed calorie balance over the past week, and how long since the last workout, mattered more than any single day's data.
- **`month`** was the single strongest predictor overall — meaning *seasonality* drives my fat loss more than any individual habit. (Likely explains why cuts tend to start/succeed in specific months — e.g. New Year, summer prep.)
- Clustering surfaced two clear behavioral modes: **"Fat Loss Mode"** (high activity, calorie deficit, better sleep) vs **"Active Maintenance"** — and the data shows clearly which one actually overlaps with real fat-loss weeks.
- Phase detection (unsupervised, no manual labeling) found 6 distinct "life phases" over 6 years — e.g. an "Active Cutting Phase" and a "Maintenance Plateau" — purely from how my 30-day rolling behavior shifted, which lines up with what I remember actually happening.
- The goal-simulation model showed: at my current average (+201 kcal/day surplus), I'm not losing fat — I'm in maintenance. Under an optimal -500 kcal/day deficit scenario, it'd take ~32 weeks to hit my goal weight.

## Data storytelling approach (how I presented it)
- Didn't lead with model internals — led with **the question a normal person asks**: "what actually causes my fat loss weeks?"
- Used the EDA charts first to build trust in the data (show the raw weight journey, the plateaus, the seasonal pattern) *before* showing any ML output — so the model's conclusions feel grounded, not like a black box.
- Translated model output into **recommendations**, not just numbers: "add a 300–500 kcal deficit, +3,000 steps/day, 2+ workouts/week, 7.5 hrs sleep" — directly actionable, not just "feature X has importance 0.094."
- Was explicit about what's real vs. synthetic data (e.g. sleep was mostly imputed) — this matters for credibility when someone asks how trustworthy the conclusions are.

## The honest answer on the low accuracy score (44.8%)
This is the question to be ready for. Say it like this, calmly, not defensively:

> "The fat-loss week classifier hit 44.8% accuracy on a 3-class problem — better than the 33% random baseline, but not high. There are real reasons for that ceiling, not just a tuning issue:
> 1. It's a 3-class problem with an already-weak baseline (33% random).
> 2. A lot of the training data is **fabricated**, not measured — e.g. sleep was only ~4% real, the rest was statistically imputed. The model partly learns patterns I generated, not patterns from real behavior.
> 3. Weight change itself is noisy — water retention, sodium, hormones — none of which are in the dataset, so even a perfect model can't explain that noise.
> 4. I trained on 2019–2024 and tested on 2025 — a genuinely harder generalization task than a random split, since my routine changed across years.
>
> The more defensible number is the **regression model** — predicting next week's weight came out to 0.34kg mean absolute error, which is a tight, meaningful result. I'd lead with that, not the classifier accuracy, if asked which model performed best."

## What I'm most proud of (have 2-3 of these ready)
- **Catching the leakage problem myself.** When I built the feature-selection step, I deliberately excluded `weight`, `bmi`, and other weight-derived columns from the candidate features before training — because including them would let the model "cheat" by basically looking at the answer. That's a subtlety beginners often miss.
- **Being honest about synthetic data instead of hiding it.** Most of the dataset's missing values were filled in with statistical models, not real measurements (e.g. sleep was only ~4% real). I tracked and reported that fill rate explicitly rather than presenting augmented data as if it were all real — that's the kind of thing that builds trust with a technical reviewer.
- **The goal-simulation model is a genuinely usable output, not just a chart.** It doesn't just describe the past — it answers a forward-looking question ("how long until I hit my goal under X scenario") using the same model trained on historical data. That's the difference between an analysis and a tool.
- **Doing feature selection as a measured experiment, not a guess.** Instead of just trimming features and hoping it helped, I retrained the model before and after (20.1% → 22.0%) on the exact same train/test split to actually prove the cut helped rather than assuming it did.
- **6 life phases emerged with zero manual labeling.** I didn't tell the model where my "cutting phases" or "plateaus" were — it found them purely from how my 30-day rolling behavior shifted, and they lined up with what I actually remember happening in my life. That's a nice "the data confirms the story" moment.

## Methodology questions to be ready for

**"Why Random Forest, not deep learning / XGBoost / something fancier?"**
> "With 2,400 rows, a deep learning model would overfit and gives no interpretability — I needed feature importance, not just a prediction. Random Forest handles nonlinear relationships, doesn't need feature scaling, and gives a ranked importance output that I could turn into actual recommendations. The dataset size and the goal (interpretability) drove the choice, not unfamiliarity with bigger models."

**"Why K-Means for clustering, not DBSCAN or hierarchical?"**
> "K-Means is the right default when you expect roughly round, similarly-sized groups and want a fixed, interpretable number of clusters. I validated the choice of K using the elbow method plus silhouette score rather than guessing a number. DBSCAN is better suited to irregular-shaped or noisy clusters, which wasn't the situation here."

**"Why train on 2019-2024 and test on 2025 instead of a random split?"**
> "Because this is time-series, behavioral data — a random split would leak future information into training (e.g. a 2025 lag feature calculated from a 2025 value, evaluated on a 'past' row). A time-based split is the only honest way to test whether the model generalizes to data it hasn't seen yet, which is also why it's a harder bar than a random split — and partly why accuracy looks lower than people expect."

## Scope & limitations (say this proactively, don't wait to be asked)
- **This is N=1 data — it's about me, not a general population.** The behavioral drivers I found (month, calorie balance, rest days) describe my own patterns. I wouldn't claim they generalize to anyone else without testing on other people's data. I'd say this clearly if anyone asks "does this apply to everyone?"
- **Putting the 0.34kg MAE in context:** normal day-to-day body weight fluctuates by roughly 0.5-1kg just from water retention and sodium. A model that's off by only 0.34kg on average is performing *within* that natural noise floor — which is actually a strong result, not just a small-sounding number.

## If asked "what would you do differently / what's next"
- Use only real (non-fabricated) data for a cleaner, smaller, more trustworthy training set
- Add real food-log data instead of estimated calories
- Build the feature-selection step I added later (`04b_feature_selection.py`) into the main pipeline by default, not as a side experiment
- Turn the static charts into an interactive Streamlit/Power BI dashboard

## One-line closer
"This wasn't about getting the highest accuracy — it was about building a complete, honest pipeline: clean data, real findings, and a model good enough to turn into an actual decision tool."
