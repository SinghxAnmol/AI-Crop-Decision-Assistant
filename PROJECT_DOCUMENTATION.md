# PROJECT DOCUMENTATION

## AI Crop Decision Assistant – Intelligent Crop Selection and Profit Prediction System

**Project Type:** Innovation
**Domain:** Artificial Intelligence & Agriculture
**SDG Alignment:** Goal 2 — Zero Hunger
**Targeted Outcome:** Optimization

---

## 1. Abstract

Agriculture forms the backbone of many economies, yet many farmers still
choose which crop to grow based on tradition, habit or guesswork rather than
data. This project, *AI Crop Decision Assistant*, is a web-based decision
support system that uses Machine Learning to recommend the most suitable
crop for a farmer's specific land, soil, water availability, season and
budget. Alongside the recommendation, the system predicts expected yield,
estimates cost, revenue and profit, assigns a risk level, suggests a basic
fertilizer plan, and shows a market demand score. A "What-If Analysis"
feature further allows the farmer to explore how increasing irrigation would
change the outcome. The system is built with Python, Flask, scikit-learn and
a simple HTML/CSS/JavaScript frontend, keeping the implementation simple,
explainable, and fully runnable offline for academic demonstration.

## 2. Introduction

Crop selection is one of the most important decisions a farmer makes each
season, directly affecting income, resource use and food security. Poor
crop choices — driven by lack of information about soil suitability, water
requirements, market demand or expected profitability — can lead to reduced
yields, wasted resources and financial loss. With the growth of data science
and machine learning, it is now possible to build lightweight, accessible
tools that use historical agricultural data to guide such decisions. This
project explores that idea at an academic scale: a simple, explainable
Machine Learning pipeline wrapped in an easy-to-use web dashboard.

## 3. Problem Statement

Farmers frequently lack quick access to a tool that combines their land's
specific soil, water, season and budget characteristics with historical
agricultural data to answer: *"Which crop should I grow, and what yield,
cost, profit and risk can I expect?"* Existing decision-making is often
informal, based on past personal experience or neighbouring farms, without
systematic comparison of alternatives.

## 4. Objectives

1. Build a machine learning model that recommends the most suitable crop
   for given land and environmental conditions.
2. Predict expected yield for the recommended and alternative crops.
3. Provide a transparent profit estimate (revenue − cost).
4. Provide an explainable, rule-based risk assessment.
5. Provide a simple fertilizer recommendation.
6. Provide a market demand score to give the farmer market context.
7. Allow comparison between the recommended crop and alternatives.
8. Provide a "What-If Analysis" so the farmer can explore the effect of
   increased irrigation before making a decision.
9. Package the entire system as a simple, offline, easy-to-explain academic
   project suitable for a Master's-level viva.

## 5. Existing System

Traditionally, crop selection relies on:
- Personal experience and family/community tradition.
- Advice from local agricultural extension officers (where available).
- Observing what neighbouring farms are growing.

These approaches are valuable but are informal, slow to adapt to changing
soil/market conditions, and do not systematically combine multiple factors
(soil, water, season, budget, market demand) into a single, comparable
recommendation. There is limited use of historical data-driven prediction at
the individual farmer level, especially for small and marginal farmers.

## 6. Proposed System

The proposed system, *AI Crop Decision Assistant*, is a Flask-based web
application backed by two Machine Learning models trained on an agricultural
dataset:

1. A **Crop Recommendation Model** (Random Forest Classifier) that predicts
   the most suitable crop, along with a Suitability Score for each candidate
   crop.
2. A **Yield Prediction Model** (Random Forest Regressor) that estimates the
   expected yield (quintals/acre) for the recommended crop and alternatives.

These predictions feed into simple, transparent formulas for profit and
risk, plus rule-based fertilizer guidance and a market demand score, all
presented through a clean dashboard with charts and comparison tables. A
What-If Analysis module lets the farmer simulate increased irrigation and
see the resulting change in yield, profit, suitability and risk.

## 7. Methodology

The project follows a standard applied Machine Learning workflow:

1. **Data Collection / Generation** — a realistic sample agricultural
   dataset is generated (`generate_dataset.py`) with 1200 records across 10
   crops, 5 states and 6 soil types, using realistic ranges for soil
   nutrients (N, P, K), pH, rainfall, temperature, market price and cost.
2. **Data Preprocessing** — missing value handling, categorical encoding
   (Label Encoding) and feature selection (`train_model.py`).
3. **Model Training** — two Random Forest models are trained: one
   classifier (crop recommendation) and one regressor (yield prediction).
4. **Model Evaluation** — classification metrics (accuracy, precision,
   recall, F1-score) and regression metrics (MAE, RMSE, R²) are computed on
   a held-out 20% test set.
5. **Model Persistence** — trained models are saved with `joblib` so they
   are trained once and simply loaded (not retrained) by the web app.
6. **Application Layer** — a Flask backend exposes `/`, `/predict`,
   `/what-if` and `/dashboard-data` routes; a plain HTML/CSS/JavaScript
   frontend collects farmer input and renders results, comparisons and
   charts (Chart.js).
7. **Derived Calculations** — profit, risk level, fertilizer tips and market
   demand are computed using simple, explainable formulas/rules layered on
   top of the ML predictions (not additional ML models), keeping the system
   easy to explain and maintain.

## 8. System Architecture

```
                +-------------------------+
                |     Farmer (Browser)     |
                +------------+-------------+
                             |
                     HTML / CSS / JS UI
                             |
                    (fetch API - JSON)
                             |
                +------------v-------------+
                |        Flask App          |
                |         (app.py)          |
                |                            |
                |  /            -> dashboard |
                |  /predict     -> recommend |
                |  /what-if     -> simulate  |
                |  /dashboard-data -> stats  |
                +------------+-------------+
                             |
        +--------------------+---------------------+
        |                                           |
+-------v--------+                        +---------v---------+
|  Trained ML     |                        |  Rule-based logic  |
|  Models (.pkl)  |                        |  (profit, risk,     |
|  - Crop model   |                        |   fertilizer,        |
|  - Yield model  |                        |   market demand)      |
+-------+--------+                        +---------+---------+
        |                                            |
        +------------------+-------------------------+
                             |
                +------------v-------------+
                |  agriculture_dataset.csv  |
                |     (data/ folder)         |
                +---------------------------+
```

## 9. Data Flow

1. Farmer submits the input form (state, district, soil type, land size,
   budget, water availability, season, previous crop).
2. Flask validates the input and builds a feature row, filling in
   NPK/pH/rainfall/temperature from dataset averages for the chosen soil
   type (since the simple form does not ask for those individually).
3. The Crop Recommendation Model predicts a probability for every crop; the
   top crops become the Recommended Crop + Alternatives.
4. For each candidate crop, the Yield Prediction Model estimates yield,
   which feeds into the profit formula together with the per-crop average
   market price and cost.
5. The risk formula, fertilizer rules and market demand lookup are applied.
6. The JSON result is sent back to the browser and rendered as result cards,
   a comparison table and charts.
7. For What-If Analysis, the same pipeline is re-run with an adjusted water
   availability value, and the before/after results are compared.

## 10. Algorithms

**Random Forest** (used for both models): an ensemble learning method that
builds many decision trees during training, each on a random subset of the
data and features (bagging). For classification, the final prediction is
the majority vote across trees (and `predict_proba` gives the fraction of
trees voting for each class, used here as the Suitability Score). For
regression, the final prediction is the average across trees.

Random Forest was chosen because it:
- Handles a mix of categorical (encoded) and numeric features well.
- Is robust to noise and less prone to overfitting than a single decision
  tree.
- Requires no feature scaling.
- Is well supported in scikit-learn and simple to explain in a viva.

## 11. Machine Learning Models

| Model | Type | Features | Target | Purpose |
|-------|------|----------|--------|---------|
| Crop Recommendation | RandomForestClassifier | State, District, Season, Soil Type, Water Availability, N, P, K, pH, Rainfall, Temperature | Crop | Recommend the most suitable crop + Suitability Score |
| Yield Prediction | RandomForestRegressor | Crop, Season, Soil Type, Water Availability, N, P, K, pH, Rainfall, Temperature | Yield Ratio (normalised) | Estimate expected yield in quintals/acre |

Both models are trained with an 80/20 train-test split and evaluated with
standard metrics (see Section 13 — Results).

## 12. Dataset

- **File:** `data/agriculture_dataset.csv`
- **Records:** 1200
- **Type:** Synthetically generated sample data using realistic agricultural
  ranges (not official government data — see `generate_dataset.py` for the
  exact generation logic and crop profiles used).
- **Coverage:** 5 states, 15 districts, 6 soil types, 3 seasons, 10 crops.
- **Fields:** State, District, Season, Soil_Type, Crop, Land_Size,
  Water_Availability, N, P, K, pH, Rainfall, Temperature, Yield,
  Market_Price, Cost_Per_Acre, Water_Requirement, Market_Demand, Risk.

## 13. Implementation

- **Backend:** `app.py` (Flask) loads the trained models once at startup and
  exposes JSON API routes for prediction and what-if analysis. Input
  validation prevents empty fields, non-numeric land size/budget, and
  negative values; unknown categorical values fall back gracefully instead
  of crashing.
- **Model Training:** `train_model.py` performs data loading, cleaning,
  encoding, feature selection, train/test split, model training, evaluation
  and saving — each as a clearly separated, commented step.
- **Frontend:** `templates/index.html`, `static/style.css`,
  `static/script.js` implement a tabbed single-page dashboard (Dashboard /
  Recommend / What-If / About) using plain JavaScript `fetch()` calls to the
  Flask API, with Chart.js for bar charts.

## 14. Results

On the generated dataset (1200 records, 80/20 split), typical results are:

**Crop Recommendation Model (Classification)**
- Accuracy: ≈ 0.90
- Precision (weighted): ≈ 0.90
- Recall (weighted): ≈ 0.90
- F1-score (weighted): ≈ 0.90

**Yield Prediction Model (Regression, on normalised yield ratio)**
- MAE: ≈ 0.03–0.04
- RMSE: ≈ 0.04–0.05
- R²: ≈ 0.45–0.55

(Exact values vary slightly if the dataset is regenerated, since generation
uses randomness; run `python train_model.py` to see the exact figures for
your copy of the dataset.)

The classification accuracy is high because soil type, season and water
availability strongly determine which crops are agronomically viable. The
yield R² is moderate, which is realistic — actual yield depends on many
real-world factors (weather variability, farming practices, pest incidence)
beyond what a simple feature set can fully capture, and the dataset
deliberately includes some random noise to reflect that.

## 15. Advantages

- Simple, explainable pipeline suitable for an academic viva.
- Runs fully offline after installation — no external API dependency.
- Modular but minimal: only the files that are genuinely needed.
- Clear separation between ML predictions and transparent business-logic
  formulas (profit, risk, fertilizer), making every number traceable.
- Interactive What-If Analysis adds a genuinely useful, easy-to-demo
  feature beyond a static prediction.

## 16. Limitations

- Uses a synthetically generated sample dataset, not verified real-world
  government agricultural data.
- Market prices and weather are static averages from the dataset, not
  live/real-time data.
- The simple 8-field form does not collect NPK/pH/rainfall/temperature
  directly from the farmer; these are approximated using dataset averages
  for the chosen soil type.
- Risk and fertilizer logic are rule-based approximations, not validated
  agronomic science.
- All outputs are estimates and should be validated with local agricultural
  experts before real-world use.

## 17. Future Scope

- Integrate a real-time weather API for live rainfall/temperature input.
- Integrate live mandi/market price data instead of dataset averages.
- Add multilingual and voice-based interaction for wider farmer accessibility.
- Build a companion mobile application.
- Incorporate satellite/remote-sensing data for soil and crop health
  monitoring.
- Use larger, region-specific, verified real-world datasets.
- Add more advanced time-series-based yield forecasting.
- Maintain a personalized history per farmer across multiple seasons.

## 18. Conclusion

*AI Crop Decision Assistant* demonstrates how a simple, explainable Machine
Learning pipeline — two Random Forest models combined with transparent
business-logic formulas — can provide meaningful, data-driven crop decision
support. By keeping the technology stack minimal (Python, Flask,
scikit-learn, plain HTML/CSS/JS) and every calculation traceable, the
project remains fully understandable for an MCA-level academic
demonstration while still addressing a genuine real-world problem aligned
with UN SDG 2: Zero Hunger. With access to real agricultural datasets and
live market/weather data, the same architecture could be extended into a
practical tool for real farmers.
