# AI Crop Decision Assistant

**Intelligent Crop Selection and Profit Prediction System**

An MCA final-year project that helps a farmer decide *"What crop should I grow?"*
based on their state, district, soil type, land size, budget, water availability,
season and previous crop — using Machine Learning trained on agricultural data.

- **Project Type:** Innovation
- **Domain:** Artificial Intelligence & Agriculture
- **SDG Alignment:** Goal 2 — Zero Hunger
- **Targeted Outcome:** Optimization

---

## 1. Introduction

Farmers often decide which crop to grow based on habit or guesswork, without
data-backed insight into expected yield, cost, profit, risk or market demand.
This project uses historical agricultural data and simple, explainable Machine
Learning models to recommend the most suitable crop for a farmer's specific
land and conditions, along with a full profit and risk analysis.

## 2. Problem Statement

Given a farmer's location, soil type, land size, budget, water availability,
season and previous crop, recommend the most suitable crop to grow, and
estimate its yield, cost, revenue, profit, risk, and market demand — so the
farmer can make an informed, data-driven decision instead of relying only on
tradition or guesswork.

## 3. Objectives

1. Recommend the most suitable crop for given land/soil/climate conditions.
2. Predict expected yield using a trained regression model.
3. Calculate estimated cost, revenue and profit.
4. Provide a simple, explainable risk level (Low / Medium / High).
5. Suggest a basic fertilizer recommendation.
6. Show a market demand score for the recommended crop.
7. Allow the farmer to compare the recommended crop against alternatives.
8. Provide a "What-If" analysis to see how more irrigation changes the outcome.

## 4. Features

- Simple 8-field input form (state, district, soil type, land size, budget,
  water availability, season, previous crop).
- AI-based crop recommendation with a Suitability Score.
- Top alternative crops with a side-by-side comparison table and chart.
- Expected Yield, Estimated Cost, Revenue and Profit (in ₹).
- Explainable Risk Level (Low / Medium / High).
- AI-based estimated Fertilizer Recommendation.
- Market Demand Score (0–100).
- What-If Analysis: see how increasing irrigation changes yield/profit/risk.
- Dashboard with dataset overview and charts (Chart.js).
- Clean, responsive, agriculture-themed dashboard UI.
- Runs completely offline after installation — no external API required.

## 5. Technology Stack

| Layer          | Technology                                  |
|----------------|----------------------------------------------|
| Frontend       | HTML, CSS, JavaScript, Chart.js               |
| Backend        | Python, Flask                                 |
| Machine Learning | Python, Pandas, NumPy, Scikit-learn         |
| Data Storage   | CSV dataset (`data/agriculture_dataset.csv`)  |
| Model Storage  | joblib (`.pkl` files in `models/`)            |

> The project structure keeps everything in a single lightweight Flask app
> with a CSV dataset — this was a deliberate choice to keep the project easy
> to run, explain and demonstrate, as required for an MCA academic project.

## 6. Folder Structure

```
AI-Crop-Decision-Assistant/
│
├── app.py                     # Flask backend (routes, prediction logic)
├── train_model.py             # Trains and saves the ML models
├── generate_dataset.py        # Creates the sample dataset (already run once)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── PROJECT_DOCUMENTATION.md   # Full academic documentation
│
├── data/
│   └── agriculture_dataset.csv   # Sample agricultural dataset (1200 records)
│
├── models/                    # Trained models (created by train_model.py)
│   ├── crop_model.pkl
│   ├── yield_model.pkl
│   ├── encoders.pkl
│   ├── crop_lookup.pkl
│   ├── crop_features.pkl
│   └── yield_features.pkl
│
├── templates/
│   └── index.html             # Main dashboard UI (Jinja2 template)
│
└── static/
    ├── style.css               # Styling
    └── script.js                # Frontend logic (tabs, form, charts, what-if)
```

## 7. Dataset Explanation

**File:** `data/agriculture_dataset.csv` (1200 records)

This is a **synthetically generated sample dataset** (see `generate_dataset.py`),
built using realistic agricultural ranges (typical NPK values, rainfall,
temperature, market price bands and cost-per-acre for 10 common Indian
crops). **It is not official government data** — it was created so the ML
models have enough consistent, realistic data to learn from and to
demonstrate the full project end-to-end without depending on an external
API or a hard-to-access dataset.

| Column               | Description                                       |
|----------------------|-----------------------------------------------------|
| State, District      | Location of the farm                                 |
| Season               | Kharif / Rabi / Zaid                                  |
| Soil_Type            | Alluvial / Black / Red / Loamy / Sandy / Clay          |
| Crop                 | Crop grown (target for recommendation model)           |
| Land_Size            | Size of land in acres                                  |
| Water_Availability   | Irrigation availability (0–100 scale)                   |
| N, P, K              | Nitrogen, Phosphorus, Potassium levels in soil           |
| pH                   | Soil pH value                                             |
| Rainfall             | Rainfall in mm                                             |
| Temperature          | Average temperature in °C                                   |
| Yield                | Actual yield (quintals/acre) — target for yield model         |
| Market_Price         | Market price per quintal (₹)                                   |
| Cost_Per_Acre        | Cost of cultivation per acre (₹)                                 |
| Water_Requirement    | Typical water requirement label for that crop (Low/Med/High)      |
| Market_Demand        | Market demand score (0–100)                                        |
| Risk                 | Rule-based risk label (Low/Medium/High) for that record              |

Yield in the dataset is generated so that it genuinely depends on how well
the soil/season/water/pH/rainfall/temperature match each crop's ideal
growing conditions — this is what makes the ML models learn a meaningful
pattern instead of memorising noise.

## 8. Machine Learning Approach

Two simple, explainable models are used (both **Random Forest**, from
scikit-learn):

### a) Crop Recommendation — `RandomForestClassifier`
- **Features:** State, District, Season, Soil Type, Water Availability, N, P, K, pH, Rainfall, Temperature
- **Target:** Crop
- Outputs a **probability for every crop**; the crop with the highest
  probability becomes the "Recommended Crop", and its probability (×100)
  becomes the **Suitability Score**. The next-highest crops become
  "Alternative Crops".

### b) Yield Prediction — `RandomForestRegressor`
- **Features:** Crop, Season, Soil Type, Water Availability, N, P, K, pH, Rainfall, Temperature
- **Target:** a normalised `Yield_Ratio` (Yield ÷ that crop's average yield in
  the dataset), because different crops have very different yield scales
  (e.g. Sugarcane ≈ 400 quintals/acre vs Gram ≈ 10 quintals/acre). The
  prediction is converted back to real units in `app.py`.

### Why Random Forest?
- Works well on a mix of categorical + numeric tabular data.
- Easy to explain in a viva: *"many decision trees vote/average together, which
  reduces overfitting compared to a single decision tree."*
- Does not require feature scaling.
- Provides `predict_proba` for classification, which is used directly as the
  Suitability Score.

### Model Evaluation (printed by `train_model.py`)
- **Classification:** Accuracy, Precision, Recall, F1-score (weighted average across 10 crop classes)
- **Regression:** MAE, RMSE, R² (on the normalised yield ratio)

Typical results on this dataset: ~90% classification accuracy and R² around
0.5 for yield (yield naturally has more randomness/noise than "which crop
fits this land", which is expected and worth mentioning in the viva).

### Not Machine Learning (by design, for simplicity + explainability)
- **Profit** = a direct formula (see below), not a model.
- **Risk Level** = a transparent rule-based formula (see `calculate_risk()`
  in `app.py`), not a model.
- **Fertilizer Recommendation** = simple rule-based comparison against each
  crop's average NPK levels in the dataset.
- **Market Price / Cost / Demand** for the profit formula = looked up from a
  per-crop average table built from the dataset, not re-predicted — this
  keeps the whole pipeline simple enough to fully explain.

## 9. Key Calculations

**Profit calculation:**
```
Estimated Revenue = Predicted Yield × Market Price × Land Size
Estimated Cost    = Cost Per Acre × Land Size
Estimated Profit  = Estimated Revenue − Estimated Cost
```

**Risk calculation** (0–100 risk score, rule-based, then bucketed into
Low/Medium/High):
```
risk_score = 0.35 × (100 − water_availability)
           + 0.25 × (rainfall distance from the 500–1500mm normal range)
           + 0.20 × (100 − market_demand)
           + 0.20 × (100 − suitability_score)
```

## 10. Installation

```
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (macOS/Linux)

pip install -r requirements.txt
```

## 11. How to Train the Model

The dataset is already included, but you can regenerate it and retrain the
models any time:

```
python generate_dataset.py      # (optional) regenerates data/agriculture_dataset.csv
python train_model.py           # trains and saves the models into models/
```

This prints the evaluation metrics (accuracy, precision, recall, F1, MAE,
RMSE, R²) to the console.

## 12. How to Run the Application

```
python app.py
```

Then open your browser at: **http://127.0.0.1:5000/**

## 13. How to Use the Application

1. Open the **Dashboard** tab to see an overview of the dataset.
2. Go to the **Recommend** tab and fill in your state, district, soil type,
   land size, budget, water availability, season and (optionally) previous
   crop, then click **Get Recommendation**.
3. Review the recommended crop, suitability score, yield, cost, revenue,
   profit, risk, water requirement, market demand and fertilizer tips, and
   compare it against alternative crops.
4. Go to the **What-If** tab, adjust the irrigation increase slider, and
   click **Recalculate** to see how more irrigation would change the yield,
   profit, suitability and risk for the same crop.
5. Read the **About** tab for project background, SDG alignment and the
   disclaimer.

## 14. Expected Output (example)

```
Recommended Crop:      Maize
Suitability:           87%
Expected Yield:        22 Quintals/Acre
Estimated Cost:        ₹35,000
Estimated Revenue:     ₹70,000
Estimated Profit:      ₹35,000
Risk:                  Medium
Water Requirement:     Medium
Market Demand:         High (e.g. 82/100)
```

(Exact figures vary based on the inputs you enter — they are not hard-coded.)

## 15. Limitations

- Predictions depend on the quality and realism of the (sample) dataset.
- Market prices can change; the app uses average prices from the dataset.
- Weather conditions are uncertain and not fetched live.
- All results are **estimates**, not guarantees of actual farm outcomes.
- Actual field conditions may differ from the averages used for NPK/pH/
  rainfall/temperature (the simple form does not ask for these individually).
- Recommendations should always be validated with a local agricultural
  expert and current mandi/market prices before making real decisions.

## 17. Troubleshooting (Windows)

**"Preparing metadata (pyproject.toml) did not run successfully" / meson/vswhere error while installing pandas:**
This happens when pip can't find a pre-built wheel for your Python version and
tries to compile pandas from source, which needs Visual Studio build tools you
likely don't have. Fix: use `requirements.txt` as provided (it uses `>=`
instead of pinned exact versions) and make sure you're on Python 3.10–3.12
(run `python --version`). If you're on a very new Python version (3.13+),
either install Python 3.12 instead, or run:
```
pip install --upgrade pip
pip install -r requirements.txt
```
Upgrading pip first usually lets it find a compatible pre-built wheel and
skip compiling from source entirely.

**Charts don't appear / console shows "Chart is not defined":**
Fixed — Chart.js is now bundled locally at `static/chart.umd.js` and no
longer loaded from an external CDN, so it works fully offline and isn't
affected by network/firewall restrictions.

**"Could not reach the server" when clicking Get Recommendation:**
This message is shown by the browser when a request fails at the network
level, or when the server sends back a non-JSON error page. `app.py` now
includes a global error handler that guarantees every error is returned as
JSON, so if you still see this: (1) confirm the terminal running `python
app.py` shows `Running on http://127.0.0.1:5000` with no errors, (2) make
sure you're opening `http://127.0.0.1:5000/` in the browser (not a `file://`
path), and (3) check your terminal for a printed `Error in /predict: ...`
line right after you click the button — that tells you the exact cause.

## 18. Future Scope

- Real-time weather API integration.
- Live mandi/market price integration.
- Multilingual support for farmers.
- Voice-based interaction.
- Mobile application version.
- Satellite/remote-sensing data for soil and crop health.
- Larger, region-specific real datasets.
- Advanced yield forecasting (e.g. time-series weather trends).
- Personalized farmer history and recommendations over multiple seasons.

---

## VIVA PREPARATION

**1. What is the purpose of this project?**
To help a farmer decide which crop to grow, using Machine Learning trained
on agricultural data, and to show the expected yield, cost, profit, risk and
market demand for that crop.

**2. What is machine learning?**
A way of teaching a computer program to find patterns in existing data (the
training data) so it can make predictions on new, unseen data, instead of
being explicitly programmed with fixed rules for every case.

**3. Why did you use Random Forest?**
Random Forest builds many decision trees on random subsets of the data and
features, then combines their votes/averages. This reduces overfitting
compared to a single decision tree, works well with a mix of categorical and
numeric data, and needs no feature scaling — while staying easy to explain.

**4. What is a feature?**
An input variable given to the model to help it make a prediction — in this
project, examples are Soil Type, Water Availability, N, P, K, pH, Rainfall
and Temperature.

**5. What is a target variable?**
The output the model is trying to predict — "Crop" for the recommendation
model, and "Yield" (as a normalised ratio) for the yield prediction model.

**6. What is train-test split?**
Splitting the dataset into two parts: a larger part (80% here) used to train
the model, and a smaller part (20% here) kept aside and never seen during
training, used only to test how well the model performs on new data.

**7. What is accuracy?**
For classification, the percentage of predictions the model got exactly
right out of all predictions made on the test set.

**8. What is RMSE?**
Root Mean Squared Error — a regression metric that measures how far, on
average, the model's predictions are from the actual values (it squares the
errors first, which penalises larger mistakes more, then takes the square
root to bring it back to the original unit).

**9. How is profit calculated?**
Estimated Revenue = Predicted Yield × Market Price × Land Size.
Estimated Cost = Cost Per Acre × Land Size.
Estimated Profit = Estimated Revenue − Estimated Cost.

**10. How does crop recommendation work?**
A trained Random Forest Classifier looks at the farmer's location, soil,
water availability and typical soil/weather values for that soil type, and
outputs a probability for every crop in the dataset. The crop with the
highest probability is recommended, and that probability becomes the
Suitability Score.

**11. What is the What-If Analysis?**
A feature that lets the farmer see what would happen if their irrigation/
water availability increased by a chosen percentage — the system
recalculates suitability, yield, profit and risk for the same crop under the
new water availability value and shows the before/after difference.

**12. What are the limitations?**
The dataset is a realistic sample, not official real-world data; market
prices and weather are not live; and all outputs are estimates that should
be validated with local agricultural experts before real decisions are made.

**13. Why is this project useful for agriculture?**
It gives farmers a simple, data-driven starting point for choosing a crop —
supporting SDG 2 (Zero Hunger) by helping improve agricultural decision-
making, which can lead to better yields and incomes over time.
