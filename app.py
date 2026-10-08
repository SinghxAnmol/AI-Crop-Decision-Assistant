"""
app.py
------
Flask backend for the AI Crop Decision Assistant.

Routes:
    /            -> Loads the main dashboard/UI (index.html) with some dashboard stats
    /predict     -> Takes farmer's input, returns crop recommendation + full analysis
    /what-if     -> Recalculates results after changing irrigation/water level
    /dashboard-data -> Returns chart data for the dashboard tab

The trained ML models (crop_model.pkl, yield_model.pkl) are loaded ONCE when the
app starts (NOT re-trained on every request) - this is important for performance
and is a common viva question.
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained models and supporting files (once, at startup)
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
DATA_PATH = os.path.join("data", "agriculture_dataset.csv")

try:
    crop_model = joblib.load(os.path.join(MODEL_DIR, "crop_model.pkl"))
    yield_model = joblib.load(os.path.join(MODEL_DIR, "yield_model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    crop_lookup = joblib.load(os.path.join(MODEL_DIR, "crop_lookup.pkl"))
    crop_features = joblib.load(os.path.join(MODEL_DIR, "crop_features.pkl"))
    yield_features = joblib.load(os.path.join(MODEL_DIR, "yield_features.pkl"))
    MODELS_LOADED = True
except Exception as e:
    # If models are missing, we don't crash the whole app - we show a clear
    # error message on the page instead (see the / route).
    print(f"WARNING: Could not load trained models ({e}).")
    print("Run 'python train_model.py' first.")
    MODELS_LOADED = False

df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()

if MODELS_LOADED:
    print("Models loaded successfully. Ready to serve predictions.")


@app.errorhandler(Exception)
def handle_any_error(e):
    """
    Safety net: if ANY route raises an exception we didn't anticipate,
    always return JSON (never Flask/Werkzeug's raw HTML error page).
    This matters because the frontend calls res.json() on every response -
    an HTML error page would fail to parse and show a confusing
    "could not reach the server" message even though the server WAS
    reached, it just crashed. The real error is still printed to the
    terminal running app.py so it can be debugged.
    """
    print(f"Unhandled error: {e}")
    return jsonify({"error": "An unexpected server error occurred. Check the terminal running 'python app.py' for details."}), 500

FERTILIZER_GUIDE = {
    # crop: (advice if N low, advice if P low, advice if K low, general note)
    "default_low_n": "Apply Urea / nitrogen-rich fertilizer to boost early growth.",
    "default_low_p": "Apply DAP / Single Super Phosphate to support root development.",
    "default_low_k": "Apply Muriate of Potash (MOP) to improve crop strength and yield.",
    "default_ok": "Current soil NPK levels look adequate for this crop; use a balanced maintenance dose.",
}


# ---------------------------------------------------------------------------
# Helper functions (kept simple and readable on purpose - viva friendly)
# ---------------------------------------------------------------------------

def safe_encode(encoder, value, fallback_index=0):
    """Encode a categorical value using a saved LabelEncoder.
    If the value was never seen during training, fall back to index 0
    instead of crashing (simple, beginner-friendly error handling)."""
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        return fallback_index


def water_label(value):
    if value < 40:
        return "Low"
    elif value < 70:
        return "Medium"
    return "High"


def water_value_from_label(label):
    return {"Low": 25, "Medium": 55, "High": 85}.get(label, 55)


def calculate_risk(water, rainfall, temperature, demand, suitability_pct):
    """
    Simple, explainable rule-based RISK calculation (0-100 risk score,
    then bucketed into Low / Medium / High).

    Higher risk comes from:
      - Low water availability (35% weight)
      - Rainfall far from a "normal" range for farming, 500-1500mm (25% weight)
      - Low market demand (20% weight)
      - Low crop suitability score from the model (20% weight)

    This is NOT a machine learning model - it is a transparent formula so it
    can be explained line-by-line in the viva.
    """
    water_risk = (100 - water) * 0.35

    if 500 <= rainfall <= 1500:
        rainfall_risk = 0
    else:
        rainfall_risk = min(100, abs(rainfall - 1000) / 15) * 0.25

    demand_risk = (100 - demand) * 0.20
    suitability_risk = (100 - suitability_pct) * 0.20

    total_risk = water_risk + rainfall_risk + demand_risk + suitability_risk

    if total_risk < 35:
        return "Low", round(total_risk, 1)
    elif total_risk < 60:
        return "Medium", round(total_risk, 1)
    return "High", round(total_risk, 1)


def recommend_fertilizer(crop, n, p, k):
    """Simple rule-based fertilizer recommendation using average NPK values
    of that crop in the dataset as the 'ideal' reference point."""
    crop_rows = df[df["Crop"] == crop]
    if crop_rows.empty:
        return [FERTILIZER_GUIDE["default_ok"]]

    ideal_n = crop_rows["N"].mean()
    ideal_p = crop_rows["P"].mean()
    ideal_k = crop_rows["K"].mean()

    tips = []
    if n < ideal_n * 0.85:
        tips.append(FERTILIZER_GUIDE["default_low_n"])
    if p < ideal_p * 0.85:
        tips.append(FERTILIZER_GUIDE["default_low_p"])
    if k < ideal_k * 0.85:
        tips.append(FERTILIZER_GUIDE["default_low_k"])

    if not tips:
        tips.append(FERTILIZER_GUIDE["default_ok"])

    return tips


def build_input_row(form):
    """Convert the raw farmer input (from the form) into the numeric feature
    row that the ML models expect. Uses dataset averages for NPK/pH/rainfall/
    temperature since the simple form does not ask the farmer for those
    (keeps the UI short); a more advanced version could let the user enter
    them directly."""
    soil = form["soil_type"]
    season = form["season"]
    state = form["state"]
    district = form["district"]
    water = float(form["water_availability"])

    soil_rows = df[df["Soil_Type"] == soil]
    ref = soil_rows if not soil_rows.empty else df

    n = float(ref["N"].mean())
    p = float(ref["P"].mean())
    k = float(ref["K"].mean())
    ph = float(ref["pH"].mean())
    rainfall = float(ref["Rainfall"].mean())
    temperature = float(ref["Temperature"].mean())

    return {
        "state": state, "district": district, "soil": soil, "season": season,
        "water": water, "n": n, "p": p, "k": k, "ph": ph,
        "rainfall": rainfall, "temperature": temperature,
    }


def predict_crops(row, top_n=4):
    """Run the crop recommendation model and return the top-N most suitable
    crops with their suitability score (from predict_proba)."""
    features = {
        "State_enc": safe_encode(encoders["State"], row["state"]),
        "District_enc": safe_encode(encoders["District"], row["district"]),
        "Season_enc": safe_encode(encoders["Season"], row["season"]),
        "Soil_Type_enc": safe_encode(encoders["Soil_Type"], row["soil"]),
        "Water_Availability": row["water"],
        "N": row["n"], "P": row["p"], "K": row["k"], "pH": row["ph"],
        "Rainfall": row["rainfall"], "Temperature": row["temperature"],
    }
    X = pd.DataFrame([features])[crop_features]
    probabilities = crop_model.predict_proba(X)[0]
    crop_classes = encoders["Crop"].inverse_transform(crop_model.classes_)

    ranked = sorted(zip(crop_classes, probabilities), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def predict_yield(crop, row):
    """Predict yield (quintals/acre) for a given crop + conditions.
    The model was trained on a normalised Yield_Ratio, so we convert the
    prediction back into real units using that crop's average yield."""
    features = {
        "Crop_enc": safe_encode(encoders["Crop"], crop),
        "Season_enc": safe_encode(encoders["Season"], row["season"]),
        "Soil_Type_enc": safe_encode(encoders["Soil_Type"], row["soil"]),
        "Water_Availability": row["water"],
        "N": row["n"], "P": row["p"], "K": row["k"], "pH": row["ph"],
        "Rainfall": row["rainfall"], "Temperature": row["temperature"],
    }
    X = pd.DataFrame([features])[yield_features]
    ratio = yield_model.predict(X)[0]
    avg_yield = crop_lookup.get(crop, {}).get("Avg_Yield", 10)
    predicted_yield = max(0.5, ratio * avg_yield)
    return round(predicted_yield, 2)


def build_full_result(crop, suitability_pct, row, land_size, budget):
    """Given a chosen crop, compute yield/revenue/cost/profit/risk/demand/
    fertilizer - i.e. everything shown on a result card."""
    lookup = crop_lookup.get(crop, {})
    market_price = lookup.get("Market_Price", 2000)
    cost_per_acre = lookup.get("Cost_Per_Acre", 20000)
    demand = lookup.get("Market_Demand", 50)
    water_requirement = lookup.get("Water_Requirement", "Medium")

    predicted_yield = predict_yield(crop, row)

    # --- Profit calculation (Section 7 of the spec) ---
    revenue = round(predicted_yield * market_price * land_size, 2)
    cost = round(cost_per_acre * land_size, 2)
    profit = round(revenue - cost, 2)

    risk_level, risk_score = calculate_risk(
        row["water"], row["rainfall"], row["temperature"], demand, suitability_pct
    )

    fertilizer_tips = recommend_fertilizer(crop, row["n"], row["p"], row["k"])

    affordable = cost <= budget if budget else True

    return {
        "crop": crop,
        "suitability": round(suitability_pct, 1),
        "expected_yield": predicted_yield,
        "estimated_revenue": revenue,
        "estimated_cost": cost,
        "estimated_profit": profit,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "water_requirement": water_requirement,
        "market_demand": round(demand, 1),
        "market_price_per_quintal": round(market_price, 2),
        "fertilizer_tips": fertilizer_tips,
        "affordable_within_budget": affordable,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    dropdown_data = {
        "states": sorted(df["State"].unique().tolist()) if not df.empty else [],
        "districts": sorted(df["District"].unique().tolist()) if not df.empty else [],
        "soil_types": sorted(df["Soil_Type"].unique().tolist()) if not df.empty else [],
        "seasons": sorted(df["Season"].unique().tolist()) if not df.empty else [],
        "crops": sorted(df["Crop"].unique().tolist()) if not df.empty else [],
    }
    return render_template("index.html", models_loaded=MODELS_LOADED, dropdowns=dropdown_data)


@app.route("/dashboard-data")
def dashboard_data():
    """Returns simple aggregate stats + chart data for the Dashboard tab."""
    if df.empty:
        return jsonify({"error": "Dataset not found."}), 500

    try:
        crop_stats = df.groupby("Crop").agg(
            avg_yield=("Yield", "mean"),
            avg_demand=("Market_Demand", "mean"),
            avg_cost=("Cost_Per_Acre", "mean"),
            avg_price=("Market_Price", "mean"),
        ).round(1)
        crop_stats["avg_profit_per_acre"] = (
            (crop_stats["avg_yield"] * crop_stats["avg_price"]) - crop_stats["avg_cost"]
        ).round(0)

        return jsonify({
            "total_crops": int(df["Crop"].nunique()),
            "total_records": int(len(df)),
            "avg_yield": round(float(df["Yield"].mean()), 1),
            "avg_profit": round(float(crop_stats["avg_profit_per_acre"].mean()), 0),
            "crops": crop_stats.index.tolist(),
            "demand_by_crop": crop_stats["avg_demand"].tolist(),
            "yield_by_crop": crop_stats["avg_yield"].tolist(),
            "profit_by_crop": crop_stats["avg_profit_per_acre"].tolist(),
        })
    except Exception as e:
        print(f"Error in /dashboard-data: {e}")
        return jsonify({"error": "Could not load dashboard data."}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_LOADED:
        return jsonify({"error": "Models are not trained yet. Please run 'python train_model.py' first."}), 500

    try:
        form = request.get_json(force=True)

        # ---- basic input validation (Section 18 of the spec) ----
        required_fields = ["state", "district", "soil_type", "land_size",
                            "budget", "water_availability", "season"]
        for field in required_fields:
            if field not in form or str(form[field]).strip() == "":
                return jsonify({"error": f"Please provide a value for '{field}'."}), 400

        try:
            land_size = float(form["land_size"])
            budget = float(form["budget"])
        except ValueError:
            return jsonify({"error": "Land size and budget must be valid numbers."}), 400

        if land_size <= 0:
            return jsonify({"error": "Land size must be greater than 0 acres."}), 400
        if budget < 0:
            return jsonify({"error": "Budget cannot be negative."}), 400

        water_label_input = form["water_availability"]
        water_numeric = water_value_from_label(water_label_input) if water_label_input in \
            ("Low", "Medium", "High") else float(water_label_input)
        form["water_availability"] = water_numeric

        row = build_input_row(form)
        ranked_crops = predict_crops(row, top_n=4)

        results = []
        for crop_name, probability in ranked_crops:
            suitability_pct = round(float(probability) * 100, 1)
            results.append(build_full_result(crop_name, suitability_pct, row, land_size, budget))

        response = {
            "recommended": results[0],
            "alternatives": results[1:],
            "input_summary": {
                "state": form["state"], "district": form["district"],
                "soil_type": form["soil_type"], "season": form["season"],
                "land_size": land_size, "budget": budget,
                "water_availability": water_label(water_numeric),
                "previous_crop": form.get("previous_crop", "N/A"),
            },
            "raw_row": row,  # used later by /what-if
        }
        return jsonify(response)

    except Exception as e:
        # Never expose raw Python errors to the user (Section 18)
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Something went wrong while generating the recommendation. Please check your inputs and try again."}), 500


@app.route("/what-if", methods=["POST"])
def what_if():
    """
    Simple What-If Analysis (Section 3 of the spec):
    The user increases irrigation/water availability by a percentage and we
    recalculate suitability / yield / profit / risk for the SAME crop, so
    the farmer can see how the numbers change.
    """
    if not MODELS_LOADED:
        return jsonify({"error": "Models are not trained yet. Please run 'python train_model.py' first."}), 500

    try:
        payload = request.get_json(force=True)
        crop = payload["crop"]
        row = payload["raw_row"]
        land_size = float(payload["land_size"])
        budget = float(payload.get("budget", 0))
        irrigation_increase_pct = float(payload.get("irrigation_increase_pct", 20))

        # ---- BEFORE ----
        ranked_before = predict_crops(row, top_n=len(crop_lookup))
        suitability_before = next((p for c, p in ranked_before if c == crop), 0) * 100
        before = build_full_result(crop, suitability_before, row, land_size, budget)

        # ---- AFTER: bump up water availability ----
        new_row = dict(row)
        new_row["water"] = min(100, row["water"] * (1 + irrigation_increase_pct / 100))

        ranked_after = predict_crops(new_row, top_n=len(crop_lookup))
        suitability_after = next((p for c, p in ranked_after if c == crop), 0) * 100
        after = build_full_result(crop, suitability_after, new_row, land_size, budget)

        difference = {
            "yield_change": round(after["expected_yield"] - before["expected_yield"], 2),
            "profit_change": round(after["estimated_profit"] - before["estimated_profit"], 2),
            "suitability_change": round(after["suitability"] - before["suitability"], 1),
        }

        return jsonify({
            "crop": crop,
            "irrigation_increase_pct": irrigation_increase_pct,
            "before": before,
            "after": after,
            "difference": difference,
        })

    except Exception as e:
        print(f"Error in /what-if: {e}")
        return jsonify({"error": "Could not run the what-if analysis. Please generate a recommendation first."}), 500


if __name__ == "__main__":
    # debug=False so raw Python errors/stack traces are never shown to the
    # user (Section 18 of the spec). Set to True only while developing.
    # threaded=True lets the dev server handle the dashboard/chart requests
    # and a prediction request without one blocking the other.
    app.run(debug=False, port=5000, threaded=True)
