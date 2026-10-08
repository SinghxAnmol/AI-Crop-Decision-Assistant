"""
train_model.py
---------------
Trains the two Machine Learning models used by this project:

1. Crop Recommendation Model  -> RandomForestClassifier
   Predicts WHICH crop is most suitable for given land/soil/weather conditions.

2. Yield Prediction Model     -> RandomForestRegressor
   Predicts the expected YIELD (quintals per acre) for a given crop and
   growing conditions.

Why Random Forest?
- It works well on tabular data with a mix of categorical + numeric features.
- It is easy to explain in a viva: "many decision trees vote / average together".
- It does not need feature scaling and handles non-linear relationships well.
- It gives feature_importances_, so we can explain WHY it predicted something.

Run this file once before starting the Flask app:
    python train_model.py

It reads:  data/agriculture_dataset.csv
It writes: models/crop_model.pkl
           models/yield_model.pkl
           models/encoders.pkl
           models/crop_lookup.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

DATA_PATH = os.path.join("data", "agriculture_dataset.csv")
MODEL_DIR = "models"

# ---------------------------------------------------------------------------
# STEP 1: Load the dataset
# ---------------------------------------------------------------------------
print("STEP 1: Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# STEP 2: Clean data (handle any missing values, if present)
# ---------------------------------------------------------------------------
print("STEP 2: Cleaning data...")
before = len(df)
df = df.dropna()  # simple approach: drop any incomplete rows
print(f"  Dropped {before - len(df)} rows with missing values")

# ---------------------------------------------------------------------------
# STEP 3: Encode categorical columns (ML models need numbers, not text)
# ---------------------------------------------------------------------------
print("STEP 3: Encoding categorical features...")
categorical_cols = ["State", "District", "Season", "Soil_Type", "Crop"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------------------------------------------------------
# STEP 4: Feature selection
# ---------------------------------------------------------------------------
print("STEP 4: Selecting features...")

# Features used to recommend a crop (we do NOT include Crop itself here)
crop_features = [
    "State_enc", "District_enc", "Season_enc", "Soil_Type_enc",
    "Water_Availability", "N", "P", "K", "pH", "Rainfall", "Temperature",
]
crop_target = "Crop_enc"

# Features used to predict yield (Crop IS included, because yield depends on
# which crop is grown)
yield_features = [
    "Crop_enc", "Season_enc", "Soil_Type_enc",
    "Water_Availability", "N", "P", "K", "pH", "Rainfall", "Temperature",
]

# NOTE ON SCALE: different crops have very different yield scales
# (e.g. Sugarcane ~ 350-450 quintals/acre vs Gram ~ 8-12 quintals/acre).
# If we train directly on raw "Yield", the model's error is dominated by
# the high-scale crops. So instead we train on a normalised target:
#   Yield_Ratio = Yield / (average yield of that crop in the dataset)
# and convert the prediction back to real units afterwards
# (predicted_yield = predicted_ratio * crop_average_yield).
# This is a simple, explainable trick, not a different algorithm.
crop_avg_yield = df.groupby("Crop")["Yield"].transform("mean")
df["Yield_Ratio"] = df["Yield"] / crop_avg_yield
yield_target = "Yield_Ratio"

print(f"  Crop model features:  {crop_features}")
print(f"  Yield model features: {yield_features}")

# ---------------------------------------------------------------------------
# STEP 5: Train/test split (80% train, 20% test)
# ---------------------------------------------------------------------------
print("STEP 5: Splitting data (80% train / 20% test)...")
X_crop = df[crop_features]
y_crop = df[crop_target]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42, stratify=y_crop
)

X_yield = df[yield_features]
y_yield = df[yield_target]
Xy_train, Xy_test, yy_train, yy_test = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# STEP 6: Train models
# ---------------------------------------------------------------------------
print("STEP 6: Training models...")

crop_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42
)
crop_model.fit(Xc_train, yc_train)

yield_model = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42
)
yield_model.fit(Xy_train, yy_train)

# ---------------------------------------------------------------------------
# STEP 7: Evaluate models
# ---------------------------------------------------------------------------
print("STEP 7: Evaluating models...\n")

# --- Crop recommendation (classification) ---
yc_pred = crop_model.predict(Xc_test)
acc = accuracy_score(yc_test, yc_pred)
prec = precision_score(yc_test, yc_pred, average="weighted", zero_division=0)
rec = recall_score(yc_test, yc_pred, average="weighted", zero_division=0)
f1 = f1_score(yc_test, yc_pred, average="weighted", zero_division=0)

print("Crop Recommendation Model (RandomForestClassifier)")
print(f"  Accuracy : {acc:.3f}")
print(f"  Precision: {prec:.3f}")
print(f"  Recall   : {rec:.3f}")
print(f"  F1-score : {f1:.3f}\n")

# --- Yield prediction (regression) ---
yy_pred = yield_model.predict(Xy_test)
mae = mean_absolute_error(yy_test, yy_pred)
rmse = np.sqrt(mean_squared_error(yy_test, yy_pred))
r2 = r2_score(yy_test, yy_pred)

print("Yield Prediction Model (RandomForestRegressor, trained on normalised Yield_Ratio)")
print(f"  MAE  : {mae:.3f}  (ratio units, e.g. 0.05 = ~5% off the crop's average yield)")
print(f"  RMSE : {rmse:.3f}  (ratio units)")
print(f"  R2   : {r2:.3f}\n")

# ---------------------------------------------------------------------------
# STEP 8: Build simple per-crop lookup tables (price, cost, demand, water)
# These come directly from the dataset (average per crop) and are used by
# app.py for the profit / demand / water-requirement calculations, instead
# of re-predicting them with another model -> keeps the project simple and
# explainable.
# ---------------------------------------------------------------------------
print("STEP 8: Building per-crop lookup table (avg price, cost, demand)...")
crop_lookup = df.groupby("Crop").agg(
    Market_Price=("Market_Price", "mean"),
    Cost_Per_Acre=("Cost_Per_Acre", "mean"),
    Market_Demand=("Market_Demand", "mean"),
    Avg_Yield=("Yield", "mean"),
    Water_Requirement=("Water_Requirement", lambda x: x.mode()[0]),
).round(2).to_dict(orient="index")

# ---------------------------------------------------------------------------
# STEP 9: Save trained models + encoders + lookup table
# ---------------------------------------------------------------------------
print("STEP 9: Saving models to disk...")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(crop_model, os.path.join(MODEL_DIR, "crop_model.pkl"))
joblib.dump(yield_model, os.path.join(MODEL_DIR, "yield_model.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
joblib.dump(crop_lookup, os.path.join(MODEL_DIR, "crop_lookup.pkl"))
joblib.dump(crop_features, os.path.join(MODEL_DIR, "crop_features.pkl"))
joblib.dump(yield_features, os.path.join(MODEL_DIR, "yield_features.pkl"))

print("\nAll models trained and saved successfully in the 'models/' folder.")
print("You can now run:  python app.py")
