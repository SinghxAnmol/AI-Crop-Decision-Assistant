"""
generate_dataset.py
--------------------
This script CREATES the sample dataset used by this project (data/agriculture_dataset.csv).

IMPORTANT (for viva): This is SYNTHETIC / SAMPLE data generated using realistic
agricultural ranges (typical NPK values, rainfall, temperature, market price bands,
cost per acre, etc. for common Indian crops). It is NOT official government data.
It is created only so the machine learning models have enough consistent data to
learn from and to demonstrate the project end-to-end.

The generation logic on purpose ties Yield / Market_Demand / Risk to the input
features (soil, water, season, NPK, rainfall, temperature) so that the trained
models actually learn a meaningful pattern instead of memorising random noise.
"""

import random
import csv
import os

random.seed(42)  # fixed seed -> same dataset every time this script is run

# ---------------------------------------------------------------------------
# 1. Reference data: states/districts, soil types, seasons
# ---------------------------------------------------------------------------

STATE_DISTRICTS = {
    "Punjab": ["Ludhiana", "Amritsar", "Patiala"],
    "Haryana": ["Karnal", "Hisar", "Panipat"],
    "Uttar Pradesh": ["Lucknow", "Meerut", "Varanasi"],
    "Maharashtra": ["Nashik", "Pune", "Nagpur"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Ujjain"],
}

SOIL_TYPES = ["Alluvial", "Black", "Red", "Loamy", "Sandy", "Clay"]
SEASONS = ["Kharif", "Rabi", "Zaid"]

# ---------------------------------------------------------------------------
# 2. Crop profiles
#    Each crop has "ideal" growing conditions. These ideal ranges are used to
#    (a) decide which crop is likely grown for a given soil/season, and
#    (b) calculate a realistic yield based on how close the actual
#        conditions are to the ideal conditions.
# ---------------------------------------------------------------------------

CROP_PROFILES = {
    "Maize":     {"soil": ["Alluvial", "Loamy", "Red"],   "season": ["Kharif"],        "water": 55,
                  "n": (80, 120), "p": (40, 60), "k": (30, 50), "ph": (5.8, 7.2),
                  "rain": (600, 1100), "temp": (21, 32),
                  "yield": (18, 26), "price": (1700, 2200), "cost": (18000, 24000), "demand": 75},
    "Rice":      {"soil": ["Alluvial", "Clay"],            "season": ["Kharif"],        "water": 85,
                  "n": (100, 140), "p": (40, 60), "k": (40, 60), "ph": (5.5, 7.0),
                  "rain": (1000, 2000), "temp": (22, 34),
                  "yield": (20, 30), "price": (1800, 2300), "cost": (22000, 30000), "demand": 82},
    "Wheat":     {"soil": ["Alluvial", "Loamy", "Black"],  "season": ["Rabi"],          "water": 50,
                  "n": (100, 130), "p": (50, 70), "k": (30, 50), "ph": (6.0, 7.5),
                  "rain": (400, 800), "temp": (10, 24),
                  "yield": (16, 24), "price": (2000, 2400), "cost": (16000, 22000), "demand": 80},
    "Cotton":    {"soil": ["Black", "Red", "Loamy"],       "season": ["Kharif"],        "water": 45,
                  "n": (80, 120), "p": (30, 50), "k": (30, 60), "ph": (6.0, 8.0),
                  "rain": (600, 1100), "temp": (21, 35),
                  "yield": (6, 12), "price": (5500, 7000), "cost": (25000, 34000), "demand": 70},
    "Sugarcane": {"soil": ["Alluvial", "Black", "Loamy"],  "season": ["Kharif", "Zaid"], "water": 90,
                  "n": (120, 160), "p": (50, 80), "k": (60, 100), "ph": (6.0, 7.5),
                  "rain": (1100, 1800), "temp": (24, 35),
                  "yield": (350, 450), "price": (280, 340), "cost": (40000, 55000), "demand": 68},
    "Soybean":   {"soil": ["Black", "Loamy"],              "season": ["Kharif"],        "water": 50,
                  "n": (30, 60), "p": (40, 70), "k": (30, 50), "ph": (6.0, 7.5),
                  "rain": (600, 1200), "temp": (20, 32),
                  "yield": (8, 14), "price": (3800, 4500), "cost": (18000, 24000), "demand": 64},
    "Groundnut": {"soil": ["Sandy", "Red", "Loamy"],       "season": ["Kharif", "Zaid"], "water": 40,
                  "n": (20, 40), "p": (40, 60), "k": (40, 60), "ph": (6.0, 7.0),
                  "rain": (500, 1000), "temp": (24, 33),
                  "yield": (10, 16), "price": (5000, 6000), "cost": (22000, 28000), "demand": 60},
    "Bajra":     {"soil": ["Sandy", "Red", "Black"],       "season": ["Kharif"],        "water": 25,
                  "n": (40, 70), "p": (20, 40), "k": (20, 40), "ph": (6.5, 8.0),
                  "rain": (300, 600), "temp": (25, 35),
                  "yield": (10, 16), "price": (1900, 2300), "cost": (10000, 15000), "demand": 55},
    "Gram":      {"soil": ["Black", "Loamy", "Alluvial"],  "season": ["Rabi"],          "water": 30,
                  "n": (20, 40), "p": (40, 60), "k": (20, 40), "ph": (6.0, 7.5),
                  "rain": (300, 650), "temp": (10, 26),
                  "yield": (8, 12), "price": (4800, 5600), "cost": (14000, 19000), "demand": 58},
    "Mustard":   {"soil": ["Alluvial", "Loamy", "Sandy"],  "season": ["Rabi"],          "water": 30,
                  "n": (60, 90), "p": (30, 50), "k": (20, 40), "ph": (6.0, 7.5),
                  "rain": (300, 650), "temp": (10, 25),
                  "yield": (8, 12), "price": (5200, 6000), "cost": (13000, 18000), "demand": 57},
}

CROP_NAMES = list(CROP_PROFILES.keys())


def pick_crop(soil, season):
    """Pick a crop that realistically suits the given soil + season.
    80% of the time we pick a crop that actually matches (so the data has a
    learnable pattern). 20% of the time we pick any crop at random, because
    in the real world farmers sometimes grow a less-suited crop too -
    this keeps the dataset realistic instead of "too perfect"."""
    matching = [c for c, p in CROP_PROFILES.items()
                if soil in p["soil"] and season in p["season"]]
    if matching and random.random() < 0.8:
        return random.choice(matching)
    return random.choice(CROP_NAMES)


def suitability_factor(crop, soil, season, water, n, p, k, ph, rainfall, temp):
    """Return a 0-1 score describing how close the actual growing conditions
    are to this crop's ideal conditions. Used only to make Yield/Demand/Risk
    realistic - the ML models never see this function, they learn the
    pattern purely from the data it produces."""
    profile = CROP_PROFILES[crop]
    score = 0
    total = 6

    score += 1 if soil in profile["soil"] else 0.3
    score += 1 if season in profile["season"] else 0.3
    score += 1 - min(abs(water - profile["water"]) / 100, 1)

    def range_score(value, low, high):
        if low <= value <= high:
            return 1
        spread = high - low if high > low else 1
        dist = min(abs(value - low), abs(value - high))
        return max(0, 1 - dist / spread)

    score += range_score(ph, *profile["ph"])
    score += range_score(rainfall, *profile["rain"])
    score += range_score(temp, *profile["temp"])

    return round(score / total, 3)


def water_availability_label(value):
    if value < 40:
        return "Low"
    elif value < 70:
        return "Medium"
    return "High"


def calculate_risk(water, rainfall_ok_score, demand, suitability):
    """Simple, explainable rule-based risk score (0-100, higher = riskier),
    then bucketed into Low / Medium / High. Mirrors the risk logic used
    later in app.py so the dataset and the app stay consistent."""
    risk_score = 0
    risk_score += (100 - water) * 0.35          # less irrigation -> more risk
    risk_score += (1 - rainfall_ok_score) * 100 * 0.25   # rainfall mismatch -> more risk
    risk_score += (100 - demand) * 0.20          # low market demand -> more risk
    risk_score += (1 - suitability) * 100 * 0.20  # poor suitability -> more risk

    if risk_score < 35:
        return "Low"
    elif risk_score < 60:
        return "Medium"
    return "High"


def generate_row():
    state = random.choice(list(STATE_DISTRICTS.keys()))
    district = random.choice(STATE_DISTRICTS[state])
    soil = random.choice(SOIL_TYPES)
    season = random.choice(SEASONS)
    crop = pick_crop(soil, season)
    profile = CROP_PROFILES[crop]

    land_size = round(random.uniform(1, 20), 1)

    # water availability (irrigation coverage, 0-100) noisy around crop's ideal
    water = max(5, min(100, int(random.gauss(profile["water"], 15))))

    n = round(random.uniform(*profile["n"]) + random.gauss(0, 8), 1)
    p = round(random.uniform(*profile["p"]) + random.gauss(0, 5), 1)
    k = round(random.uniform(*profile["k"]) + random.gauss(0, 5), 1)
    ph = round(random.uniform(*profile["ph"]) + random.gauss(0, 0.3), 2)
    rainfall = round(random.uniform(*profile["rain"]) + random.gauss(0, 80), 1)
    temperature = round(random.uniform(*profile["temp"]) + random.gauss(0, 2), 1)

    n, p, k = max(0, n), max(0, p), max(0, k)
    ph = max(3.5, min(9.0, ph))
    rainfall = max(50, rainfall)
    temperature = max(5, min(45, temperature))

    suit = suitability_factor(crop, soil, season, water, n, p, k, ph, rainfall, temperature)

    rain_low, rain_high = profile["rain"]
    rainfall_ok_score = 1 if rain_low <= rainfall <= rain_high else max(
        0, 1 - min(abs(rainfall - rain_low), abs(rainfall - rain_high)) / (rain_high - rain_low))

    # base_yield is fixed per crop (its typical mid-range yield). The ACTUAL
    # yield then depends mainly on how suitable the real conditions are
    # (soil/season/water/pH/rainfall/temperature match), plus a small amount
    # of random noise. This keeps yield genuinely learnable from the features
    # instead of being dominated by unrelated randomness.
    base_yield = sum(profile["yield"]) / 2
    yield_per_acre = round(max(1, base_yield * (0.5 + 0.7 * suit) + random.gauss(0, base_yield * 0.04)), 2)

    market_price = round(random.uniform(*profile["price"]) * (0.95 + 0.1 * random.random()), 2)
    cost_per_acre = round(random.uniform(*profile["cost"]), 2)

    demand = round(max(10, min(100, profile["demand"] + random.gauss(0, 8))), 1)
    risk = calculate_risk(water, rainfall_ok_score, demand, suit)

    return {
        "State": state,
        "District": district,
        "Season": season,
        "Soil_Type": soil,
        "Crop": crop,
        "Land_Size": land_size,
        "Water_Availability": water,
        "N": n,
        "P": p,
        "K": k,
        "pH": ph,
        "Rainfall": rainfall,
        "Temperature": temperature,
        "Yield": yield_per_acre,
        "Market_Price": market_price,
        "Cost_Per_Acre": cost_per_acre,
        "Water_Requirement": water_availability_label(profile["water"]),
        "Market_Demand": demand,
        "Risk": risk,
    }


def main(num_records=1200):
    os.makedirs("data", exist_ok=True)
    rows = [generate_row() for _ in range(num_records)]

    fieldnames = list(rows[0].keys())
    out_path = os.path.join("data", "agriculture_dataset.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {num_records} records -> {out_path}")


if __name__ == "__main__":
    main()
