import ast
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MacroChefAI", page_icon="🍽️", layout="wide")

EPS = 1e-6

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
CANDIDATE_DATA_PATHS = [
    CURRENT_DIR / "processed_data" / "dataset.parquet",
    CURRENT_DIR.parent / "processed_data" / "dataset.parquet",
    CURRENT_DIR / "dataset.parquet",
    CURRENT_DIR.parent / "dataset.parquet",
]


# -------------------------------------------------------------------
# Parsing / normalization helpers
# -------------------------------------------------------------------

def safe_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        if text.startswith("c(") and text.endswith(")"):
            inner = text[2:-1].strip()
            parts = re.split(r'\s*,\s*', inner)
            cleaned = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
            return [x for x in cleaned if x]
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]
        return [text]
    return []


def ensure_numeric(df, cols):
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_text(x):
    x = html.unescape(str(x)).lower().strip()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9\s_]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_ingredient(text):
    text = normalize_text(text)
    stop_words = {
        "fresh", "ground", "large", "small", "medium", "extra", "virgin",
        "divided", "optional", "finely", "chopped", "diced", "minced",
        "sliced", "to", "taste", "or", "for", "and"
    }
    tokens = [t for t in text.split() if t not in stop_words]
    return "_".join(tokens).strip("_")


def normalize_ingredient_list(items):
    cleaned = []
    for item in safe_list(items):
        token = clean_ingredient(item)
        if token:
            cleaned.append(token)
    return cleaned


def first_existing(df, names, default=None):
    for name in names:
        if name in df.columns:
            return name
    return default


# -------------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------------

def find_data_path():
    for path in CANDIDATE_DATA_PATHS:
        if path.exists():
            return path
    return None


def add_weight_loss_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["calories", "protein", "fat", "carbs", "fiber", "sugar"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["protein_per_100kcal"] = df["protein"] / (df["calories"] / 100 + EPS)
    df["fiber_per_100kcal"] = df["fiber"] / (df["calories"] / 100 + EPS)
    df["sugar_per_100kcal"] = df["sugar"] / (df["calories"] / 100 + EPS)

    df["weight_loss_score"] = (
        3.0 * df["protein_per_100kcal"]
        + 2.5 * df["fiber_per_100kcal"]
        - 2.0 * df["sugar_per_100kcal"]
        - 0.003 * df["calories"]
    )
    return df


def classify_protein(p):
    if pd.isna(p):
        return np.nan
    if p < 0.12:
        return "low"
    if p < 0.20:
        return "moderate"
    return "high"


def classify_carb(p):
    if pd.isna(p):
        return np.nan
    if p < 0.45:
        return "low"
    if p <= 0.65:
        return "moderate"
    return "high"


def classify_fat(p):
    if pd.isna(p):
        return np.nan
    if p < 0.20:
        return "low"
    if p <= 0.35:
        return "moderate"
    return "high"


def classify_sodium(s):
    if pd.isna(s):
        return np.nan
    if s <= 140:
        return "low"
    if s <= 400:
        return "moderate"
    return "high"


def add_macro_classifications(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["calories"].notna() & (df["calories"] > 0)].copy()

    df["protein_kcal"] = df["protein"] * 4
    df["carb_kcal"] = df["carbs"] * 4
    df["fat_kcal"] = df["fat"] * 9

    df["protein_pct"] = df["protein_kcal"] / df["calories"]
    df["carb_pct"] = df["carb_kcal"] / df["calories"]
    df["fat_pct"] = df["fat_kcal"] / df["calories"]

    df["protein_level"] = df["protein_pct"].apply(classify_protein)
    df["carb_level"] = df["carb_pct"].apply(classify_carb)
    df["fat_level"] = df["fat_pct"].apply(classify_fat)
    df["sodium_level"] = df["sodium"].apply(classify_sodium)

    df["balanced"] = (
        df["protein_pct"].between(0.10, 0.35)
        & df["carb_pct"].between(0.45, 0.65)
        & df["fat_pct"].between(0.20, 0.35)
    ).astype(int)
    df["balanced_label"] = df["balanced"].map({1: "balanced", 0: "not_balanced"})
    return df


def add_medical_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["sugar", "sodium", "sat_fat", "cholest", "protein", "carbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["risk_diabetes"] = (df["sugar"] > 15).astype(int)
    df["risk_hypertension"] = (df["sodium"] > 600).astype(int)
    df["risk_heart_disease"] = (df["sat_fat"] > 5).astype(int)
    df["risk_cholesterol"] = (df["cholest"] > 100).astype(int)
    df["risk_kidney"] = (df["protein"] > 40).astype(int)
    df["risk_keto_violation"] = (df["carbs"] > 30).astype(int)

    risk_cols = [
        "risk_diabetes",
        "risk_hypertension",
        "risk_heart_disease",
        "risk_cholesterol",
        "risk_kidney",
        "risk_keto_violation",
    ]
    df["medical_risk_score"] = df[risk_cols].sum(axis=1)

    def classify(score):
        if score == 0:
            return "low"
        if score <= 2:
            return "moderate"
        return "high"

    df["medical_risk_level"] = df["medical_risk_score"].apply(classify)

    def reasons(row):
        out = []
        if row["risk_diabetes"]:
            out.append("high sugar")
        if row["risk_hypertension"]:
            out.append("high sodium")
        if row["risk_heart_disease"]:
            out.append("high saturated fat")
        if row["risk_cholesterol"]:
            out.append("high cholesterol")
        if row["risk_kidney"]:
            out.append("very high protein")
        if row["risk_keto_violation"]:
            out.append("high carbohydrates (not keto friendly)")
        return ", ".join(out) if out else "no major medical risk"

    df["medical_risk_reason"] = df.apply(reasons, axis=1)
    return df


@st.cache_data
def load_data():
    data_path = find_data_path()
    if data_path is None:
        st.error("dataset.parquet not found.")
        st.info("Place the processed parquet file in processed_data/dataset.parquet or next to the app file.")
        st.stop()

    df = pd.read_parquet(data_path).copy()

    rename_map = {}
    alias_groups = {
        "name": ["name", "title", "recipe_name", "Name"],
        "description": ["description", "Description"],
        "calories": ["calories", "Calories"],
        "protein": ["protein", "ProteinContent"],
        "fat": ["fat", "FatContent"],
        "carbs": ["carbs", "CarbohydrateContent"],
        "fiber": ["fiber", "FiberContent"],
        "sugar": ["sugar", "SugarContent"],
        "sat_fat": ["sat_fat", "SaturatedFatContent"],
        "sodium": ["sodium", "sod", "SodiumContent"],
        "cholest": ["cholest", "cholesterol", "CholesterolContent"],
        "cook_time": ["cook_time", "total_time_minutes", "TotalTimeMinutes"],
        "ingredients_clean": ["ingredients_clean", "ingredients_list"],
        "instructions_list": ["instructions_list", "RecipeInstructions"],
        "food_tags": ["food_tags", "tags", "diet_tags", "RecipeCategory"],
        "ingredient_quantities": ["ingredient_quantities", "RecipeIngredientQuantities"],
        "ingredients_text": ["ingredients_text"],
    }
    for canonical, aliases in alias_groups.items():
        existing = first_existing(df, aliases)
        if existing and existing != canonical:
            rename_map[existing] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    # Text cleanup
    for col in ["name", "description"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: html.unescape(x) if pd.notna(x) else x)

    # Columns with defaults
    if "name" not in df.columns:
        df["name"] = [f"Recipe {i+1}" for i in range(len(df))]

    if "food_tags" not in df.columns:
        df["food_tags"] = [[] for _ in range(len(df))]
    else:
        df["food_tags"] = df["food_tags"].apply(safe_list)

    if "ingredients_clean" in df.columns:
        df["ingredients_clean"] = df["ingredients_clean"].apply(normalize_ingredient_list)
    elif "RecipeIngredientParts" in df.columns:
        df["ingredients_clean"] = df["RecipeIngredientParts"].apply(normalize_ingredient_list)
    else:
        df["ingredients_clean"] = [[] for _ in range(len(df))]

    if "instructions_list" not in df.columns:
        df["instructions_list"] = [[] for _ in range(len(df))]
    else:
        df["instructions_list"] = df["instructions_list"].apply(safe_list)
        df["instructions_list"] = df["instructions_list"].apply(
            lambda items: [html.unescape(str(x)) for x in items]
        )

    if "ingredient_quantities" not in df.columns:
        df["ingredient_quantities"] = [[] for _ in range(len(df))]
    else:
        df["ingredient_quantities"] = df["ingredient_quantities"].apply(safe_list)
        def to_num_list(items):
            out = []
            for x in items:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(np.nan)
            return out
        df["ingredient_quantities"] = df["ingredient_quantities"].apply(to_num_list)

    if "ingredients_text" not in df.columns:
        df["ingredients_text"] = df["ingredients_clean"].apply(lambda x: " ".join(x))
    else:
        df["ingredients_text"] = df["ingredients_text"].fillna("").apply(normalize_text)

    df = ensure_numeric(
        df,
        ["calories", "protein", "fat", "carbs", "fiber", "sugar", "sat_fat", "sodium", "cholest", "cook_time"],
    )

    df["total_relative_quantity"] = df["ingredient_quantities"].apply(
        lambda vals: float(np.nansum(vals)) if isinstance(vals, list) and len(vals) else 0.0
    )
    df["avg_relative_quantity"] = df["ingredient_quantities"].apply(
        lambda vals: float(np.nanmean(vals)) if isinstance(vals, list) and len(vals) else 0.0
    )
    df["has_quantity_data"] = df["ingredient_quantities"].apply(
        lambda vals: int(isinstance(vals, list) and len(vals) > 0)
    )

    df = add_weight_loss_features(df)
    df = add_macro_classifications(df)
    df = add_medical_risk_flags(df)

    return df.reset_index(drop=True)


@st.cache_resource
def build_models(df: pd.DataFrame):
    macro_X = df[["protein", "fat", "carbs"]].fillna(0.0)
    scaler = StandardScaler()
    macro_X_scaled = scaler.fit_transform(macro_X)
    knn = NearestNeighbors(n_neighbors=min(50, len(df)), metric="euclidean")
    knn.fit(macro_X_scaled)

    tfidf_df = df[df["ingredients_text"].fillna("").str.strip().astype(bool)].copy()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    ingredient_matrix = vectorizer.fit_transform(tfidf_df["ingredients_text"])
    return scaler, knn, tfidf_df, vectorizer, ingredient_matrix


df = load_data()
macro_scaler, knn_model, tfidf_df, vectorizer, ingredient_matrix = build_models(df)


# -------------------------------------------------------------------
# Fitness target helpers
# -------------------------------------------------------------------

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m ** 2)


def calculate_bmr(weight, height, age, sex):
    if sex == "male":
        return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
    return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161


def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderate": 1.55,
        "very_active": 1.725,
        "extra_active": 1.9,
    }
    return bmr * activity_multipliers.get(activity_level, 1.2)


def adjust_calories(tdee, goal, sex):
    if goal == "weight_loss":
        target = tdee * 0.8
        floor = 1500 if sex == "male" else 1200
        return max(target, floor)
    if goal == "weight_gain":
        return tdee * 1.1
    return tdee


def calculate_macros(calories, goal, weight_kg):
    if goal == "weight_loss":
        protein = 1.8 * weight_kg
        fat = 0.8 * weight_kg
        carbs = max((calories - protein * 4 - fat * 9) / 4, 0)
    elif goal == "weight_gain":
        protein = 1.6 * weight_kg
        fat = 0.9 * weight_kg
        carbs = max((calories - protein * 4 - fat * 9) / 4, 0)
    else:
        protein = 1.6 * weight_kg
        fat = 0.8 * weight_kg
        carbs = max((calories - protein * 4 - fat * 9) / 4, 0)
    return protein, fat, carbs


def build_user_targets(profile, meals_per_day=3):
    bmi = calculate_bmi(profile["weight"], profile["height"])
    bmr = calculate_bmr(profile["weight"], profile["height"], profile["age"], profile["sex"])
    tdee = calculate_tdee(bmr, profile["activity_level"])
    target_calories = adjust_calories(tdee, profile["goal"], profile["sex"])
    target_protein, target_fat, target_carbs = calculate_macros(target_calories, profile["goal"], profile["weight"])
    return {
        "bmi": bmi,
        "bmr": bmr,
        "tdee": tdee,
        "target_calories": target_calories,
        "target_protein": target_protein,
        "target_fat": target_fat,
        "target_carbs": target_carbs,
        "meal_calories": target_calories / meals_per_day,
        "meal_protein": target_protein / meals_per_day,
        "meal_fat": target_fat / meals_per_day,
        "meal_carbs": target_carbs / meals_per_day,
    }


# -------------------------------------------------------------------
# Recommendation helpers
# -------------------------------------------------------------------

def knn_candidates(data, protein, fat, carbs, n_neighbors=50):
    n_neighbors = max(1, min(n_neighbors, len(data)))
    user_vec = pd.DataFrame([{"protein": protein, "fat": fat, "carbs": carbs}])
    user_scaled = macro_scaler.transform(user_vec)
    distances, indices = knn_model.kneighbors(user_scaled, n_neighbors=n_neighbors)
    candidates = data.iloc[indices[0]].copy()
    candidates["macro_distance"] = distances[0]
    candidates["macro_score"] = 1 / (1 + candidates["macro_distance"])
    return candidates


def ingredient_similarity(candidates, user_ingredients):
    candidates = candidates.copy()
    cleaned_user = sorted(set(normalize_ingredient_list(user_ingredients)))

    if not cleaned_user:
        candidates["ingredient_overlap_score"] = 0.0
        candidates["ingredient_tfidf_score"] = 0.0
        candidates["ingredient_score"] = 0.0
        candidates["matched_ingredients"] = [[] for _ in range(len(candidates))]
        candidates["missing_ingredients"] = candidates["ingredients_clean"].apply(
            lambda x: x[:10] if isinstance(x, list) else []
        )
        return candidates

    user_set = set(cleaned_user)
    user_text = " ".join(cleaned_user)
    user_vec = vectorizer.transform([user_text])
    sims = cosine_similarity(user_vec, ingredient_matrix).flatten()
    sim_map = pd.Series(sims, index=tfidf_df.index)

    def matched(recipe_ingredients):
        recipe_set = {x for x in recipe_ingredients if x}
        return sorted(recipe_set.intersection(user_set))

    def missing(recipe_ingredients):
        recipe_set = {x for x in recipe_ingredients if x}
        return sorted(recipe_set.difference(user_set))

    def overlap(recipe_ingredients):
        recipe_set = {x for x in recipe_ingredients if x}
        if not recipe_set:
            return 0.0
        return len(recipe_set.intersection(user_set)) / len(recipe_set)

    candidates["matched_ingredients"] = candidates["ingredients_clean"].apply(matched)
    candidates["missing_ingredients"] = candidates["ingredients_clean"].apply(missing)
    candidates["ingredient_overlap_score"] = candidates["ingredients_clean"].apply(overlap)
    candidates["ingredient_tfidf_score"] = candidates.index.map(sim_map).fillna(0.0)
    candidates["ingredient_score"] = (
        0.7 * candidates["ingredient_overlap_score"] +
        0.3 * candidates["ingredient_tfidf_score"]
    )
    return candidates


def preferred_food_match(tags, preferred_food_types):
    if not preferred_food_types:
        return True
    tags_norm = {normalize_text(t) for t in safe_list(tags)}
    mapping = {
        "vegan": {"vegan"},
        "vegetarian": {"vegetarian", "veggie"},
        "chicken": {"chicken", "poultry"},
        "meat": {"beef", "lamb", "pork", "meat"},
        "seafood": {"seafood", "fish", "shrimp", "salmon", "tuna"},
        "breakfast": {"breakfast", "brunch"},
        "lunch": {"lunch"},
        "dinner": {"dinner", "main dish", "main course", "main_dish"},
        "snack": {"snack", "appetizer"},
    }
    prefs = {normalize_text(x) for x in preferred_food_types}
    for pref in prefs:
        allowed = mapping.get(pref, {pref})
        if tags_norm.intersection(allowed):
            return True
    return False


def filter_by_preferences(candidates, preferred_food_types=None, max_missing_ingredients=None, max_cook_time=None):
    filtered = candidates.copy()
    if preferred_food_types:
        filtered = filtered[
            filtered["food_tags"].apply(lambda tags: preferred_food_match(tags, preferred_food_types))
        ].copy()
    if max_missing_ingredients is not None and "missing_ingredients" in filtered.columns:
        filtered = filtered[filtered["missing_ingredients"].apply(len) <= max_missing_ingredients].copy()
    if max_cook_time is not None and "cook_time" in filtered.columns:
        filtered = filtered[filtered["cook_time"].fillna(np.inf) <= max_cook_time].copy()
    return filtered


def filter_recipes_by_health(data: pd.DataFrame, user_conditions=None, strict=True) -> pd.DataFrame:
    filtered = data.copy()
    user_conditions = {str(x).strip().lower() for x in (user_conditions or []) if str(x).strip()}
    if not user_conditions:
        return filtered
    if strict:
        if "diabetes" in user_conditions:
            filtered = filtered[filtered["risk_diabetes"] == 0]
        if "hypertension" in user_conditions:
            filtered = filtered[filtered["risk_hypertension"] == 0]
        if "heart_disease" in user_conditions:
            filtered = filtered[filtered["risk_heart_disease"] == 0]
        if "cholesterol" in user_conditions:
            filtered = filtered[filtered["risk_cholesterol"] == 0]
        if "kidney_disease" in user_conditions:
            filtered = filtered[filtered["risk_kidney"] == 0]
        if "keto" in user_conditions:
            filtered = filtered[filtered["risk_keto_violation"] == 0]
    return filtered


def add_weight_loss_priority(candidates, full_df):
    candidates = candidates.copy()
    score_min = float(full_df["weight_loss_score"].min())
    score_max = float(full_df["weight_loss_score"].max())
    denom = max(score_max - score_min, EPS)
    candidates["weight_loss_priority"] = (candidates["weight_loss_score"] - score_min) / denom
    candidates["weight_loss_priority"] = candidates["weight_loss_priority"].clip(0, 1)
    return candidates


def final_rank(candidates):
    candidates = candidates.copy()
    medical_penalty = candidates["medical_risk_score"].fillna(0) / 6.0
    candidates["final_score"] = (
        0.45 * candidates["macro_score"] +
        0.25 * candidates["ingredient_score"] +
        0.20 * candidates["weight_loss_priority"] -
        0.10 * medical_penalty
    )
    return candidates.sort_values("final_score", ascending=False)


def recommend_recipes(protein, fat, carbs, user_ingredients=None, preferred_food_types=None,
                      user_conditions=None, strict_health_filter=True,
                      n_results=5, n_candidates=50, max_missing_ingredients=None,
                      max_cook_time=None):
    candidates = knn_candidates(df, protein, fat, carbs, n_neighbors=n_candidates)
    candidates = ingredient_similarity(candidates, user_ingredients)
    candidates = filter_by_preferences(
        candidates,
        preferred_food_types=preferred_food_types,
        max_missing_ingredients=max_missing_ingredients,
        max_cook_time=max_cook_time,
    )
    candidates = filter_recipes_by_health(
        candidates,
        user_conditions=user_conditions,
        strict=strict_health_filter,
    )

    if candidates.empty:
        return pd.DataFrame()

    candidates = add_weight_loss_priority(candidates, df)
    candidates = final_rank(candidates)

    cols = [
        "name", "description", "calories", "protein", "fat", "carbs", "fiber", "sugar",
        "sat_fat", "sodium", "cholest", "cook_time", "food_tags",
        "protein_level", "carb_level", "fat_level", "sodium_level", "balanced_label",
        "macro_score", "ingredient_overlap_score", "ingredient_tfidf_score",
        "ingredient_score", "weight_loss_score", "weight_loss_priority",
        "medical_risk_score", "medical_risk_level", "medical_risk_reason",
        "matched_ingredients", "missing_ingredients",
        "ingredients_clean", "instructions_list", "final_score",
    ]
    cols = [c for c in cols if c in candidates.columns]
    return candidates[cols].head(n_results)


# -------------------------------------------------------------------
# Sidebar inputs
# -------------------------------------------------------------------
st.sidebar.title("User Profile")

weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0)
height = st.sidebar.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=175.0)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
activity_level = st.sidebar.selectbox(
    "Activity level",
    ["sedentary", "lightly_active", "moderate", "very_active", "extra_active"],
)
goal = st.sidebar.selectbox("Goal", ["weight_loss", "maintenance", "weight_gain"])
meals_per_day = st.sidebar.slider("Meals per day", 2, 6, 3)

preferred_food_types = st.sidebar.multiselect(
    "Preferred food types / meal tags",
    ["chicken", "meat", "seafood", "vegetarian", "vegan", "breakfast", "lunch", "dinner", "snack"],
)

health_conditions = st.sidebar.multiselect(
    "Health conditions",
    ["diabetes", "hypertension", "heart_disease", "cholesterol", "kidney_disease", "keto"],
)
strict_health_filter = st.sidebar.toggle("Strict health filtering", value=True)
max_cook_time = st.sidebar.slider("Max cooking time (minutes)", 10, 180, 60)
max_missing_ingredients = st.sidebar.slider("Max missing ingredients", 0, 12, 6)
top_n = st.sidebar.slider("Number of recommendations", 1, 20, 5)

ingredients_text = st.sidebar.text_area(
    "Available ingredients at home",
    value="chicken, rice, onion, garlic, yogurt, broccoli",
)
surprise_me = st.sidebar.button("Surprise me")

user_profile = {
    "weight": weight,
    "height": height,
    "age": age,
    "sex": sex,
    "activity_level": activity_level,
    "goal": goal,
}
user_targets = build_user_targets(user_profile, meals_per_day=meals_per_day)
user_ingredients = [x.strip() for x in ingredients_text.split(",") if x.strip()]

recs = recommend_recipes(
    protein=user_targets["meal_protein"],
    fat=user_targets["meal_fat"],
    carbs=user_targets["meal_carbs"],
    user_ingredients=user_ingredients,
    preferred_food_types=preferred_food_types,
    user_conditions=health_conditions,
    strict_health_filter=strict_health_filter,
    max_missing_ingredients=max_missing_ingredients,
    max_cook_time=max_cook_time,
    n_results=top_n,
    n_candidates=60,
)

if surprise_me and not recs.empty:
    recs = recs.sample(min(top_n, len(recs))).sort_values("final_score", ascending=False)


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("🍽️ MacroChefAI")
st.caption("Rule-based weight-loss scoring, macro classification, and health-aware recipe recommendations.")

summary1, summary2, summary3, summary4 = st.columns(4)
summary1.metric("Recipes loaded", f"{len(df):,}")
summary2.metric("Low risk recipes", f"{(df['medical_risk_level'] == 'low').sum():,}")
summary3.metric("Balanced recipes", f"{(df['balanced'] == 1).sum():,}")
summary4.metric("With quantity data", f"{(df['has_quantity_data'] == 1).sum():,}")

tab1, tab2, tab3 = st.tabs(["Overview", "Recommendations", "Dataset"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BMI", f"{user_targets['bmi']:.1f}")
    col2.metric("BMR", f"{user_targets['bmr']:.0f} kcal")
    col3.metric("TDEE", f"{user_targets['tdee']:.0f} kcal")
    col4.metric("Daily Calories", f"{user_targets['target_calories']:.0f} kcal")

    st.subheader("Daily Macro Targets")
    d1, d2, d3 = st.columns(3)
    d1.metric("Protein", f"{user_targets['target_protein']:.0f} g")
    d2.metric("Fat", f"{user_targets['target_fat']:.0f} g")
    d3.metric("Carbs", f"{user_targets['target_carbs']:.0f} g")

    st.subheader("Per-Meal Targets")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calories", f"{user_targets['meal_calories']:.0f} kcal")
    m2.metric("Protein", f"{user_targets['meal_protein']:.0f} g")
    m3.metric("Fat", f"{user_targets['meal_fat']:.0f} g")
    m4.metric("Carbs", f"{user_targets['meal_carbs']:.0f} g")

    macro_df = pd.DataFrame(
        {
            "macro": ["Protein", "Fat", "Carbs"],
            "grams": [
                user_targets["target_protein"],
                user_targets["target_fat"],
                user_targets["target_carbs"],
            ],
        }
    )
    st.bar_chart(macro_df.set_index("macro"))

    st.subheader("How recommendation filtering works")
    st.write(
        "Recipes are first matched to your per-meal protein, fat, and carbohydrate targets. "
        "They are then re-ranked using pantry ingredient similarity, the rule-based weight-loss score, "
        "and a medical-risk penalty. If strict health filtering is on, recipes that violate your selected "
        "health conditions are removed before ranking."
    )

with tab2:
    st.subheader("Top Recipe Recommendations")

    if recs.empty:
        st.warning("No recipes found. Try fewer health restrictions, a larger missing-ingredient limit, or broader food-type preferences.")
    else:
        for _, recipe in recs.iterrows():
            with st.expander(f"{recipe.get('name', 'Untitled Recipe')} — score {recipe['final_score']:.3f}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Calories", f"{recipe['calories']:.0f}")
                c2.metric("Protein", f"{recipe['protein']:.1f} g")
                c3.metric("Cook Time", f"{recipe['cook_time']:.0f} min")
                c4.metric("Medical Risk", recipe.get("medical_risk_level", "—"))

                s1, s2, s3 = st.columns(3)
                s1.metric("Macro match", f"{recipe.get('macro_score', 0):.3f}")
                s2.metric("Ingredient match", f"{recipe.get('ingredient_score', 0):.3f}")
                s3.metric("Weight-loss priority", f"{recipe.get('weight_loss_priority', 0):.3f}")

                st.write(
                    f"**Macro classes:** protein={recipe.get('protein_level', '—')}, "
                    f"carbs={recipe.get('carb_level', '—')}, fat={recipe.get('fat_level', '—')}, "
                    f"sodium={recipe.get('sodium_level', '—')}, balanced={recipe.get('balanced_label', '—')}"
                )
                st.write(f"**Food tags:** {', '.join(safe_list(recipe.get('food_tags', []))) or '—'}")
                st.write(f"**Medical risk reason:** {recipe.get('medical_risk_reason', '—')}")
                st.write(f"**Matched ingredients:** {', '.join(recipe.get('matched_ingredients', [])[:15]) or '—'}")
                st.write(f"**Missing ingredients:** {', '.join(recipe.get('missing_ingredients', [])[:15]) or '—'}")

                if pd.notna(recipe.get("description", np.nan)) and str(recipe.get("description", "")).strip():
                    st.write(f"**Description:** {recipe.get('description')}")

                st.markdown("**Ingredients**")
                ingredients = safe_list(recipe.get("ingredients_clean", []))
                if ingredients:
                    for ing in ingredients[:25]:
                        st.write(f"- {ing}")
                else:
                    st.write("No ingredient list available.")

                st.markdown("**Instructions**")
                steps = safe_list(recipe.get("instructions_list", []))
                if steps:
                    for i, step in enumerate(steps[:10], start=1):
                        st.write(f"{i}. {step}")
                else:
                    st.write("No instructions available.")

with tab3:
    st.subheader("Dataset preview")
    preview_cols = [
        c for c in [
            "name", "calories", "protein", "carbs", "fat", "fiber", "sugar",
            "protein_level", "carb_level", "fat_level", "sodium_level",
            "balanced_label", "weight_loss_score", "medical_risk_level", "medical_risk_reason",
            "has_quantity_data", "total_relative_quantity",
        ] if c in df.columns
    ]
    st.dataframe(df[preview_cols].head(200), use_container_width=True)

st.caption("These health and medical flags are dietary heuristics for awareness and filtering. They are not medical advice or diagnosis.")
