"""MacroChefAI data-processing pipeline.

This module contains ALL the feature-engineering, scoring, save / load,
and model-fitting functions extracted **verbatim** from the original
``macrochefai_final.ipynb`` notebook.  No logic has been changed.

Usage (from a notebook or script)::

    from src.pipeline import (
        load_and_process_recipe_csv,
        save_processed_recipe_data,
        fit_ingredient_tfidf,
        fit_macro_knn,
    )

    df = load_and_process_recipe_csv("raw_data/merged_usable_cal_per_g.csv")
    save_processed_recipe_data(df)
    fit_ingredient_tfidf(df)
    fit_macro_knn(df)
"""

import os
import ast
import json
import re
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=SyntaxWarning)

EPS = 1e-9

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = str(PROJECT_ROOT / "raw_data" / "merged_usable_cal_per_g.csv")

PROCESSED_MODEL_PATH = str(PROJECT_ROOT / "processed_data" / "recipes_model_ready.parquet")
PROCESSED_DISPLAY_PATH = str(PROJECT_ROOT / "processed_data" / "recipes_display_ready.parquet")
COMPACT_MODEL_PATH = str(PROJECT_ROOT / "processed_data" / "recipes_model_compact.parquet")

TFIDF_PATH = str(PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib")
INGREDIENT_MATRIX_PATH = str(PROJECT_ROOT / "models" / "ingredient_matrix.npz")
KNN_MODEL_PATH = str(PROJECT_ROOT / "models" / "macro_knn.joblib")

os.makedirs(PROJECT_ROOT / "processed_data", exist_ok=True)
os.makedirs(PROJECT_ROOT / "models", exist_ok=True)



# ============================================================================
# Notebook cell 3
# ============================================================================

"""Helper functions for parsing list-like columns, time strings, and ingredient text."""

def _safe_list(x):
    """Safely convert many list-like formats into Python lists.

    Handles:
    - existing Python lists
    - JSON-style list strings
    - Python-style list strings
    - R-style c("a","b")
    - simple comma-separated strings

    Avoids calling ast.literal_eval on arbitrary text such as:
    '1/2-1 lb corned beef', which can trigger SyntaxWarning.
    """
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    s = str(x).strip()
    if not s:
        return []

    if s.startswith("c(") and s.endswith(")"):
        inner = s[2:-1]
        parts = re.findall(r'"([^"]+)"|\'([^\']+)\'', inner)
        vals = [a or b for a, b in parts if (a or b)]
        return vals if vals else []

    if s.startswith("[") and s.endswith("]"):
        try:
            val = json.loads(s)
            if isinstance(val, list):
                return val
        except Exception:
            pass

        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return val
        except Exception:
            pass

    if "," in s:
        return [item.strip() for item in s.split(",") if item.strip()]

    return [s]


def _split_serialized_list_tolerant(text: str):
    """
    Parse strings like:
    ["step1", "step2 with D"Annuzios", "step3 with 1/2" oil"]

    It treats a quote as the END of an item only when the next non-space
    character is a comma or the end of the list.
    """
    s = text.strip()

    if not (s.startswith("[") and s.endswith("]")):
        return [s]

    inner = s[1:-1]
    items = []
    buf = []
    in_item = False
    i = 0
    n = len(inner)

    while i < n:
        ch = inner[i]

        if not in_item:
            # skip whitespace and commas between items
            if ch in " \t\r\n,":
                i += 1
                continue

            # quoted item starts
            if ch == '"':
                in_item = True
                buf = []
                i += 1
                continue

            # unquoted fallback item
            start = i
            while i < n and inner[i] != ",":
                i += 1
            token = inner[start:i].strip().strip('"').strip("'")
            if token:
                items.append(token)
            continue

        else:
            # inside quoted item
            if ch == '"':
                # look ahead to decide whether this quote closes the item
                j = i + 1
                while j < n and inner[j].isspace():
                    j += 1

                # closing quote if followed by comma or end of string
                if j >= n or inner[j] == ",":
                    item = "".join(buf).strip()
                    if item:
                        items.append(item)

                    in_item = False
                    buf = []

                    # skip trailing spaces and comma
                    i = j
                    if i < n and inner[i] == ",":
                        i += 1
                    continue

                # otherwise this quote is part of the text
                buf.append(ch)
                i += 1
                continue

            else:
                buf.append(ch)
                i += 1

    # flush unfinished item
    if buf:
        item = "".join(buf).strip()
        if item:
            items.append(item)

    return items if items else [text]


def parse_instructions(value):
    """Robust parser for recipe instruction fields."""

    # numpy arrays first
    if isinstance(value, np.ndarray):
        value = value.tolist()

    # already a list
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]

        # unwrap one-item list containing serialized list text
        if len(cleaned) == 1:
            only = cleaned[0]
            if only.startswith("[") and only.endswith("]"):
                return parse_instructions(only)

        return cleaned

    # nulls
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    # R-style c("a","b")
    if text.startswith("c(") and text.endswith(")"):
        inner = text[2:-1]
        parts = re.findall(r'"([^"]+)"|\'([^\']+)\'', inner)
        vals = [(a or b).strip() for a, b in parts if (a or b)]
        return vals if vals else []

    # serialized list-like text
    if text.startswith("[") and text.endswith("]"):
        # 1. valid JSON
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parse_instructions(parsed)
        except Exception:
            pass

        # 2. valid Python literal
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parse_instructions(parsed)
        except Exception:
            pass

        # 3. tolerant manual parser
        parsed = _split_serialized_list_tolerant(text)
        return [str(x).strip() for x in parsed if str(x).strip()]

    # plain text fallback
    return [p.strip() for p in re.split(r"\.\s+|\n+", text) if p.strip()]


def parse_quantity_value(x):
    """Parse a quantity string into a float when possible.

    Examples:
    - '1 1/2' -> 1.5
    - '1/2-1' -> 0.75
    - '2' -> 2.0

    Parameters
    ----------
    x : Any
        Input quantity text.

    Returns
    -------
    float
        Parsed numeric quantity, or NaN when parsing fails.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()
    if not s or s in {"none", "nan", "na", "to taste", "as needed"}:
        return np.nan

    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\([^\)]*\)", "", s).strip()

    try:
        if " " in s:
            parts = s.split()
            if len(parts) == 2 and "/" in parts[1]:
                return float(parts[0]) + float(Fraction(parts[1]))
        if "-" in s:
            vals = [parse_quantity_value(part) for part in s.split("-") if part.strip()]
            vals = [v for v in vals if pd.notna(v)]
            if vals:
                return float(np.mean(vals))
        return float(Fraction(s))
    except Exception:
        try:
            return float(s)
        except Exception:
            return np.nan


def normalize_quantity_list(q_list):
    """Normalize a list-like quantity field into numeric values.

    Parameters
    ----------
    q_list : Any
        List-like quantity field.

    Returns
    -------
    list[float]
        Parsed numeric values with missing entries replaced by 0.0.
    """
    values = [parse_quantity_value(x) for x in _safe_list(q_list)]
    return [0.0 if pd.isna(v) else float(v) for v in values]


def parse_time_to_minutes(x):
    """Convert ISO-8601-like duration strings such as PT1H15M into minutes.

    Parameters
    ----------
    x : Any
        Raw time value.

    Returns
    -------
    float
        Minutes, or NaN when parsing fails.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?$", s)
    if match:
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        return h * 60 + m

    try:
        return float(s)
    except Exception:
        return np.nan

def build_display_ingredient_rows(row: pd.Series) -> list[dict]:
    """Build display-friendly ingredient rows with quantity, raw text, and normalized name."""
    raw_list = row.get("ingredients_raw_list", [])
    clean_list = row.get("ingredients_clean", [])
    qty_list = row.get("ingredient_quantities", [])

    raw_list = raw_list if isinstance(raw_list, list) else []
    clean_list = clean_list if isinstance(clean_list, list) else []
    qty_list = qty_list if isinstance(qty_list, list) else []

    n = max(len(raw_list), len(clean_list), len(qty_list))
    rows = []

    for i in range(n):
        rows.append({
            "quantity": qty_list[i] if i < len(qty_list) else None,
            "raw_ingredient": raw_list[i] if i < len(raw_list) else None,
            "normalized_ingredient": clean_list[i] if i < len(clean_list) else None,
        })

    return rows

def extract_first_image_url(x):
    """Return the first usable image URL from a raw Images field."""
    vals = _safe_list(x)
    for val in vals:
        s = str(val).strip()
        if s and s.lower().startswith(("http://", "https://")):
            return s
    s = str(x).strip()
    return s if s.lower().startswith(("http://", "https://")) else ""


def normalize_recipe_meal_type(category, keywords=None):
    """Map recipe category / keywords to a meal type for app filtering."""
    category_text = str(category).strip().lower()
    keyword_vals = keywords if isinstance(keywords, list) else _safe_list(keywords)
    keyword_text = " ".join(str(x).strip().lower() for x in keyword_vals if str(x).strip())

    text = f"{category_text} {keyword_text}".strip()

    if any(term in text for term in ["breakfast", "brunch"]):
        return "breakfast"
    if "lunch" in text:
        return "lunch"
    if any(term in text for term in ["dinner", "supper"]):
        return "dinner"
    if any(term in text for term in ["snack", "appetizer"]):
        return "snack"
    return "other"



# ============================================================================
# Notebook cell 4
# ============================================================================

"""Ingredient normalization dictionaries and helper functions."""

INGREDIENT_ALIASES = {
    "garbanzo bean": "chickpea",
    "garbanzo beans": "chickpea",
    "chickpeas": "chickpea",
    "black beans": "black bean",
    "kidney beans": "kidney bean",
    "white beans": "white bean",
    "cannellini beans": "cannellini bean",
    "spring onions": "green onion",
    "scallions": "green onion",
    "red onions": "red onion",
    "yellow onions": "yellow onion",
    "brown onions": "onion",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "sweet potatoes": "sweet potato",
    "bell peppers": "bell pepper",
    "capsicum": "bell pepper",
    "courgette": "zucchini",
    "aubergine": "eggplant",
    "cilantro": "coriander",
    "confectioners sugar": "powdered sugar",
    "icing sugar": "powdered sugar",
    "caster sugar": "sugar",
    "extra virgin olive oil": "olive oil",
    "virgin olive oil": "olive oil",
    "canola oil": "rapeseed oil",
    "chicken breasts": "chicken breast",
    "chicken thighs": "chicken thigh",
    "ground beef": "beef",
    "minced beef": "beef",
    "ground turkey": "turkey",
    "minced turkey": "turkey",
}

DESCRIPTOR_WORDS = {
    "fresh", "frozen", "canned", "dried", "raw", "cooked",
    "chopped", "diced", "minced", "sliced", "grated", "shredded",
    "boneless", "skinless", "lean", "large", "small", "medium",
    "organic", "optional", "plain", "whole", "halved", "crushed",
    "toasted", "softened", "melted", "warm", "cold", "drained",
    "rinsed", "beaten", "packed", "extra", "virgin", "firmly",
    "choice", "chef", "reserve", "reserved", "flavored", "flavoured"
}

UNIT_WORDS = {
    "cup", "cups", "tbsp", "tsp", "teaspoon", "teaspoons",
    "tablespoon", "tablespoons", "oz", "ounce", "ounces",
    "lb", "lbs", "pound", "pounds", "gram", "grams", "g",
    "kg", "ml", "l", "pinch", "clove", "cloves", "slice", "slices",
    "can", "cans", "package", "packages", "pack", "packs",
    "carton", "cartons", "jar", "jars", "inch", "inches"
}


def normalize_ingredient_name(text: str) -> str:
    """Normalize raw ingredient text for matching, scoring, and filtering.

    Parameters
    ----------
    text : str
        Raw ingredient text.

    Returns
    -------
    str
        Normalized ingredient phrase.
    """
    if not isinstance(text, str):
        return ""

    s = text.lower().strip()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\b\d+\s*/\s*\d+\b", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    words = [w for w in s.split() if w not in DESCRIPTOR_WORDS and w not in UNIT_WORDS]
    s = " ".join(words).strip()

    if s in INGREDIENT_ALIASES:
        s = INGREDIENT_ALIASES[s]

    singular_map = {
        "tomatoes": "tomato",
        "potatoes": "potato",
        "onions": "onion",
        "carrots": "carrot",
        "peppers": "pepper",
        "beans": "bean",
        "lentils": "lentil",
        "pecans": "pecan",
        "walnuts": "walnut",
        "almonds": "almond",
        "eggs": "egg",
    }
    s = " ".join(singular_map.get(w, w) for w in s.split()).strip()

    if s in INGREDIENT_ALIASES:
        s = INGREDIENT_ALIASES[s]

    return s


def clean_ingredient(token: str) -> str:
    """Compatibility wrapper around ingredient normalization.

    Parameters
    ----------
    token : str
        Raw ingredient token.

    Returns
    -------
    str
        Normalized ingredient text.
    """
    return normalize_ingredient_name(token)


def normalize_user_ingredients(items):
    """Normalize user pantry, must-include, or must-exclude ingredients.

    Parameters
    ----------
    items : str | list[str] | None
        User-provided ingredient input.

    Returns
    -------
    set[str]
        Normalized ingredient set.
    """
    if items is None:
        return set()

    if isinstance(items, str):
        items = [items]

    normalized = set()
    for item in items:
        norm = normalize_ingredient_name(item)
        if norm:
            normalized.add(norm)
    return normalized


# ============================================================================
# Notebook cell 8
# ============================================================================

"""Feature engineering for food tags, nutrition features, Nutri-Score, NRF score, and health flags."""

def infer_food_tags(row):
    """Infer food-type tags from normalized ingredients, category, and keywords.

    Returns tags such as:
    - vegan
    - vegetarian
    - meat
    - poultry
    - chicken
    - seafood
    - dairy
    - egg
    - dessert
    - salad
    - breakfast
    """
    tags = set()

    ingredients = row.get("ingredients_clean", [])
    ingredients = [str(x).lower() for x in ingredients] if isinstance(ingredients, list) else []

    category = str(row.get("RecipeCategory", "")).lower()
    keywords = row.get("Keywords", [])
    keywords = [str(x).lower() for x in keywords] if isinstance(keywords, list) else []

    text = " ".join(ingredients + [category] + keywords)

    meat_terms = {"beef", "pork", "lamb", "bacon", "ham", "sausage"}
    poultry_terms = {"chicken", "turkey", "duck"}
    seafood_terms = {"fish", "salmon", "tuna", "shrimp", "prawn", "cod", "crab"}
    dairy_terms = {"milk", "cheese", "yogurt", "butter", "cream"}
    egg_terms = {"egg"}
    vegan_blockers = meat_terms | poultry_terms | seafood_terms | dairy_terms | egg_terms

    if any(t in text for t in poultry_terms):
        tags.update({"chicken", "poultry"})
    if any(t in text for t in seafood_terms):
        tags.add("seafood")
    if any(t in text for t in meat_terms):
        tags.add("meat")
    if any(t in text for t in dairy_terms):
        tags.add("dairy")
    if any(t in text for t in egg_terms):
        tags.add("egg")

    if not any(t in text for t in meat_terms | poultry_terms | seafood_terms):
        tags.add("vegetarian")

    if not any(t in text for t in vegan_blockers):
        tags.add("vegan")

    if "dessert" in category:
        tags.add("dessert")
    if "salad" in category:
        tags.add("salad")
    if "breakfast" in category:
        tags.add("breakfast")

    return sorted(tags)


def add_weight_loss_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-100g, density, ratio, and macro-label features used in nutrition scoring.

    Parameters
    ----------
    df : pandas.DataFrame
        Recipe dataframe with nutrition and serving columns.

    Returns
    -------
    pandas.DataFrame
        Dataframe with added nutrition-engineered features.
    """
    df = df.copy()

    numeric_cols = [
        "calories", "protein", "fat", "sat_fat", "carbs",
        "sugar", "fiber", "sodium", "serving_g", "cal_per_g", "cook_time"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = [
        "calories", "protein", "fat", "sat_fat", "carbs",
        "sugar", "fiber", "sodium", "serving_g", "cal_per_g"
    ]
    df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    df = df[(df["calories"] > 0) & (df["serving_g"] > 0) & (df["cal_per_g"] > 0)].copy()

    df["energy_kcal_100g"] = df["cal_per_g"] * 100
    df["protein_100g"] = df["protein"] / (df["serving_g"] + EPS) * 100
    df["fat_100g"] = df["fat"] / (df["serving_g"] + EPS) * 100
    df["sat_fat_100g"] = df["sat_fat"] / (df["serving_g"] + EPS) * 100
    df["carbs_100g"] = df["carbs"] / (df["serving_g"] + EPS) * 100
    df["sugar_100g"] = df["sugar"] / (df["serving_g"] + EPS) * 100
    df["fiber_100g"] = df["fiber"] / (df["serving_g"] + EPS) * 100
    df["sodium_100g"] = df["sodium"] / (df["serving_g"] + EPS) * 100

    df["protein_per_kcal"] = df["protein"] / (df["calories"] + EPS)
    df["fiber_per_kcal"] = df["fiber"] / (df["calories"] + EPS)
    df["sugar_per_kcal"] = df["sugar"] / (df["calories"] + EPS)

    df["low_calorie_density"] = (df["cal_per_g"] <= 1.5).astype(int)
    df["high_protein_flag"] = (df["protein_100g"] >= 8).astype(int)
    df["high_fiber_flag"] = (df["fiber_100g"] >= 3).astype(int)
    df["high_sugar_flag"] = (df["sugar_100g"] >= 10).astype(int)
    df["high_sat_fat_flag"] = (df["sat_fat_100g"] >= 5).astype(int)
    df["high_sodium_flag"] = (df["sodium_100g"] >= 600).astype(int)

    df["protein_kcal"] = df["protein"] * 4
    df["carb_kcal"] = df["carbs"] * 4
    df["fat_kcal"] = df["fat"] * 9

    df["protein_pct"] = df["protein_kcal"] / (df["calories"] + EPS)
    df["carb_pct"] = df["carb_kcal"] / (df["calories"] + EPS)
    df["fat_pct"] = df["fat_kcal"] / (df["calories"] + EPS)

    def classify_protein(p):
        """ These thresholds align with the AMDR: 10–35 % of calories from protein, 45–65 % from carbohydrates and 20–35 % from fats"""
        if pd.isna(p):
            return np.nan
        if p < 0.10:
            return "low"
        if p < 0.35:
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

    df["protein_level"] = df["protein_pct"].apply(classify_protein)
    df["carb_level"] = df["carb_pct"].apply(classify_carb)
    df["fat_level"] = df["fat_pct"].apply(classify_fat)

    df["macro_labels"] = df.apply(
        lambda r: [
            lbl for lbl, ok in [
                ("High Protein", r["protein_level"] == "high"),
                ("Fiber Friendly", r["fiber_100g"] >= 3),
                ("Low Sugar", r["sugar_100g"] <= 5),
                ("Low Sodium", r["sodium_100g"] <= 120),
                ("Low Calorie Density", r["low_calorie_density"] == 1),
            ] if ok
        ],
        axis=1,
    )

    return df


def add_goal_alignment_score(df: pd.DataFrame, target_meal_calories: float) -> pd.DataFrame:
    """Score recipes by closeness to a goal-adjusted per-meal calorie target.

    This score is user-specific and should be computed only after user targets
    are known. A value of 100 means the recipe is exactly on the goal meal
    calories, while lower values indicate a poorer fit.
    """
    df = df.copy()
    target_meal_calories = max(float(target_meal_calories), 1.0)

    ratio = (pd.to_numeric(df["calories"], errors="coerce") - target_meal_calories) / (target_meal_calories + EPS)
    df["goal_alignment_score"] = (100 * (1 - ratio.abs())).clip(0, 100).round(1)
    return df



FVP_WORDS = {
    "apple", "banana", "orange", "lemon", "lime", "grape", "mango", "papaya",
    "pineapple", "peach", "pear", "plum", "cherry", "berry", "blueberry",
    "strawberry", "raspberry", "blackberry", "apricot", "fig", "date",
    "raisin", "coconut", "avocado", "tomato", "carrot", "spinach", "kale",
    "lettuce", "cabbage", "broccoli", "cauliflower", "zucchini", "eggplant",
    "bell pepper", "pepper", "onion", "garlic", "ginger", "celery", "cucumber",
    "pumpkin", "squash", "yam", "sweet potato", "potato", "corn", "mushroom",
    "pea", "green bean", "okra", "beet", "radish", "turnip", "artichoke"
}

PULSE_WORDS = {
    "lentil", "chickpea", "black bean", "kidney bean", "white bean",
    "cannellini bean", "pinto bean", "split pea", "pea", "bean"
}

NUT_WORDS = {
    "almond", "walnut", "pecan", "cashew", "hazelnut", "pistachio",
    "macadamia", "peanut", "nut"
}

HEALTHY_OIL_WORDS = {"olive oil", "walnut oil", "rapeseed oil", "canola oil"}


def add_nutriscore_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ingredient-group proxy features used for approximate Nutri-Score.

    Because exact ingredient-by-weight percentages are unavailable, fruits,
    vegetables, pulses, nuts, and healthy oils are estimated using counts and ratios.
    """
    df = df.copy()

    def count_groups(ings):
        if not isinstance(ings, list):
            ings = []

        fvp = sum(any(term in ing for term in FVP_WORDS) for ing in ings)
        pulses = sum(any(term in ing for term in PULSE_WORDS) for ing in ings)
        nuts = sum(any(term in ing for term in NUT_WORDS) for ing in ings)
        healthy_oils = sum(any(term in ing for term in HEALTHY_OIL_WORDS) for ing in ings)

        total = max(len(ings), 1)
        return pd.Series({
            "fvp_count": fvp,
            "pulse_count": pulses,
            "nut_count": nuts,
            "healthy_oil_count": healthy_oils,
            "fvp_ratio": fvp / total,
            "pulse_ratio": pulses / total,
            "nut_ratio": nuts / total,
            "healthy_oil_ratio": healthy_oils / total,
        })

    counts = df["ingredients_clean"].apply(count_groups)
    df = pd.concat([df, counts], axis=1)
    df["beneficial_ingredient_ratio"] = (
        df["fvp_ratio"] + df["pulse_ratio"] + df["nut_ratio"] + df["healthy_oil_ratio"]
    ).clip(upper=1.0)
    return df


def _points_from_thresholds(value, thresholds):
    """Assign points based on ordered thresholds."""
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)


def add_nutriscore_points(df: pd.DataFrame) -> pd.DataFrame:
    """Compute approximate Nutri-Score numeric points and A-to-E labels.

    Notes
    -----
    This is a recipe-oriented approximation. FVPN percentage is estimated using
    ingredient ratios instead of exact weight shares.
    """
    df = df.copy()

    df["energy_kj_100g"] = df["energy_kcal_100g"] * 4.184

    df["ns_energy_points"] = df["energy_kj_100g"].apply(
        lambda x: _points_from_thresholds(x, [335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350])
    )
    df["ns_sugar_points"] = df["sugar_100g"].apply(
        lambda x: _points_from_thresholds(x, [4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45])
    )
    df["ns_satfat_points"] = df["sat_fat_100g"].apply(
        lambda x: _points_from_thresholds(x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    )
    df["ns_sodium_points"] = df["sodium_100g"].apply(
        lambda x: _points_from_thresholds(x, [90, 180, 270, 360, 450, 540, 630, 720, 810, 900])
    )

    df["ns_negative_points"] = (
        df["ns_energy_points"] +
        df["ns_sugar_points"] +
        df["ns_satfat_points"] +
        df["ns_sodium_points"]
    )

    def fvpn_points(r):
        if r >= 0.80:
            return 5
        if r >= 0.60:
            return 2
        if r >= 0.40:
            return 1
        return 0

    def fiber_points(x):
        return _points_from_thresholds(x, [0.9, 1.9, 2.8, 3.7, 4.7])

    def protein_points(x):
        return _points_from_thresholds(x, [1.6, 3.2, 4.8, 6.4, 8.0])

    df["ns_fvpn_points"] = df["beneficial_ingredient_ratio"].apply(fvpn_points)
    df["ns_fiber_points"] = df["fiber_100g"].apply(fiber_points)
    df["ns_protein_points"] = df["protein_100g"].apply(protein_points)

    df["nutri_score_numeric"] = (
        df["ns_negative_points"] -
        df["ns_fvpn_points"] -
        df["ns_fiber_points"] -
        df["ns_protein_points"]
    )

    def label_food(score):
        if score <= -1:
            return "A"
        if score <= 2:
            return "B"
        if score <= 10:
            return "C"
        if score <= 18:
            return "D"
        return "E"

    df["nutri_score_label"] = df["nutri_score_numeric"].apply(label_food)
    return df


def compute_nutrient_density_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a stable nutrient-density indicator from per-100g features.

    Important:
    - This is NOT canonical NRF23.
    - It is a reduced nutrient-density model based on the nutrients
      available in this dataset.
    - Score is standardized per 100 kcal, which is closer to NRF logic.

    Required columns
    ----------------
    energy_kcal_100g
    protein_100g
    fiber_100g
    sugar_100g
    sat_fat_100g
    sodium_100g
    """
    df = df.copy()

    required = [
        "energy_kcal_100g",
        "protein_100g",
        "fiber_100g",
        "sugar_100g",
        "sat_fat_100g",
        "sodium_100g",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows that can be scored
    df = df.dropna(subset=required).copy()
    df = df[df["energy_kcal_100g"] > 0].copy()

    # Safety clipping against extreme bad values
    df["protein_100g"] = df["protein_100g"].clip(0, 100)
    df["fiber_100g"] = df["fiber_100g"].clip(0, 100)
    df["sugar_100g"] = df["sugar_100g"].clip(0, 100)
    df["sat_fat_100g"] = df["sat_fat_100g"].clip(0, 100)
    df["sodium_100g"] = df["sodium_100g"].clip(0, 10000)

    # Convert per-100g to per-100kcal
    kcal_factor = 100.0 / (df["energy_kcal_100g"] + EPS)

    df["protein_per_100kcal"] = df["protein_100g"] * kcal_factor
    df["fiber_per_100kcal"] = df["fiber_100g"] * kcal_factor
    df["sugar_per_100kcal"] = df["sugar_100g"] * kcal_factor
    df["sat_fat_per_100kcal"] = df["sat_fat_100g"] * kcal_factor
    df["sodium_per_100kcal"] = df["sodium_100g"] * kcal_factor

    # Reference values (practical FDA-style anchors)
    # Protein has no mandatory %DV on many labels, but 50 g is a common reference intake.
    protein_ref = 50.0      # g/day
    fiber_ref = 28.0        # g/day
    sugar_max = 50.0        # g/day (proxy; ideally use added sugar if available)
    satfat_max = 20.0       # g/day
    sodium_max = 2300.0     # mg/day

    # Positive nutrients: cap each at 1.0
    df["nd_pos_protein"] = (df["protein_per_100kcal"] / protein_ref).clip(0, 1.0)
    df["nd_pos_fiber"] = (df["fiber_per_100kcal"] / fiber_ref).clip(0, 1.0)

    # Negative nutrients: cap each at 1.0
    df["nd_neg_sugar"] = (df["sugar_per_100kcal"] / sugar_max).clip(0, 1.0)
    df["nd_neg_satfat"] = (df["sat_fat_per_100kcal"] / satfat_max).clip(0, 1.0)
    df["nd_neg_sodium"] = (df["sodium_per_100kcal"] / sodium_max).clip(0, 1.0)

    # Raw score: theoretical range = [-3, +2]
    df["nutrient_density_raw"] = (
        df["nd_pos_protein"] +
        df["nd_pos_fiber"] -
        df["nd_neg_sugar"] -
        df["nd_neg_satfat"] -
        df["nd_neg_sodium"]
    )

    # Fixed mapping to 0-100 (stable across datasets)
    raw_min, raw_max = -3.0, 2.0
    df["nutrient_density_score"] = (
        (df["nutrient_density_raw"] - raw_min) / (raw_max - raw_min) * 100.0
    ).clip(0, 100).round(1)

    # Interpretable class labels
    def classify(score):
        if pd.isna(score):
            return np.nan
        if score >= 70:
            return "high nutrient density"
        if score >= 40:
            return "moderate nutrient density"
        return "low nutrient density"

    df["nutrient_density_class"] = df["nutrient_density_score"].apply(classify)

    # Optional short indicator for UI badges
    def badge(score):
        if pd.isna(score):
            return np.nan
        if score >= 70:
            return "ND-A"
        if score >= 55:
            return "ND-B"
        if score >= 40:
            return "ND-C"
        if score >= 25:
            return "ND-D"
        return "ND-E"

    df["nutrient_density_badge"] = df["nutrient_density_score"].apply(badge)

    return df


def add_medical_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic medical risk flags for common conditions such as diabetes and hypertension."""
    df = df.copy()

    for col in ["sugar", "sodium", "sat_fat", "cholest", "protein", "carbs"]:
        if col in df.columns:
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

    def classify_medical_risk(score):
        if score == 0:
            return "low"
        if score <= 2:
            return "moderate"
        return "high"

    df["medical_risk_level"] = df["medical_risk_score"].apply(classify_medical_risk)

    def risk_reason(row):
        reasons = []
        if row["risk_diabetes"]:
            reasons.append("high sugar")
        if row["risk_hypertension"]:
            reasons.append("high sodium")
        if row["risk_heart_disease"]:
            reasons.append("high saturated fat")
        if row["risk_cholesterol"]:
            reasons.append("high cholesterol")
        if row["risk_kidney"]:
            reasons.append("very high protein")
        if row["risk_keto_violation"]:
            reasons.append("high carbohydrates")
        return ", ".join(reasons) if reasons else "no major medical risk"

    df["medical_risk_reason"] = df.apply(risk_reason, axis=1)
    return df


def filter_recipes_by_health(df: pd.DataFrame, user_conditions=None, strict=True) -> pd.DataFrame:
    """Filter recipes by health-condition risk flags.

    If a required risk column is missing, that condition is skipped instead of raising an error.
    """
    filtered = df.copy()
    user_conditions = {str(x).strip().lower() for x in (user_conditions or [])}

    if not user_conditions:
        return filtered

    condition_to_col = {
        "diabetes": "risk_diabetes",
        "hypertension": "risk_hypertension",
        "heart disease": "risk_heart_disease",
        "cholesterol": "risk_cholesterol",
        "kidney": "risk_kidney",
    }

    for condition, col in condition_to_col.items():
        if condition in user_conditions and strict and col in filtered.columns:
            filtered = filtered[filtered[col] == 0]

    return filtered


# ============================================================================
# Notebook cell 10
# ============================================================================

"""Load the merged recipe CSV, standardize fields, and add app-ready features."""

def load_and_process_recipe_csv(path_csv: str) -> pd.DataFrame:
    """Load and preprocess the merged recipe CSV for modeling and app serving.

    Parameters
    ----------
    path_csv : str
        Path to the merged recipe dataset.

    Returns
    -------
    pandas.DataFrame
        Processed dataframe with recipe display, filtering, and scoring features.
    """
    df = pd.read_csv(path_csv)

    required_cols = [
        "id",
        "name_final",
        "ingredients",
        "steps",
        "Calories",
        "FatContent",
        "SaturatedFatContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "ProteinContent",
        "CholesterolContent",
        "SodiumContent",
        "servings",
        "serving_g",
        "cal_per_g",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["recipe_id"] = df["id"]
    df["final_name"] = df["name_final"].fillna("").astype(str).str.strip()
    df["name"] = df["final_name"]

    if "description_final" in df.columns:
        df["final_description"] = df["description_final"].fillna("").astype(str).str.strip()
    else:
        df["final_description"] = ""

    # ingredients / instructions
    df["ingredients_list"] = df["ingredients"].apply(_safe_list)

    if "ingredients_raw" in df.columns:
        df["ingredients_raw_list"] = df["ingredients_raw"].apply(_safe_list)
    else:
        df["ingredients_raw_list"] = df["ingredients_list"]

    # optional quantities if available from source tables
    if "RecipeIngredientQuantities" in df.columns:
        df["ingredient_quantities"] = df["RecipeIngredientQuantities"].apply(_safe_list)
    else:
        df["ingredient_quantities"] = [[] for _ in range(len(df))]

    df["ingredients_normalized"] = df["ingredients_list"].apply(
        lambda lst: [normalize_ingredient_name(x) for x in lst if normalize_ingredient_name(x)]
    )

    df["ingredients_clean"] = df["ingredients_normalized"].apply(
        lambda lst: sorted(list(set(lst))) if isinstance(lst, list) else []
    )

    df["ingredients_text"] = df["ingredients_clean"].apply(
        lambda lst: " ".join(lst) if isinstance(lst, list) else ""
    )

    df["instructions_list"] = df["steps"].apply(parse_instructions)

    if "Keywords" in df.columns:
        df["Keywords"] = df["Keywords"].apply(_safe_list)
    else:
        df["Keywords"] = [[] for _ in range(len(df))]

    if "RecipeCategory" not in df.columns:
        df["RecipeCategory"] = ""
    else:
        df["RecipeCategory"] = df["RecipeCategory"].fillna("").astype(str).str.strip()

    if "Images" not in df.columns:
        df["Images"] = ""
    else:
        df["Images"] = df["Images"].fillna("").astype(str).str.strip()

    df["image_url"] = df["Images"].apply(extract_first_image_url)
    df["dd_meal_type"] = df.apply(
        lambda row: normalize_recipe_meal_type(
            row.get("RecipeCategory", ""),
            row.get("Keywords", []),
        ),
        axis=1,
    )

    if "TotalTime" in df.columns:
        df["cook_time"] = df["TotalTime"].apply(parse_time_to_minutes)
    else:
        df["cook_time"] = np.nan

    df = df.rename(columns={
        "Calories": "calories",
        "FatContent": "fat",
        "SaturatedFatContent": "sat_fat",
        "CarbohydrateContent": "carbs",
        "FiberContent": "fiber",
        "SugarContent": "sugar",
        "ProteinContent": "protein",
        "CholesterolContent": "cholest",
        "SodiumContent": "sodium",
    })

    numeric_cols = [
        "calories", "fat", "sat_fat", "carbs", "fiber", "sugar",
        "protein", "cholest", "sodium", "servings", "serving_g", "cal_per_g", "cook_time"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[
        "final_name", "calories", "fat", "sat_fat", "carbs",
        "fiber", "sugar", "protein", "sodium", "serving_g", "cal_per_g"
    ]).copy()

    df = df[
        (df["final_name"] != "") &
        (df["calories"] > 0) &
        (df["serving_g"] > 0) &
        (df["cal_per_g"] > 0)
    ].copy()

    df["food_tags"] = df.apply(infer_food_tags, axis=1)
    df = add_weight_loss_features(df)
    df = add_nutriscore_features(df)
    df = add_nutriscore_points(df)
    df = compute_nutrient_density_score(df)
    df = add_medical_risk_flags(df)

    df = df.drop_duplicates(subset=["final_name", "ingredients_text"]).reset_index(drop=True)
    return df



# ============================================================================
# Notebook cell 12
# ============================================================================

"""Functions for pantry matching, TF-IDF ingredient matching, KNN macro matching, and final ranking.

This revised block fixes:
- categorical fillna/filter errors in preference filters
- missing passes_include_exclude at inference time
- robust ingredient parsing from saved processed files
- TF-IDF row alignment using source_index instead of filtered dataframe index
"""
def _safe_lower_text(series: pd.Series) -> pd.Series:
    """Convert any pandas dtype, including Categorical, to lowercase strings safely."""
    return series.astype("string").fillna("").str.strip().str.lower()


def _normalize_text_token(value):
    """Normalize a free-text token for ingredient comparison."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _coerce_ingredient_list(value):
    """Convert ingredient fields into a normalized list of strings.

    Handles Python lists, JSON-like strings, newline-separated text, and plain strings.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    items = None

    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [text]
            except Exception:
                if "\n" in text:
                    items = [part.strip() for part in text.split("\n") if part.strip()]
                else:
                    items = [text]
        elif "\n" in text:
            items = [part.strip() for part in text.split("\n") if part.strip()]
        elif "|" in text:
            items = [part.strip() for part in text.split("|") if part.strip()]
        else:
            items = [text]
    else:
        items = [str(value)]

    cleaned = []
    for item in (items or []):
        norm = (
            normalize_ingredient_name(item)
            if "normalize_ingredient_name" in globals()
            else _normalize_text_token(item)
        )
        if norm:
            cleaned.append(norm)

    return cleaned


def _get_ingredient_series(df: pd.DataFrame) -> pd.Series:
    """Pick the best available ingredient column from saved processed data."""
    for col in ["ingredients_clean", "ingredients_raw_list", "ingredients_list", "ingredients_text"]:
        if col in df.columns:
            return df[col]
    return pd.Series([[] for _ in range(len(df))], index=df.index, dtype="object")


def ingredient_overlap(recipe_ingredients, user_ingredients):
    """Return matched and missing ingredients between a recipe and user pantry."""
    recipe_set = set(_coerce_ingredient_list(recipe_ingredients))
    user_set = set(
        _coerce_ingredient_list(list(user_ingredients) if isinstance(user_ingredients, set) else user_ingredients)
    )

    matched = sorted(recipe_set & user_set)
    missing = sorted(recipe_set - user_set)
    coverage = len(matched) / max(len(recipe_set), 1)

    return {
        "matched_ingredients": matched,
        "missing_ingredients": missing,
        "matched_count": len(matched),
        "missing_count": len(missing),
        "ingredient_coverage": coverage,
    }


def recipe_passes_ingredient_rules(recipe_ingredients, must_include=None, must_exclude=None):
    """Check whether a recipe satisfies must-include and must-exclude rules."""
    recipe_set = set(_coerce_ingredient_list(recipe_ingredients))
    must_include = normalize_user_ingredients(must_include)
    must_exclude = normalize_user_ingredients(must_exclude)

    include_ok = must_include.issubset(recipe_set) if must_include else True
    exclude_ok = recipe_set.isdisjoint(must_exclude) if must_exclude else True
    return include_ok and exclude_ok


def compute_ingredient_score(recipe_ingredients, pantry_ingredients=None, must_include=None, must_exclude=None):
    """Fallback exact-match ingredient score used alongside TF-IDF similarity."""
    recipe_set = set(_coerce_ingredient_list(recipe_ingredients))
    pantry_set = normalize_user_ingredients(pantry_ingredients)
    include_set = normalize_user_ingredients(must_include)
    exclude_set = normalize_user_ingredients(must_exclude)

    matched = recipe_set & pantry_set
    include_hits = recipe_set & include_set
    exclude_hits = recipe_set & exclude_set

    coverage = len(matched) / max(len(recipe_set), 1)
    include_bonus = len(include_hits) / max(len(include_set), 1) if include_set else 0
    exclude_penalty = len(exclude_hits) / max(len(exclude_set), 1) if exclude_set else 0

    score = 0.7 * coverage + 0.3 * include_bonus - 1.0 * exclude_penalty
    return max(0.0, round(score, 4))


def add_user_ingredient_features(df, pantry_ingredients=None, must_include=None, must_exclude=None):
    """Add exact matched/missing ingredients plus a fallback overlap score.

    This version is robust to saved parquet/list-string columns and always creates
    passes_include_exclude so downstream filtering does not fail.
    """
    out = df.copy()

    pantry_set = normalize_user_ingredients(pantry_ingredients)
    include_set = normalize_user_ingredients(must_include)
    exclude_set = normalize_user_ingredients(must_exclude)

    ingredient_series = _get_ingredient_series(out)

    def process_row(ings):
        normalized_ings = _coerce_ingredient_list(ings)
        result = ingredient_overlap(normalized_ings, pantry_set | include_set)
        result["ingredient_score_overlap"] = compute_ingredient_score(
            normalized_ings,
            pantry_ingredients=pantry_set,
            must_include=include_set,
            must_exclude=exclude_set,
        )
        result["passes_include_exclude"] = recipe_passes_ingredient_rules(
            normalized_ings,
            must_include=include_set,
            must_exclude=exclude_set,
        )
        return pd.Series(result)

    features = ingredient_series.apply(process_row)
    out = pd.concat([out, features], axis=1)

    if "passes_include_exclude" not in out.columns:
        out["passes_include_exclude"] = True

    return out

# Replace the existing fit/load/KNN scoring block with this version.
# This uses ONLY protein, fat, carbs in KNN space.

def fit_macro_knn(
    df: pd.DataFrame,
    feature_cols=("protein", "fat", "carbs"),
    model_path: str = KNN_MODEL_PATH,
):
    """Fit and save a KNN model for macro-based candidate retrieval using protein/fat/carbs only."""
    macro_X = (
        df.loc[:, list(feature_cols)]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    macro_scaler = StandardScaler()
    macro_X_scaled = macro_scaler.fit_transform(macro_X)

    knn_model = NearestNeighbors(metric="euclidean", algorithm="auto")
    knn_model.fit(macro_X_scaled)

    bundle = {
        "model": knn_model,
        "scaler": macro_scaler,
        "feature_cols": list(feature_cols),
        "fit_index": macro_X.index.to_numpy(),
        "n_rows": len(macro_X),
    }
    joblib.dump(bundle, model_path)
    print("Saved:", model_path)
    return bundle


def load_macro_knn(model_path: str = KNN_MODEL_PATH):
    """Load the saved KNN macro model bundle."""
    return joblib.load(model_path)


def knn_candidates(
    df: pd.DataFrame,
    protein: float,
    fat: float,
    carbs: float,
    n_neighbors: int = 250,
    knn_bundle=None,
):
    """Retrieve macro-nearest recipes in standardized protein+fat+carb space."""
    if knn_bundle is None:
        knn_bundle = load_macro_knn()

    feature_cols = knn_bundle.get("feature_cols", ["protein", "fat", "carbs"])

    user_vec = pd.DataFrame([{
        "protein": protein,
        "fat": fat,
        "carbs": carbs,
    }])[feature_cols]

    user_scaled = knn_bundle["scaler"].transform(user_vec)
    n_neighbors = int(min(max(n_neighbors, 1), knn_bundle.get("n_rows", len(df))))

    distances, indices = knn_bundle["model"].kneighbors(user_scaled, n_neighbors=n_neighbors)

    fit_index = knn_bundle.get("fit_index")
    if fit_index is not None:
        real_indices = fit_index[indices[0]]
        candidates = df.loc[real_indices].copy()
    else:
        candidates = df.iloc[indices[0]].copy()

    candidates["source_index"] = candidates.index
    candidates["macro_distance"] = distances[0]
    candidates["macro_score"] = (1 / (1 + candidates["macro_distance"])).round(4)
    return candidates


def add_knn_macro_features(df: pd.DataFrame, user_targets: dict, knn_bundle=None) -> pd.DataFrame:
    """Score any dataframe against the user's protein/fat/carb targets using the saved KNN scaler geometry."""
    if knn_bundle is None:
        knn_bundle = load_macro_knn()

    feature_cols = knn_bundle.get("feature_cols", ["protein", "fat", "carbs"])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing macro columns required by KNN model: {missing}")

    target_map = {
        "protein": float(user_targets.get("meal_protein", 0)),
        "fat": float(user_targets.get("meal_fat", 0)),
        "carbs": float(user_targets.get("meal_carbs", 0)),
    }
    user_vec = pd.DataFrame([{c: target_map[c] for c in feature_cols}])

    recipe_scaled = knn_bundle["scaler"].transform(df[feature_cols])
    user_scaled = knn_bundle["scaler"].transform(user_vec)[0]
    distances = np.linalg.norm(recipe_scaled - user_scaled, axis=1)

    out = df.copy()
    out["macro_distance"] = distances
    out["macro_score"] = (1 / (1 + out["macro_distance"])).round(4)
    return out


def compute_macro_score(df: pd.DataFrame, user_targets: dict, knn_bundle=None) -> pd.DataFrame:
    """Compatibility wrapper: macro score now comes from KNN distance in protein+fat+carb space."""
    return add_knn_macro_features(df, user_targets=user_targets, knn_bundle=knn_bundle)


def build_ingredient_query_text(pantry_ingredients=None, must_include=None):
    """Build a text query from pantry and must-include ingredients for TF-IDF matching."""
    pantry = sorted(normalize_user_ingredients(pantry_ingredients))
    include = sorted(normalize_user_ingredients(must_include))
    return " ".join(pantry + include).strip()


def add_tfidf_ingredient_match(
    df: pd.DataFrame,
    pantry_ingredients=None,
    must_include=None,
    vectorizer=None,
    matrix=None,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add TF-IDF ingredient similarity between user pantry query and recipes.

    Uses source_index when available so filtered candidate rows still map back to the
    original saved TF-IDF matrix correctly.
    """
    out = df.copy()
    query_text = build_ingredient_query_text(
        pantry_ingredients=pantry_ingredients,
        must_include=must_include,
    )

    if not query_text.strip():
        out["ingredient_similarity"] = 0.0
        if "ingredient_score_overlap" in out.columns:
            out["ingredient_score"] = out["ingredient_score_overlap"].astype(float).round(4)
        else:
            out["ingredient_score"] = 0.0
        return out

    if vectorizer is None or matrix is None:
        vectorizer, matrix = load_ingredient_tfidf()

    row_ids = out["source_index"].to_numpy() if "source_index" in out.columns else out.index.to_numpy()
    query_vec = vectorizer.transform([query_text])
    subset_matrix = matrix[row_ids]
    sims = cosine_similarity(query_vec, subset_matrix).ravel()

    out["ingredient_similarity"] = np.round(sims, 4)
    overlap_score = out.get("ingredient_score_overlap", pd.Series(0.0, index=out.index)).astype(float)
    out["ingredient_score"] = (0.75 * out["ingredient_similarity"] + 0.25 * overlap_score).round(4)
    return out


def apply_preference_filters(
    df,
    preferred_food_types=None,
    preferred_meal_type=None,
    max_cook_time=None,
    protein_pref=None,
    fat_pref=None,
    carb_pref=None,
):
    """Filter recipes using food type, meal type, cook time, and optional macro level labels."""
    out = df.copy()

    if preferred_food_types and "food_tags" in out.columns:
        preferred_food_types = {str(x).strip().lower() for x in preferred_food_types if str(x).strip()}

        def _has_preferred_food_type(tags):
            tag_list = tags if isinstance(tags, list) else _coerce_ingredient_list(tags)
            tag_set = {str(t).strip().lower() for t in tag_list if str(t).strip()}
            return bool(preferred_food_types & tag_set)

        out = out[out["food_tags"].apply(_has_preferred_food_type)]

    if preferred_meal_type and "dd_meal_type" in out.columns:
        out = out[_safe_lower_text(out["dd_meal_type"]) == str(preferred_meal_type).strip().lower()]

    if max_cook_time is not None:
        time_col = (
            "cook_time"
            if "cook_time" in out.columns
            else ("cook_time_mins" if "cook_time_mins" in out.columns else None)
        )
        if time_col:
            out = out[pd.to_numeric(out[time_col], errors="coerce").fillna(np.inf) <= max_cook_time]

    macro_pref_map = {
        "high": "high",
        "low": "low",
        "moderate": "moderate",
        None: None,
        "": None,
        "none": None,
    }

    pref_checks = {
        "protein_level": macro_pref_map.get(str(protein_pref).lower() if protein_pref is not None else None),
        "fat_level": macro_pref_map.get(str(fat_pref).lower() if fat_pref is not None else None),
        "carb_level": macro_pref_map.get(str(carb_pref).lower() if carb_pref is not None else None),
    }

    for col, pref in pref_checks.items():
        if pref and col in out.columns:
            out = out[_safe_lower_text(out[col]) == pref]

    return out

def add_calorie_balance_features(df, user_targets):
    """
    Compute calorie alignment between recipe calories and target meal calories.
    """

    out = df.copy()

    target = float(user_targets.get("meal_calories", 0))

    if "calories" not in out.columns or target <= 0:
        out["calorie_diff"] = 0.0
        out["calorie_balance_label"] = "unknown"
        out["goal_alignment_score"] = 0.0
        return out

    out["calorie_diff"] = out["calories"] - target
    out["calorie_diff_pct"] = out["calorie_diff"] / target

    def classify(diff_pct):
        if abs(diff_pct) <= 0.10:
            return "on target"
        elif diff_pct < 0:
            return "calorie deficit"
        else:
            return "calorie surplus"

    out["calorie_balance_label"] = out["calorie_diff_pct"].apply(classify)

    # convert distance to score
    out["goal_alignment_score"] = (1 / (1 + abs(out["calorie_diff_pct"]))).round(4)

    return out

def sort_recommendations(df, sort_by="final_score", ascending=False):
    """
    Sort recommendation results safely.
    Falls back to final_score if the requested column does not exist.
    """

    if df.empty:
        return df

    if sort_by not in df.columns:
        sort_by = "final_score"

    return df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

def build_recommendations(
    df,
    user_profile,
    pantry_ingredients=None,
    must_include=None,
    must_exclude=None,
    preferred_food_types=None,
    preferred_meal_type=None,
    max_cook_time=None,
    meals_per_day=3,
    n_recommendations=10,
    protein_pref=None,
    fat_pref=None,
    carb_pref=None,
    health_issues=None,
    sort_by="final_score",
    knn_bundle=None,
    tfidf_vectorizer=None,
    ingredient_matrix=None,
    candidate_pool_size=250,
    verbose=True,
):
    """
    Create the final recommendation table using:
    1) KNN candidate retrieval on calories+macros
    2) strict health filtering
    3) progressively relaxed preference filtering
    4) strict include/exclude ingredient rules
    5) TF-IDF ingredient scoring + final ranking

    Relaxation order:
    - Stage 1: all preferences strict
    - Stage 2: remove macro label filters
    - Stage 3: remove meal type filter too
    - Stage 4: remove food type filter too
    - Stage 5: relax cook time too
    """

    user_targets = build_user_targets(user_profile, meals_per_day=meals_per_day)

    if knn_bundle is None:
        knn_bundle = load_macro_knn()
    if tfidf_vectorizer is None or ingredient_matrix is None:
        tfidf_vectorizer, ingredient_matrix = load_ingredient_tfidf()

    candidate_pool_size = int(min(max(candidate_pool_size, n_recommendations * 10), len(df)))

    # Step 1: KNN candidate retrieval
    base = knn_candidates(
        df,
        protein=float(user_targets.get("meal_protein", 0)),
        fat=float(user_targets.get("meal_fat", 0)),
        carbs=float(user_targets.get("meal_carbs", 0)),
        n_neighbors=candidate_pool_size,
        knn_bundle=knn_bundle,
    )

    if verbose:
        print(f"After KNN: {len(base)}")

    # Step 2: strict health filtering
    base = filter_recipes_by_health(base, user_conditions=health_issues, strict=True)

    if verbose:
        print(f"After health filter: {len(base)}")

    if base.empty:
        return base

    # Step 3: progressive preference relaxation
    preference_stages = [
        {
            "stage": "strict_all_preferences",
            "preferred_food_types": preferred_food_types,
            "preferred_meal_type": preferred_meal_type,
            "max_cook_time": max_cook_time,
            "protein_pref": protein_pref,
            "fat_pref": fat_pref,
            "carb_pref": carb_pref,
        },
        {
            "stage": "relax_macro_labels",
            "preferred_food_types": preferred_food_types,
            "preferred_meal_type": preferred_meal_type,
            "max_cook_time": max_cook_time,
            "protein_pref": None,
            "fat_pref": None,
            "carb_pref": None,
        },
        {
            "stage": "relax_meal_type_and_macro_labels",
            "preferred_food_types": preferred_food_types,
            "preferred_meal_type": None,
            "max_cook_time": max_cook_time,
            "protein_pref": None,
            "fat_pref": None,
            "carb_pref": None,
        },
        {
            "stage": "relax_food_type_meal_type_macro_labels",
            "preferred_food_types": None,
            "preferred_meal_type": None,
            "max_cook_time": max_cook_time,
            "protein_pref": None,
            "fat_pref": None,
            "carb_pref": None,
        },
        {
            "stage": "relax_all_preferences_except_health_and_ingredient_rules",
            "preferred_food_types": None,
            "preferred_meal_type": None,
            "max_cook_time": None,
            "protein_pref": None,
            "fat_pref": None,
            "carb_pref": None,
        },
    ]

    out = pd.DataFrame()
    selected_stage = None

    for stage_cfg in preference_stages:
        candidate = apply_preference_filters(
            base.copy(),
            preferred_food_types=stage_cfg["preferred_food_types"],
            preferred_meal_type=stage_cfg["preferred_meal_type"],
            max_cook_time=stage_cfg["max_cook_time"],
            protein_pref=stage_cfg["protein_pref"],
            fat_pref=stage_cfg["fat_pref"],
            carb_pref=stage_cfg["carb_pref"],
        )

        if verbose:
            print(f'After preference filters [{stage_cfg["stage"]}]: {len(candidate)}')

        if not candidate.empty:
            out = candidate
            selected_stage = stage_cfg["stage"]
            break

    if out.empty:
        return out

    # Step 4: strict ingredient include/exclude rules
    out = add_user_ingredient_features(
        out,
        pantry_ingredients=pantry_ingredients,
        must_include=must_include,
        must_exclude=must_exclude,
    )

    if "passes_include_exclude" not in out.columns:
        out["passes_include_exclude"] = True

    if verbose:
        print(f"After ingredient features: {len(out)}")

    out = out[out["passes_include_exclude"] == True].copy()

    if verbose:
        print(f"After include/exclude: {len(out)}")

    if out.empty:
        return out

    # Step 5: ingredient TF-IDF score
    out = add_tfidf_ingredient_match(
        out,
        pantry_ingredients=pantry_ingredients,
        must_include=must_include,
        vectorizer=tfidf_vectorizer,
        matrix=ingredient_matrix,
        reference_df=df,
    )

    # Step 6: macro score
    out = compute_macro_score(out, user_targets=user_targets, knn_bundle=knn_bundle)

    # Step 7: calorie alignment
    out = add_calorie_balance_features(out, user_targets=user_targets)

    # Step 8: soft bonus scores for the original requested preferences
    out = out.copy()
    out["meal_type_bonus"] = 0.0
    out["food_type_bonus"] = 0.0
    out["macro_pref_bonus"] = 0.0

    if preferred_meal_type and "dd_meal_type" in out.columns:
        target = str(preferred_meal_type).strip().lower()
        out["meal_type_bonus"] = (
            _safe_lower_text(out["dd_meal_type"]).str.contains(target, na=False).astype(float) * 0.05
        )

    if preferred_food_types and "food_tags" in out.columns:
        prefs = [str(x).strip().lower() for x in preferred_food_types if str(x).strip()]

        def _food_bonus(tags):
            if isinstance(tags, (list, tuple, set, np.ndarray)):
                txt = " ".join(map(str, tags)).lower()
            else:
                txt = str(tags).lower()
            return 1.0 if any(p in txt for p in prefs) else 0.0

        out["food_type_bonus"] = out["food_tags"].apply(_food_bonus).astype(float) * 0.05

    macro_bonus_parts = []
    if protein_pref and "protein_level" in out.columns:
        macro_bonus_parts.append(
            _safe_lower_text(out["protein_level"]).str.contains(str(protein_pref).strip().lower(), na=False).astype(float)
        )
    if fat_pref and "fat_level" in out.columns:
        macro_bonus_parts.append(
            _safe_lower_text(out["fat_level"]).str.contains(str(fat_pref).strip().lower(), na=False).astype(float)
        )
    if carb_pref and "carb_level" in out.columns:
        macro_bonus_parts.append(
            _safe_lower_text(out["carb_level"]).str.contains(str(carb_pref).strip().lower(), na=False).astype(float)
        )

    if macro_bonus_parts:
        macro_bonus = sum(macro_bonus_parts) / len(macro_bonus_parts)
        out["macro_pref_bonus"] = macro_bonus.astype(float) * 0.08

    # Step 9: final score
    nutrient_density = out.get("nutrient_density_score", pd.Series(0.0, index=out.index)).fillna(0).astype(float)
    density_scaled = (nutrient_density / max(float(nutrient_density.max()), 1.0)).clip(0, 1)

    out["final_score"] = (
        0.55 * out["macro_score"].astype(float)
        + 0.25 * out["ingredient_score"].astype(float)
        + 0.10 * density_scaled
        + out["meal_type_bonus"].astype(float)
        + out["food_type_bonus"].astype(float)
        + out["macro_pref_bonus"].astype(float)
    ).round(4)

    out["filter_stage_used"] = selected_stage

    out = sort_recommendations(out, sort_by=sort_by, ascending=False)

    display_cols = [
        "recipe_id", "name", "final_name", "final_description", "RecipeCategory", "dd_meal_type",
        "image_url", "nutri_score_label", "macro_labels", "macro_labels_display",
        "macro_score", "macro_distance", "goal_alignment_score", "calorie_balance_label",
        "ingredient_score", "ingredient_similarity", "ingredient_score_overlap",
        "meal_type_bonus", "food_type_bonus", "macro_pref_bonus",
        "final_score", "filter_stage_used",
        "nutrient_density_score", "nutrient_density_class",
        "matched_ingredients", "missing_ingredients",
        "ingredients_list", "ingredients_clean", "ingredients_text",
        "instructions_list", "food_tags", "cook_time", "calories", "protein", "fat", "carbs",
        "protein_level", "fat_level", "carb_level",
    ]
    display_cols = [c for c in display_cols if c in out.columns]

    return out[display_cols].head(n_recommendations)


# ============================================================================
# Notebook cell 14
# ============================================================================

def add_calorie_balance_sort_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility helper kept for notebooks that expect a feature-postprocess step.

    The recommendation-time calorie balance fields are now computed with
    ``add_calorie_balance_features`` because they depend on the user's goal and
    per-meal calorie target. This function returns the dataframe unchanged.
    """
    return df.copy()



# ============================================================================
# Notebook cell 16
# ============================================================================

"""Build and use ingredient-text TF-IDF features for content-based similarity search."""

def fit_ingredient_tfidf(df: pd.DataFrame, vectorizer_path: str = TFIDF_PATH, matrix_path: str = INGREDIENT_MATRIX_PATH):
    """Fit a TF-IDF model on normalized ingredient text and save both vectorizer and matrix."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )
    matrix = vectorizer.fit_transform(df["ingredients_text"].fillna(""))
    joblib.dump(vectorizer, vectorizer_path)
    save_npz(matrix_path, matrix)
    print(f"Saved TF-IDF vectorizer -> {vectorizer_path}")
    print(f"Saved ingredient matrix -> {matrix_path}")
    return vectorizer, matrix


def load_ingredient_tfidf(vectorizer_path: str = TFIDF_PATH, matrix_path: str = INGREDIENT_MATRIX_PATH):
    """Load the saved ingredient TF-IDF vectorizer and sparse matrix."""
    vectorizer = joblib.load(vectorizer_path)
    matrix = load_npz(matrix_path)
    return vectorizer, matrix


def similar_recipes_by_ingredients(df: pd.DataFrame, recipe_name: str, top_k: int = 10, vectorizer=None, matrix=None):
    """Return recipes with similar ingredient profiles to a given recipe name."""
    if vectorizer is None or matrix is None:
        vectorizer, matrix = load_ingredient_tfidf()

    name_series = df["name"].fillna("").str.lower()
    idx_matches = df.index[name_series == str(recipe_name).lower()].tolist()
    if not idx_matches:
        raise ValueError(f"Recipe not found: {recipe_name}")

    idx = idx_matches[0]
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    order = np.argsort(-sims)

    out = df.iloc[order].copy()
    out["ingredient_similarity"] = sims[order]
    out = out[out.index != idx]

    cols = [c for c in ["name", "ingredient_similarity", "nutri_score_label", "nutrient_density_score"] if c in out.columns]
    return out[cols].head(top_k)


# ============================================================================
# Notebook cell 18
# ============================================================================

"""Save and load processed recipe data using separate model and display files."""

LIST_LIKE_COLUMNS = [
    "ingredients_list",
    "ingredients_raw_list",
    "ingredient_quantities",
    "ingredients_normalized",
    "ingredients_clean",
    "instructions_list",
    "food_tags",
    "Keywords",
    "macro_labels",
    "matched_ingredients",
    "missing_ingredients",
    "ingredient_rows",
]

DISPLAY_COLUMNS = [
    "recipe_id",
    "name",
    "final_name",
    "final_description",
    "RecipeCategory",
    "Images",
    "image_url",
    "dd_meal_type",

    # full recipe output
    "ingredients_list",
    "ingredients_raw_list",
    "ingredient_quantities",
    "ingredients_clean",
    "ingredient_rows",
    "instructions_list",

    # display tags / metadata
    "food_tags",
    "cook_time",
    "servings",
    "serving_g",

    # nutrition shown to user
    "calories",
    "protein",
    "fat",
    "carbs",
    "fiber",
    "sugar",
    "sodium",

    # recommendation-time calorie goal fields (present after user matching)
    "goal_meal_calories",
    "goal_alignment_score",
    "calorie_balance_pct",
    "calorie_balance_label",

    # scores shown to user
    "nutrient_density_score",
    "nutrient_density_class",
    "nutrient_density_badge",
    "nutri_score_numeric",
    "nutri_score_label",
    "medical_risk_score",
    "medical_risk_level",
    "medical_risk_reason",
    "protein_level",
    "fat_level",
    "carb_level",
    "macro_labels",
]

MODEL_COLUMNS = [
    "recipe_id",
    "name",
    "final_name",
    "RecipeCategory",
    "dd_meal_type",
    "image_url",
    "ingredients_text",
    "cook_time",
    "servings",
    "serving_g",
    "calories",
    "protein",
    "fat",
    "sat_fat",
    "carbs",
    "fiber",
    "sugar",
    "cholest",
    "sodium",
    "cal_per_g",
    "energy_kcal_100g",
    "energy_kj_100g",
    "protein_100g",
    "fat_100g",
    "sat_fat_100g",
    "carbs_100g",
    "fiber_100g",
    "sugar_100g",
    "sodium_100g",
    "protein_per_kcal",
    "fiber_per_kcal",
    "sugar_per_kcal",
    "protein_per_100kcal",
    "fiber_per_100kcal",
    "sugar_per_100kcal",
    "food_tags",
    "Keywords",
    "protein_level",
    "fat_level",
    "carb_level",
    "macro_labels",
    "nutrient_density_score",
    "nutrient_density_class",
    "nutrient_density_badge",
    "nutri_score_numeric",
    "nutri_score_label",
    "medical_risk_score",
    "medical_risk_level",
    "medical_risk_reason",
    "ingredient_rows",
    "ingredients_clean",
]

COMPACT_MODEL_COLUMNS = [
    "recipe_id",
    "name",
    "final_name",
    "ingredients_text",
    "cook_time",
    "servings",
    "serving_g",

    "calories",
    "protein",
    "fat",
    "sat_fat",
    "carbs",
    "fiber",
    "sugar",
    "sodium",

    "cal_per_g",
    "energy_kcal_100g",
    "protein_100g",
    "fat_100g",
    "sat_fat_100g",
    "carbs_100g",
    "fiber_100g",
    "sugar_100g",
    "sodium_100g",

    "protein_level",
    "fat_level",
    "carb_level",

    # goal-aware fields
    "goal_meal_calories",
    "calorie_balance_pct",

    # nutrition scoring
    "nutrient_density_score",
    "nutrient_density_class",
    "nutrient_density_badge",
    "nutri_score_numeric",
    "nutri_score_label",

    # medical scoring
    "risk_diabetes",
    "risk_hypertension",
    "risk_heart_disease",
    "risk_cholesterol",
    "risk_kidney",
    "risk_keto_violation",
    "medical_risk_score",
    "medical_risk_level",
    "medical_risk_reason",
]

def _serialize_list_columns(df: pd.DataFrame, cols):
    """Serialize list columns as JSON strings for stable parquet storage."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)
    return out


def _deserialize_list_columns(df: pd.DataFrame, cols):
    """Deserialize JSON-string list columns back into Python lists."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") and x.endswith("]") else x
            )
    return out


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.select_dtypes(include=["int", "int64", "int32"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")

    for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")

    for col in out.select_dtypes(include=["object"]).columns:
        sample = out[col].dropna().head(20)

        # Skip list/array-like object columns
        if sample.apply(lambda x: isinstance(x, (list, tuple, set, dict, np.ndarray))).any():
            continue

        nunique = out[col].nunique(dropna=True)
        total = len(out[col])

        if total > 0 and nunique / total < 0.5:
            out[col] = out[col].astype("category")

    return out


def save_processed_recipe_data(
    df: pd.DataFrame,
    model_path: str = PROCESSED_MODEL_PATH,
    display_path: str = PROCESSED_DISPLAY_PATH,
    compact_model_path: str = COMPACT_MODEL_PATH,
):
    """Save processed recipe data into compact model and display files."""
    model_cols = list(dict.fromkeys([c for c in MODEL_COLUMNS if c in df.columns]))
    display_cols = list(dict.fromkeys([c for c in DISPLAY_COLUMNS if c in df.columns]))
    compact_cols = list(dict.fromkeys([c for c in COMPACT_MODEL_COLUMNS if c in df.columns]))

    df_model = optimize_dataframe_dtypes(df[model_cols].copy())
    df_compact = optimize_dataframe_dtypes(df[compact_cols].copy())
    df_display = df[display_cols].copy()
    df_display = _serialize_list_columns(df_display, LIST_LIKE_COLUMNS)

    df_model.to_parquet(model_path, index=False, compression="zstd")
    df_compact.to_parquet(compact_model_path, index=False, compression="zstd")
    df_display.to_parquet(display_path, index=False, compression="zstd")

    print(f"Saved model file   -> {model_path}")
    print(f"Saved compact file -> {compact_model_path}")
    print(f"Saved display file -> {display_path}")


def load_processed_recipe_data(
    model_path: str = PROCESSED_MODEL_PATH,
    display_path: str = PROCESSED_DISPLAY_PATH,
    include_display: bool = True,
    model_columns: list[str] | None = None,
    display_columns: list[str] | None = None,
):
    """Load processed recipe data safely.

    Args:
        model_path: Path to lightweight model-ready parquet.
        display_path: Path to display-ready parquet.
        include_display: Whether to also load display columns.
        model_columns: Optional subset of model columns to load.
        display_columns: Optional subset of display columns to load.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df_model = pd.read_parquet(model_path, columns=model_columns)

    if not include_display:
        return df_model

    df_display = pd.read_parquet(display_path, columns=display_columns)
    df_display = _deserialize_list_columns(df_display, LIST_LIKE_COLUMNS)

    join_key = "recipe_id" if "recipe_id" in df_model.columns and "recipe_id" in df_display.columns else "final_name"
    merged = df_model.merge(df_display, on=join_key, how="left", suffixes=("", "_display"))
    return merged
