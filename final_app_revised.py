import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import streamlit as st

def render_info(text: str = ""):
    """Render a compact score explainer with inline tap/hover superscript references."""
    st.markdown(
        """
        <style>
        .score-info-box {
            background:#f8fbff;
            border:1px solid #d9e7ff;
            border-radius:18px;
            padding:14px 16px;
            margin-top:10px;
            margin-bottom:10px;
            box-shadow:0 8px 20px rgba(15,23,42,0.04);
        }
        .score-info-title {
            font-size:1rem;
            font-weight:800;
            color:#111827;
            margin-bottom:8px;
        }
        .score-info-row {
            display:flex;
            flex-wrap:wrap;
            gap:10px 14px;
            color:#334155;
            font-size:0.93rem;
            line-height:1.5;
        }
        .score-item {
            display:inline-flex;
            align-items:center;
            gap:2px;
            white-space:nowrap;
        }
        .score-ref {
            position:relative;
            display:inline-flex;
            align-items:flex-start;
            cursor:pointer;
            outline:none;
        }
        .score-ref sup {
            color:#1f6fff;
            font-weight:800;
            font-size:0.68rem;
            line-height:1;
        }
        .score-ref .score-tooltip {
            position:absolute;
            left:0;
            top:1.25rem;
            min-width:220px;
            max-width:min(78vw, 320px);
            background:#111827;
            color:#fff;
            border-radius:12px;
            padding:10px 12px;
            font-size:0.8rem;
            line-height:1.45;
            box-shadow:0 12px 28px rgba(15,23,42,0.24);
            opacity:0;
            visibility:hidden;
            transform:translateY(4px);
            transition:opacity .18s ease, transform .18s ease, visibility .18s ease;
            z-index:20;
            white-space:normal;
        }
        .score-ref:hover .score-tooltip,
        .score-ref:focus .score-tooltip,
        .score-ref:focus-within .score-tooltip,
        .score-ref:active .score-tooltip {
            opacity:1;
            visibility:visible;
            transform:translateY(0);
        }
        @media (max-width: 640px) {
            .score-ref .score-tooltip {
                left:auto;
                right:0;
                top:1.35rem;
                max-width:min(82vw, 300px);
            }
        }
        </style>
        <div class="score-info-box">
          <div class="score-info-title">How the scores work</div>
          <div class="score-info-row">
            <span class="score-item"><strong>Nutri-Score</strong><span class="score-ref" tabindex="0"><sup>1</sup><span class="score-tooltip">A simple nutrition grade from A to E. In general, A is more favorable and E is less favorable.</span></span></span>
            <span class="score-item"><strong>Nutrient density</strong><span class="score-ref" tabindex="0"><sup>2</sup><span class="score-tooltip">Estimates how much nutritional value the recipe provides for its calories. Higher is usually better.</span></span></span>
            <span class="score-item"><strong>Calorie alignment</strong><span class="score-ref" tabindex="0"><sup>3</sup><span class="score-tooltip">Compares recipe calories with your per-meal target and labels it as on target, deficit, or surplus.</span></span></span>
            <span class="score-item"><strong>Macro fit</strong><span class="score-ref" tabindex="0"><sup>4</sup><span class="score-tooltip">Based on KNN distance in calorie + protein + carbs + fat space. Higher values mean the recipe is closer to your target macros.</span></span></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

###############################################################################
# Paths                                                                      #
###############################################################################

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"
COMPACT_DATA_CANDIDATES = [
    PROCESSED_DIR / "recipes_model_compact.parquet",
    PROCESSED_DIR / "recipes_model.parquet",
]
DISPLAY_DATA_CANDIDATES = [
    PROCESSED_DIR / "recipes_display_ready.parquet",
    PROCESSED_DIR / "recipes_display.parquet",
]

BASE_DISPLAY_COLS = [
    "recipe_id",
    "name",
    "final_name",
    "final_description",
    "ingredients_clean",
    "food_tags",
    "cook_time",
    "servings",
    "serving_g",
    "calories",
    "protein",
    "fat",
    "carbs",
    "fiber",
    "sugar",
    "sodium",
    "nutrient_density_score",
    "nutrient_density_class",
    "nutri_score_numeric",
    "nutri_score_label",
    "protein_level",
    "fat_level",
    "carb_level",
    "RecipeCategory",
    "dd_meal_type",
    "meal_type",
    "Images",
    "image_url",
    "ingredients_text",
    "ingredients_raw_list",
    "ingredient_quantities",
    "instructions_list",
    "macro_labels",
    "calorie_balance_pct",
    "calorie_balance_label",
    "dd_calorie_balance_pct",
    "dd_calorie_balance_label",
    "goal_meal_calories",
    "dd_goal_meal_calories",
]

DETAIL_DISPLAY_COLS = [
    "recipe_id",
    "name",
    "final_name",
    "final_description",
    "ingredients_raw_list",
    "ingredient_quantities",
    "ingredients_clean",
    "instructions_list",
    "food_tags",
    "macro_labels",
    "Images",
    "image_url",
]

LIST_COLUMNS_BASE = ["ingredients_clean", "food_tags", "ingredients_raw_list", "ingredient_quantities", "instructions_list", "macro_labels"]
LIST_COLUMNS_DETAIL = LIST_COLUMNS_BASE

CATEGORY_COLUMNS = [
    "protein_level",
    "fat_level",
    "carb_level",
    "nutrient_density_class",
    "nutri_score_label",
    "dd_meal_type",
    "meal_type",
]

###############################################################################
# Nutrition / recommendation helpers                                          #
###############################################################################


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)



def calculate_bmr(weight_kg: float, height_cm: float, age: float, sex: str) -> float:
    sex = str(sex).lower()
    if sex == "male":
        return (9.99 * weight_kg) + (6.25 * height_cm) - (4.92 * age) + 5
    return (9.99 * weight_kg) + (6.25 * height_cm) - (4.92 * age) - 161



def calculate_tdee(bmr: float, activity_level: str) -> float:
    multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderate": 1.55,
        "very_active": 1.725,
        "extra_active": 1.9,
    }
    return bmr * multipliers.get(activity_level, 1.2)



def adjust_calories(tdee: float, goal: str, sex: str) -> float:
    goal = str(goal).lower()
    sex = str(sex).lower()
    if goal in {"weight_loss", "loss", "lose"}:
        target = tdee * 0.8
        floor = 1500 if sex == "male" else 1200
        return max(target, floor)
    if goal in {"weight_gain", "gain"}:
        return tdee * 1.1
    return tdee



def calculate_macros(calories: float, goal: str, weight_kg: float) -> Dict[str, float]:
    goal = str(goal).lower()
    if goal in {"weight_loss", "loss", "lose"}:
        protein_g = 1.8 * weight_kg
        fat_g = 0.8 * weight_kg
    elif goal in {"weight_gain", "gain"}:
        protein_g = 1.6 * weight_kg
        fat_g = 0.9 * weight_kg
    else:
        protein_g = 1.6 * weight_kg
        fat_g = 0.8 * weight_kg

    carbs_g = max((calories - protein_g * 4 - fat_g * 9) / 4, 0)
    return {"protein_g": protein_g, "fat_g": fat_g, "carbs_g": carbs_g}



def classify_macro_levels(protein_pct: float, carb_pct: float, fat_pct: float) -> Dict[str, str]:
    def classify_protein(p: float) -> str:
        if p < 0.12:
            return "Low"
        if p < 0.20:
            return "Moderate"
        return "High"

    def classify_carb(p: float) -> str:
        if p < 0.45:
            return "Low"
        if p <= 0.65:
            return "Moderate"
        return "High"

    def classify_fat(p: float) -> str:
        if p < 0.20:
            return "Low"
        if p <= 0.35:
            return "Moderate"
        return "High"

    return {
        "protein_level": classify_protein(float(protein_pct or 0.0)),
        "carb_level": classify_carb(float(carb_pct or 0.0)),
        "fat_level": classify_fat(float(fat_pct or 0.0)),
    }



def match_ingredients(recipe_ingredients: List[str], available: List[str]) -> Tuple[int, int, List[str], List[str]]:
    recipe_set = {str(ing).strip().lower() for ing in recipe_ingredients if str(ing).strip()}
    user_set = {str(ing).strip().lower() for ing in available if str(ing).strip()}
    matched = sorted(recipe_set.intersection(user_set))
    missing = sorted(recipe_set.difference(user_set))
    return len(matched), len(missing), matched, missing


MEASUREMENT_WORDS = {
    "teaspoon", "teaspoons", "tsp", "tablespoon", "tablespoons", "tbsp", "cup", "cups",
    "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs", "gram", "grams", "g",
    "kg", "ml", "l", "liter", "liters", "clove", "cloves", "slice", "slices", "can", "cans",
    "package", "packages", "pkg", "pkgs"
}

DESCRIPTOR_WORDS = {
    "optional", "lightly", "toasted", "minced", "dried", "crushed", "fresh", "packed", "raw",
    "regular", "carefully", "scraped", "off", "flaked", "for", "bones", "substituted", "equal",
    "amount", "of", "an", "a", "the", "medium", "large", "small", "chopped", "grated", "drained",
    "baby", "skinless", "boneless"
}

STOP_IF_ONLY_GENERIC = {"ingredient", "ingredients", "item", "items"}


def _strip_quantity_prefix(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("–", "-")
    s = re.sub(r"^\s*[\d\s/.-]+", "", s)
    changed = True
    while changed:
        changed = False
        parts = s.split()
        if parts and parts[0] in MEASUREMENT_WORDS:
            s = " ".join(parts[1:])
            changed = True
            continue
        if parts and re.fullmatch(r"[\d/.-]+", parts[0] or ""):
            s = " ".join(parts[1:])
            changed = True
    return re.sub(r"\s+", " ", s).strip(" ,;:-")



def _normalize_ingredient_option(text: str) -> str:
    s = text.strip().lower()
    if not s:
        return ""
    s = s.split(",")[0]
    s = _strip_quantity_prefix(s)
    s = s.replace("-", " ")
    tokens = []
    for token in re.findall(r"[a-zA-Z]+", s):
        if token in DESCRIPTOR_WORDS:
            continue
        tokens.append(token)
    while tokens and tokens[0] in MEASUREMENT_WORDS:
        tokens.pop(0)
    if not tokens:
        return ""
    phrase = " ".join(tokens).strip()
    if phrase in STOP_IF_ONLY_GENERIC:
        return ""
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase



def build_ingredient_requirements(raw_ingredients: List[str], clean_ingredients: List[str]) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []

    raw_list = [str(x).strip() for x in raw_ingredients if str(x).strip()]
    if raw_list:
        for raw in raw_list:
            first_part = re.split(r"\s+and/or\s+|\s+or\s+", raw, flags=re.IGNORECASE)
            options = []
            for part in first_part:
                normalized = _normalize_ingredient_option(part)
                if normalized:
                    options.append(normalized)
            options = list(dict.fromkeys(options))
            if options:
                label = " or ".join(options)
                requirements.append({"label": label, "options": options})

    if requirements:
        return requirements

    clean_list = []
    for ing in clean_ingredients:
        normalized = _normalize_ingredient_option(str(ing))
        if normalized:
            clean_list.append(normalized)
    clean_list = list(dict.fromkeys(clean_list))
    return [{"label": ing, "options": [ing]} for ing in clean_list]



def ingredient_matches_option(user_item: str, option: str) -> bool:
    u = _normalize_ingredient_option(user_item)
    o = _normalize_ingredient_option(option)
    if not u or not o:
        return False
    return u == o or u in o or o in u



def match_ingredient_requirements(raw_ingredients: List[str], clean_ingredients: List[str], available: List[str]) -> Tuple[int, int, List[str], List[str], int]:
    requirements = build_ingredient_requirements(raw_ingredients, clean_ingredients)
    normalized_available = [str(x).strip() for x in available if str(x).strip()]

    matched_labels: List[str] = []
    missing_labels: List[str] = []
    for req in requirements:
        options = req.get("options", [])
        label = req.get("label", "")
        is_match = any(
            ingredient_matches_option(user_item, option)
            for user_item in normalized_available
            for option in options
        )
        if is_match:
            matched_labels.append(label)
        else:
            missing_labels.append(label)

    total = len(requirements)
    return len(matched_labels), len(missing_labels), matched_labels, missing_labels, total



def normalize_meal_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "breakfast": "breakfast",
        "brunch": "breakfast",
        "lunch": "lunch",
        "dinner": "dinner",
        "supper": "dinner",
    }
    return mapping.get(text, "other")



def build_calorie_balance_label(
    calories: float,
    goal_meal_calories: float,
    tolerance_pct: float = 0.01,
    tolerance_kcal: float = 5.0,
) -> str:
    """Return a user-facing calorie alignment label.

    A recipe is only considered "On calorie target" when it is close in both
    relative and absolute terms. This avoids cases like 543 kcal vs 518 kcal
    being labeled on-target merely because the percentage difference is just
    under a loose 5% threshold.
    """
    if pd.isna(calories) or pd.isna(goal_meal_calories) or goal_meal_calories <= 0:
        return "On calorie target"

    calories = float(calories)
    goal_meal_calories = float(goal_meal_calories)
    delta = calories - goal_meal_calories
    pct = abs(delta) / max(goal_meal_calories, 1e-9) * 100

    if abs(delta) <= tolerance_kcal and pct <= tolerance_pct * 100:
        return "On calorie target"
    if delta < 0:
        return f"{pct:.0f}% calorie deficit"
    return f"{pct:.0f}% calorie surplus"


###############################################################################
# Parsing helpers                                                             #
###############################################################################


def _split_serialized_list_tolerant(text: str) -> List[str]:
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return [s]

    inner = s[1:-1]
    items: List[str] = []
    buf: List[str] = []
    in_item = False
    i = 0
    n = len(inner)

    while i < n:
        ch = inner[i]

        if not in_item:
            if ch in " \t\r\n,":
                i += 1
                continue
            if ch == '"':
                in_item = True
                buf = []
                i += 1
                continue

            start = i
            while i < n and inner[i] != ",":
                i += 1
            token = inner[start:i].strip().strip('"').strip("'")
            if token:
                items.append(token)
            continue

        if ch == '"':
            j = i + 1
            while j < n and inner[j].isspace():
                j += 1
            if j >= n or inner[j] == ",":
                item = "".join(buf).strip()
                if item:
                    items.append(item)
                in_item = False
                buf = []
                i = j
                if i < n and inner[i] == ",":
                    i += 1
                continue

        buf.append(ch)
        i += 1

    if buf:
        item = "".join(buf).strip()
        if item:
            items.append(item)

    return items if items else [text]



def parse_instructions(value: Any) -> List[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        if len(cleaned) == 1:
            only = cleaned[0]
            if only.startswith("[") and only.endswith("]"):
                return parse_instructions(only)
        return cleaned

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("c(") and text.endswith(")"):
        inner = text[2:-1]
        parts = re.findall(r'"([^"]+)"|\'([^\']+)\'', inner)
        vals = [(a or b).strip() for a, b in parts if (a or b)]
        return vals if vals else []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parse_instructions(parsed)
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parse_instructions(parsed)
        except Exception:
            pass

        parsed = _split_serialized_list_tolerant(text)
        return [str(x).strip() for x in parsed if str(x).strip()]

    return [p.strip() for p in re.split(r"\.\s+|\n+", text) if p.strip()]



def _parse_list_like(x: Any) -> List[Any]:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

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
            return val if isinstance(val, list) else [s]
        except Exception:
            pass
        try:
            val = ast.literal_eval(s)
            return val if isinstance(val, list) else [s]
        except Exception:
            return [s]

    if "," in s:
        return [item.strip() for item in s.split(",") if item.strip()]

    return [s]


###############################################################################
# Data loading                                                                #
###############################################################################


def _first_existing_path(candidates: List[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Missing {label} file. Looked in:\n{searched}")



def _read_parquet_columns(path: Path, columns: List[str]) -> pd.DataFrame:
    full_probe = pd.read_parquet(path)
    available = set(full_probe.columns)
    keep = [c for c in columns if c in available]
    return full_probe[keep].copy()


@st.cache_data(show_spinner=False)
def load_base_dataset() -> pd.DataFrame:
    compact_path = _first_existing_path(COMPACT_DATA_CANDIDATES, "compact data")
    display_path = _first_existing_path(DISPLAY_DATA_CANDIDATES, "display data")

    compact_df = pd.read_parquet(compact_path)
    display_df = _read_parquet_columns(display_path, BASE_DISPLAY_COLS)

    for col in LIST_COLUMNS_BASE:
        if col in display_df.columns:
            if col == "instructions_list":
                display_df[col] = display_df[col].apply(parse_instructions)
            else:
                display_df[col] = display_df[col].apply(_parse_list_like)

    if "recipe_id" in compact_df.columns and "recipe_id" in display_df.columns:
        df = compact_df.merge(display_df, on="recipe_id", how="left", suffixes=("", "_display"))
    else:
        df = compact_df.copy()
        for col in display_df.columns:
            if col not in df.columns:
                df[col] = display_df[col]

    if "name" not in df.columns and "final_name" in df.columns:
        df["name"] = df["final_name"]
    elif "name" in df.columns:
        df["name"] = df["name"].fillna(df.get("final_name"))

    alias_map = {
        "Calories": "calories",
        "ProteinContent": "protein",
        "FatContent": "fat",
        "CarbohydrateContent": "carbs",
        "FiberContent": "fiber",
        "SugarContent": "sugar",
        "SodiumContent": "sodium",
        "SaturatedFatContent": "sat_fat",
    }
    for source, target in alias_map.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]

    numeric_cols = [
        "calories", "protein", "fat", "carbs", "fiber", "sugar", "sodium", "sat_fat",
        "cal_per_g", "energy_kcal_100g", "protein_100g", "fat_100g", "carbs_100g", "fiber_100g",
        "sugar_100g", "sodium_100g", "sat_fat_100g", "cook_time", "servings", "serving_g",
        "nutri_score_numeric", "nutrient_density_score", "calorie_balance_pct", "dd_calorie_balance_pct",
        "goal_meal_calories", "dd_goal_meal_calories",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            try:
                df[col] = df[col].astype("category")
            except Exception:
                pass

    for col in ["ingredients_clean", "food_tags", "ingredients_raw_list", "ingredient_quantities", "instructions_list", "macro_labels"]:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]

    if "image_url" not in df.columns:
        if "Images" in df.columns:
            df["image_url"] = df["Images"].astype(str).str.strip()
        else:
            df["image_url"] = ""
    df["image_url"] = df["image_url"].fillna("").astype(str).str.strip()

    if "dd_meal_type" not in df.columns:
        source = None
        if "meal_type" in df.columns:
            source = df["meal_type"]
        elif "RecipeCategory" in df.columns:
            source = df["RecipeCategory"]
        if source is not None:
            df["dd_meal_type"] = source.apply(normalize_meal_type)
        else:
            df["dd_meal_type"] = "other"
    else:
        df["dd_meal_type"] = df["dd_meal_type"].apply(normalize_meal_type)

    df = df[df["calories"].fillna(0) > 0].copy()

    if {"protein", "fat", "carbs", "calories"}.issubset(df.columns):
        if "protein_pct" not in df.columns:
            df["protein_pct"] = (df["protein"] * 4.0) / df["calories"]
        if "carb_pct" not in df.columns:
            df["carb_pct"] = (df["carbs"] * 4.0) / df["calories"]
        if "fat_pct" not in df.columns:
            df["fat_pct"] = (df["fat"] * 9.0) / df["calories"]

    if "nutrient_density_norm" not in df.columns:
        if "nutrient_density_score" in df.columns:
            min_n = df["nutrient_density_score"].min()
            max_n = df["nutrient_density_score"].max()
            if pd.isna(min_n) or pd.isna(max_n) or min_n == max_n:
                df["nutrient_density_norm"] = 0.5
            else:
                df["nutrient_density_norm"] = (df["nutrient_density_score"] - min_n) / (max_n - min_n)
        else:
            df["nutrient_density_norm"] = 0.5

    return df


@st.cache_data(show_spinner=False)
def load_recipe_details(recipe_ids: Tuple[int, ...]) -> pd.DataFrame:
    if not recipe_ids:
        return pd.DataFrame(columns=DETAIL_DISPLAY_COLS)

    display_path = _first_existing_path(DISPLAY_DATA_CANDIDATES, "display data")
    filters = [("recipe_id", "in", list(recipe_ids))]
    try:
        detail_df = pd.read_parquet(display_path, columns=DETAIL_DISPLAY_COLS, filters=filters)
    except Exception:
        detail_df = _read_parquet_columns(display_path, DETAIL_DISPLAY_COLS)
        if "recipe_id" in detail_df.columns:
            detail_df = detail_df[detail_df["recipe_id"].isin(recipe_ids)].copy()

    for col in LIST_COLUMNS_DETAIL:
        if col in detail_df.columns:
            if col == "instructions_list":
                detail_df[col] = detail_df[col].apply(parse_instructions)
            else:
                detail_df[col] = detail_df[col].apply(_parse_list_like)

    return detail_df


###############################################################################
# Recommendation engine                                                       #
###############################################################################


DEFAULT_FOOD_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="600" viewBox="0 0 900 600">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fff8ef"/>
      <stop offset="100%" stop-color="#fde7cf"/>
    </linearGradient>
    <linearGradient id="plate" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f5f0ea"/>
    </linearGradient>
  </defs>
  <rect width="900" height="600" rx="32" fill="url(#bg)"/>
  <circle cx="450" cy="320" r="190" fill="url(#plate)" stroke="#e7d8c8" stroke-width="18"/>
  <circle cx="450" cy="320" r="130" fill="#fffaf3"/>
  <ellipse cx="450" cy="350" rx="135" ry="42" fill="#dceccf"/>
  <circle cx="395" cy="292" r="42" fill="#ff8a65"/>
  <circle cx="505" cy="280" r="38" fill="#ff7043"/>
  <circle cx="452" cy="252" r="34" fill="#ffb74d"/>
  <circle cx="475" cy="352" r="30" fill="#8bc34a"/>
  <circle cx="390" cy="360" r="28" fill="#66bb6a"/>
  <ellipse cx="535" cy="340" rx="48" ry="24" fill="#8d6e63"/>
  <ellipse cx="360" cy="255" rx="54" ry="24" fill="#a1887f"/>
  <text x="450" y="108" text-anchor="middle" font-family="Arial, sans-serif" font-size="34" font-weight="700" fill="#8a5a2b">MacroChefAI</text>
  <text x="450" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#9b6f45">Recipe image unavailable</text>
</svg>
"""
DEFAULT_FOOD_IMAGE_URL = "data:image/svg+xml;utf8," + quote(DEFAULT_FOOD_SVG)


def get_recipe_image_url(row: pd.Series) -> str:
    for col in ["image_url", "Images"]:
        value = str(row.get(col, "") or "").strip()
        if value and value.lower() not in {"nan", "none"}:
            return value
    return DEFAULT_FOOD_IMAGE_URL


@st.cache_resource(show_spinner=False)
def build_macro_knn(df: pd.DataFrame) -> Dict[str, Any]:
    feature_cols = ["calories", "protein", "fat", "carbs"]
    macro_df = df.dropna(subset=feature_cols).copy()
    macro_X = macro_df[feature_cols].astype(float)

    scaler = StandardScaler()
    macro_X_scaled = scaler.fit_transform(macro_X)

    model = NearestNeighbors(metric="euclidean")
    model.fit(macro_X_scaled)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "index_array": macro_df.index.to_numpy(),
    }


def get_knn_candidates(df: pd.DataFrame, target_vec: Dict[str, float], n_neighbors: int) -> pd.DataFrame:
    bundle = build_macro_knn(df)
    feature_cols = bundle["feature_cols"]
    safe_n = max(1, min(int(n_neighbors), len(bundle["index_array"])))

    user_df = pd.DataFrame([{c: float(target_vec.get(c, 0.0)) for c in feature_cols}])
    user_scaled = bundle["scaler"].transform(user_df[feature_cols])
    distances, indices = bundle["model"].kneighbors(user_scaled, n_neighbors=safe_n)

    selected_index = bundle["index_array"][indices[0]]
    candidates = df.loc[selected_index].copy()
    candidates["macro_distance"] = distances[0]
    candidates["macro_score"] = (1.0 / (1.0 + candidates["macro_distance"]).clip(lower=0)).round(4)
    return candidates


def recommend_recipes(
    df: pd.DataFrame,
    user_profile: Dict[str, Any],
    available: List[str],
    include: List[str],
    exclude: List[str],
    preferred_food_types: List[str],
    health_conditions: List[str],
    max_cook_time: float,
    macro_prefs: List[str],
    nutri_filter: List[str],
    nutrient_density_filter: str,
    meal_type_filter: str,
    n_recommendations: int,
    sort_by: str,
) -> pd.DataFrame:
    tdee = calculate_tdee(
        calculate_bmr(user_profile["weight"], user_profile["height"], user_profile["age"], user_profile["sex"]),
        user_profile["activity_level"],
    )
    target_cal = adjust_calories(tdee, user_profile["goal"], user_profile["sex"])
    targets = calculate_macros(target_cal, user_profile["goal"], user_profile["weight"])
    meals_per_day = user_profile.get("meals_per_day", 3)
    prot_t = targets["protein_g"] / meals_per_day
    fat_t = targets["fat_g"] / meals_per_day
    carb_t = targets["carbs_g"] / meals_per_day
    cal_t = target_cal / meals_per_day

    candidate_pool = min(len(df), max(300, int(n_recommendations) * 80))
    data = get_knn_candidates(
        df,
        {"calories": cal_t, "protein": prot_t, "fat": fat_t, "carbs": carb_t},
        n_neighbors=candidate_pool,
    )

    if meal_type_filter != "Any" and "dd_meal_type" in data.columns:
        filtered = data[data["dd_meal_type"] == meal_type_filter]
        if not filtered.empty:
            data = filtered

    if include:
        include_set = {ing.strip().lower() for ing in include}
        filtered = data[data["ingredients_clean"].apply(lambda x: include_set.issubset({str(i).lower() for i in x}))]
        if not filtered.empty:
            data = filtered

    if exclude:
        exclude_set = {ing.strip().lower() for ing in exclude}
        data = data[data["ingredients_clean"].apply(lambda x: not bool({str(i).lower() for i in x}.intersection(exclude_set)))]

    if preferred_food_types:
        pref_set = {t.lower() for t in preferred_food_types}
        filtered = data[data["food_tags"].apply(lambda x: bool(pref_set.intersection({str(t).lower() for t in x})))]
        if not filtered.empty:
            data = filtered

    conditions = {c.lower() for c in health_conditions}
    if conditions:
        if "diabetes" in conditions:
            data = data[data["risk_diabetes"] == 0] if "risk_diabetes" in data.columns else data[data["sugar"].fillna(np.inf) <= 15]
        if "hypertension" in conditions:
            data = data[data["risk_hypertension"] == 0] if "risk_hypertension" in data.columns else data[data["sodium"].fillna(np.inf) <= 600]
        if "heart disease" in conditions:
            data = data[data["risk_heart_disease"] == 0] if "risk_heart_disease" in data.columns else data[data["sat_fat"].fillna(np.inf) <= 5]
        if "high cholesterol" in conditions and "risk_cholesterol" in data.columns:
            data = data[data["risk_cholesterol"] == 0]
        if "kidney disease" in conditions and "risk_kidney" in data.columns:
            data = data[data["risk_kidney"] == 0]
        if "keto" in conditions and "risk_keto_violation" in data.columns:
            data = data[data["risk_keto_violation"] == 0]

    if max_cook_time > 0 and "cook_time" in data.columns:
        filtered = data[data["cook_time"].fillna(np.inf) <= max_cook_time]
        if not filtered.empty:
            data = filtered

    if nutri_filter and "nutri_score_label" in data.columns:
        allowed = {x.strip() for x in nutri_filter if x.strip()}
        filtered = data[data["nutri_score_label"].astype(str).isin(allowed)]
        if not filtered.empty:
            data = filtered

    if nutrient_density_filter != "Any" and "nutrient_density_class" in data.columns:
        allowed = {x.strip() for x in nutrient_density_filter.split(",") if x.strip()}
        filtered = data[data["nutrient_density_class"].astype(str).isin(allowed)]
        if not filtered.empty:
            data = filtered

    available_list = [ing.strip() for ing in available if ing.strip()]
    matched_counts: List[int] = []
    missing_counts: List[int] = []
    matched_items: List[List[str]] = []
    missing_items: List[List[str]] = []
    ingredient_totals: List[int] = []
    for _, row in data.iterrows():
        matched, missing, matched_list, missing_list, total = match_ingredient_requirements(
            row.get("ingredients_raw_list", []),
            row.get("ingredients_clean", []),
            available_list,
        )
        matched_counts.append(matched)
        missing_counts.append(missing)
        matched_items.append(matched_list)
        missing_items.append(missing_list)
        ingredient_totals.append(total)

    data["matched"] = matched_counts
    data["missing"] = missing_counts
    data["ingredient_total"] = ingredient_totals
    data["matched_ingredients"] = matched_items
    data["missing_ingredients"] = missing_items
    data["ingredient_score"] = np.where(data["ingredient_total"] > 0, data["matched"] / data["ingredient_total"], 0.0)

    data["per_meal_target_calories"] = cal_t
    data["remaining_daily_calories"] = target_cal - data["calories"]
    data["calorie_balance_label_live"] = data["calories"].apply(lambda x: build_calorie_balance_label(x, cal_t))
    data["calorie_balance_pct_live"] = ((data["calories"] - cal_t).abs() / max(cal_t, 1e-9) * 100).round(0)

    # Soft preference bonus for macro labels rather than replacing KNN macro score.
    macro_pref_bonus = pd.Series(0.0, index=data.index)
    for pref in [str(p).strip().lower() for p in macro_prefs if str(p).strip()]:
        if pref == "high protein":
            macro_pref_bonus += np.where(
                data["protein_level"].astype(str).str.lower() == "high" if "protein_level" in data.columns else data["protein_pct"] >= 0.20,
                0.08,
                0.0,
            )
        elif pref == "moderate protein":
            macro_pref_bonus += np.where(data["protein_level"].astype(str).str.lower() == "moderate", 0.05, 0.0) if "protein_level" in data.columns else 0.0
        elif pref == "low protein":
            macro_pref_bonus += np.where(data["protein_level"].astype(str).str.lower() == "low", 0.05, 0.0) if "protein_level" in data.columns else 0.0
        elif pref == "high carb":
            macro_pref_bonus += np.where(
                data["carb_level"].astype(str).str.lower() == "high" if "carb_level" in data.columns else data["carb_pct"] > 0.65,
                0.08,
                0.0,
            )
        elif pref == "moderate carb":
            macro_pref_bonus += np.where(data["carb_level"].astype(str).str.lower() == "moderate", 0.05, 0.0) if "carb_level" in data.columns else 0.0
        elif pref == "low carb":
            macro_pref_bonus += np.where(
                data["carb_level"].astype(str).str.lower() == "low" if "carb_level" in data.columns else data["carb_pct"] < 0.45,
                0.08,
                0.0,
            )
        elif pref == "high fat":
            macro_pref_bonus += np.where(
                data["fat_level"].astype(str).str.lower() == "high" if "fat_level" in data.columns else data["fat_pct"] > 0.35,
                0.08,
                0.0,
            )
        elif pref == "moderate fat":
            macro_pref_bonus += np.where(data["fat_level"].astype(str).str.lower() == "moderate", 0.05, 0.0) if "fat_level" in data.columns else 0.0
        elif pref == "low fat":
            macro_pref_bonus += np.where(data["fat_level"].astype(str).str.lower() == "low", 0.05, 0.0) if "fat_level" in data.columns else 0.0
        elif pref == "high fiber":
            macro_pref_bonus += np.where(data["fiber"].fillna(0) >= 5, 0.05, 0.0)

    data["macro_pref_bonus"] = np.clip(macro_pref_bonus, 0, 0.2)
    data["final_score"] = (
        0.65 * data["macro_score"].astype(float)
        + 0.25 * data["ingredient_score"].astype(float)
        + 0.10 * data["nutrient_density_norm"].astype(float)
        + data["macro_pref_bonus"].astype(float)
    ).round(4)

    if sort_by == "Macro score":
        data = data.sort_values(
            ["macro_score", "final_score", "ingredient_score", "ingredient_total"],
            ascending=[False, False, False, True],
        )
    elif sort_by == "Ingredient score":
        data = data.sort_values(
            ["ingredient_score", "ingredient_total", "final_score", "macro_score"],
            ascending=[False, True, False, False],
        )
    elif sort_by == "Calories":
        data = data.sort_values(["calories", "ingredient_score", "ingredient_total"], ascending=[True, False, True])
    else:
        data = data.sort_values(
            ["nutrient_density_norm", "final_score", "ingredient_score", "ingredient_total"],
            ascending=[False, False, False, True],
        )

    return data.head(n_recommendations)


###############################################################################
# Rendering                                                                   #
###############################################################################


def _format_num(value: Any, suffix: str = "", digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def safe_float(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


APP_CSS = """
<style>
:root {
    --bg: #f6f8fb;
    --surface: #ffffff;
    --surface-2: #f1f5fb;
    --text: #111827;
    --muted: #6b7280;
    --line: #e5e7eb;
    --blue: #1f6fff;
    --teal: #22b8b2;
    --purple: #7e22ce;
    --gold: #d18b08;
    --green: #1f9d55;
}

.stApp {
    background:
      radial-gradient(circle at top right, rgba(31,111,255,0.10), transparent 28%),
      radial-gradient(circle at top left, rgba(34,184,178,0.12), transparent 30%),
      var(--bg);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
    border-right: 1px solid rgba(229,231,235,0.9);
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.02em;
}

.hero-shell {
    background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(245,249,255,0.98));
    border: 1px solid rgba(226,232,240,0.9);
    box-shadow: 0 16px 48px rgba(15, 23, 42, 0.08);
    border-radius: 28px;
    padding: 26px 28px;
    margin-bottom: 1.15rem;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--blue);
    background: rgba(31,111,255,0.08);
    border: 1px solid rgba(31,111,255,0.12);
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 2.45rem;
    line-height: 1.02;
    font-weight: 800;
    margin: 0 0 8px 0;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.55;
    max-width: 760px;
    margin-bottom: 18px;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-top: 8px;
}

.metric-card {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(226,232,240,0.9);
    border-radius: 22px;
    padding: 16px 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

.metric-label {
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.4rem;
    line-height: 1;
    font-weight: 800;
    color: var(--text);
}

.metric-subvalue {
    margin-top: 6px;
    color: var(--muted);
    font-size: 0.82rem;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
    margin-bottom: 6px;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: white;
    border: 1px solid rgba(148, 163, 184, 0.5);
    color: #475569;
    border-radius: 999px;
    padding: 9px 14px;
    font-size: 0.92rem;
    font-weight: 600;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 800;
    margin: 0.2rem 0 0.7rem;
}

.recipe-shell {
    background: var(--surface);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 28px;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    margin-bottom: 1.2rem;
    overflow: hidden;
}

.recipe-header {
    padding: 1rem 1.25rem 0.35rem;
}

.recipe-title {
    font-size: 2rem;
    line-height: 1.08;
    font-weight: 800;
    margin: 0.4rem 0 0.2rem;
}

.recipe-sub {
    color: var(--muted);
    font-size: 0.98rem;
    margin-bottom: 0.35rem;
}

.recipe-grid {
    display: grid;
    grid-template-columns: 1.35fr 1fr;
    gap: 1rem;
    padding: 0 1.25rem 1.25rem;
}

.image-frame {
    border-radius: 26px;
    overflow: hidden;
    background: linear-gradient(135deg, #fff9f0, #f1f6ff);
    border: 1px solid rgba(226,232,240,0.92);
    min-height: 380px;
}

.insight-card {
    background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 24px;
    padding: 18px;
    margin-bottom: 12px;
}

.mini-stat-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 10px;
}

.mini-stat {
    background: var(--surface-2);
    border-radius: 18px;
    padding: 12px;
}

.mini-stat .label {
    color: var(--muted);
    font-size: 0.8rem;
}

.mini-stat .value {
    color: var(--text);
    font-size: 1.16rem;
    font-weight: 800;
    margin-top: 3px;
}

.inline-note {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.45;
}

.stButton > button {
    border-radius: 999px !important;
    min-height: 3.2rem;
    border: none !important;
    background: linear-gradient(135deg, #1f6fff, #0f62f2) !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    box-shadow: 0 14px 28px rgba(31,111,255,0.28);
}

.stTextInput input, .stTextArea textarea, .stMultiSelect div[data-baseweb="select"], .stSelectbox div[data-baseweb="select"], .stNumberInput input {
    border-radius: 16px !important;
}

.result-kicker {
    color: var(--blue);
    font-weight: 800;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    margin-bottom: 0.15rem;
}

.result-caption {
    color: var(--muted);
    font-size: 0.9rem;
}

.compact-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
}

.compact-card {
    background: white;
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 24px;
    padding: 16px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    height: 100%;
}

@media (max-width: 980px) {
  .metric-strip, .compact-grid, .recipe-grid {
      grid-template-columns: 1fr;
  }
  .hero-title { font-size: 2rem; }
  .recipe-title { font-size: 1.55rem; }
}
</style>
"""


def inject_app_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def html_escape(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def pill_chip(text: str) -> str:
    return f'<span class="chip">{html_escape(text)}</span>'


def summarize_filters(preferred_types: List[str], health_conditions: List[str], macro_prefs: List[str], meal_type: str) -> str:
    chips = []
    if meal_type and meal_type != "Any":
        chips.append(pill_chip(meal_type.title()))
    for item in preferred_types[:3]:
        chips.append(pill_chip(item.title()))
    for item in health_conditions[:2]:
        chips.append(pill_chip(f"{item.title()} friendly"))
    for item in macro_prefs[:2]:
        chips.append(pill_chip(item))
    return "".join(chips) or pill_chip("Personalized picks")


def build_badges(row: pd.Series, per_meal_calories: float) -> List[str]:
    badges: List[str] = []
    calories = pd.to_numeric(row.get("calories"), errors="coerce")
    if pd.notna(calories):
        if calories <= 500:
            badges.append("Under 500 Calories")
        elif abs(float(calories) - float(per_meal_calories)) <= 50:
            badges.append("Close to target")
    meal_type = str(row.get("dd_meal_type", "")).strip().lower()
    if meal_type and meal_type != "other":
        badges.append(meal_type.title())
    elif str(row.get("RecipeCategory", "")).strip():
        badges.append(str(row.get("RecipeCategory")).strip())
    if str(row.get("protein_level", "")).strip().lower() == "high":
        badges.append("High Protein")
    elif pd.to_numeric(row.get("fiber"), errors="coerce") >= 5:
        badges.append("High Fiber")
    if str(row.get("nutri_score_label", "")).strip() in {"A", "B"}:
        badges.append(f"Nutri-Score {str(row.get('nutri_score_label')).strip()}")
    return badges[:4]


def macro_percentages(row: pd.Series) -> Tuple[float, float, float]:
    protein = max(float(pd.to_numeric(row.get("protein"), errors="coerce") or 0), 0.0)
    carbs = max(float(pd.to_numeric(row.get("carbs"), errors="coerce") or 0), 0.0)
    fat = max(float(pd.to_numeric(row.get("fat"), errors="coerce") or 0), 0.0)
    protein_cal = protein * 4.0
    carb_cal = carbs * 4.0
    fat_cal = fat * 9.0
    total = max(protein_cal + carb_cal + fat_cal, 1e-9)
    return carb_cal / total, fat_cal / total, protein_cal / total


def nutrition_donut_html(row: pd.Series) -> str:
    """Return a simple macro summary block that renders reliably in Streamlit."""
    carb_pct, fat_pct, protein_pct = macro_percentages(row)
    calories = int(round(safe_float(row.get("calories"), 0.0)))
    serving_g = pd.to_numeric(row.get("serving_g"), errors="coerce")
    serving_text = f"Serving size: {int(round(serving_g))} g" if pd.notna(serving_g) and serving_g > 0 else "Serving size unavailable"

    items = [
        ("Carbs", safe_float(row.get("carbs"), 0.0), carb_pct, "#2bb8b2"),
        ("Fat", safe_float(row.get("fat"), 0.0), fat_pct, "#7e22ce"),
        ("Protein", safe_float(row.get("protein"), 0.0), protein_pct, "#d18b08"),
    ]
    blocks = []
    for label, grams, frac, color in items:
        width = max(min(frac * 100.0, 100.0), 0.0)
        blocks.append(
            f"""
            <div class="mini-stat" style="background:rgba(255,255,255,0.96);">
              <div class="label">{label}</div>
              <div class="value" style="color:{color};">{grams:.1f} g</div>
              <div class="label" style="margin-top:4px;">{width:.0f}% of macro calories</div>
              <div style="margin-top:8px;height:8px;background:#e8eef5;border-radius:999px;overflow:hidden;">
                <div style="height:8px;width:{width:.0f}%;background:{color};border-radius:999px;"></div>
              </div>
            </div>
            """
        )

    return f"""
    <div class="mini-stat-grid" style="margin-top:8px;">
      <div class="mini-stat" style="background:rgba(255,255,255,0.96);">
        <div class="label">Calories</div>
        <div class="value">{calories:.0f} kcal</div>
        <div class="label" style="margin-top:4px;">{serving_text}</div>
      </div>
      {''.join(blocks)}
    </div>
    """


def traffic_light_color(value: float, low_cutoff: float, high_cutoff: float) -> str:
    if pd.isna(value):
        return "#94a3b8"
    if value >= high_cutoff:
        return "#dc2626"
    if value <= low_cutoff:
        return "#16a34a"
    return "#f59e0b"


def nutrient_warning_html(label: str, value: float, unit: str, low_cutoff: float, high_cutoff: float) -> str:
    color = traffic_light_color(value, low_cutoff, high_cutoff)
    state = "High" if pd.notna(value) and value >= high_cutoff else ("Low" if pd.notna(value) and value <= low_cutoff else "Moderate")
    val_txt = "N/A" if pd.isna(value) else (f"{value:.1f} {unit}" if unit == "g" else f"{int(round(value))} {unit}")
    return (
        f"<div class='mini-stat' style='background:rgba(255,255,255,0.92);'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value' style='color:{color};'>{val_txt}</div>"
        f"<div class='label' style='margin-top:3px;color:{color};font-weight:700;'>{state}</div>"
        f"</div>"
    )


def calorie_progress_html(recipe_calories: float, target_calories: float, daily_target: float) -> str:
    """Return a stable calorie summary card without SVG or nested unsupported markup."""
    recipe_calories = max(float(recipe_calories or 0), 0.0)
    target_calories = max(float(target_calories or 0), 1.0)
    daily_target = max(float(daily_target or target_calories), target_calories)

    remaining_after = max(daily_target - recipe_calories, 0.0)
    meal_pct_raw = (recipe_calories / target_calories) * 100.0
    daily_pct = min((recipe_calories / daily_target) * 100.0, 100.0)
    calorie_delta = recipe_calories - target_calories

    if abs(calorie_delta) <= 5:
        status_label = "On meal target"
        status_fg = "#15803d"
        diff_text = "On target"
    elif calorie_delta < 0:
        status_label = f"{abs(calorie_delta):.0f} kcal below target"
        status_fg = "#15803d"
        diff_text = f"{abs(calorie_delta):.0f} kcal under"
    else:
        status_label = f"{calorie_delta:.0f} kcal above target"
        status_fg = "#b45309" if meal_pct_raw <= 115 else "#b91c1c"
        diff_text = f"{calorie_delta:.0f} kcal over"

    meal_width = max(min(meal_pct_raw, 100.0), 0.0)
    daily_width = max(min(daily_pct, 100.0), 0.0)

    return f"""
    <div style="margin-top:12px;padding-top:12px;border-top:1px solid #eef2f7;">
      <div class="mini-stat-grid">
        <div class="mini-stat"><div class="label">Meal target</div><div class="value">{target_calories:.0f} kcal</div></div>
        <div class="mini-stat"><div class="label">Remaining today</div><div class="value">{remaining_after:.0f} kcal</div></div>
        <div class="mini-stat"><div class="label">Status</div><div class="value" style="color:{status_fg};">{status_label}</div></div>
      </div>
      <div style="margin-top:12px;font-size:0.9rem;color:#475569;line-height:1.55;">
        This meal uses {meal_pct_raw:.0f}% of your meal target and {daily_pct:.0f}% of your full-day target. Difference: <strong style="color:{status_fg};">{diff_text}</strong>.
      </div>
      <div style="margin-top:10px;">
        <div style="font-size:0.8rem;color:#6b7280;margin-bottom:4px;">Meal target progress</div>
        <div style="height:8px;background:#e8eef5;border-radius:999px;overflow:hidden;">
          <div style="height:8px;width:{meal_width:.0f}%;background:#1f6fff;border-radius:999px;"></div>
        </div>
      </div>
      <div style="margin-top:10px;">
        <div style="font-size:0.8rem;color:#6b7280;margin-bottom:4px;">Daily target usage</div>
        <div style="height:8px;background:#e8eef5;border-radius:999px;overflow:hidden;">
          <div style="height:8px;width:{daily_width:.0f}%;background:#22b8b2;border-radius:999px;"></div>
        </div>
      </div>
    </div>
    """





def render_hero_panel(target_calories: float, per_meal_calories: float, per_meal_protein: float, per_meal_carbs: float, per_meal_fat: float, chip_html: str) -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="hero-kicker">✨ KNN macro fit + ingredient matching</div>
          <div class="hero-title">MacroChefAI</div>
          <div class="hero-subtitle">Discover recipe suggestions with a polished nutrition-first layout: calorie-aware, macro-aligned, ingredient-aware, and easier to browse on mobile.</div>
          <div class="chip-row">{chip_html}</div>
          <div class="metric-strip">
            <div class="metric-card"><div class="metric-label">Daily target</div><div class="metric-value">{target_calories:.0f} kcal</div><div class="metric-subvalue">Personalized from your profile</div></div>
            <div class="metric-card"><div class="metric-label">Per-meal calories</div><div class="metric-value">{per_meal_calories:.0f}</div><div class="metric-subvalue">Close-match recipes preferred</div></div>
            <div class="metric-card"><div class="metric-label">Protein target</div><div class="metric-value">{per_meal_protein:.0f} g</div><div class="metric-subvalue">Carbs {per_meal_carbs:.0f} g · Fat {per_meal_fat:.0f} g</div></div>
            <div class="metric-card"><div class="metric-label">Recommendation engine</div><div class="metric-value">KNN + TF-IDF</div><div class="metric-subvalue">Macro distance + pantry match</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nutrient_status_label(value: float, low_cutoff: float, high_cutoff: float) -> tuple[str, str]:
    if pd.isna(value):
        return "⚪", "Unknown"
    if value >= high_cutoff:
        return "🔴", "High"
    if value <= low_cutoff:
        return "🟢", "Good"
    return "🟠", "Moderate"


def render_progress_bar(value_pct: float, color: str = "#1f6fff") -> None:
    value_pct = max(min(float(value_pct), 100.0), 0.0)
    st.markdown(
        f"""
        <div style="height:8px;background:#e8eef5;border-radius:999px;overflow:hidden;">
          <div style="height:8px;width:{value_pct:.0f}%;background:{color};border-radius:999px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nutrition_section_streamlit(row: pd.Series, per_meal_calories: float, daily_target_calories: float) -> None:
    carb_pct, fat_pct, protein_pct = macro_percentages(row)
    calories = safe_float(row.get("calories"), 0.0)
    serving_g = pd.to_numeric(row.get("serving_g"), errors="coerce")
    serving_text = f"{int(round(serving_g))} g" if pd.notna(serving_g) and serving_g > 0 else "N/A"

    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Nutrition per serving</div>', unsafe_allow_html=True)

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        st.caption("Calories")
        st.markdown(f"**{calories:.0f} kcal**")
        st.caption(f"Serving: {serving_text}")
    with c1:
        st.caption("Carbs")
        st.markdown(f"**{safe_float(row.get('carbs'), 0.0):.1f} g**")
        st.caption(f"{carb_pct*100:.0f}% of macro calories")
        render_progress_bar(carb_pct * 100.0, "#2bb8b2")
    with c2:
        st.caption("Fat")
        st.markdown(f"**{safe_float(row.get('fat'), 0.0):.1f} g**")
        st.caption(f"{fat_pct*100:.0f}% of macro calories")
        render_progress_bar(fat_pct * 100.0, "#7e22ce")
    with c3:
        st.caption("Protein")
        st.markdown(f"**{safe_float(row.get('protein'), 0.0):.1f} g**")
        st.caption(f"{protein_pct*100:.0f}% of macro calories")
        render_progress_bar(protein_pct * 100.0, "#d18b08")

    target_calories = max(float(per_meal_calories), 1.0)
    remaining_after = max(daily_target_calories - calories, 0.0)
    calorie_delta = calories - target_calories
    meal_pct_raw = (calories / target_calories) * 100.0
    daily_pct = (calories / daily_target_calories) * 100.0 if daily_target_calories > 0 else 0.0

    if abs(calorie_delta) < 1:
        status_label = "On meal target"
        diff_text = "On target"
    elif calorie_delta < 0:
        status_label = f"{abs(calorie_delta):.0f} kcal below target"
        diff_text = f"{abs(calorie_delta):.0f} kcal under"
    else:
        status_label = f"{calorie_delta:.0f} kcal above target"
        diff_text = f"{calorie_delta:.0f} kcal over"

    st.markdown("<hr style='margin:12px 0;border:none;border-top:1px solid #eef2f7;'>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.caption("Meal target")
        st.markdown(f"**{target_calories:.0f} kcal**")
    with s2:
        st.caption("Remaining today")
        st.markdown(f"**{remaining_after:.0f} kcal**")
    with s3:
        st.caption("Status")
        st.markdown(f"**{status_label}**")

    st.caption(f"This meal uses {meal_pct_raw:.0f}% of your meal target and {daily_pct:.0f}% of your full-day target. Difference: {diff_text}.")
    st.caption("Meal target progress")
    render_progress_bar(meal_pct_raw, "#1f6fff")
    st.caption("Daily target usage")
    render_progress_bar(daily_pct, "#22b8b2")
    st.markdown("</div>", unsafe_allow_html=True)


def render_why_recipe_streamlit(row: pd.Series, per_meal_calories: float, daily_target_calories: float,
                                calories_value: float, remaining_daily_value: float,
                                macro_distance_value: float, fiber_value: float,
                                matched: int, ingredient_total: int) -> None:
    sugar_val = pd.to_numeric(row.get("sugar"), errors="coerce")
    sodium_val = pd.to_numeric(row.get("sodium"), errors="coerce")
    serving_g = pd.to_numeric(row.get("serving_g"), errors="coerce")
    serving_text = f"{int(round(serving_g))} g" if pd.notna(serving_g) and serving_g > 0 else "N/A"
    remaining_daily_cal = safe_float(row.get("remaining_daily_calories"), max(daily_target_calories - calories_value, 0.0))

    sugar_icon, sugar_status = nutrient_status_label(float(sugar_val) if pd.notna(sugar_val) else np.nan, 5, 15)
    sodium_icon, sodium_status = nutrient_status_label(float(sodium_val) if pd.notna(sodium_val) else np.nan, 140, 600)

    meal_pct = (calories_value / max(float(per_meal_calories), 1.0)) * 100.0
    remaining_pct = (remaining_daily_cal / max(float(daily_target_calories), 1.0)) * 100.0

    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why this recipe</div>', unsafe_allow_html=True)
    st.write(
        f"This recipe has {calories_value:.0f} kcal, leaves {remaining_daily_value:.0f} kcal remaining today after this meal, "
        f"and compares against your {per_meal_calories:.0f} kcal meal target."
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        st.caption("Calorie fit")
        st.markdown(f"**{str(row.get('calorie_balance_label_live', 'On target'))}**")
    with r2:
        st.caption("Ingredient match")
        st.markdown(f"**{matched}/{ingredient_total}**")
    with r3:
        st.caption("Nutri-Score")
        st.markdown(f"**{str(row.get('nutri_score_label', 'N/A'))}**")

    r4, r5, r6 = st.columns(3)
    with r4:
        st.caption("Macro distance")
        st.markdown(f"**{macro_distance_value:.2f}**")
    with r5:
        st.caption("Fiber")
        st.markdown(f"**{fiber_value:.1f} g**")
    with r6:
        st.caption("Serving size")
        st.markdown(f"**{serving_text}**")

    r7, r8, r9 = st.columns(3)
    with r7:
        st.caption("Sugar")
        if pd.notna(sugar_val):
            st.markdown(f"**{sugar_icon} {float(sugar_val):.1f} g · {sugar_status}**")
        else:
            st.markdown("**⚪ N/A**")
    with r8:
        st.caption("Sodium")
        if pd.notna(sodium_val):
            st.markdown(f"**{sodium_icon} {float(sodium_val):.0f} mg · {sodium_status}**")
        else:
            st.markdown("**⚪ N/A**")
    with r9:
        st.caption("Remaining daily cal")
        st.markdown(f"**{remaining_daily_cal:.0f} kcal**")

    b1, b2 = st.columns(2)
    with b1:
        st.caption(f"Calories vs meal target ({meal_pct:.0f}%)")
        render_progress_bar(meal_pct, "#1f6fff")
    with b2:
        st.caption(f"Remaining today ({remaining_pct:.0f}% of daily target)")
        render_progress_bar(remaining_pct, "#22b8b2")

    st.markdown("</div>", unsafe_allow_html=True)


def render_recipe_card(row: pd.Series, per_meal_calories: float, daily_target_calories: float, is_featured: bool = False) -> None:
    title = row.get("final_name") or row.get("name") or "Recipe"
    desc = str(row.get("final_description") or "").strip()
    image_url = get_recipe_image_url(row)
    calories_value = safe_float(row.get("calories"), 0.0)
    remaining_daily_value = safe_float(row.get("remaining_daily_calories"), max(daily_target_calories - calories_value, 0.0))
    macro_distance_value = safe_float(row.get("macro_distance"), 0.0)
    fiber_value = safe_float(row.get("fiber"), 0.0)
    badges = ''.join(pill_chip(x) for x in build_badges(row, per_meal_calories))
    ingredient_total = int(row.get("ingredient_total", row.get("matched", 0) + row.get("missing", 0)))
    matched = int(row.get("matched", 0))
    cook_time = pd.to_numeric(row.get("cook_time"), errors="coerce")
    servings = pd.to_numeric(row.get("servings"), errors="coerce")

    st.markdown('<div class="recipe-shell">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="recipe-header">
          <div class="result-kicker">{'FEATURED MATCH' if is_featured else 'RECIPE MATCH'}</div>
          <div class="recipe-title">{html_escape(title)}</div>
          <div class="recipe-sub">Serves {int(servings) if pd.notna(servings) and servings > 0 else '—'} · {int(cook_time) if pd.notna(cook_time) else '—'} min · {f"{int(round(float(pd.to_numeric(row.get('serving_g'), errors='coerce') or 0)))} g per serving" if pd.notna(pd.to_numeric(row.get('serving_g'), errors='coerce')) and float(pd.to_numeric(row.get('serving_g'), errors='coerce') or 0) > 0 else 'Serving weight unavailable'} · Macro score {float(pd.to_numeric(row.get('macro_score'), errors='coerce') or 0):.2f}</div>
          <div class="chip-row">{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1.0], gap="large")
    with left:
        try:
            st.image(image_url, use_container_width=True)
        except Exception:
            st.image(DEFAULT_FOOD_IMAGE_URL, use_container_width=True)
        if desc:
            st.markdown(f"<p class='inline-note' style='padding:12px 2px 0 2px;'>{html_escape(desc)}</p>", unsafe_allow_html=True)
    with right:
        render_nutrition_section_streamlit(row, per_meal_calories, daily_target_calories)
        render_why_recipe_streamlit(
            row,
            per_meal_calories,
            daily_target_calories,
            calories_value,
            remaining_daily_value,
            macro_distance_value,
            fiber_value,
            matched,
            ingredient_total,
        )

    tabs = st.tabs(["Ingredients", "Instructions", "Nutrition details"])
    with tabs[0]:
        matched_items = row.get("matched_ingredients", []) or []
        missing_items = row.get("missing_ingredients", []) or []
        if matched_items:
            st.success("Matched pantry items: " + ", ".join(map(str, matched_items[:20])))
        if missing_items:
            st.info("Missing items: " + ", ".join(map(str, missing_items[:20])))
        ingredient_display = row.get("ingredients_raw_list", [])
        if isinstance(ingredient_display, list) and ingredient_display:
            for ing in ingredient_display:
                st.write(f"• {ing}")
        else:
            recipe_ings = row.get("ingredients_clean", []) or []
            for ing in recipe_ings:
                st.write(f"• {ing}")
    with tabs[1]:
        instructions = row.get("instructions_list", []) or []
        if instructions:
            for i, step in enumerate(instructions, start=1):
                st.write(f"{i}. {step}")
        else:
            st.caption("Instructions were not available for this recipe.")
    with tabs[2]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories", _format_num(row.get("calories"), " kcal", 0))
        c2.metric("Protein", _format_num(row.get("protein"), " g", 1))
        c3.metric("Carbs", _format_num(row.get("carbs"), " g", 1))
        c4.metric("Fat", _format_num(row.get("fat"), " g", 1))
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Nutrient density", _format_num(row.get("nutrient_density_score"), "", 1))
        c6.metric("Ingredient score", _format_num(row.get("ingredient_score"), "", 2))
        c7.metric("Remaining daily cal", _format_num(row.get("remaining_daily_calories"), " kcal", 0))
        c8.metric("Serving size", _format_num(row.get("serving_g"), " g", 0))
        sugar_val = pd.to_numeric(row.get("sugar"), errors="coerce")
        sodium_val = pd.to_numeric(row.get("sodium"), errors="coerce")
        st.markdown(
            f"<div class='mini-stat-grid' style='margin-top:12px;'>{nutrient_warning_html('Sugar', float(sugar_val) if pd.notna(sugar_val) else np.nan, 'g', 5, 15)}{nutrient_warning_html('Sodium', float(sodium_val) if pd.notna(sodium_val) else np.nan, 'mg', 140, 600)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


###############################################################################
# Main app                                                                    #
###############################################################################


def main() -> None:
    st.set_page_config(page_title="MacroChefAI", page_icon="🍽️", layout="wide")
    inject_app_css()

    st.sidebar.markdown("## Personalize your picks")
    st.sidebar.caption("Tune your targets, pantry, and preferences. The app ranks recipes using KNN macro distance plus ingredient fit.")

    age = st.sidebar.number_input("Age (years)", min_value=10, max_value=100, value=30)
    weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.sidebar.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    activity = st.sidebar.selectbox(
        "Activity Level",
        options=[
            ("sedentary", "Little or no exercise"),
            ("lightly_active", "Lightly active (1–3 days/week)"),
            ("moderate", "Moderately active (3–5 days/week)"),
            ("very_active", "Very active (6–7 days/week)"),
            ("extra_active", "Extra active (very intense exercise)"),
        ],
        format_func=lambda x: x[1],
    )
    activity_level = activity[0]
    goal = st.sidebar.selectbox(
        "Fitness Goal",
        options=[("weight_loss", "Lose weight"), ("maintenance", "Maintain"), ("weight_gain", "Gain weight")],
        format_func=lambda x: x[1],
    )[0]
    meals_per_day = st.sidebar.number_input("Meals per day", min_value=1, max_value=6, value=3)
    meal_type_filter = st.sidebar.selectbox("Meal type", ["Any", "breakfast", "lunch", "dinner"])
    preferred_types = st.sidebar.multiselect(
        "Preferred food types",
        ["Vegetarian", "Vegan", "Chicken", "Seafood", "Meat", "Other"],
    )
    health_conditions = st.sidebar.multiselect(
        "Health conditions",
        ["Diabetes", "Hypertension", "Heart disease", "High cholesterol", "Kidney disease", "Keto"],
    )
    max_cook_time = st.sidebar.number_input("Maximum cook time (minutes)", min_value=0, max_value=240, value=45)
    macro_prefs = st.sidebar.multiselect(
        "Macro preferences",
        [
            "High Protein", "Moderate Protein", "Low Protein",
            "High Fiber",
            "High Carb", "Moderate Carb", "Low Carb",
            "High Fat", "Moderate Fat", "Low Fat",
        ],
        help="Combine preferences such as High Protein and Low Carb.",
    )
    nutri_filter = st.sidebar.multiselect("Nutri-Score preference", ["A", "B", "C", "D", "E"], default=["A", "B", "C"])
    nutrient_density_filter = st.sidebar.selectbox(
        "Nutrient density filter",
        ["Any", "Excellent", "Excellent,Good", "Good", "Good,Fair", "Fair", "Fair,Poor", "Poor"],
    )
    num_recipes = st.sidebar.slider("Number of recommendations", min_value=1, max_value=20, value=6)
    sort_by = st.sidebar.selectbox("Sort recommendations by", ["Macro score", "Ingredient score", "Nutrient density", "Calories"])

    bmi = calculate_bmi(weight, height)
    bmr = calculate_bmr(weight, height, age, sex.lower())
    tdee = calculate_tdee(bmr, activity_level)
    target_calories = adjust_calories(tdee, goal, sex.lower())
    macros = calculate_macros(target_calories, goal, weight)
    per_meal_calories = target_calories / meals_per_day
    per_meal_protein = macros["protein_g"] / meals_per_day
    per_meal_fat = macros["fat_g"] / meals_per_day
    per_meal_carbs = macros["carbs_g"] / meals_per_day

    chip_html = summarize_filters([x.lower() for x in preferred_types], [x.lower() for x in health_conditions], macro_prefs, meal_type_filter)
    render_hero_panel(target_calories, per_meal_calories, per_meal_protein, per_meal_carbs, per_meal_fat, chip_html)
    render_info("What do the scores mean?")

    control_col, image_col = st.columns([1.25, 0.95], gap="large")
    with control_col:
        st.markdown("### Ingredients & pantry")
        available_ingredients = st.text_area(
            "What do you have right now?",
            "olive oil, tomato, onion, chicken breast, garlic",
            height=120,
            help="Separate ingredients with commas.",
        )
        c_a, c_b = st.columns(2)
        with c_a:
            include_ingredients = st.text_input("Must include", "")
        with c_b:
            exclude_ingredients = st.text_input("Exclude", "peanut")
        btn_left, btn_mid, btn_right = st.columns([1, 1.6, 1])
        with btn_mid:
            submitted = st.button("Get Recommendations", use_container_width=True)
    with image_col:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.image(DEFAULT_FOOD_IMAGE_URL, use_container_width=True)
        st.markdown(
            """
            <div class="section-title" style="margin-top:10px;">Designed like a nutrition app</div>
            <div class="inline-note">Large hero image, rounded nutrition cards, clear pill tags, and a featured recipe layout inspired by polished mobile nutrition apps.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Your targets")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("BMI", f"{bmi:.1f}")
    t2.metric("BMR", f"{bmr:.0f} kcal")
    t3.metric("TDEE", f"{tdee:.0f} kcal")
    t4.metric("Per meal", f"{per_meal_calories:.0f} kcal")

    if not submitted:
        st.info("Update your pantry and filters, then click Get Recommendations to see the redesigned recipe cards.")
        return

    user_profile = {
        "age": age,
        "weight": weight,
        "height": height,
        "sex": sex.lower(),
        "activity_level": activity_level,
        "goal": goal,
        "meals_per_day": meals_per_day,
    }

    available_list = [x.strip() for x in available_ingredients.split(",") if x.strip()]
    include_list = [x.strip() for x in include_ingredients.split(",") if x.strip()]
    exclude_list = [x.strip() for x in exclude_ingredients.split(",") if x.strip()]
    preferred_list = [x.lower() for x in preferred_types]
    conditions = [x.lower() for x in health_conditions]
    macro_pref_list = [x.strip() for x in macro_prefs]

    try:
        base_df = load_base_dataset()
    except Exception as exc:
        st.error(f"Failed to load processed recipe data: {exc}")
        st.stop()

    recommendations = recommend_recipes(
        base_df,
        user_profile,
        available_list,
        include_list,
        exclude_list,
        preferred_list,
        conditions,
        max_cook_time,
        macro_pref_list,
        nutri_filter,
        nutrient_density_filter,
        meal_type_filter,
        num_recipes,
        sort_by,
    )

    if recommendations.empty:
        st.warning("No recipes matched those filters. Try a broader meal type, remove a macro preference, or increase cook time.")
        return

    detail_ids = tuple(int(x) for x in recommendations["recipe_id"].tolist()) if "recipe_id" in recommendations.columns else tuple()
    try:
        detail_df = load_recipe_details(detail_ids)
    except Exception:
        detail_df = pd.DataFrame()
    if not detail_df.empty and "recipe_id" in recommendations.columns:
        merged = recommendations.merge(detail_df, on="recipe_id", how="left", suffixes=("", "_detail"))
    else:
        merged = recommendations.copy()

    for base_col in ["final_name", "final_description", "ingredients_raw_list", "instructions_list", "image_url"]:
        detail_col = f"{base_col}_detail"
        if detail_col in merged.columns:
            if base_col in merged.columns:
                merged[base_col] = merged[base_col].where(merged[base_col].notna(), merged[detail_col])
            else:
                merged[base_col] = merged[detail_col]

    st.markdown("### Recommended recipes")
    st.caption("The first card is the featured match. Remaining recipes keep the same polished layout for easier comparison.")

    merged = merged.reset_index(drop=True)
    render_recipe_card(merged.iloc[0], per_meal_calories, target_calories, is_featured=True)
    for idx in range(1, len(merged)):
        render_recipe_card(merged.iloc[idx], per_meal_calories, target_calories, is_featured=False)


if __name__ == "__main__":
    main()
