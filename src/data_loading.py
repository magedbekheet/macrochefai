"""Data loading and parsing helpers for parquet recipe files."""

import ast
import json
import re
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.ingredients import match_ingredient_requirements


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
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
    "recipe_id", "name", "final_name", "final_description",
    "ingredients_clean", "food_tags", "cook_time", "servings", "serving_g",
    "calories", "protein", "fat", "carbs", "fiber", "sugar", "sodium",
    "nutrient_density_score", "nutrient_density_class",
    "nutri_score_numeric", "nutri_score_label",
    "protein_level", "fat_level", "carb_level",
    "RecipeCategory", "dd_meal_type", "meal_type",
    "Images", "image_url", "ingredients_text",
    "ingredients_raw_list", "ingredient_quantities",
    "instructions_list", "macro_labels",
    "calorie_balance_pct", "calorie_balance_label",
    "dd_calorie_balance_pct", "dd_calorie_balance_label",
    "goal_meal_calories", "dd_goal_meal_calories",
]

DETAIL_DISPLAY_COLS = [
    "recipe_id", "name", "final_name", "final_description",
    "ingredients_raw_list", "ingredient_quantities",
    "ingredients_clean", "instructions_list",
    "food_tags", "macro_labels", "Images", "image_url",
]

LIST_COLUMNS_BASE = [
    "ingredients_clean", "food_tags", "ingredients_raw_list",
    "ingredient_quantities", "instructions_list", "macro_labels",
]
LIST_COLUMNS_DETAIL = LIST_COLUMNS_BASE

CATEGORY_COLUMNS = [
    "protein_level", "fat_level", "carb_level",
    "nutrient_density_class", "nutri_score_label",
    "dd_meal_type", "meal_type",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


@st.cache_resource(show_spinner=False)
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

    # Downcast float64 → float32 to halve numeric memory
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")

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


@st.cache_resource(show_spinner=False)
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
