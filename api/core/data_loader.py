"""Load and preprocess the dataset, build ML models."""

from __future__ import annotations

import ast
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

EPS = 1e-6

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
DATA_CANDIDATES = [
    BASE_DIR / "data" / "processed_data" / "dataset.parquet",
    BASE_DIR / "processed_data" / "dataset.parquet",
    BASE_DIR / "dataset.parquet",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def normalize_text(x: str) -> str:
    x = html.unescape(str(x)).lower().strip()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9\s_]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


_STOP_WORDS = {
    "fresh", "ground", "large", "small", "medium", "extra", "virgin",
    "divided", "optional", "finely", "chopped", "diced", "minced",
    "sliced", "to", "taste", "or", "for", "and",
}


def clean_ingredient(text: str) -> str:
    text = normalize_text(text)
    tokens = [t for t in text.split() if t not in _STOP_WORDS]
    return "_".join(tokens).strip("_")


def normalize_ingredient_list(items) -> list[str]:
    cleaned = []
    for item in safe_list(items):
        token = clean_ingredient(item)
        if token:
            cleaned.append(token)
    return cleaned


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def first_existing(df: pd.DataFrame, names: list[str]):
    for name in names:
        if name in df.columns:
            return name
    return None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

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


def _classify_protein(p):
    if pd.isna(p): return np.nan
    if p < 0.12:   return "low"
    if p < 0.20:   return "moderate"
    return "high"

def _classify_carb(p):
    if pd.isna(p): return np.nan
    if p < 0.45:   return "low"
    if p <= 0.65:  return "moderate"
    return "high"

def _classify_fat(p):
    if pd.isna(p): return np.nan
    if p < 0.20:   return "low"
    if p <= 0.35:  return "moderate"
    return "high"

def _classify_sodium(s):
    if pd.isna(s): return np.nan
    if s <= 140:   return "low"
    if s <= 400:   return "moderate"
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
    df["protein_level"] = df["protein_pct"].apply(_classify_protein)
    df["carb_level"] = df["carb_pct"].apply(_classify_carb)
    df["fat_level"] = df["fat_pct"].apply(_classify_fat)
    df["sodium_level"] = df["sodium"].apply(_classify_sodium)
    macro_sum = df["protein_pct"] + df["carb_pct"] + df["fat_pct"]
    df["balanced"] = (
        df["protein_pct"].between(0.10, 0.35)
        & df["carb_pct"].between(0.45, 0.65)
        & df["fat_pct"].between(0.20, 0.35)
        & macro_sum.between(0.85, 1.15)
    ).astype(int)
    df["balanced_label"] = df["balanced"].map({1: "balanced", 0: "not_balanced"})
    return df


def add_medical_risk_flags_v3(df: pd.DataFrame) -> pd.DataFrame:
    """V3: per-RECIPE thresholds."""
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
        "risk_diabetes", "risk_hypertension", "risk_heart_disease",
        "risk_cholesterol", "risk_kidney", "risk_keto_violation",
    ]
    df["medical_risk_score"] = df[risk_cols].sum(axis=1)
    df["medical_risk_level"] = df["medical_risk_score"].apply(
        lambda s: "low" if s == 0 else ("moderate" if s <= 2 else "high")
    )
    return df





# ---------------------------------------------------------------------------
# Data loading (called once at startup)
# ---------------------------------------------------------------------------

def _find_data_path() -> Path | None:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    alias_groups = {
        "name": ["name", "title", "recipe_name", "Name"],
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
        "servings": ["servings", "RecipeServings"],
        "ingredients_clean": ["ingredients_clean", "ingredients_list"],
        "instructions_list": ["instructions_list", "RecipeInstructions"],
        "food_tags": ["food_tags", "tags", "diet_tags", "RecipeCategory"],
        "ingredients_text": ["ingredients_text"],
        "description": ["description", "Description"],
    }
    rename_map = {}
    for canonical, aliases in alias_groups.items():
        existing = first_existing(df, aliases)
        if existing and existing != canonical:
            rename_map[existing] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_dataset(version: str = "v3") -> pd.DataFrame:
    """Load and preprocess the parquet dataset."""
    data_path = _find_data_path()
    if data_path is None:
        raise FileNotFoundError(
            "dataset.parquet not found. Searched: " + ", ".join(str(p) for p in DATA_CANDIDATES)
        )

    df = pd.read_parquet(data_path).copy()
    df = _apply_aliases(df)

    for col in ["name", "description"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: html.unescape(x) if pd.notna(x) else x)

    if "name" not in df.columns:
        df["name"] = [f"Recipe {i+1}" for i in range(len(df))]

    if "food_tags" not in df.columns:
        df["food_tags"] = [[] for _ in range(len(df))]
    else:
        df["food_tags"] = df["food_tags"].apply(safe_list)

    if "ingredients_clean" in df.columns:
        df["ingredients_clean"] = df["ingredients_clean"].apply(normalize_ingredient_list)
    else:
        df["ingredients_clean"] = [[] for _ in range(len(df))]

    if "instructions_list" not in df.columns:
        df["instructions_list"] = [[] for _ in range(len(df))]
    else:
        df["instructions_list"] = df["instructions_list"].apply(safe_list)
        df["instructions_list"] = df["instructions_list"].apply(
            lambda items: [html.unescape(str(x)) for x in items]
        )

    if "ingredients_text" not in df.columns:
        df["ingredients_text"] = df["ingredients_clean"].apply(lambda x: " ".join(x))
    else:
        df["ingredients_text"] = df["ingredients_text"].fillna("").apply(normalize_text)

    df = ensure_numeric(
        df,
        ["calories", "protein", "fat", "carbs", "fiber", "sugar",
         "sat_fat", "sodium", "cholest", "cook_time", "servings"],
    )

    df = add_weight_loss_features(df)
    df = add_macro_classifications(df)
    df = add_medical_risk_flags_v3(df)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ML models (built once at startup)
# ---------------------------------------------------------------------------

def build_models(df: pd.DataFrame) -> dict:
    """Build KNN + TF-IDF models. Returns a dict of model artifacts."""
    # KNN on macros
    macro_X = df[["protein", "fat", "carbs"]].fillna(0.0)
    scaler = StandardScaler()
    macro_X_scaled = scaler.fit_transform(macro_X)
    knn = NearestNeighbors(n_neighbors=min(200, len(df)), metric="euclidean")
    knn.fit(macro_X_scaled)

    # TF-IDF on ingredients
    tfidf_df = df[df["ingredients_text"].fillna("").str.strip().astype(bool)].copy()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    ingredient_matrix = vectorizer.fit_transform(tfidf_df["ingredients_text"])

    return {
        "scaler": scaler,
        "knn": knn,
        "tfidf_df": tfidf_df,
        "vectorizer": vectorizer,
        "ingredient_matrix": ingredient_matrix,
    }
