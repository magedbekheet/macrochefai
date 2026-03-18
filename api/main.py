"""FastAPI application — recipe recommendation API.

Runs as a background thread inside the Streamlit process,
sharing the same cached DataFrames via src.data_loading.

Endpoints
---------
POST /api/v1/recommend   — Get personalized recipe recommendations
GET  /api/v1/health      — Health-check / readiness probe
GET  /api/v1/recipe/{id} — Single recipe details
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.nutrition import calculate_bmi, calculate_bmr, calculate_tdee, adjust_calories, calculate_macros
from src.recommender import recommend_recipes
from src.schemas import (
    RecommendRequest,
    RecommendResponse,
    RecipeSummary,
    NutritionInfo,
    UserTargets,
    HealthResponse,
    RecipeDetailResponse,
)


# ---------------------------------------------------------------------------
# Shared data access — uses the same @st.cache_resource as Streamlit
# ---------------------------------------------------------------------------

def _get_base_df() -> pd.DataFrame:
    """Get the base DataFrame from the shared Streamlit cache."""
    from src.data_loading import load_base_dataset
    return load_base_dataset()


def _get_display_df() -> pd.DataFrame:
    """Get display DataFrame for recipe detail lookups."""
    from src.data_loading import load_recipe_details
    # Return empty df — details are fetched on demand
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MacroChefAI API",
    version="1.0.0",
    description="Personalized recipe recommendations powered by KNN macro-matching and ingredient fit.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url="/api/redoc",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v, default: float = 0.0) -> float:
    n = pd.to_numeric(v, errors="coerce")
    return default if pd.isna(n) else float(n)


def _safe_list(v) -> list:
    if isinstance(v, list):
        return v
    return []


def _row_to_summary(row: pd.Series) -> RecipeSummary:
    return RecipeSummary(
        recipe_id=int(row["recipe_id"]) if pd.notna(row.get("recipe_id")) else None,
        name=str(row.get("final_name") or row.get("name") or ""),
        description=str(row.get("final_description") or ""),
        image_url=str(row.get("image_url") or ""),
        cook_time=_safe_float(row.get("cook_time"), None),
        servings=_safe_float(row.get("servings"), None),
        serving_g=_safe_float(row.get("serving_g"), None),
        nutrition=NutritionInfo(
            calories=_safe_float(row.get("calories")),
            protein=_safe_float(row.get("protein")),
            fat=_safe_float(row.get("fat")),
            carbs=_safe_float(row.get("carbs")),
            fiber=_safe_float(row.get("fiber")),
            sugar=_safe_float(row.get("sugar")),
            sodium=_safe_float(row.get("sodium")),
        ),
        nutri_score_label=str(row.get("nutri_score_label") or ""),
        nutrient_density_score=_safe_float(row.get("nutrient_density_score"), None),
        macro_score=_safe_float(row.get("macro_score"), None),
        ingredient_score=_safe_float(row.get("ingredient_score"), None),
        calorie_balance_label=str(row.get("calorie_balance_label_live") or ""),
        matched_ingredients=_safe_list(row.get("matched_ingredients")),
        missing_ingredients=_safe_list(row.get("missing_ingredients")),
        food_tags=_safe_list(row.get("food_tags")),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    try:
        df = _get_base_df()
        count = len(df)
    except Exception:
        count = 0
    return HealthResponse(status="ok", recipes_loaded=count)


@app.post("/api/v1/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    try:
        df = _get_base_df()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Data not loaded yet: {e}")

    up = req.user_profile
    f = req.filters

    # Compute user targets
    bmi = calculate_bmi(up.weight, up.height)
    bmr = calculate_bmr(up.weight, up.height, up.age, up.sex)
    tdee = calculate_tdee(bmr, up.activity_level)
    target_cal = adjust_calories(tdee, up.goal, up.sex)
    macros = calculate_macros(target_cal, up.goal, up.weight)
    per_meal_cal = target_cal / up.meals_per_day

    user_profile_dict = {
        "age": up.age,
        "weight": up.weight,
        "height": up.height,
        "sex": up.sex,
        "activity_level": up.activity_level,
        "goal": up.goal,
        "meals_per_day": up.meals_per_day,
    }

    recs = recommend_recipes(
        df,
        user_profile_dict,
        f.available_ingredients,
        f.include_ingredients,
        f.exclude_ingredients,
        [x.lower() for x in f.preferred_food_types],
        [x.lower() for x in f.health_conditions],
        f.max_cook_time,
        f.macro_prefs,
        f.nutri_filter,
        f.nutrient_density_filter,
        f.meal_type,
        f.num_recipes,
        f.sort_by,
    )

    recipes = [_row_to_summary(recs.iloc[i]) for i in range(len(recs))]

    return RecommendResponse(
        user_targets=UserTargets(
            bmi=round(bmi, 1),
            bmr=round(bmr, 0),
            tdee=round(tdee, 0),
            daily_calories=round(target_cal, 0),
            per_meal_calories=round(per_meal_cal, 0),
            protein_g=round(macros["protein_g"], 1),
            fat_g=round(macros["fat_g"], 1),
            carbs_g=round(macros["carbs_g"], 1),
        ),
        recipes=recipes,
        total_results=len(recipes),
    )


@app.get("/api/v1/recipe/{recipe_id}", response_model=RecipeDetailResponse)
async def get_recipe(recipe_id: int):
    try:
        df = _get_base_df()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Data not loaded yet: {e}")

    # Find in base dataset
    match = df[df["recipe_id"] == recipe_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Recipe {recipe_id} not found")

    row = match.iloc[0]
    summary = _row_to_summary(row)

    ingredients_raw = _safe_list(row.get("ingredients_raw_list"))
    instructions = _safe_list(row.get("instructions_list"))

    return RecipeDetailResponse(
        recipe=summary,
        ingredients_raw=ingredients_raw,
        instructions=instructions,
    )
