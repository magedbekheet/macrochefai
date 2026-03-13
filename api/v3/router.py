"""V3 recommendation router — original logic (per-recipe risk, weak penalty)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from api.core.data_loader import normalize_ingredient_list, normalize_text, safe_list
from api.core.models import (
    NutritionInfo,
    RecipeResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from api.core.user_targets import build_user_targets

router = APIRouter()

EPS = 1e-6


# -------------------------------------------------------------------
# V3 recommendation helpers (per-recipe risk, weak penalty, post-filter)
# -------------------------------------------------------------------

def _preferred_food_match(tags, preferred_food_types):
    if not preferred_food_types:
        return True
    tags_norm = {normalize_text(t) for t in safe_list(tags)}
    mapping = {
        "vegan": {"vegan"}, "vegetarian": {"vegetarian", "veggie"},
        "chicken": {"chicken", "poultry"},
        "meat": {"beef", "lamb", "pork", "meat", "steak", "bacon", "turkey"},
        "seafood": {"seafood", "fish", "shrimp", "salmon", "tuna", "cod"},
        "breakfast": {"breakfast", "brunch"}, "lunch": {"lunch"},
        "dinner": {"dinner", "main dish", "main course", "main_dish"},
        "snack": {"snack", "appetizer"},
    }
    prefs = {normalize_text(x) for x in preferred_food_types}
    for pref in prefs:
        if tags_norm.intersection(mapping.get(pref, {pref})):
            return True
    return False


def _knn_candidates(data, protein, fat, carbs, n_neighbors=50):
    """V3: smaller candidate pool (50), fitted on given data slice."""
    n_neighbors = max(1, min(n_neighbors, len(data)))
    macro_cols = data[["protein", "fat", "carbs"]].fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(macro_cols)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(scaled)
    user_vec = pd.DataFrame([{"protein": protein, "fat": fat, "carbs": carbs}])
    user_scaled = scaler.transform(user_vec)
    distances, indices = knn.kneighbors(user_scaled, n_neighbors=n_neighbors)
    cands = data.iloc[indices[0]].copy()
    cands["macro_distance"] = distances[0]
    cands["macro_score"] = 1 / (1 + cands["macro_distance"])
    return cands


def _ingredient_similarity(candidates, user_ingredients, models):
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
    vectorizer = models["vectorizer"]
    ingredient_matrix = models["ingredient_matrix"]
    tfidf_df = models["tfidf_df"]
    user_vec = vectorizer.transform([user_text])

    candidate_idx_in_tfidf = []
    for idx in candidates.index:
        if idx in tfidf_df.index:
            candidate_idx_in_tfidf.append((idx, tfidf_df.index.get_loc(idx)))

    if candidate_idx_in_tfidf:
        original_ids, matrix_locs = zip(*candidate_idx_in_tfidf)
        candidate_matrix = ingredient_matrix[list(matrix_locs)]
        sims = cosine_similarity(user_vec, candidate_matrix).flatten()
        sim_map = pd.Series(sims, index=list(original_ids))
    else:
        sim_map = pd.Series(dtype=float)

    candidates["matched_ingredients"] = candidates["ingredients_clean"].apply(
        lambda r: sorted({x for x in r if x}.intersection(user_set))
    )
    candidates["missing_ingredients"] = candidates["ingredients_clean"].apply(
        lambda r: sorted({x for x in r if x}.difference(user_set))
    )
    candidates["ingredient_overlap_score"] = candidates["ingredients_clean"].apply(
        lambda r: len({x for x in r if x}.intersection(user_set)) / max(len({x for x in r if x}), 1)
    )
    candidates["ingredient_tfidf_score"] = candidates.index.map(sim_map).fillna(0.0)
    candidates["ingredient_score"] = (
        0.7 * candidates["ingredient_overlap_score"]
        + 0.3 * candidates["ingredient_tfidf_score"]
    )
    return candidates


def _filter_by_preferences(candidates, preferred_food_types=None,
                           max_missing=None, max_cook_time=None):
    """V3: food type filtering happens HERE (post-KNN)."""
    f = candidates.copy()
    if preferred_food_types:
        f = f[f["food_tags"].apply(lambda t: _preferred_food_match(t, preferred_food_types))].copy()
    if max_missing is not None and "missing_ingredients" in f.columns:
        f = f[f["missing_ingredients"].apply(len) <= max_missing].copy()
    if max_cook_time is not None and "cook_time" in f.columns:
        f = f[f["cook_time"].fillna(np.inf) <= max_cook_time].copy()
    return f


def _filter_health(data, conditions, strict=True):
    if not conditions or not strict:
        return data
    f = data.copy()
    conds = {str(x).strip().lower() for x in conditions if str(x).strip()}
    if "diabetes" in conds:       f = f[f["risk_diabetes"] == 0]
    if "hypertension" in conds:   f = f[f["risk_hypertension"] == 0]
    if "heart_disease" in conds:  f = f[f["risk_heart_disease"] == 0]
    if "cholesterol" in conds:    f = f[f["risk_cholesterol"] == 0]
    if "kidney_disease" in conds: f = f[f["risk_kidney"] == 0]
    if "keto" in conds:           f = f[f["risk_keto_violation"] == 0]
    return f


def _wl_priority(candidates, full_df):
    candidates = candidates.copy()
    mn = float(full_df["weight_loss_score"].min())
    mx = float(full_df["weight_loss_score"].max())
    d = max(mx - mn, EPS)
    candidates["weight_loss_priority"] = ((candidates["weight_loss_score"] - mn) / d).clip(0, 1)
    return candidates


def _final_rank(candidates):
    """V3: fixed 0.10 penalty regardless of conditions."""
    candidates = candidates.copy()
    penalty = candidates["medical_risk_score"].fillna(0) / 6.0
    candidates["final_score"] = (
        0.50 * candidates["macro_score"]
        + 0.25 * candidates["ingredient_score"]
        + 0.25 * candidates["weight_loss_priority"]
        - 0.10 * penalty
    )
    return candidates.sort_values("final_score", ascending=False)


def _recommend_v3(df, models, protein, fat, carbs,
                  user_ingredients=None, preferred_food_types=None,
                  user_conditions=None, strict_health_filter=True,
                  n_results=5, n_candidates=50,
                  max_missing=None, max_cook_time=None):
    """V3: KNN on full dataset → post-filter food type → fallback relaxes ALL."""
    candidates = _knn_candidates(df, protein, fat, carbs, n_neighbors=n_candidates)
    candidates = _ingredient_similarity(candidates, user_ingredients or [], models)

    result = _filter_by_preferences(
        candidates, preferred_food_types, max_missing, max_cook_time
    )
    result = _filter_health(result, user_conditions, strict_health_filter)

    # V3 fallback: silently disable ALL filters
    if result.empty:
        result = candidates.copy()

    if result.empty:
        return result, 0

    result = _wl_priority(result, df)
    result = _final_rank(result)
    return result.head(n_results), len(candidates)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

def _recipe_to_response(row) -> RecipeResponse:
    return RecipeResponse(
        name=str(row.get("name", "Untitled")),
        nutrition=NutritionInfo(
            calories=float(row.get("calories", 0)),
            protein=float(row.get("protein", 0)),
            fat=float(row.get("fat", 0)),
            carbs=float(row.get("carbs", 0)),
            fiber=float(row.get("fiber", 0)),
            sugar=float(row.get("sugar", 0)),
            sat_fat=float(row.get("sat_fat", 0)),
            sodium=float(row.get("sodium", 0)),
            cholesterol=float(row.get("cholest", 0)),
        ),
        servings=float(row["servings"]) if pd.notna(row.get("servings")) else None,
        cook_time=float(row["cook_time"]) if pd.notna(row.get("cook_time")) else None,
        food_tags=safe_list(row.get("food_tags", [])),
        macro_score=float(row.get("macro_score", 0)),
        ingredient_score=float(row.get("ingredient_score", 0)),
        weight_loss_priority=float(row.get("weight_loss_priority", 0)),
        medical_risk_level=str(row.get("medical_risk_level", "unknown")),
        medical_risk_reason=str(row.get("medical_risk_reason", "")),
        matched_ingredients=safe_list(row.get("matched_ingredients", [])),
        missing_ingredients=safe_list(row.get("missing_ingredients", [])),
        ingredients=safe_list(row.get("ingredients_clean", [])),
        instructions=safe_list(row.get("instructions_list", [])),
        final_score=float(row.get("final_score", 0)),
    )


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest, req: Request):
    df = req.app.state.df_v3
    models = req.app.state.models_v3

    p = request.profile
    targets = build_user_targets(
        p.weight, p.height, p.age, p.sex,
        p.activity_level, p.goal, request.meals_per_day,
    )

    recs, total = _recommend_v3(
        df, models,
        protein=targets["meal_protein"],
        fat=targets["meal_fat"],
        carbs=targets["meal_carbs"],
        user_ingredients=request.available_ingredients,
        preferred_food_types=request.preferred_food_types,
        user_conditions=request.health_conditions,
        strict_health_filter=request.strict_health_filter,
        n_results=request.n_results,
        n_candidates=50,
        max_missing=request.max_missing_ingredients,
        max_cook_time=request.max_cook_time,
    )

    recipes = [_recipe_to_response(row) for _, row in recs.iterrows()] if not recs.empty else []

    return RecommendationResponse(
        recipes=recipes,
        total_candidates=total,
        relaxed_filters=[],
        user_targets=targets,
    )


@router.post("/user-targets")
def user_targets(profile: dict):
    return build_user_targets(
        weight=profile["weight"],
        height=profile["height"],
        age=profile["age"],
        sex=profile["sex"],
        activity_level=profile["activity_level"],
        goal=profile["goal"],
        meals_per_day=profile.get("meals_per_day", 3),
    )
