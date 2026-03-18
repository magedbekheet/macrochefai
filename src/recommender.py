"""KNN recommendation engine, scoring, and calorie-balance utilities."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.nutrition import calculate_bmr, calculate_tdee, adjust_calories, calculate_macros
from src.ingredients import match_ingredient_requirements


def build_calorie_balance_label(
    calories: float,
    goal_meal_calories: float,
    tolerance_pct: float = 0.01,
    tolerance_kcal: float = 5.0,
) -> str:
    """Return a user-facing calorie alignment label."""
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

    # Soft preference bonus for macro labels.
    macro_pref_bonus = pd.Series(0.0, index=data.index)
    for pref in [str(p).strip().lower() for p in macro_prefs if str(p).strip()]:
        if pref == "high protein":
            macro_pref_bonus += np.where(
                data["protein_level"].astype(str).str.lower() == "high" if "protein_level" in data.columns else data["protein_pct"] >= 0.20,
                0.08, 0.0,
            )
        elif pref == "moderate protein":
            macro_pref_bonus += np.where(data["protein_level"].astype(str).str.lower() == "moderate", 0.05, 0.0) if "protein_level" in data.columns else 0.0
        elif pref == "low protein":
            macro_pref_bonus += np.where(data["protein_level"].astype(str).str.lower() == "low", 0.05, 0.0) if "protein_level" in data.columns else 0.0
        elif pref == "high carb":
            macro_pref_bonus += np.where(
                data["carb_level"].astype(str).str.lower() == "high" if "carb_level" in data.columns else data["carb_pct"] > 0.65,
                0.08, 0.0,
            )
        elif pref == "moderate carb":
            macro_pref_bonus += np.where(data["carb_level"].astype(str).str.lower() == "moderate", 0.05, 0.0) if "carb_level" in data.columns else 0.0
        elif pref == "low carb":
            macro_pref_bonus += np.where(
                data["carb_level"].astype(str).str.lower() == "low" if "carb_level" in data.columns else data["carb_pct"] < 0.45,
                0.08, 0.0,
            )
        elif pref == "high fat":
            macro_pref_bonus += np.where(
                data["fat_level"].astype(str).str.lower() == "high" if "fat_level" in data.columns else data["fat_pct"] > 0.35,
                0.08, 0.0,
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
