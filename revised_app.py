import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


###############################################################################
# Helper functions                                                           #
###############################################################################


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Return body-mass index (kg/m²)."""
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)



def calculate_bmr(weight_kg: float, height_cm: float, age: float, sex: str) -> float:
    """Estimate the basal metabolic rate using the Mifflin-St Jeor equation."""
    sex = str(sex).lower()
    if sex == "male":
        return (9.99 * weight_kg) + (6.25 * height_cm) - (4.92 * age) + 5
    return (9.99 * weight_kg) + (6.25 * height_cm) - (4.92 * age) - 161



def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Estimate TDEE from BMR and an activity multiplier."""
    multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderate": 1.55,
        "very_active": 1.725,
        "extra_active": 1.9,
    }
    return bmr * multipliers.get(activity_level, 1.2)



def adjust_calories(tdee: float, goal: str, sex: str) -> float:
    """Adjust TDEE for weight loss, maintenance, or weight gain."""
    goal = goal.lower()
    sex = sex.lower()
    if goal == "weight_loss":
        target = tdee * 0.8
        floor = 1500 if sex == "male" else 1200
        return max(target, floor)
    if goal == "weight_gain":
        return tdee * 1.1
    return tdee



def calculate_macros(calories: float, goal: str, weight_kg: float) -> Dict[str, float]:
    """Derive daily protein, fat, and carbohydrate targets."""
    goal = goal.lower()
    if goal == "weight_loss":
        protein_g = 1.8 * weight_kg
        fat_g = 0.8 * weight_kg
    elif goal == "weight_gain":
        protein_g = 1.6 * weight_kg
        fat_g = 0.9 * weight_kg
    else:
        protein_g = 1.6 * weight_kg
        fat_g = 0.8 * weight_kg

    carbs_g = max((calories - protein_g * 4 - fat_g * 9) / 4, 0)
    return {"protein_g": protein_g, "fat_g": fat_g, "carbs_g": carbs_g}



def classify_macro_levels(protein_pct: float, carb_pct: float, fat_pct: float) -> Dict[str, str]:
    """Classify macro percentages into low, moderate, or high."""

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
        "protein_level": classify_protein(protein_pct),
        "carb_level": classify_carb(carb_pct),
        "fat_level": classify_fat(fat_pct),
    }



def match_ingredients(recipe_ingredients: List[str], available: List[str]) -> Tuple[int, int, List[str], List[str]]:
    """Count matched and missing ingredients between a recipe and the user's pantry."""
    recipe_set = {str(ing).strip().lower() for ing in recipe_ingredients if str(ing).strip()}
    user_set = {str(ing).strip().lower() for ing in available if str(ing).strip()}
    matched = sorted(recipe_set.intersection(user_set))
    missing = sorted(recipe_set.difference(user_set))
    return len(matched), len(missing), matched, missing



def _parse_list_like(x: Any) -> List[Any]:
    """Safely convert serialized list-like values back to Python lists."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            val = json.loads(s)
            return val if isinstance(val, list) else [s]
        except Exception:
            return [s]

    return [s]


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_data"
COMPACT_DATA_PATH = PROCESSED_DIR / "recipes_model_compact.parquet"
DISPLAY_DATA_PATH = PROCESSED_DIR / "recipes_display_ready.parquet"

DISPLAY_LIST_COLUMNS = [
    "ingredients_list",
    "ingredients_raw_list",
    "ingredient_quantities",
    "ingredients_clean",
    "instructions_list",
    "food_tags",
    "macro_labels",
    "Keywords",
]


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load compact and display recipe files, then merge them for app use."""
    if not COMPACT_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing compact data file: {COMPACT_DATA_PATH}")
    if not DISPLAY_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing display data file: {DISPLAY_DATA_PATH}")

    compact_df = pd.read_parquet(COMPACT_DATA_PATH)

    display_schema = pd.read_parquet(DISPLAY_DATA_PATH, columns=["recipe_id"]).columns.tolist()
    del display_schema

    display_probe = pd.read_parquet(DISPLAY_DATA_PATH)
    available_display_cols = set(display_probe.columns)
    desired_display_cols = [
        "recipe_id",
        "final_name",
        "final_description",
        "ingredients_list",
        "ingredients_raw_list",
        "ingredient_quantities",
        "ingredients_clean",
        "instructions_list",
        "food_tags",
        "macro_labels",
    ]
    load_cols = [c for c in desired_display_cols if c in available_display_cols]
    display_df = display_probe[load_cols].copy()

    for col in DISPLAY_LIST_COLUMNS:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(_parse_list_like)

    df = compact_df.merge(display_df, on="recipe_id", how="left", suffixes=("", "_display"))

    if "name" not in df.columns and "final_name" in df.columns:
        df["name"] = df["final_name"]
    else:
        df["name"] = df["name"].fillna(df.get("final_name"))

    if "calories" not in df.columns and "Calories" in df.columns:
        df["calories"] = df["Calories"]
    if "protein" not in df.columns and "ProteinContent" in df.columns:
        df["protein"] = df["ProteinContent"]
    if "fat" not in df.columns and "FatContent" in df.columns:
        df["fat"] = df["FatContent"]
    if "carbs" not in df.columns and "CarbohydrateContent" in df.columns:
        df["carbs"] = df["CarbohydrateContent"]
    if "fiber" not in df.columns and "FiberContent" in df.columns:
        df["fiber"] = df["FiberContent"]
    if "sugar" not in df.columns and "SugarContent" in df.columns:
        df["sugar"] = df["SugarContent"]
    if "sodium" not in df.columns and "SodiumContent" in df.columns:
        df["sodium"] = df["SodiumContent"]
    if "sat_fat" not in df.columns and "SaturatedFatContent" in df.columns:
        df["sat_fat"] = df["SaturatedFatContent"]

    for col in [
        "calories", "protein", "fat", "carbs", "fiber", "sugar", "sodium", "sat_fat",
        "cal_per_g", "energy_kcal_100g", "protein_100g", "fat_100g", "carbs_100g", "fiber_100g",
        "sugar_100g", "sodium_100g", "sat_fat_100g", "cook_time", "servings", "serving_g",
        "weight_loss_score", "weight_loss_score_pct", "weight_loss_probability", "health_score",
        "nutri_score_numeric"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "weight_loss_probability" not in df.columns:
        if "weight_loss_score_pct" in df.columns:
            df["weight_loss_probability"] = df["weight_loss_score_pct"].clip(0, 1)
        elif "weight_loss_score" in df.columns:
            mn = df["weight_loss_score"].min()
            mx = df["weight_loss_score"].max()
            if pd.isna(mn) or pd.isna(mx) or mn == mx:
                df["weight_loss_probability"] = 0.5
            else:
                df["weight_loss_probability"] = (df["weight_loss_score"] - mn) / (mx - mn)
        else:
            df["weight_loss_probability"] = 0.0

    if "ingredients_clean" not in df.columns:
        df["ingredients_clean"] = [[] for _ in range(len(df))]
    if "ingredients_raw_list" not in df.columns:
        df["ingredients_raw_list"] = [[] for _ in range(len(df))]
    if "instructions_list" not in df.columns:
        df["instructions_list"] = [[] for _ in range(len(df))]
    if "food_tags" not in df.columns:
        df["food_tags"] = [[] for _ in range(len(df))]
    if "macro_labels" not in df.columns:
        df["macro_labels"] = [[] for _ in range(len(df))]

    return df



def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure app-facing helper columns exist on the loaded processed dataset."""
    out = df.copy()
    out = out[out["calories"].fillna(0) > 0].copy()

    if "protein_pct" not in out.columns and {"protein", "fat", "carbs", "calories"}.issubset(out.columns):
        out["protein_pct"] = (out["protein"] * 4.0) / out["calories"]
        out["carb_pct"] = (out["carbs"] * 4.0) / out["calories"]
        out["fat_pct"] = (out["fat"] * 9.0) / out["calories"]

    if "protein_per_100kcal" not in out.columns:
        out["protein_per_100kcal"] = out["protein"] * 100.0 / out["calories"]
    if "fiber_per_100kcal" not in out.columns:
        out["fiber_per_100kcal"] = out["fiber"] * 100.0 / out["calories"]
    if "sugar_per_100kcal" not in out.columns:
        out["sugar_per_100kcal"] = out["sugar"] * 100.0 / out["calories"]

    if "health_score_norm" not in out.columns:
        min_h = out["health_score"].min()
        max_h = out["health_score"].max()
        if pd.isna(min_h) or pd.isna(max_h) or min_h == max_h:
            out["health_score_norm"] = 0.5
        else:
            out["health_score_norm"] = (out["health_score"] - min_h) / (max_h - min_h)

    if "weight_loss_norm" not in out.columns:
        min_w = out["weight_loss_score"].min()
        max_w = out["weight_loss_score"].max()
        if pd.isna(min_w) or pd.isna(max_w) or min_w == max_w:
            out["weight_loss_norm"] = out["weight_loss_probability"].fillna(0.5)
        else:
            out["weight_loss_norm"] = (out["weight_loss_score"] - min_w) / (max_w - min_w)

    return out



def recommend_recipes(
    df: pd.DataFrame,
    user_profile: Dict[str, Any],
    available: List[str],
    include: List[str],
    exclude: List[str],
    preferred_food_types: List[str],
    health_conditions: List[str],
    max_cook_time: float,
    macro_pref: str,
    nutri_filter: str,
    n_recommendations: int,
    sort_by: str,
) -> pd.DataFrame:
    """Filter and rank recipes according to the user profile and app controls."""
    data = df.copy()

    if include:
        include_set = {ing.strip().lower() for ing in include}
        data = data[data["ingredients_clean"].apply(lambda x: include_set.issubset({str(i).lower() for i in x}))]

    if exclude:
        exclude_set = {ing.strip().lower() for ing in exclude}
        data = data[data["ingredients_clean"].apply(lambda x: not bool({str(i).lower() for i in x}.intersection(exclude_set)))]

    if preferred_food_types:
        pref_set = {t.lower() for t in preferred_food_types}
        data = data[data["food_tags"].apply(lambda x: bool(pref_set.intersection({str(t).lower() for t in x})))]

    conditions = {c.lower() for c in health_conditions}
    if conditions:
        if "diabetes" in conditions:
            if "risk_diabetes" in data.columns:
                data = data[data["risk_diabetes"] == 0]
            else:
                data = data[data["sugar"].fillna(np.inf) <= 15]
        if "hypertension" in conditions:
            if "risk_hypertension" in data.columns:
                data = data[data["risk_hypertension"] == 0]
            else:
                data = data[data["sodium"].fillna(np.inf) <= 600]
        if "heart disease" in conditions:
            if "risk_heart_disease" in data.columns:
                data = data[data["risk_heart_disease"] == 0]
            else:
                data = data[data["sat_fat"].fillna(np.inf) <= 5]
        if "high cholesterol" in conditions and "risk_cholesterol" in data.columns:
            data = data[data["risk_cholesterol"] == 0]
        if "kidney disease" in conditions and "risk_kidney" in data.columns:
            data = data[data["risk_kidney"] == 0]
        if "keto" in conditions and "risk_keto_violation" in data.columns:
            data = data[data["risk_keto_violation"] == 0]

    if max_cook_time > 0 and "cook_time" in data.columns:
        data = data[data["cook_time"].fillna(np.inf) <= max_cook_time]

    if nutri_filter and nutri_filter != "Any" and "nutri_score_label" in data.columns:
        allowed = {x.strip() for x in nutri_filter.split(",") if x.strip()}
        data = data[data["nutri_score_label"].isin(allowed)]

    available_list = [ing.strip().lower() for ing in available if ing.strip()]
    matched_counts = []
    missing_counts = []
    matched_items = []
    missing_items = []
    for ingredients in data["ingredients_clean"]:
        matched, missing, matched_list, missing_list = match_ingredients(ingredients, available_list)
        matched_counts.append(matched)
        missing_counts.append(missing)
        matched_items.append(matched_list)
        missing_items.append(missing_list)

    data = data.copy()
    data["matched"] = matched_counts
    data["missing"] = missing_counts
    data["matched_ingredients"] = matched_items
    data["missing_ingredients"] = missing_items
    data["ingredient_score"] = data["matched"] / (data["matched"] + data["missing"] + 1e-6)

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

    data["macro_score"] = -(
        (data["protein"] - prot_t) ** 2 +
        (data["fat"] - fat_t) ** 2 +
        (data["carbs"] - carb_t) ** 2 +
        ((data["calories"] - cal_t) / 10.0) ** 2
    )

    if macro_pref:
        pref = macro_pref.lower()
        if pref == "high protein":
            if "protein_level" in data.columns:
                data = data[data["protein_level"].str.lower() == "high"]
            else:
                data = data[data["protein_pct"] >= 0.20]
        elif pref == "moderate protein" and "protein_level" in data.columns:
            data = data[data["protein_level"].str.lower() == "moderate"]
        elif pref == "low protein" and "protein_level" in data.columns:
            data = data[data["protein_level"].str.lower() == "low"]
        elif pref == "high carb":
            if "carb_level" in data.columns:
                data = data[data["carb_level"].str.lower() == "high"]
            else:
                data = data[data["carb_pct"] > 0.65]
        elif pref == "moderate carb" and "carb_level" in data.columns:
            data = data[data["carb_level"].str.lower() == "moderate"]
        elif pref == "low carb":
            if "carb_level" in data.columns:
                data = data[data["carb_level"].str.lower() == "low"]
            else:
                data = data[data["carb_pct"] < 0.45]
        elif pref == "high fat":
            if "fat_level" in data.columns:
                data = data[data["fat_level"].str.lower() == "high"]
            else:
                data = data[data["fat_pct"] > 0.35]
        elif pref == "moderate fat" and "fat_level" in data.columns:
            data = data[data["fat_level"].str.lower() == "moderate"]
        elif pref == "low fat" and "fat_level" in data.columns:
            data = data[data["fat_level"].str.lower() == "low"]
        elif pref == "high fiber":
            data = data[data["fiber"].fillna(0) >= 5]

    if sort_by == "Macro score":
        data = data.sort_values("macro_score", ascending=False)
    elif sort_by == "Ingredient score":
        data = data.sort_values("ingredient_score", ascending=False)
    elif sort_by == "Weight-loss":
        data = data.sort_values("weight_loss_probability", ascending=False)
    else:
        data = data.sort_values("health_score_norm", ascending=False)

    return data.head(n_recommendations)


###############################################################################
# Streamlit UI                                                              #
###############################################################################


def main() -> None:
    """Render the Streamlit app while preserving the original layout."""
    st.title("MacroChefAI: Personalised Recipe Recommendations")
    st.write(
        """Input your personal information, fitness goals and available ingredients to
        receive recipe suggestions tailored to your macro targets and dietary
        preferences.
        """
    )

    with st.sidebar:
        st.header("User Profile")
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=30)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
        sex = st.selectbox("Sex", ["Male", "Female"])
        activity = st.selectbox(
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
        goal = st.selectbox(
            "Fitness Goal",
            options=[("weight_loss", "Lose weight"), ("maintenance", "Maintain"), ("weight_gain", "Gain weight")],
            format_func=lambda x: x[1],
        )[0]
        meals_per_day = st.number_input("Meals per day", min_value=1, max_value=6, value=3)
        preferred_types = st.multiselect("Preferred food types", ["Vegetarian", "Vegan", "Chicken", "Seafood", "Meat", "Other"])
        health_conditions = st.multiselect(
            "Health conditions (for filtering)",
            ["Diabetes", "Hypertension", "Heart disease", "High cholesterol", "Kidney disease", "Keto"],
        )
        max_cook_time = st.number_input("Maximum cook time (minutes)", min_value=0, max_value=240, value=0)
        macro_pref = st.selectbox(
            "Macro preference (optional)",
            [
                "None",
                "High Protein", "Moderate Protein", "Low Protein",
                "High Fiber",
                "High Carb", "Moderate Carb", "Low Carb",
                "High Fat", "Moderate Fat", "Low Fat",
            ],
        )
        nutri_filter = st.selectbox(
            "Nutri-Score filter",
            ["Any", "A", "A,B", "B", "B,C", "C", "C,D", "D", "D,E", "E"],
        )
        num_recipes = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)
        sort_by = st.selectbox("Sort recommendations by", ["Macro score", "Ingredient score", "Weight-loss", "Health"])

    st.header("Ingredients")
    available_ingredients = st.text_area(
        "Ingredients you have (comma-separated)", "e.g., chicken breast, rice, olive oil, broccoli"
    )
    include_ingredients = st.text_input(
        "Ingredients that must be included (comma-separated)", ""
    )
    exclude_ingredients = st.text_input(
        "Ingredients to exclude (comma-separated)", ""
    )

    if st.button("Get Recommendations"):
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

        try:
            df = load_dataset()
        except Exception as exc:
            st.error(f"Failed to load processed recipe data: {exc}")
            return

        df_prepared = prepare_features(df)
        recommendations = recommend_recipes(
            df_prepared,
            user_profile,
            available_list,
            include_list,
            exclude_list,
            preferred_list,
            conditions,
            max_cook_time,
            macro_pref if macro_pref != "None" else "",
            nutri_filter,
            num_recipes,
            sort_by,
        )

        bmi = calculate_bmi(weight, height)
        bmr = calculate_bmr(weight, height, age, sex.lower())
        tdee = calculate_tdee(bmr, activity_level)
        target_calories = adjust_calories(tdee, goal, sex.lower())
        macros = calculate_macros(target_calories, goal, weight)
        per_meal_calories = target_calories / meals_per_day
        per_meal_protein = macros["protein_g"] / meals_per_day
        per_meal_fat = macros["fat_g"] / meals_per_day
        per_meal_carbs = macros["carbs_g"] / meals_per_day

        st.subheader("Targets")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BMI", f"{bmi:.1f}")
        c2.metric("BMR", f"{bmr:.0f} kcal")
        c3.metric("TDEE", f"{tdee:.0f} kcal")
        c4.metric("Daily target", f"{target_calories:.0f} kcal")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Per meal kcal", f"{per_meal_calories:.0f}")
        c6.metric("Per meal protein", f"{per_meal_protein:.0f} g")
        c7.metric("Per meal fat", f"{per_meal_fat:.0f} g")
        c8.metric("Per meal carbs", f"{per_meal_carbs:.0f} g")

        if recommendations.empty:
            st.warning("No recipes found with the current filters. Try relaxing some criteria.")
        else:
            st.subheader("Recommended Recipes")
            for _, row in recommendations.iterrows():
                macro_labels = classify_macro_levels(row.get("protein_pct", 0.0), row.get("carb_pct", 0.0), row.get("fat_pct", 0.0))
                title = row.get("name", row.get("final_name", "Recipe"))
                st.markdown(f"### {title}")

                cols = st.columns(4)
                cols[0].write(f"Calories: {row['calories']:.0f} kcal")
                cols[1].write(f"Calories / meal target: {per_meal_calories:.0f} kcal")
                cols[2].write(f"Calories / 100 g: {row.get('energy_kcal_100g', np.nan):.1f} kcal" if pd.notna(row.get('energy_kcal_100g', np.nan)) else "Calories / 100 g: N/A")
                cols[3].write(f"Nutri-Score: {row.get('nutri_score_label', 'N/A')}")

                st.write(
                    f"Protein: {row['protein']:.1f} g | Carbs: {row['carbs']:.1f} g | Fat: {row['fat']:.1f} g"
                )
                st.write(
                    f"Macro levels: Protein {macro_labels['protein_level']}, Carbs {macro_labels['carb_level']}, Fat {macro_labels['fat_level']}"
                )
                st.write(
                    f"Fiber: {row.get('fiber', np.nan):.1f} g | Sugar: {row.get('sugar', np.nan):.1f} g | Sodium: {row.get('sodium', np.nan):.1f} mg"
                )
                st.write(
                    f"Weight-loss score: {row.get('weight_loss_score', np.nan):.2f} | Health score: {row.get('health_score_norm', 0.0) * 100:.1f}%"
                )
                st.write(
                    f"Ingredient score: {row.get('ingredient_score', 0.0):.2f} | Weight-loss probability: {row.get('weight_loss_probability', 0.0):.2f}"
                )
                st.write(
                    f"Ingredient match: {int(row.get('matched', 0))} matched / {int(row.get('matched', 0) + row.get('missing', 0))} total"
                )

                matched_items = row.get("matched_ingredients", [])
                missing_items = row.get("missing_ingredients", [])
                if matched_items:
                    st.write(f"Matched ingredients: {', '.join(matched_items)}")
                if missing_items:
                    st.write(f"Missing ingredients: {', '.join(missing_items[:20])}")

                if isinstance(row.get("macro_labels", []), list) and row.get("macro_labels"):
                    st.write(f"Macro labels: {', '.join(row['macro_labels'])}")

                ingredient_display = row.get("ingredients_raw_list", [])
                if isinstance(ingredient_display, list) and ingredient_display:
                    st.write("Ingredients:")
                    for ing in ingredient_display:
                        st.write(f"- {ing}")
                else:
                    recipe_ings = ", ".join([str(ing).strip() for ing in row.get("ingredients_clean", [])])
                    st.write(f"Ingredients: {recipe_ings}")

                instructions = row.get("instructions_list", [])
                if isinstance(instructions, list) and instructions:
                    with st.expander("Instructions"):
                        for i, step in enumerate(instructions, start=1):
                            st.write(f"{i}. {step}")

                st.markdown("---")


if __name__ == "__main__":
    main()
