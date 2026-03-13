"""MacroChefAI Streamlit App — thin UI client that calls the FastAPI backend."""

import os

import requests
import streamlit as st

st.set_page_config(page_title="MacroChefAI", page_icon="🍽️", layout="wide")

# -------------------------------------------------------------------
# API configuration
# -------------------------------------------------------------------
API_BASE_URL = os.getenv("MACROCHEF_API_URL", "http://localhost:8000")
RECOMMEND_ENDPOINT = f"{API_BASE_URL}/api/v3/recommend"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


# -------------------------------------------------------------------
# API helpers
# -------------------------------------------------------------------

def check_api_health() -> dict | None:
    """Ping the FastAPI health endpoint. Returns the JSON or None."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def get_recommendations(payload: dict) -> dict | None:
    """POST to /api/v3/recommend. Returns the full JSON response or None."""
    try:
        resp = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot connect to the API server. "
            f"Make sure FastAPI is running at **{API_BASE_URL}**.\n\n"
            "Start it with: `uv run uvicorn api.main:app --reload`"
        )
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} — {exc.response.text}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None


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

# -------------------------------------------------------------------
# Build the API request payload
# -------------------------------------------------------------------
payload = {
    "profile": {
        "weight": weight,
        "height": height,
        "age": age,
        "sex": sex,
        "activity_level": activity_level,
        "goal": goal,
    },
    "meals_per_day": meals_per_day,
    "available_ingredients": [x.strip() for x in ingredients_text.split(",") if x.strip()],
    "preferred_food_types": preferred_food_types,
    "health_conditions": health_conditions,
    "strict_health_filter": strict_health_filter,
    "max_cook_time": max_cook_time,
    "max_missing_ingredients": max_missing_ingredients,
    "n_results": top_n,
}

# -------------------------------------------------------------------
# Call the API
# -------------------------------------------------------------------
api_data = get_recommendations(payload)

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.title("🍽️ MacroChefAI")
st.caption("Health-aware recipe recommendations powered by the MacroChefAI API.")

if api_data is None:
    st.stop()

recipes = api_data.get("recipes", [])
user_targets = api_data.get("user_targets", {})
total_candidates = api_data.get("total_candidates", 0)

# --- Health check badge ---
health = check_api_health()
if health:
    v3_count = health.get("v3_recipes", "?")
    st.sidebar.success(f"✅ API connected — {v3_count} recipes loaded")
else:
    st.sidebar.warning("⚠️ API health check failed")

# --- Summary metrics ---
s1, s2, s3 = st.columns(3)
s1.metric("Recipes found", len(recipes))
s2.metric("Candidates scanned", total_candidates)
s3.metric("API version", "V3")

tab1, tab2 = st.tabs(["Overview", "Recommendations"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BMI", f"{user_targets.get('bmi', 0):.1f}")
    col2.metric("BMR", f"{user_targets.get('bmr', 0):.0f} kcal")
    col3.metric("TDEE", f"{user_targets.get('tdee', 0):.0f} kcal")
    col4.metric("Daily Calories", f"{user_targets.get('target_calories', 0):.0f} kcal")

    st.subheader("Daily Macro Targets")
    d1, d2, d3 = st.columns(3)
    d1.metric("Protein", f"{user_targets.get('target_protein', 0):.0f} g")
    d2.metric("Fat", f"{user_targets.get('target_fat', 0):.0f} g")
    d3.metric("Carbs", f"{user_targets.get('target_carbs', 0):.0f} g")

    st.subheader("Per-Meal Targets")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calories", f"{user_targets.get('meal_calories', 0):.0f} kcal")
    m2.metric("Protein", f"{user_targets.get('meal_protein', 0):.0f} g")
    m3.metric("Fat", f"{user_targets.get('meal_fat', 0):.0f} g")
    m4.metric("Carbs", f"{user_targets.get('meal_carbs', 0):.0f} g")

    st.subheader("How it works")
    st.write(
        "Your profile is sent to the MacroChefAI API, which calculates per-meal macro targets "
        "and finds the best recipes using KNN macro matching, TF-IDF ingredient similarity, "
        "a weight-loss priority score, and medical risk filtering."
    )

with tab2:
    st.subheader("Top Recipe Recommendations")

    if not recipes:
        st.warning(
            "No recipes found. Try fewer health restrictions, "
            "a larger missing-ingredient limit, or broader food-type preferences."
        )
    else:
        for recipe in recipes:
            nutrition = recipe.get("nutrition", {})
            name = recipe.get("name", "Untitled Recipe")
            score = recipe.get("final_score", 0)

            with st.expander(f"{name} — score {score:.3f}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Calories", f"{nutrition.get('calories', 0):.0f}")
                c2.metric("Protein", f"{nutrition.get('protein', 0):.1f} g")
                cook_time = recipe.get("cook_time")
                c3.metric("Cook Time", f"{cook_time:.0f} min" if cook_time else "—")
                c4.metric("Medical Risk", recipe.get("medical_risk_level", "—"))

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Macro match", f"{recipe.get('macro_score', 0):.3f}")
                sc2.metric("Ingredient match", f"{recipe.get('ingredient_score', 0):.3f}")
                sc3.metric("Weight-loss priority", f"{recipe.get('weight_loss_priority', 0):.3f}")

                st.write(f"**Food tags:** {', '.join(recipe.get('food_tags', [])) or '—'}")
                st.write(f"**Medical risk reason:** {recipe.get('medical_risk_reason', '—')}")
                st.write(
                    f"**Matched ingredients:** "
                    f"{', '.join(recipe.get('matched_ingredients', [])[:15]) or '—'}"
                )
                st.write(
                    f"**Missing ingredients:** "
                    f"{', '.join(recipe.get('missing_ingredients', [])[:15]) or '—'}"
                )

                st.markdown("**Ingredients**")
                ingredients_list = recipe.get("ingredients", [])
                if ingredients_list:
                    for ing in ingredients_list[:25]:
                        st.write(f"- {ing}")
                else:
                    st.write("No ingredient list available.")

                st.markdown("**Instructions**")
                steps = recipe.get("instructions", [])
                if steps:
                    for i, step in enumerate(steps[:10], start=1):
                        st.write(f"{i}. {step}")
                else:
                    st.write("No instructions available.")

st.caption(
    "These health and medical flags are dietary heuristics for awareness "
    "and filtering. They are not medical advice or diagnosis."
)
