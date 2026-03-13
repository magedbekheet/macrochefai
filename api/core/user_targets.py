"""User profile calculations: BMI, BMR, TDEE, macros."""

from __future__ import annotations


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)


def calculate_bmr(weight: float, height: float, age: int, sex: str) -> float:
    if sex == "male":
        return (9.99 * weight) + (6.25 * height) - (4.92 * age) + 5
    return (9.99 * weight) + (6.25 * height) - (4.92 * age) - 161


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
    if goal == "weight_loss":
        target = tdee * 0.8
        floor = 1500 if sex == "male" else 1200
        return max(target, floor)
    if goal == "weight_gain":
        return tdee * 1.1
    return tdee


def calculate_macros(
    calories: float, goal: str, weight_kg: float
) -> tuple[float, float, float]:
    """Returns (protein_g, fat_g, carbs_g). Scales down if protein+fat > 90% budget."""
    if goal == "weight_loss":
        protein = 1.8 * weight_kg
        fat = 0.8 * weight_kg
    elif goal == "weight_gain":
        protein = 1.6 * weight_kg
        fat = 0.9 * weight_kg
    else:
        protein = 1.6 * weight_kg
        fat = 0.8 * weight_kg

    pf_kcal = protein * 4 + fat * 9
    if pf_kcal > calories * 0.90:
        scale = (calories * 0.90) / pf_kcal
        protein *= scale
        fat *= scale

    carbs = max((calories - protein * 4 - fat * 9) / 4, 0)
    return protein, fat, carbs


def build_user_targets(
    weight: float,
    height: float,
    age: int,
    sex: str,
    activity_level: str,
    goal: str,
    meals_per_day: int = 3,
) -> dict:
    bmi = calculate_bmi(weight, height)
    bmr = calculate_bmr(weight, height, age, sex)
    tdee = calculate_tdee(bmr, activity_level)
    target_calories = adjust_calories(tdee, goal, sex)
    target_protein, target_fat, target_carbs = calculate_macros(
        target_calories, goal, weight
    )
    return {
        "bmi": round(bmi, 1),
        "bmr": round(bmr, 0),
        "tdee": round(tdee, 0),
        "target_calories": round(target_calories, 0),
        "target_protein": round(target_protein, 1),
        "target_fat": round(target_fat, 1),
        "target_carbs": round(target_carbs, 1),
        "meal_calories": round(target_calories / meals_per_day, 0),
        "meal_protein": round(target_protein / meals_per_day, 1),
        "meal_fat": round(target_fat / meals_per_day, 1),
        "meal_carbs": round(target_carbs / meals_per_day, 1),
    }
