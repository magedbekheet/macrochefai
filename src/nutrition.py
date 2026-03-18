"""BMI, BMR, TDEE, calorie adjustment, and macro calculations."""

from typing import Dict


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
