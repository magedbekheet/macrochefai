from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

class UserProfile(BaseModel):
    weight: float          # kg
    height: float          # cm
    age: int
    sex: Literal["male", "female"]
    activity_level: Literal["sedentary", "lightly_active", "moderate", "very_active", "extra_active"]
    goal: Literal["weight_loss", "maintenance", "weight_gain"]

class RecommendationRequest(BaseModel):
    profile: UserProfile
    meals_per_day: int = 3
    available_ingredients: list[str] = []
    preferred_food_types: list[str] = []     # e.g. ["chicken", "seafood"]
    health_conditions: list[str] = []         # e.g. ["diabetes", "hypertension"]
    strict_health_filter: bool = True
    max_cook_time: int | None = 60            # minutes
    max_missing_ingredients: int | None = 6
    n_results: int = 5

class NutritionInfo(BaseModel):
    calories: float
    protein: float
    fat: float
    carbs: float
    fiber: float
    sugar: float
    sat_fat: float
    sodium: float
    cholesterol: float

class RecipeResponse(BaseModel):
    name: str
    nutrition: NutritionInfo
    servings: float | None
    cook_time: float | None
    food_tags: list[str]
    macro_score: float
    ingredient_score: float
    weight_loss_priority: float
    medical_risk_level: str
    medical_risk_reason: str
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    ingredients: list[str]
    instructions: list[str]
    final_score: float

class RecommendationResponse(BaseModel):
    recipes: list[RecipeResponse]
    total_candidates: int
    relaxed_filters: list[str]
    user_targets: dict                  # BMI, TDEE, per-meal macros
