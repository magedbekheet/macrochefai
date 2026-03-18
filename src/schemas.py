"""Pydantic models for FastAPI request / response validation."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    age: int = Field(..., ge=10, le=100, description="Age in years")
    weight: float = Field(..., ge=30, le=200, description="Weight in kg")
    height: float = Field(..., ge=120, le=220, description="Height in cm")
    sex: str = Field(..., pattern="^(male|female)$", description="Biological sex")
    activity_level: str = Field(
        "sedentary",
        description="One of: sedentary, lightly_active, moderate, very_active, extra_active",
    )
    goal: str = Field(
        "maintenance",
        description="One of: weight_loss, maintenance, weight_gain",
    )
    meals_per_day: int = Field(3, ge=1, le=6, description="Meals per day")


class Filters(BaseModel):
    available_ingredients: List[str] = Field(
        default_factory=list, description="Pantry ingredients (comma-separated before sending)"
    )
    include_ingredients: List[str] = Field(default_factory=list, description="Must-include ingredients")
    exclude_ingredients: List[str] = Field(default_factory=list, description="Must-exclude ingredients")
    preferred_food_types: List[str] = Field(default_factory=list, description="e.g. Vegetarian, Chicken")
    health_conditions: List[str] = Field(default_factory=list, description="e.g. Diabetes, Hypertension")
    max_cook_time: int = Field(45, ge=0, le=240, description="Max cooking time in minutes")
    macro_prefs: List[str] = Field(default_factory=list, description="e.g. High Protein, Low Carb")
    nutri_filter: List[str] = Field(default=["A", "B", "C"], description="Nutri-Score grades to include")
    nutrient_density_filter: str = Field("Any", description="Nutrient density filter")
    meal_type: str = Field("Any", description="Meal type filter: Any, breakfast, lunch, dinner")
    num_recipes: int = Field(6, ge=1, le=20, description="Number of recommendations")
    sort_by: str = Field("Macro score", description="Sort by: Macro score, Ingredient score, etc.")


class RecommendRequest(BaseModel):
    user_profile: UserProfile
    filters: Filters = Field(default_factory=Filters)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class NutritionInfo(BaseModel):
    calories: float = 0.0
    protein: float = 0.0
    fat: float = 0.0
    carbs: float = 0.0
    fiber: float = 0.0
    sugar: float = 0.0
    sodium: float = 0.0


class RecipeSummary(BaseModel):
    recipe_id: Optional[int] = None
    name: str = ""
    description: str = ""
    image_url: str = ""
    cook_time: Optional[float] = None
    servings: Optional[float] = None
    serving_g: Optional[float] = None
    nutrition: NutritionInfo = Field(default_factory=NutritionInfo)
    nutri_score_label: str = ""
    nutrient_density_score: Optional[float] = None
    macro_score: Optional[float] = None
    ingredient_score: Optional[float] = None
    calorie_balance_label: str = ""
    matched_ingredients: List[str] = Field(default_factory=list)
    missing_ingredients: List[str] = Field(default_factory=list)
    food_tags: List[str] = Field(default_factory=list)


class UserTargets(BaseModel):
    bmi: float
    bmr: float
    tdee: float
    daily_calories: float
    per_meal_calories: float
    protein_g: float
    fat_g: float
    carbs_g: float


class RecommendResponse(BaseModel):
    user_targets: UserTargets
    recipes: List[RecipeSummary]
    total_results: int


class HealthResponse(BaseModel):
    status: str = "ok"
    recipes_loaded: int = 0


class RecipeDetailResponse(BaseModel):
    recipe: RecipeSummary
    ingredients_raw: List[str] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)
