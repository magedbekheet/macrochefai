"""Ingredient normalization and matching logic."""

import re
from typing import Any, Dict, List, Tuple


MEASUREMENT_WORDS = {
    "teaspoon", "teaspoons", "tsp", "tablespoon", "tablespoons", "tbsp", "cup", "cups",
    "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs", "gram", "grams", "g",
    "kg", "ml", "l", "liter", "liters", "clove", "cloves", "slice", "slices", "can", "cans",
    "package", "packages", "pkg", "pkgs",
}

DESCRIPTOR_WORDS = {
    "optional", "lightly", "toasted", "minced", "dried", "crushed", "fresh", "packed", "raw",
    "regular", "carefully", "scraped", "off", "flaked", "for", "bones", "substituted", "equal",
    "amount", "of", "an", "a", "the", "medium", "large", "small", "chopped", "grated", "drained",
    "baby", "skinless", "boneless",
}

STOP_IF_ONLY_GENERIC = {"ingredient", "ingredients", "item", "items"}


def _strip_quantity_prefix(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = s.replace("–", "-")
    s = re.sub(r"^\s*[\d\s/.-]+", "", s)
    changed = True
    while changed:
        changed = False
        parts = s.split()
        if parts and parts[0] in MEASUREMENT_WORDS:
            s = " ".join(parts[1:])
            changed = True
            continue
        if parts and re.fullmatch(r"[\d/.-]+", parts[0] or ""):
            s = " ".join(parts[1:])
            changed = True
    return re.sub(r"\s+", " ", s).strip(" ,;:-")


def _normalize_ingredient_option(text: str) -> str:
    s = text.strip().lower()
    if not s:
        return ""
    s = s.split(",")[0]
    s = _strip_quantity_prefix(s)
    s = s.replace("-", " ")
    tokens = []
    for token in re.findall(r"[a-zA-Z]+", s):
        if token in DESCRIPTOR_WORDS:
            continue
        tokens.append(token)
    while tokens and tokens[0] in MEASUREMENT_WORDS:
        tokens.pop(0)
    if not tokens:
        return ""
    phrase = " ".join(tokens).strip()
    if phrase in STOP_IF_ONLY_GENERIC:
        return ""
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def build_ingredient_requirements(raw_ingredients: List[str], clean_ingredients: List[str]) -> List[Dict[str, Any]]:
    requirements: List[Dict[str, Any]] = []

    raw_list = [str(x).strip() for x in raw_ingredients if str(x).strip()]
    if raw_list:
        for raw in raw_list:
            first_part = re.split(r"\s+and/or\s+|\s+or\s+", raw, flags=re.IGNORECASE)
            options = []
            for part in first_part:
                normalized = _normalize_ingredient_option(part)
                if normalized:
                    options.append(normalized)
            options = list(dict.fromkeys(options))
            if options:
                label = " or ".join(options)
                requirements.append({"label": label, "options": options})

    if requirements:
        return requirements

    clean_list = []
    for ing in clean_ingredients:
        normalized = _normalize_ingredient_option(str(ing))
        if normalized:
            clean_list.append(normalized)
    clean_list = list(dict.fromkeys(clean_list))
    return [{"label": ing, "options": [ing]} for ing in clean_list]


def ingredient_matches_option(user_item: str, option: str) -> bool:
    u = _normalize_ingredient_option(user_item)
    o = _normalize_ingredient_option(option)
    if not u or not o:
        return False
    return u == o or u in o or o in u


def match_ingredients(recipe_ingredients: List[str], available: List[str]) -> Tuple[int, int, List[str], List[str]]:
    recipe_set = {str(ing).strip().lower() for ing in recipe_ingredients if str(ing).strip()}
    user_set = {str(ing).strip().lower() for ing in available if str(ing).strip()}
    matched = sorted(recipe_set.intersection(user_set))
    missing = sorted(recipe_set.difference(user_set))
    return len(matched), len(missing), matched, missing


def match_ingredient_requirements(
    raw_ingredients: List[str],
    clean_ingredients: List[str],
    available: List[str],
) -> Tuple[int, int, List[str], List[str], int]:
    requirements = build_ingredient_requirements(raw_ingredients, clean_ingredients)
    normalized_available = [str(x).strip() for x in available if str(x).strip()]

    matched_labels: List[str] = []
    missing_labels: List[str] = []
    for req in requirements:
        options = req.get("options", [])
        label = req.get("label", "")
        is_match = any(
            ingredient_matches_option(user_item, option)
            for user_item in normalized_available
            for option in options
        )
        if is_match:
            matched_labels.append(label)
        else:
            missing_labels.append(label)

    total = len(requirements)
    return len(matched_labels), len(missing_labels), matched_labels, missing_labels, total
