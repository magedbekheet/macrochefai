# MacroChefAI

A recipe recommendation app that matches meals to your nutritional goals. You tell it your age, weight, fitness goal, and what's in your fridge — it finds the best recipes out of 486K+ options using a KNN-based scoring engine.

**Try it:** [macrochef-app-664804155729.europe-west1.run.app](https://macrochef-app-664804155729.europe-west1.run.app)

---

## What it does

- Calculates your daily calorie and macro targets based on your body stats, activity level, and goal
- Searches through 486K recipes to find ones that closely match your per-meal macros using K-Nearest Neighbors
- Scores recipes on ingredient overlap with your pantry (so you know what you already have vs. what you need)
- Filters by health conditions (diabetes, hypertension, etc.), meal type, food preferences, and cook time
- Shows Nutri-Score (A–E) and nutrient density for each recipe
- Combines everything into a weighted final score to rank the best matches

---

## Project structure

```
macrochefai/
├── streamlit_app.py              # Main UI
├── api/
│   └── main.py                   # REST API (3 endpoints)
├── src/
│   ├── nutrition.py              # BMI, BMR, TDEE, macro calculations
│   ├── ingredients.py            # Pantry-to-recipe matching
│   ├── data_loading.py           # Loads and caches the recipe dataset
│   ├── recommender.py            # KNN model + scoring logic
│   ├── pipeline.py               # Raw data → processed parquets
│   ├── schemas.py                # API request/response models
│   └── gcs_loader.py             # Downloads data from Cloud Storage
├── notebooks/
│   ├── pipeline.ipynb            # Data processing notebook
│   └── macrochefai_final.ipynb   # Original dev notebook
├── processed_data/               # Parquet files (not in git)
├── models/                       # Trained models (not in git)
├── raw_data/                     # Source CSV (not in git)
├── Dockerfile
├── entrypoint.sh
├── nginx.conf
├── requirements.txt
└── .streamlit/config.toml
```

For a detailed breakdown of each file, see [PROJECT_FILES.md](PROJECT_FILES.md).

---

## Running locally

You'll need Python 3.12 and the processed data files. If you don't have them, run `notebooks/pipeline.ipynb` first to generate the parquet files and model artifacts.

```bash
# Setup
git clone https://github.com/magedbekheet/macrochefai.git
cd macrochefai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start the app
streamlit run streamlit_app.py
# Open http://localhost:8501
```

To also run the API separately:

```bash
uvicorn api.main:app --port 8000
# Docs at http://localhost:8000/api/docs
```

---

## Docker

```bash
docker build -t macrochefai .
docker run -p 8080:8080 macrochefai
# Open http://localhost:8080
```

This starts nginx, Streamlit, and FastAPI inside one container. Nginx routes `/api/*` to FastAPI and everything else to Streamlit.

---

## Deploying to Cloud Run

The app runs on Google Cloud Run. Data is stored in a GCS bucket and downloaded at startup to keep the Docker image small.

```bash
# Build for the right architecture
docker buildx build --platform linux/amd64 \
  -t gcr.io/<PROJECT_ID>/macrochef-app:latest --push .

# Deploy
gcloud run deploy macrochef-app \
  --image gcr.io/<PROJECT_ID>/macrochef-app:latest \
  --region europe-west1 \
  --port 8080 \
  --memory 8Gi \
  --cpu 4 \
  --allow-unauthenticated \
  --set-env-vars "GCS_BUCKET=macrochefai-data"
```

The GCS bucket (`gs://macrochefai-data/`) holds the processed parquets, model files, and raw data. The container downloads what it needs on startup, typically takes about 10 seconds.

---

## API

Three endpoints, all under `/api/v1/`:

**`GET /health`** — Returns number of loaded recipes.

**`POST /recommend`** — The main one. Send user profile + filters, get back ranked recipes.

```json
{
  "user_profile": {
    "age": 30, "weight": 70, "height": 170,
    "sex": "male", "activity_level": "moderate",
    "goal": "weight_loss", "meals_per_day": 3
  },
  "filters": {
    "available_ingredients": ["olive oil", "tomato", "chicken breast"],
    "exclude_ingredients": ["peanut"],
    "health_conditions": ["diabetes"],
    "num_recipes": 6,
    "sort_by": "Macro score"
  }
}
```

**`GET /recipe/{id}`** — Full details for a single recipe (ingredients, instructions).

Interactive docs available at `/api/docs` on the deployed app.

---

## How the recommendation engine works

1. Your profile (age, weight, height, activity, goal) goes through standard nutrition formulas — BMR via Mifflin-St Jeor, multiplied by activity level to get TDEE, adjusted for your goal, then split into per-meal protein/fat/carbs targets.

2. We fit a KNN model over all 486K recipes on their calorie/protein/fat/carbs values (standardized), then find the nearest neighbors to your target. This gives us a pool of ~300 candidates.

3. Those candidates get filtered by meal type, health conditions, cook time, Nutri-Score, and food preferences.

4. Each remaining recipe gets an ingredient score — we compare your pantry against the recipe's ingredients and calculate how much coverage you have.

5. Everything gets combined into a final score: 65% macro closeness + 25% ingredient match + 10% nutrient density, with small bonuses for matching macro preferences (like "high protein").

---

## Dataset

Source: [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

Started with ~984K rows, cleaned down to ~486K recipes that have valid calories, macros, and serving weights. We engineered a bunch of features on top: per-100g nutrition values, Nutri-Score grades, nutrient density scores, medical risk flags, food type tags, and macro labels.

---

## Tech stack

- **Frontend:** Streamlit
- **API:** FastAPI + Uvicorn
- **ML:** scikit-learn (KNN, TF-IDF, StandardScaler)
- **Data:** Pandas + PyArrow
- **Container:** Docker (python:3.12-slim + nginx)
- **Cloud:** Google Cloud Run + Cloud Storage
- **Validation:** Pydantic

---

## Team

- [Maged Bekheet](https://github.com/magedbekheet)
- [Ehtisham Aziz](https://github.com/ehtishamaziz)
- [Julian Czywil](https://github.com/Julbool)
- [Artur Klimek](https://github.com/artkl1)

Built as part of the Le Wagon Data Science bootcamp.
