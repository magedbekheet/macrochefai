# Project File Guide

Quick rundown of what each file does and how everything fits together.

---

## How the app runs

When the Docker container starts up on Cloud Run, here's what happens:

1. `entrypoint.sh` kicks off — it first downloads data files from our Google Cloud Storage bucket
2. Streamlit starts up, and as part of that, it launches FastAPI in a background thread (so they share the same memory and don't load the data twice)
3. Once both are ready, nginx starts and acts as the front door — it routes `/api/*` requests to FastAPI and everything else to Streamlit

All three run inside the same container, on a single port (8080).

---

## Root files

**`streamlit_app.py`** — This is the main app. Around 1100 lines of UI code — the sidebar with all the filters, the hero section with your daily targets, and the recipe cards with nutrition breakdowns. It also handles starting FastAPI as a thread at the top of the file.

**`Dockerfile`** — Builds the container. We install nginx and curl on top of a Python 3.12 slim image, then copy in our code. Data isn't baked in — it gets pulled from GCS at runtime, which keeps the image small (~100 MB).

**`entrypoint.sh`** — The boot script. Downloads data, starts Streamlit in the background, polls until both Streamlit and FastAPI are healthy, then hands off to nginx. Nginx runs in the foreground to keep the container alive.

**`nginx.conf`** — Routes traffic. Anything starting with `/api/` goes to FastAPI on port 8000. Everything else (the UI, static assets, WebSocket connections) goes to Streamlit on port 8501.

**`requirements.txt`** — Python dependencies. The main ones are streamlit, fastapi, uvicorn, pandas, scikit-learn, and google-cloud-storage.

**`README.md`** — Project docs — setup instructions, how to deploy, API reference, how the recommendation engine works, and dataset info.

**`.gitignore`** / **`.dockerignore`** — Keeps data directories, caches, virtual environments, and IDE files out of git and Docker builds respectively.

**`.python-version`** — Tells pyenv/uv to use Python 3.12.9.

---

## `src/` — where the actual logic lives

Both Streamlit and FastAPI import from here, so nothing is duplicated.

**`data_loading.py`** — Loads the parquet files, merges them, parses list columns, and caches the result with `@st.cache_resource`. This is the most performance-critical file — it's basically loading ~486K recipes into a single DataFrame that everything else reads from.

**`gcs_loader.py`** — Called once at startup by `entrypoint.sh`. Downloads each data file from our GCS bucket if it's not already on disk. Simple download-and-skip logic.

**`recommender.py`** — The heart of the app. Takes a user profile (age, weight, goals, etc.) and available ingredients, builds a KNN model over calorie/protein/fat/carbs space, finds the closest recipes, then filters and scores them based on ingredient match, health conditions, meal type, and preferences. Returns the top N recipes sorted by score.

**`nutrition.py`** — The math behind the targets you see in the UI. Calculates BMI, BMR (using Mifflin-St Jeor), TDEE based on activity level, adjusts calories for your goal (lose/gain/maintain), and splits into protein/fat/carbs grams.

**`ingredients.py`** — Handles matching what's in your pantry against what a recipe needs. Uses substring/fuzzy matching to figure out how many ingredients you already have vs. what you'd need to buy.

**`schemas.py`** — Pydantic models that define what the API accepts and returns. Things like `RecommendRequest`, `RecipeSummary`, `NutritionInfo`. If you're looking at the API docs, these are the shapes you'll see.

**`pipeline.py`** — The data processing pipeline. Takes the raw Food.com CSV and transforms it into our cleaned parquet files — calculates per-100g nutrition, Nutri-Scores, nutrient density, risk flags for health conditions, etc. You run this once offline, not during the app's normal operation.

---

## `api/` — the REST API

**`main.py`** — Three endpoints: `POST /api/v1/recommend` for getting recipe suggestions, `GET /api/v1/health` for checking if the service is up, and `GET /api/v1/recipe/{id}` for looking up a single recipe. It doesn't load data on its own — it calls `load_base_dataset()` from `src/data_loading.py`, which returns the same cached DataFrame that Streamlit uses.

---

## `.streamlit/`

**`config.toml`** — Streamlit settings. Sets the theme colors, enables headless mode for Docker, and disables CORS/XSRF checks so it works behind nginx on Cloud Run.

---

## `notebooks/`

**`macrochefai_final.ipynb`** — The original development notebook where we did exploratory analysis, tested feature engineering ideas, and prototyped the recommendation logic before moving it into `src/`.

**`pipeline.ipynb`** — A lighter notebook for running the data processing pipeline interactively.

---

## Data directories (stored in GCS, not in git)

**`raw_data/`** — The original Food.com CSV (~1.5 GB).

**`processed_data/`** — Cleaned parquet files output by `pipeline.py`. Three files: one with just numeric features (compact), one with everything (model-ready), and one with display columns (names, descriptions, images).

**`models/`** — Pre-fitted scikit-learn artifacts: the KNN model, TF-IDF vectorizer for ingredients, and the sparse ingredient matrix.
