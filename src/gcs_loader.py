"""Download data files from Google Cloud Storage at startup.

When running in Cloud Run, data is not baked into the image — it is
fetched from the ``macrochefai-data`` GCS bucket on first access.
Locally, if the files already exist, no download is attempted.
"""

import os
from pathlib import Path

from google.cloud import storage

GCS_BUCKET = os.environ.get("GCS_BUCKET", "macrochefai-data")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files to download: (gcs_path, local_path)
DATA_FILES = [
    ("processed_data/processed_data/recipes_model_compact.parquet", PROJECT_ROOT / "processed_data" / "recipes_model_compact.parquet"),
    ("processed_data/processed_data/recipes_display_ready.parquet", PROJECT_ROOT / "processed_data" / "recipes_display_ready.parquet"),
    ("processed_data/processed_data/recipes_model_ready.parquet", PROJECT_ROOT / "processed_data" / "recipes_model_ready.parquet"),
    ("models/models/tfidf_vectorizer.joblib", PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"),
    ("models/models/ingredient_matrix.npz", PROJECT_ROOT / "models" / "ingredient_matrix.npz"),
    ("models/models/macro_knn.joblib", PROJECT_ROOT / "models" / "macro_knn.joblib"),
]


def ensure_data_downloaded():
    """Download data from GCS if not already present locally."""
    missing = [(gcs, local) for gcs, local in DATA_FILES if not local.exists()]

    if not missing:
        print("All data files already present locally.")
        return

    print(f"Downloading {len(missing)} files from gs://{GCS_BUCKET}/ ...")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    for gcs_path, local_path in missing:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = bucket.blob(gcs_path)
        print(f"  Downloading {gcs_path} → {local_path}")
        blob.download_to_filename(str(local_path))

    print("All data files downloaded.")
