"""MacroChefAI FastAPI — entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from api.core.data_loader import build_models, load_dataset
from api.v3.router import router as v3_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("macrochefai")

app = FastAPI(
    title="MacroChefAI API",
    description="Health-aware recipe recommendation engine.",
    version="3.0.0",
)


@app.on_event("startup")
def startup():
    """Load data + build models once at startup."""
    logger.info("Loading dataset …")
    app.state.df_v3 = load_dataset(version="v3")
    logger.info("Dataset: %d recipes", len(app.state.df_v3))

    logger.info("Building models …")
    app.state.models_v3 = build_models(app.state.df_v3)

    logger.info("Startup complete ✓")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "recipes": len(app.state.df_v3),
    }


app.include_router(v3_router, prefix="/api/v3", tags=["V3 — Recommendations"])
