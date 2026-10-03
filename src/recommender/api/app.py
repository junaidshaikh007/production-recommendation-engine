from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from recommender.models.hybrid import HybridRecommender, DEFAULT_ARTIFACT_PATH
from recommender.config import PROJECT_ROOT

# Globals to hold our loaded model
ml_models = {}
MODEL_PATH = DEFAULT_ARTIFACT_PATH.with_name("hybrid_model.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    try:
        model = HybridRecommender.load(MODEL_PATH)
        ml_models["hybrid"] = model
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        ml_models["hybrid"] = None
    
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(title="Recommendation API", lifespan=lifespan)

class RecommendationRequest(BaseModel):
    user_id: str
    k: int = 10

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[str]

@app.get("/health")
def health_check():
    """Health check endpoint to ensure API and model are loaded."""
    if ml_models.get("hybrid") is None:
        return {"status": "degraded", "model_loaded": False}
    return {"status": "ok", "model_loaded": True}

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: str, k: int = 10):
    """Get Top-K recommendations for a specific user."""
    model = ml_models.get("hybrid")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")
    
    try:
        recommendations = model.recommend(user_id=user_id, k=k)
        return RecommendationResponse(user_id=user_id, recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
