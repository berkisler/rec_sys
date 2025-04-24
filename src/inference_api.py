# src/inference_api.py

import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# 1. The run_id from your local mlruns folder
#    You can copy it from the MLflow UI or the training output logs.
RUN_ID = os.getenv("MLFLOW_RUN_ID", "<your-run-id-here>")
MODEL_ARTIFACT_PATH = f"runs:/{RUN_ID}/model"

# 2. Load the model
try:
    # For Surprise's SVD, we used mlflow.sklearn.log_model, so we do:
    model = mlflow.sklearn.load_model(MODEL_ARTIFACT_PATH)
except Exception as e:
    print("Error loading model from local MLflow:", e)
    model = None

class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5

@app.get("/")
def root():
    return {"message": "Local MLflow Recommender System"}

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    if model is None:
        return {"error": "Model not loaded. Check logs or run ID."}

    # We'll assume item IDs range 1..1682 for MovieLens 100K
    items = range(1, 1683)
    predictions = []
    for item_id in items:
        # The model "predict" isn't the same as scikit-learn's .predict for arrays
        # Surprise's SVD object has `predict(user_id, item_id)`
        # But if mlflow.sklearn logged it, we might get a pickled SVD object
        # that we can call: model.predict(user_id, item_id).
        pred = model.predict(req.user_id, item_id)
        # pred has .est for the estimated rating
        predictions.append((item_id, pred.est))

    # Sort by rating descending
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Take top_n items
    top_items = [item for (item, rating) in predictions[:req.top_n]]
    return {
        "user_id": req.user_id,
        "top_n_recommendations": top_items
    }
