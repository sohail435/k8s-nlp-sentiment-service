import os
import uvicorn
import mlflow.transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="Kubernetes NLP Sentiment Microservice")

# --- MLflow Model Registry Logic for Transformers ---
MODEL_URI = "models:/DistilBERTSentimentModel/Production"
TRACKING_URI = "file:./mlruns"

def load_sentiment_model():
    """Attempts to load from MLflow registry, falls back to raw pipeline."""
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        print(f"Attempting to load model from MLflow: {MODEL_URI}")
        return mlflow.transformers.load_model(MODEL_URI)
    except Exception as e:
        print(f"Registry load failed, falling back to local pipeline: {e}")
        print("Loading Transformer Model locally...")
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load Hugging Face DistilBERT Pipeline at startup
print("Initializing Sentiment Model...")
sentiment_model = load_sentiment_model()
print("Model Initialized Successfully!")

class TextRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=500, example="Kubernetes and FastAPI make MLOps smooth!")

class SentimentResponse(BaseModel):
    label: str
    score: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "nlp-sentiment"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(payload: TextRequest):
    try:
        results = sentiment_model(payload.text)
        prediction = results[0]
        return SentimentResponse(
            label=prediction['label'],
            score=round(prediction['score'], 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# 👇 Entry point for Render / local execution
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Note: reload=False is recommended here so Transformer models aren't loaded twice in memory
    uvicorn.run("app:app", host="0.0.0.0", port=port)