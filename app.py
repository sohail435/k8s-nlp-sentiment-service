from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

app = FastAPI(title="Kubernetes NLP Sentiment Microservice")

# Load Hugging Face DistilBERT Pipeline at startup
print("Loading Transformer Model...")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("Model Loaded Successfully!")

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