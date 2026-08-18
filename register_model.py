import mlflow
import mlflow.transformers
from transformers import pipeline

# Set tracking URI locally inside the service
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("nlp-sentiment-registry")

with mlflow.start_run(run_name="distilbert-sentiment-v1"):
    # Load your fine-tuned Hugging Face sentiment pipeline or model locally
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Log configuration metrics
    mlflow.log_param("model_architecture", "DistilBERTForSequenceClassification")
    mlflow.log_metric("baseline_accuracy", 0.94)
    
    # Log using MLflow's native transformers flavor
    mlflow.transformers.log_model(
        transformers_model=sentiment_pipeline,
        artifact_path="sentiment_pipeline",
        registered_model_name="DistilBERTSentimentModel"
    )
    print("DistilBERT NLP model successfully registered to MLflow registry!")