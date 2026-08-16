FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CPU directly from PyTorch index, followed by requirements with higher timeout
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# Pre-download the model weights during build time
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]