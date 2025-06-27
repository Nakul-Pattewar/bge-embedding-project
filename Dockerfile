# Use official Python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Copy the rest of the code
COPY . .

# Set offline mode for Hugging Face/Transformers
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Start FastAPI server on container start
ENTRYPOINT ["uvicorn", "api.embedding_api:app", "--host", "0.0.0.0", "--port", "8000"] 