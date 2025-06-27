from fastapi import FastAPI, Request
from pydantic import BaseModel
from embedding_utils import BGEEmbedder
import numpy as np

app = FastAPI()
embedder = BGEEmbedder()  # Loads model once at startup

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    print("\nBGE Embedding API is running!\n")

@app.post("/embed")
def embed_query(request: QueryRequest):
    embedding = embedder.encode_texts([request.query])[0]
    return {
        "embedding": embedding.tolist(),
        "dimensions": len(embedding)
    }

@app.get("/")
def root():
    return {"message": "BGE Embedding API is running!"}