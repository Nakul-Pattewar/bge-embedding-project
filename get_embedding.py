import sys
import numpy as np
from sentence_transformers import SentenceTransformer

def get_query_embedding(query: str) -> np.ndarray:
    """Get embedding vector for a single query"""
    model_name = "BAAI/bge-large-en-v1.5"

    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name, cache_folder=None, local_files_only=True)
        
    print("Model loaded successfully!")
    embedding = model.encode([query])
    return embedding[0]  # Return the first (and only) embedding

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_embedding.py \"your query here\"")
        sys.exit(1)
    
    query = sys.argv[1]
    print(f"Query: {query}")
    
    # Get embedding
    embedding = get_query_embedding(query)
    
    print(f"Embedding dimensions: {embedding.shape}")
    print(f"Embedding vector: {embedding.tolist()}")

if __name__ == "__main__":
    main() 