from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Tuple

class BGEEmbedder:
    """Wrapper for BGE embedding model"""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        """Initialize the embedding model"""
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.model_name = model_name
        print("Model loaded successfully!")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray: 
        """Convert texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts)
        return embeddings
    
    # def find_similar(self, query: str, texts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        # """Find most similar texts to query"""
        
        # # Generate embeddings
        # query_embedding = self.encode_texts([query])
        # text_embeddings = self.encode_texts(texts)
        
        # # Calculate similarities
        # similarities = cosine_similarity(query_embedding, text_embeddings)[0]
        
        # # Get top-k results
        # top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # results = []
        # for idx in top_indices:
        #     results.append((texts[idx], similarities[idx]))
        
        # return results
    
    # def save_embeddings(self, texts: List[str], filepath: str):
        # """Save embeddings to file"""
        # embeddings = self.encode_texts(texts)
        
        # data = {
        #     'texts': texts,
        #     'embeddings': embeddings.tolist(),
        #     'model_name': self.model_name,
        #     'dimensions': embeddings.shape[1]
        # }
        
        # with open(filepath, 'w') as f:
        #     json.dump(data, f, indent=2)
        
        # print(f"Saved {len(texts)} embeddings to {filepath}")
    
    # def load_embeddings(self, filepath: str) -> Tuple[List[str], np.ndarray]:
        # """Load embeddings from file"""
        # with open(filepath, 'r') as f:
        #     data = json.load(f)
        
        # texts = data['texts']
        # embeddings = np.array(data['embeddings'])
        
        # print(f"Loaded {len(texts)} embeddings from {filepath}")
        # return texts, embeddings

# Usage functions
# def demo_similarity_search():
    # """Demo: Similarity search"""
    # print("Demo: Similarity Search")
    # print("-" *30)
    
    # embedder = BGEEmbedder()
    
    # # Sample documents
    # documents = [
    #     "Machine learning algorithms learn patterns from data",
    #     "Python is a popular programming language for AI",
    #     "Natural language processing helps computers understand text",
    #     "The weather forecast shows rain tomorrow",
    #     "Deep learning is a subset of machine learning",
    #     "JavaScript is used for web development",
    #     "Computer vision enables machines to see and interpret images"
    # ]
    
    # # Query
    # query = "artificial intelligence and programming"
    
    # # Find similar documents
    # top_k = 3;
    # results = embedder.find_similar(query, documents, top_k=top_k)
    
    # print(f"Query: '{query}'")
    # print(f"\nTop {top_k} most similar documents:")
    # for i, (doc, score) in enumerate(results, 1):
    #     print(f"{i}. Score: {score:.3f}")
    #     print(f"   Text: {doc}")

# def demo_save_load():
    # """Demo: Save and load embeddings"""
    # print("\nDemo: Save and Load Embeddings")
    # print("-" * 30)
    
    # embedder = BGEEmbedder()
    
    # # Sample texts
    # texts = [
    #     "Artificial intelligence is transforming technology",
    #     "Machine learning requires large datasets",
    #     "Neural networks mimic the human brain",
    #     "Machine learning algorithms learn patterns from data",
    #     "Python is a popular programming language for AI",
    #     "Natural language processing helps computers understand text",
    #     "The weather forecast shows rain tomorrow",
    #     "Deep learning is a subset of machine learning",
    #     "JavaScript is used for web development",
    #     "Computer vision enables machines to see and interpret images"
    # ]
    
    # # Save embeddings
    # embedder.save_embeddings(texts, "sample_embeddings_2.json")
    
    # # Load embeddings
    # loaded_texts, loaded_embeddings = embedder.load_embeddings("sample_embeddings_2.json")
    
    # print(f"Loaded embeddings shape: {loaded_embeddings.shape}")

# if __name__ == "__main__":
    # Run demos
    # demo_similarity_search()
    # demo_save_load()
    # pass