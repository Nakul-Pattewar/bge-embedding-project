import sys
from embedding_utils import BGEEmbedder
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_FILE = "sample_embeddings.json"

# Example texts for indexing
def get_example_texts():
    return [
        "Machine learning is a subset of artificial intelligence",
        "AI helps computers learn from data",
        "The weather is sunny today",
        "Deep learning uses neural networks",
        "It's raining cats and dogs",
        "Python is a popular programming language for AI",
        "Natural language processing helps computers understand text",
        "The weather forecast shows rain tomorrow",
        "JavaScript is used for web development",
        "Computer vision enables machines to see and interpret images"
    ]

def build_and_save_index(texts, embedding_file):
    embedder = BGEEmbedder()
    embedder.save_embeddings(texts, embedding_file)
    print(f"Embeddings for {len(texts)} texts saved to {embedding_file}")

def serve_query(embedding_file, user_query, top_k=3):
    embedder = BGEEmbedder()
    texts, embeddings = embedder.load_embeddings(embedding_file)
    query_embedding = embedder.encode_texts([user_query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    print(f"\nUser query: {user_query}")
    print(f"Top {top_k} most similar texts:")
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. Score: {similarities[idx]:.3f}")
        print(f"   Text: {texts[idx]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py [index|query] [optional: query_text]")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "index":
        texts = get_example_texts()
        build_and_save_index(texts, EMBEDDING_FILE)
    elif mode == "query":
        if len(sys.argv) < 3:
            print("Please provide a query string.")
            sys.exit(1)
        user_query = sys.argv[2]
        serve_query(EMBEDDING_FILE, user_query, top_k=3)
    else:
        print("Unknown mode. Use 'index' or 'query'.") 