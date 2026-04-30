"""
Script de teste para validar a pesquisa semântica no ChromaDB.
"""
import requests
import chromadb

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "movies"


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def search(query: str, top_k: int = 5):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"\n🔍 Query: \"{query}\"")
    print("=" * 70)

    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
    )

    for i, (doc_id, metadata, distance) in enumerate(zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), start=1):
        similarity = 1 - distance  # cosine: distance vai 0..2; similarity é o inverso
        print(f"\n{i}. {metadata['title']} ({metadata['year']})  [score: {similarity:.3f}]")
        print(f"   Realizador: {metadata['director']}")
        print(f"   Géneros: {metadata['genres']}")
        print(f"   Cast: {metadata['cast']}")


if __name__ == "__main__":
    queries = [
        "a sci-fi movie about virtual reality and hackers",
        "a romantic comedy set in Paris",
        "a heist movie with twists and complex plot",
        "an animated movie about animals friendship",
        "a movie like Inception with dreams",
    ]

    for q in queries:
        search(q, top_k=3)
