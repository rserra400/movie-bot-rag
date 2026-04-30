"""
Script de indexação do catálogo TMDB 5000 no ChromaDB.
Para cada filme:
  1. Constrói um documento de texto rico (título, sinopse, cast, etc.)
  2. Gera embedding via Ollama (nomic-embed-text)
  3. Guarda no ChromaDB com metadados para filtros
"""

import json
import pandas as pd
import requests
import chromadb
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "movies"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_json_load(text):
    """Lê uma string JSON; devolve [] se falhar."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return []


def get_embedding(text: str) -> list[float]:
    """Pede um embedding ao Ollama."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def build_document(movie_row, credits_row) -> tuple[str, dict]:
    """
    Constrói:
      - texto rico para gerar o embedding
      - dicionário de metadados para filtros no ChromaDB
    """
    title = movie_row["title"]
    overview = movie_row.get("overview") or ""
    tagline = movie_row.get("tagline") or ""
    release_date = movie_row.get("release_date") or ""
    year = release_date[:4] if isinstance(release_date, str) and release_date else "Unknown"

    # Géneros
    genres = [g["name"] for g in safe_json_load(movie_row["genres"])]
    # Keywords
    keywords = [k["name"] for k in safe_json_load(movie_row["keywords"])][:10]
    # Cast (top 5)
    cast = [c["name"] for c in safe_json_load(credits_row["cast"])[:5]]
    # Realizador
    crew = safe_json_load(credits_row["crew"])
    directors = [c["name"] for c in crew if c.get("job") == "Director"]

    # ----- Texto para embedding -----
    parts = [f"Title: {title}", f"Year: {year}"]
    if directors:
        parts.append(f"Director: {', '.join(directors)}")
    if genres:
        parts.append(f"Genres: {', '.join(genres)}")
    if cast:
        parts.append(f"Cast: {', '.join(cast)}")
    if tagline:
        parts.append(f"Tagline: {tagline}")
    if overview:
        parts.append(f"Overview: {overview}")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")
    document = "\n".join(parts)

    # ----- Metadados (ChromaDB só aceita tipos primitivos) -----
    metadata = {
        "title": title,
        "year": year,
        "director": directors[0] if directors else "",
        "genres": ", ".join(genres),
        "cast": ", ".join(cast),
        "vote_average": float(movie_row.get("vote_average") or 0),
        "popularity": float(movie_row.get("popularity") or 0),
    }

    return document, metadata


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    print("📂 A carregar CSVs...")
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

    # Junta os dois pelas chaves id <-> movie_id
    df = movies.merge(credits, left_on="id", right_on="movie_id", suffixes=("", "_c"))
    print(f"✅ {len(df)} filmes prontos a indexar.")

    print("🗄️  A inicializar ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Recriar a coleção do zero (útil enquanto se programa)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # similaridade por cosseno
    )

    print("🤖 A gerar embeddings com Ollama...")
    batch_ids, batch_docs, batch_embs, batch_meta = [], [], [], []
    BATCH_SIZE = 50  # número de filmes por commit no ChromaDB

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            document, metadata = build_document(row, row)
            embedding = get_embedding(document)

            batch_ids.append(str(row["id"]))
            batch_docs.append(document)
            batch_embs.append(embedding)
            batch_meta.append(metadata)

            if len(batch_ids) >= BATCH_SIZE:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embs,
                    metadatas=batch_meta,
                )
                batch_ids, batch_docs, batch_embs, batch_meta = [], [], [], []
        except Exception as e:
            print(f"⚠️  Erro no filme '{row.get('title')}': {e}")
            continue

    # Último batch
    if batch_ids:
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_meta,
        )

    print(f"\n🎉 Indexação completa! Total na coleção: {collection.count()} filmes.")


if __name__ == "__main__":
    main()
