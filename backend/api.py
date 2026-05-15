"""
Backend FastAPI para o Movie Bot.
Expõe endpoints de pesquisa semântica e filtros sobre o catálogo TMDB 5000.
"""

from typing import Optional
import chromadb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "movies"

# ---------------------------------------------------------------------------
# App + Chroma client + embedding model (singletons)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Movie Bot API",
    description="API de pesquisa de filmes com RAG sobre o catálogo TMDB 5000",
    version="0.2.0",
)

_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _chroma_client.get_collection(COLLECTION_NAME)
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    return _embed_model.encode(text).tolist()


def _format_results(raw: dict) -> list[dict]:
    """Converte o output do ChromaDB num formato amigável."""
    out = []
    if not raw["ids"] or not raw["ids"][0]:
        return out

    ids = raw["ids"][0]
    metadatas = raw["metadatas"][0]
    documents = raw["documents"][0]
    distances = raw.get("distances", [[None] * len(ids)])[0]

    for movie_id, meta, doc, dist in zip(ids, metadatas, documents, distances):
        similarity = (1 - dist) if dist is not None else None
        out.append({
            "id": movie_id,
            "title": meta.get("title"),
            "year": meta.get("year"),
            "director": meta.get("director"),
            "genres": meta.get("genres"),
            "cast": meta.get("cast"),
            "vote_average": meta.get("vote_average"),
            "popularity": meta.get("popularity"),
            "similarity": similarity,
            "document": doc,
        })
    return out


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    genre: Optional[str] = None  # filtro opcional


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Movie Bot API",
        "movies_indexed": _collection.count(),
        "endpoints": [
            "POST /search",
            "GET /movies/by-actor/{name}",
            "GET /movies/by-director/{name}",
            "GET /movies/by-title/{title}",
        ],
    }


@app.post("/search")
def search(req: SearchRequest):
    """
    Pesquisa semântica de filmes a partir de uma descrição/frase.
    Opcionalmente filtra por género.
    """
    if not req.query.strip():
        raise HTTPException(400, "Query vazia.")

    embedding = get_embedding(req.query)

    # Filtro de metadata se fornecido
    where = None
    # Filtro de metadata se fornecido
    if req.genre:
        results = _collection.query(
            query_embeddings=[embedding],
            n_results=req.top_k * 4,
        )
        formatted = _format_results(results)
        genre_low = req.genre.lower()
        formatted = [m for m in formatted if genre_low in (m["genres"] or "").lower()]
        return {"query": req.query, "results": formatted[: req.top_k]}

    results = _collection.query(
        query_embeddings=[embedding],
        n_results=req.top_k,
    )
    return {"query": req.query, "results": _format_results(results)}
@app.get("/movies/by-actor/{name}")
def by_actor(name: str, limit: int = Query(10, ge=1, le=50)):
    """Filmes onde o ator aparece no cast (top 5 do filme)."""
    all_data = _collection.get(include=["metadatas"])
    name_low = name.lower()
    matches = []
    for movie_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        if name_low in (meta.get("cast") or "").lower():
            matches.append({"id": movie_id, **meta})
    matches.sort(key=lambda m: m.get("popularity", 0), reverse=True)
    return {"actor": name, "count": len(matches), "results": matches[:limit]}


@app.get("/movies/by-director/{name}")
def by_director(name: str, limit: int = Query(10, ge=1, le=50)):
    """Filmes de um realizador específico."""
    all_data = _collection.get(include=["metadatas"])
    name_low = name.lower()
    matches = []
    for movie_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        if name_low in (meta.get("director") or "").lower():
            matches.append({"id": movie_id, **meta})
    matches.sort(key=lambda m: m.get("popularity", 0), reverse=True)
    return {"director": name, "count": len(matches), "results": matches[:limit]}


@app.get("/movies/by-title/{title}")
def by_title(title: str):
    """Procura um filme pelo título (match parcial, case-insensitive)."""
    all_data = _collection.get(include=["metadatas", "documents"])
    title_low = title.lower()
    matches = []
    for movie_id, meta, doc in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        if title_low in (meta.get("title") or "").lower():
            matches.append({"id": movie_id, **meta, "document": doc})

    if not matches:
        raise HTTPException(404, f"Nenhum filme encontrado com '{title}'.")

    matches.sort(key=lambda m: (m["title"].lower() != title_low, -m.get("popularity", 0)))
    return {"query": title, "count": len(matches), "results": matches[:10]}

@app.get("/movies/filter")
def filter_movies(
    director: Optional[str] = None,
    actor: Optional[str] = None,
    genre: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = Query(10, ge=1, le=50),
):
    """
    Filtros combinados sobre o catálogo: realizador + ator + género + intervalo de anos.
    Todos os filtros são opcionais e combinam-se com AND.
    """
    if not any([director, actor, genre, year_from, year_to]):
        raise HTTPException(400, "Pelo menos um filtro é necessário.")

    all_data = _collection.get(include=["metadatas"])
    matches = []

    director_low = director.lower() if director else None
    actor_low = actor.lower() if actor else None
    genre_low = genre.lower() if genre else None

    for movie_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        # Filtro: realizador
        if director_low and director_low not in (meta.get("director") or "").lower():
            continue
        # Filtro: ator
        if actor_low and actor_low not in (meta.get("cast") or "").lower():
            continue
        # Filtro: género
        if genre_low and genre_low not in (meta.get("genres") or "").lower():
            continue
        # Filtro: ano
        try:
            year_int = int(meta.get("year") or 0)
        except (ValueError, TypeError):
            year_int = 0
        if year_from and year_int < year_from:
            continue
        if year_to and year_int > year_to:
            continue

        matches.append({"id": movie_id, **meta})

    # ordenar por popularidade desc
    matches.sort(key=lambda m: m.get("popularity", 0), reverse=True)

    return {
        "filters": {
            "director": director,
            "actor": actor,
            "genre": genre,
            "year_from": year_from,
            "year_to": year_to,
        },
        "count": len(matches),
        "results": matches[:limit],
    }

# ---------------------------------------------------------------------------
# Sistema de franquias
# ---------------------------------------------------------------------------
# Dicionário: chave = forma como o utilizador pode chamar a franquia (em PT/EN
# minúsculas), valor = lista de palavras que devem aparecer no título inglês.
# Suporta múltiplas variantes do nome em qualquer língua.
# Estrutura: cada franquia tem "match_type" e "keywords"
#  - "all": exige que TODAS as keywords apareçam no título (ex: Fast & Furious)
#  - "any": basta UMA das keywords aparecer (ex: James Bond — qualquer título)
#  - "contains": basta a string aparecer (default para nomes simples)
FRANCHISES = {
    # Velocidade Furiosa / Fast & Furious — exige "fast" E "furious"
    "fast and furious": {"type": "all", "kw": ["fast", "furious"]},
    "fast & furious": {"type": "all", "kw": ["fast", "furious"]},
    "velocidade furiosa": {"type": "all", "kw": ["fast", "furious"]},
    "the fast and the furious": {"type": "all", "kw": ["fast", "furious"]},

    # Lord of the Rings
    "lord of the rings": {"type": "contains", "kw": ["lord of the rings"]},
    "senhor dos anéis": {"type": "contains", "kw": ["lord of the rings"]},
    "senhor dos aneis": {"type": "contains", "kw": ["lord of the rings"]},
    "lotr": {"type": "contains", "kw": ["lord of the rings"]},

    # Hobbit
    "hobbit": {"type": "contains", "kw": ["hobbit"]},
    "the hobbit": {"type": "contains", "kw": ["hobbit"]},
    "o hobbit": {"type": "contains", "kw": ["hobbit"]},

    # Harry Potter
    "harry potter": {"type": "contains", "kw": ["harry potter"]},
    "saga harry potter": {"type": "contains", "kw": ["harry potter"]},

    # Star Wars
    "star wars": {"type": "contains", "kw": ["star wars"]},
    "guerra das estrelas": {"type": "contains", "kw": ["star wars"]},
    "guerras estelares": {"type": "contains", "kw": ["star wars"]},

    # James Bond — qualquer um destes títulos
    "james bond": {"type": "any", "kw": ["007", "skyfall", "spectre", "casino royale", "quantum of solace", "goldeneye", "die another day", "tomorrow never dies"]},
    "007": {"type": "any", "kw": ["007", "skyfall", "spectre", "casino royale", "quantum of solace", "goldeneye", "die another day", "tomorrow never dies"]},
    "bond": {"type": "any", "kw": ["007", "skyfall", "spectre", "casino royale", "quantum of solace", "goldeneye", "die another day", "tomorrow never dies"]},

    # Indiana Jones
    "indiana jones": {"type": "contains", "kw": ["indiana jones"]},

    # Pirates of the Caribbean
    "pirates of the caribbean": {"type": "contains", "kw": ["pirates of the caribbean"]},
    "piratas das caraibas": {"type": "contains", "kw": ["pirates of the caribbean"]},
    "piratas das caraíbas": {"type": "contains", "kw": ["pirates of the caribbean"]},
    "piratas do caribe": {"type": "contains", "kw": ["pirates of the caribbean"]},

    # Matrix
    "matrix": {"type": "contains", "kw": ["matrix"]},

    # Avengers
    "avengers": {"type": "contains", "kw": ["avengers"]},
    "vingadores": {"type": "contains", "kw": ["avengers"]},

    # Iron Man
    "iron man": {"type": "contains", "kw": ["iron man"]},
    "homem de ferro": {"type": "contains", "kw": ["iron man"]},

    # Spider-Man
    "spider-man": {"type": "contains", "kw": ["spider-man"]},
    "spiderman": {"type": "contains", "kw": ["spider-man"]},
    "homem-aranha": {"type": "contains", "kw": ["spider-man"]},
    "homem aranha": {"type": "contains", "kw": ["spider-man"]},

    # Batman / Dark Knight
    "batman": {"type": "any", "kw": ["batman", "dark knight"]},
    "the dark knight": {"type": "any", "kw": ["batman", "dark knight"]},
    "cavaleiro das trevas": {"type": "any", "kw": ["batman", "dark knight"]},

    # Mission Impossible
    "mission impossible": {"type": "contains", "kw": ["mission: impossible"]},
    "missão impossível": {"type": "contains", "kw": ["mission: impossible"]},
    "missao impossivel": {"type": "contains", "kw": ["mission: impossible"]},

    # Hunger Games
    "hunger games": {"type": "contains", "kw": ["hunger games"]},
    "jogos da fome": {"type": "contains", "kw": ["hunger games"]},

    # Twilight
    "twilight": {"type": "contains", "kw": ["twilight"]},
    "crepusculo": {"type": "contains", "kw": ["twilight"]},
    "crepúsculo": {"type": "contains", "kw": ["twilight"]},

    # Toy Story
    "toy story": {"type": "contains", "kw": ["toy story"]},

    # Shrek
    "shrek": {"type": "contains", "kw": ["shrek"]},

    # Rocky
    "rocky": {"type": "contains", "kw": ["rocky"]},

    # Terminator
    "terminator": {"type": "contains", "kw": ["terminator"]},
    "exterminador": {"type": "contains", "kw": ["terminator"]},
    "exterminador implacavel": {"type": "contains", "kw": ["terminator"]},
    "exterminador implacável": {"type": "contains", "kw": ["terminator"]},

    # Jurassic
    "jurassic park": {"type": "contains", "kw": ["jurassic"]},
    "jurassic world": {"type": "contains", "kw": ["jurassic"]},
    "parque jurassico": {"type": "contains", "kw": ["jurassic"]},
    "parque jurássico": {"type": "contains", "kw": ["jurassic"]},

    # X-Men
    "x-men": {"type": "contains", "kw": ["x-men"]},
    "xmen": {"type": "contains", "kw": ["x-men"]},

    # Transformers
    "transformers": {"type": "contains", "kw": ["transformers"]},

    # Rambo
    "rambo": {"type": "any", "kw": ["rambo", "first blood"]},

    # Die Hard
    "die hard": {"type": "contains", "kw": ["die hard"]},

    # Bourne
    "bourne": {"type": "contains", "kw": ["bourne"]},

    # Karate Kid
    "karate kid": {"type": "contains", "kw": ["karate kid"]},
}


@app.get("/movies/by-franchise/{name}")
def by_franchise(name: str, limit: int = Query(20, ge=1, le=50)):
    """
    Procura todos os filmes de uma franquia.
    Aceita nome em PT, EN, ou variantes (ex: "Velocidade Furiosa", "Fast & Furious", "Fast and Furious").
    Faz match contra um dicionário curado de franquias e palavras-chave de título.
    """
    name_low = name.lower().strip()

    # 1. Procurar a franquia no dicionário
 # 1. Procurar a franquia no dicionário
    franchise_def = FRANCHISES.get(name_low)

    # 2. Match parcial se não houver match exato
    if not franchise_def:
        for franchise_key, franchise_val in FRANCHISES.items():
            input_words = set(name_low.split())
            key_words = set(franchise_key.split())
            if len(input_words & key_words) >= max(1, len(input_words) - 1):
                franchise_def = franchise_val
                break

    if not franchise_def:
        raise HTTPException(
            404,
            f"Franquia '{name}' não reconhecida. "
            f"Franquias disponíveis: {sorted(set(FRANCHISES.keys()))[:10]}..."
        )

    match_type = franchise_def["type"]
    keywords = franchise_def["kw"]

    # 3. Filtrar filmes do catálogo conforme tipo de match
    all_data = _collection.get(include=["metadatas"])
    matches = []
    for movie_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        title_low = (meta.get("title") or "").lower()

        if match_type == "all":
            # Todas as keywords têm de aparecer (ex: "fast" E "furious")
            matched = all(kw in title_low for kw in keywords)
        elif match_type == "any":
            # Qualquer uma das keywords (ex: "007" ou "skyfall" ou ...)
            matched = any(kw in title_low for kw in keywords)
        else:  # "contains"
            # Match simples — string deve aparecer no título
            matched = any(kw in title_low for kw in keywords)

        if matched:
            matches.append({"id": movie_id, **meta})

    # 2. Se não houver match exato, tentar match parcial (ex: "fast furious" → "fast and furious")
    if not keywords:
        for franchise_key, franchise_kws in FRANCHISES.items():
            # Se todas as palavras do input estão na chave da franquia
            input_words = set(name_low.split())
            key_words = set(franchise_key.split())
            # Match se há sobreposição significativa
            if len(input_words & key_words) >= max(1, len(input_words) - 1):
                keywords = franchise_kws
                break

    if not keywords:
        raise HTTPException(
            404,
            f"Franquia '{name}' não reconhecida. "
            f"Franquias disponíveis: {sorted(set(FRANCHISES.keys()))[:10]}..."
        )

    # 3. Filtrar filmes do catálogo cujo título contenha qualquer keyword
    all_data = _collection.get(include=["metadatas"])
    matches = []
    for movie_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        title_low = (meta.get("title") or "").lower()

	# Match: se a franquia tem 1 keyword, basta essa.
        # Se tem múltiplas (ex: ["fast", "furious"]), exige que TODAS apareçam.
        # Isto evita falsos positivos como "Breakfast" matching "fast".
        if len(keywords) == 1:
            matched = keywords[0] in title_low
        else:
            matched = all(kw in title_low for kw in keywords)
        if matched:
            matches.append({"id": movie_id, **meta})

    # Ordenar por ano (cronológico) — útil para sagas
    matches.sort(key=lambda m: m.get("year", "0"))

    return {
        "franchise": name,
        "keywords_used": keywords,
        "count": len(matches),
        "results": matches[:limit],
    }


@app.get("/movies/popular")
def popular_movies(limit: int = Query(20, ge=1, le=50)):
    """Filmes mais populares do catálogo."""
    all_data = _collection.get(include=["metadatas"])
    movies = [{"id": mid, **meta}
              for mid, meta in zip(all_data["ids"], all_data["metadatas"])]
    movies.sort(key=lambda m: float(m.get("popularity") or 0), reverse=True)
    return {"count": len(movies), "results": movies[:limit]}


@app.get("/franchises")
def list_franchises():
    """Lista todas as franquias suportadas."""
    return {
        "count": len(FRANCHISES),
        "franchises": sorted(set(FRANCHISES.keys())),
    }
