"""
Wrapper OpenAI-compatible para o Movie Bot.
Recebe pedidos de chat, decide se usa RAG, e devolve resposta da Llama via llama-cpp.
Suporta histórico de conversa para perguntas de followup.
"""

import json
import os
import re
import time
import uuid
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
LLAMA_CHAT_URL = "http://localhost:8080/v1/chat/completions"
LLM_MODEL = "llama3.1"
MOVIE_API = "http://localhost:8000"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ---------------------------------------------------------------------------
# Caches em memória com TTL
# ---------------------------------------------------------------------------
_api_cache:  dict[str, tuple[float, Optional[dict]]] = {}
_tmdb_cache: dict[str, tuple[float, Optional[dict]]] = {}
_API_TTL  = 300   # 5 min para resultados ChromaDB
_TMDB_TTL = 3600  # 1h para dados TMDB (posters não mudam)

app = FastAPI(title="Movie Bot Wrapper", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Modelos OpenAI-compatible
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None


class DiscoverRequest(BaseModel):
    genre_filter: Optional[str] = None   # nome EN: "Action", "Comedy", etc.
    genre_label: Optional[str] = None    # nome PT: "ação", "comédia", etc.
    mood: Optional[str] = None           # "leve e divertido", "intenso", etc.
    year_from: Optional[int] = None
    year_to: Optional[int] = None


# ---------------------------------------------------------------------------
# Classificador rápido de ferramentas (sem LLM)
# ---------------------------------------------------------------------------
_FRANCHISE_MAP = {
    "velocidade furiosa": "Fast and Furious",
    "velozes e furiosos": "Fast and Furious",
    "fast and furious": "Fast and Furious",
    "harry potter": "Harry Potter",
    "senhor dos anéis": "The Lord of the Rings",
    "senhor dos aneis": "The Lord of the Rings",
    "o senhor dos anéis": "The Lord of the Rings",
    "vingadores": "Avengers",
    "avengers": "Avengers",
    "star wars": "Star Wars",
    "guerra das estrelas": "Star Wars",
    "batman": "Batman",
    "superman": "Superman",
    "homem aranha": "Spider-Man",
    "homem-aranha": "Spider-Man",
    "spider-man": "Spider-Man",
    "spider man": "Spider-Man",
    "missão impossível": "Mission: Impossible",
    "missao impossivel": "Mission: Impossible",
    "jurassic park": "Jurassic Park",
    "jurassic world": "Jurassic World",
    "toy story": "Toy Story",
    "piratas das caraíbas": "Pirates of the Caribbean",
    "piratas das caribeias": "Pirates of the Caribbean",
    "piratas das caraibas": "Pirates of the Caribbean",
    "james bond": "James Bond",
    "007": "James Bond",
    "matrix": "The Matrix",
    "alien": "Alien",
    "predator": "Predator",
    "homem de ferro": "Iron Man",
    "iron man": "Iron Man",
    "thor": "Thor",
    "capitão america": "Captain America",
    "capitao america": "Captain America",
    "john wick": "John Wick",
    "mad max": "Mad Max",
    "transformers": "Transformers",
    "x-men": "X-Men",
    "x men": "X-Men",
    "deadpool": "Deadpool",
    "rocky": "Rocky",
    "rambo": "Rambo",
    "terminator": "The Terminator",
    "exterminador implacável": "The Terminator",
    "planeta dos macacos": "Planet of the Apes",
}

_GENRE_THEME_WORDS = {
    "ação", "acção", "aventura", "terror", "horror", "comédia", "comedia",
    "drama", "thriller", "ficção científica", "ficcao cientifica", "sci-fi",
    "romance", "crime", "mistério", "misterio", "animação", "animacao",
    "fantasia", "suspense", "musical", "guerra", "western", "documentário",
    "documentario", "drogas", "heist", "espionagem", "espaço", "espaco",
    "zumbis", "vampiro", "psicológico", "psicologico", "violência", "violencia",
}

_GENRE_EN_MAP = {
    "ação": "Action", "acção": "Action",
    "aventura": "Adventure",
    "terror": "Horror", "horror": "Horror",
    "comédia": "Comedy", "comedia": "Comedy",
    "drama": "Drama",
    "thriller": "Thriller",
    "ficção científica": "Science Fiction", "ficcao cientifica": "Science Fiction",
    "sci-fi": "Science Fiction",
    "romance": "Romance",
    "crime": "Crime",
    "mistério": "Mystery", "misterio": "Mystery",
    "animação": "Animation", "animacao": "Animation",
    "fantasia": "Fantasy",
    "musical": "Music",
    "guerra": "War",
    "western": "Western",
    "documentário": "Documentary", "documentario": "Documentary",
}

_MOVIE_KEYWORDS = [
    "filme", "filmes", "cinema", "movie", "ator", "atriz", "realizador",
    "diretor", "personagem", "saga", "franquia", "série", "serie",
    "oscar", "prémio", "estreia", "recomend", "assisti", "assistir",
    "género", "genero", "curta", "longa", "animaç", "documentário",
    "ver um", "ver uns", "sugere", "sugerir", "conheces algum",
]


# Tradução PT→EN para melhorar a pesquisa semântica no ChromaDB (corpus em inglês)
_PT_EN_SEARCH = [
    (r'\bde ação\b|\bde acção\b',                  'action'),
    (r'\bde aventura\b',                            'adventure'),
    (r'\bde terror\b|\bde horror\b',                'horror'),
    (r'\bde comédia\b|\bde comedia\b',              'comedy'),
    (r'\bde drama\b',                               'drama'),
    (r'\bde ficção científica\b|\bde ficcao cientifica\b|\bsci-fi\b', 'science fiction'),
    (r'\bde animação\b|\bde animacao\b',            'animation'),
    (r'\bde fantasia\b',                            'fantasy'),
    (r'\bde romance\b|\bde amor\b',                 'romance love'),
    (r'\bde crime\b',                               'crime'),
    (r'\bde mistério\b|\bde misterio\b',            'mystery'),
    (r'\bde suspense\b',                            'thriller suspense'),
    (r'\bde guerra\b',                              'war'),
    (r'\bsobre drogas\b',                           'about drugs narcotics addiction'),
    (r'\bsobre amor\b',                             'about love romance'),
    (r'\bsobre família\b|\bsobre familia\b',        'about family'),
    (r'\bsobre espionagem\b',                       'spy espionage'),
    (r'\bsobre espaço\b|\bsobre espaco\b',          'space'),
    (r'\bfilmes?\b',                                'films'),
    (r'\b(?:recomend[ae]|sugere|quero|gostava\s+de\s+ver|podes?\s+(?:recomendar|sugerir))\b', ''),
]

def _translate_search_query(text: str) -> str:
    result = text
    for pattern, repl in _PT_EN_SEARCH:
        result = re.sub(pattern, repl, result, flags=re.I)
    return ' '.join(result.split())


def _has_genre_theme(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _GENRE_THEME_WORDS)


def _resolve_franchise(raw: str) -> str:
    key = raw.lower().strip()
    if key in _FRANCHISE_MAP:
        return _FRANCHISE_MAP[key]
    for pt, en in _FRANCHISE_MAP.items():
        if pt in key:
            return en
    return raw.title()


def _extract_year_filters(text: str) -> dict:
    # "de 1990 a 2000" / "entre 1990 e 2000"
    m = re.search(r"(?:de|entre)\s+(\d{4})\s+(?:a|e|até|ate)\s+(\d{4})", text)
    if m:
        return {"year_from": int(m.group(1)), "year_to": int(m.group(2))}
    # "anos 80" / "nos anos 90s"
    m = re.search(r"anos?\s+(\d{2,4})s?", text)
    if m:
        d = int(m.group(1))
        if d < 100:
            d = (2000 if d < 30 else 1900) + d
        return {"year_from": d, "year_to": d + 9}
    # "após 1990" / "depois de 1990"
    filters = {}
    m = re.search(r"(?:após|depois\s+de|desde)\s+(\d{4})", text)
    if m:
        filters["year_from"] = int(m.group(1))
    m = re.search(r"(?:antes\s+de|até|ate)\s+(\d{4})", text)
    if m:
        filters["year_to"] = int(m.group(1))
    return filters


def _extract_genre_en(text: str) -> Optional[str]:
    t = text.lower()
    for pt, en in _GENRE_EN_MAP.items():
        if pt in t:
            return en
    return None


def classify_query(messages: list[dict]) -> dict:
    """Classificador de regras sem LLM. Elimina uma chamada LLM por pedido."""
    last = messages[-1]["content"]
    t = last.lower().strip()
    has_history = sum(1 for m in messages if m["role"] in ("user", "assistant")) > 1

    # 1. FRANCHISE
    for pat in [
        r"todos\s+os\s+filmes\s+d[aeo]s?\s+(.+?)[\?!]*$",
        r"filmes?\s+todos\s+d[aeo]s?\s+(?:saga\s+)?(.+?)[\?!]*$",
        r"saga\s+(?:completa\s+)?(?:d[aeo]s?\s+)?(.+?)[\?!]*$",
        r"franquia\s+(?:d[aeo]s?\s+)?(.+?)[\?!]*$",
        r"coleção\s+completa\s+(?:d[aeo]s?\s+)?(.+?)[\?!]*$",
        r"série\s+completa\s+de\s+filmes\s+(?:d[aeo]s?\s+)?(.+?)[\?!]*$",
    ]:
        m = re.search(pat, t)
        if m:
            return {"tool": "franchise", "argument": _resolve_franchise(m.group(1).strip())}

    # 2. FOLLOWUP (só com histórico)
    _followup_markers = [
        "esse filme", "aquele filme", "esses filmes", "aqueles filmes",
        "o primeiro", "o segundo", "o terceiro", "o último",
        "dos que recomendaste", "dos que mencionaste", "dos que disseste",
        "entre eles", "entre esses", "entre aqueles",
        "qual deles", "desses", "daqueles", "que referiste", "que mencionaste",
        "o mais popular dos", "o mais recente dos", "o mais antigo dos",
        "o pior dos", "o melhor dos",
    ]
    if has_history and any(mk in t for mk in _followup_markers):
        return {"tool": "followup", "argument": ""}

    # 3. MORE_LIKE
    for pat in [
        r"(?:filmes?\s+)?(?:mais\s+)?parecid[ao]s?\s+com\s+(?:[ao]\s+)?(.+?)[\?!]*$",
        r"(?:filmes?\s+)?similar(?:es)?\s+ao?\s+(?:[ao]\s+)?(.+?)[\?!]*$",
        r"mais\s+filmes?\s+(?:como|tipo)\s+(?:[ao]\s+)?(.+?)[\?!]*$",
        r"outros?\s+(?:filmes?\s+)?(?:como|tipo)\s+(?:o\s+)?(.+?)[\?!]*$",
        r"à\s+semelhança\s+de\s+(.+?)[\?!]*$",
    ]:
        m = re.search(pat, t)
        if m:
            return {"tool": "more_like", "argument": m.group(1).strip().title()}

    # 4. BY_TITLE
    for pat in [
        r"(?:fala[- ]me|conta[- ]me)\s+(?:mais\s+)?(?:sobre|acerca\s+de)\s+(?:o\s+filme\s+)?(.+?)[\?!]*$",
        r"sinopse\s+d[aeo]\s+(.+?)[\?!]*$",
        r"de\s+que\s+(?:trata|é)\s+(?:o\s+filme\s+)?(.+?)[\?!]*$",
    ]:
        m = re.search(pat, t)
        if m:
            return {"tool": "by_title", "argument": m.group(1).strip().title()}

    # 5. FILTER_COMBINED (tem filtro de ano)
    year_f = _extract_year_filters(t)
    if year_f:
        filters: dict = {}
        genre_f = _extract_genre_en(t)
        if genre_f:
            filters["genre"] = genre_f
        filters.update(year_f)
        return {"tool": "filter_combined", "argument": json.dumps(filters)}

    # 6. BY_DIRECTOR (apenas com palavra-chave explícita)
    for pat in [
        r"filmes?\s+d[ao]\s+realizador\s+(.+?)[\?!]*$",
        r"filmes?\s+da\s+realizadora\s+(.+?)[\?!]*$",
        r"d[ao]\s+realizador\s+(.+?)[\?!]*$",
        r"d[ao]\s+diretor\s+(.+?)[\?!]*$",
        r"da\s+diretora\s+(.+?)[\?!]*$",
    ]:
        m = re.search(pat, t)
        if m:
            return {"tool": "by_director", "argument": m.group(1).strip().title()}

    # 7. BY_ACTOR ("filmes do/da/com X" sem género/tema)
    for pat in [
        r"filmes?\s+d[ao]\s+(.+?)[\?!]*$",
        r"filmes?\s+da\s+(.+?)[\?!]*$",
        r"filmes?\s+com\s+(?:[ao]\s+)?(.+?)[\?!]*$",
        r"filmografia\s+d[aeo]\s+(.+?)[\?!]*$",
    ]:
        m = re.search(pat, t)
        if m:
            raw = m.group(1).strip()
            if 1 <= len(raw.split()) <= 4 and not _has_genre_theme(raw):
                return {"tool": "by_actor", "argument": raw.title()}

    # 8. NONE (claramente fora do tema filmes)
    if not any(kw in t for kw in _MOVIE_KEYWORDS) and not has_history:
        return {"tool": "none", "argument": ""}

    # 9. SEARCH (default)
    return {"tool": "search", "argument": last.strip()}


def find_movie_and_search_similar(title: str) -> Optional[dict]:
    """
    Para 'more_like': busca um filme pelo título e depois faz pesquisa
    semântica usando os detalhes desse filme como query.
    """
    try:
        # 1. Buscar o filme original
        r = requests.get(f"{MOVIE_API}/movies/by-title/{title}", timeout=30)
        if r.status_code != 200:
            return None

        title_data = r.json()
        movies = title_data.get("results", [])
        if not movies:
            return None

        # Pegar o filme mais relevante (já vem ordenado)
        ref_movie = movies[0]
        ref_title = ref_movie.get("title", "").lower()

        # 2. Construir query rica a partir do documento do filme
        document = ref_movie.get("document", "")
        # extrair géneros, keywords, sinopse para fazer query
        query_parts = []
        if "Genres:" in document:
            genres = document.split("Genres:")[1].split("\n")[0].strip()
            query_parts.append(genres)
        if "Keywords:" in document:
            keywords = document.split("Keywords:")[1].strip()
            query_parts.append(keywords)
        if "Overview:" in document:
            overview = document.split("Overview:")[1].split("\nKeywords:")[0].strip()
            query_parts.append(overview[:200])

        query = " ".join(query_parts) if query_parts else ref_movie.get("title", "")
        print(f"🔍 Query gerada para 'more_like': {query[:200]}...")

        # 3. Fazer pesquisa semântica com essa query (mais resultados para descontar o original)
        r2 = requests.post(
            f"{MOVIE_API}/search",
            json={"query": query, "top_k": 10},
            timeout=60,
        )
        if r2.status_code != 200:
            return None

        search_data = r2.json()
        # 4. Filtrar o filme original dos resultados
        filtered = [
            m for m in search_data.get("results", [])
            if m.get("title", "").lower() != ref_title
        ][:5]

        return {"query": title, "results": filtered, "reference": ref_movie}

    except requests.RequestException as e:
        print(f"⚠️  Erro em more_like: {e}")
        return None


def call_movie_api(tool: str, argument: str) -> Optional[dict]:
    """Chama o endpoint apropriado da Movie API."""
    try:
        if tool == "search":
            query = _translate_search_query(argument)
            print(f"🔤 Search query (traduzida): {query[:80]}")
            r = requests.post(
                f"{MOVIE_API}/search",
                json={"query": query, "top_k": 8},
                timeout=60,
            )
        elif tool == "by_actor":
            r = requests.get(f"{MOVIE_API}/movies/by-actor/{argument}?limit=10", timeout=30)
        elif tool == "by_director":
            r = requests.get(f"{MOVIE_API}/movies/by-director/{argument}?limit=10", timeout=30)
        elif tool == "by_title":
            r = requests.get(f"{MOVIE_API}/movies/by-title/{argument}", timeout=30)
        elif tool == "filter_combined":
            # argument vem como JSON string com os filtros
            try:
                filters = json.loads(argument) if isinstance(argument, str) else argument
            except json.JSONDecodeError:
                print(f"⚠️  Não foi possível parsear filtros: {argument}")
                return None
            params = {k: v for k, v in filters.items() if v}
            params["limit"] = 10
            r = requests.get(f"{MOVIE_API}/movies/filter", params=params, timeout=30)
        else:
            return None

        if r.status_code == 200:
            return r.json()
    except requests.RequestException as e:
        print(f"⚠️  Erro a chamar Movie API: {e}")
    return None


def format_context(api_data: dict, tool: str) -> str:
    """Constrói um excerto de contexto a partir dos resultados da API."""
    results = api_data.get("results", [])
    if not results:
        return "Sem resultados encontrados no catálogo."

    lines = [f"Resultados do catálogo (ferramenta: {tool}):\n"]
    for i, m in enumerate(results[:5], 1):
        line = f"{i}. {m.get('title')} ({m.get('year')}) — Realizador: {m.get('director')}"
        if m.get("genres"):
            line += f" | Géneros: {m['genres']}"
        if m.get("cast"):
            line += f" | Cast: {m['cast']}"
        if m.get("vote_average"):
            line += f" | Rating: {m['vote_average']}"
        lines.append(line)
        if m.get("document"):
            doc = m["document"]
            if "Overview:" in doc:
                overview = doc.split("Overview:")[1].split("\nKeywords:")[0].strip()
                lines.append(f"   Sinopse: {overview[:300]}")
    return "\n".join(lines)


def build_final_prompt(user_message: str, context: str) -> str:
    return f"""És um assistente especializado em filmes. Respondes em português de Portugal (não Brasil). Não uses "você", usa "tu".

REGRAS CRÍTICAS:
1. SÓ podes mencionar filmes que estão na LISTA abaixo. NUNCA inventes filmes.
2. Recomenda os 2-3 filmes da lista que melhor correspondem ao pedido, mesmo que a correspondência não seja perfeita.
3. Para cada filme inclui: título, ano, realizador, e uma frase curta a explicar porque é relevante.
4. Se NENHUM filme da lista tiver qualquer relação com o pedido, diz: "Não encontrei filmes relevantes no catálogo para o que pediste."
5. Sê directo e natural. Não uses preâmbulos longos.

LISTA DE FILMES DO CATÁLOGO:
{context}

PERGUNTA: {user_message}

RESPOSTA:"""


def build_filter_prompt(user_message: str, context: str, filters: dict) -> str:
    """Prompt dedicado para filter_combined — muito firme contra alucinações."""
    filter_desc = []
    if filters.get("director"):
        filter_desc.append(f"realizador: {filters['director']}")
    if filters.get("actor"):
        filter_desc.append(f"ator: {filters['actor']}")
    if filters.get("genre"):
        filter_desc.append(f"género: {filters['genre']}")
    if filters.get("year_from") or filters.get("year_to"):
        yf = filters.get("year_from", "?")
        yt = filters.get("year_to", "?")
        filter_desc.append(f"anos: {yf}-{yt}")
    filter_str = ", ".join(filter_desc)

    return f"""És um assistente especializado em filmes. Respondes em português de Portugal (não Brasil). Não uses "você", usa "tu".

O utilizador fez uma pesquisa filtrada com estes critérios: {filter_str}

LISTA EXAUSTIVA dos filmes que correspondem (esta é a ÚNICA fonte de verdade):

{context}

REGRAS ESTRITAS — viola e a resposta é inválida:
1. SÓ podes mencionar filmes que estão na lista acima. NUNCA menciones outros filmes (mesmo que conheças do teu treino).
2. NÃO repitas o mesmo filme duas vezes na resposta.
3. Se a lista tiver poucos filmes (1-3), apresenta TODOS. Se tiver muitos, escolhe os 3-5 mais relevantes/populares.
4. Para cada filme: título, ano, realizador (e ator se for relevante), e uma frase curta a justificar.
5. NÃO inventes filmes que NÃO ESTÃO na lista. Se mencionares um filme que não está na lista acima, isso é um ERRO GRAVE.
6. Sê natural e conversacional, não uses preâmbulos longos.

PERGUNTA DO UTILIZADOR: {user_message}

RESPOSTA:"""


def get_tmdb_data(title: str, year) -> Optional[dict]:
    if not TMDB_API_KEY or not title:
        return None
    try:
        params = {"api_key": TMDB_API_KEY, "query": title, "language": "pt-PT"}
        if year:
            params["year"] = str(year)[:4]
        r = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params, timeout=10
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        m = results[0]
        return {
            "title": title,
            "year": year,
            "tmdb_id": m.get("id"),
            "poster_url": f"{TMDB_IMAGE_BASE}{m['poster_path']}" if m.get("poster_path") else None,
            "tmdb_rating": round(m.get("vote_average", 0), 1),
            "overview": m.get("overview", ""),
        }
    except Exception as e:
        print(f"⚠️  TMDB lookup falhou para '{title}': {e}")
        return None


def search_tmdb_franchise(name: str) -> list[dict]:
    """Vai ao TMDB buscar TODOS os filmes de uma colecção/saga."""
    if not TMDB_API_KEY:
        return []
    try:
        # Tenta em PT primeiro, depois EN
        collection_id = None
        for lang in ["pt-PT", "en-US"]:
            r = requests.get(
                "https://api.themoviedb.org/3/search/collection",
                params={"api_key": TMDB_API_KEY, "query": name, "language": lang},
                timeout=10,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if results:
                collection_id = results[0]["id"]
                collection_name = results[0].get("name", name)
                break

        if not collection_id:
            return []

        r2 = requests.get(
            f"https://api.themoviedb.org/3/collection/{collection_id}",
            params={"api_key": TMDB_API_KEY, "language": "pt-PT"},
            timeout=10,
        )
        r2.raise_for_status()
        parts = r2.json().get("parts", [])
        print(f"🎬 Franchise '{collection_name}': {len(parts)} filmes encontrados no TMDB")

        movies = []
        for m in sorted(parts, key=lambda x: x.get("release_date", "") or ""):
            movies.append({
                "title": m.get("title") or m.get("original_title"),
                "year": (m.get("release_date") or "")[:4] or None,
                "tmdb_id": m.get("id"),
                "poster_url": f"{TMDB_IMAGE_BASE}{m['poster_path']}" if m.get("poster_path") else None,
                "tmdb_rating": round(m.get("vote_average", 0), 1),
                "overview": m.get("overview", ""),
                "director": None,
            })
        return movies
    except Exception as e:
        print(f"⚠️  Franchise search falhou para '{name}': {e}")
        return []


def call_llama(messages: list[dict]) -> str:
    """Chama a Llama via llama-cpp e devolve a resposta de texto."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 300,
    }
    r = requests.post(LLAMA_CHAT_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _stream_llama(messages: list[dict]):
    """Generator de tokens — streaming real da Llama token a token."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 300,
    }
    try:
        with requests.post(LLAMA_CHAT_URL, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
    except Exception as e:
        print(f"⚠️  Streaming error: {e}")


def get_tmdb_data_cached(title: str, year) -> Optional[dict]:
    key = f"{title}:{year}"
    if key in _tmdb_cache:
        ts, val = _tmdb_cache[key]
        if time.time() - ts < _TMDB_TTL:
            return val
    val = get_tmdb_data(title, year)
    _tmdb_cache[key] = (time.time(), val)
    return val


def call_movie_api_cached(tool: str, argument: str) -> Optional[dict]:
    key = f"{tool}:{argument}"
    if key in _api_cache:
        ts, val = _api_cache[key]
        if time.time() - ts < _API_TTL:
            print(f"💾 Cache hit: {tool}:{argument[:40]}")
            return val
    val = call_movie_api(tool, argument)
    if val:
        _api_cache[key] = (time.time(), val)
    return val


# ---------------------------------------------------------------------------
# Endpoints OpenAI-compatible
# ---------------------------------------------------------------------------

@app.get("/movies/popular")
def popular_movies():
    """Filmes populares enriquecidos com posters TMDB (cached)."""
    try:
        r = requests.get(f"{MOVIE_API}/movies/popular?limit=12", timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])

        def _enrich(movie):
            tmdb = get_tmdb_data_cached(movie.get("title"), movie.get("year"))
            return {
                "title": movie.get("title"),
                "year": movie.get("year"),
                "director": movie.get("director"),
                "genres": movie.get("genres"),
                "vote_average": movie.get("vote_average"),
                "poster_url": tmdb.get("poster_url") if tmdb else None,
                "tmdb_rating": tmdb.get("tmdb_rating") if tmdb else None,
                "tmdb_id": tmdb.get("tmdb_id") if tmdb else None,
                "overview": tmdb.get("overview") if tmdb else None,
            }

        with ThreadPoolExecutor(max_workers=6) as ex:
            enriched = list(ex.map(_enrich, results))
        return {"results": enriched}
    except Exception as e:
        print(f"⚠️  Erro em /movies/popular: {e}")
        return {"results": []}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "movie-bot",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "movie-bot",
            }
        ],
    }


@app.post("/v1/discover")
def discover_movies(req: DiscoverRequest):
    """Endpoint dedicado para o modo Descobrir — filtra por género + época, ordena por humor."""
    params: dict = {"limit": 15}
    if req.genre_filter:
        params["genre"] = req.genre_filter
    if req.year_from:
        params["year_from"] = req.year_from
    if req.year_to:
        params["year_to"] = req.year_to

    try:
        r = requests.get(f"{MOVIE_API}/movies/filter", params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        print(f"⚠️  Discover filter falhou: {e}")
        return {"text": "Erro ao pesquisar filmes.", "movies": []}

    if not results:
        return {"text": "Não encontrei filmes para estes critérios no catálogo.", "movies": []}

    lines = []
    for i, m in enumerate(results[:10], 1):
        line = f"{i}. {m.get('title')} ({m.get('year')}) — Realizador: {m.get('director')}"
        if m.get("genres"):
            line += f" | Géneros: {m['genres']}"
        if m.get("vote_average"):
            line += f" | Rating: {m['vote_average']}"
        lines.append(line)
    context = "\n".join(lines)

    genre_desc = req.genre_label or req.genre_filter or "filme"
    mood_desc = req.mood or "interessante"

    prompt = f"""És um assistente especializado em filmes. Respondes em português de Portugal (não Brasil). Não uses "você", usa "tu".

Da seguinte lista de filmes de {genre_desc}, recomenda 3-4 que sejam mais {mood_desc}.

LISTA:
{context}

REGRAS:
1. SÓ uses filmes desta lista. NUNCA inventes.
2. Para cada filme: título, ano, realizador, e uma frase curta sobre porque é {mood_desc}.
3. Sê directo e natural.

RESPOSTA:"""

    try:
        response_text = call_llama([{"role": "user", "content": prompt}])
    except Exception:
        response_text = "Não foi possível gerar a recomendação."

    def _enrich_d(movie):
        return get_tmdb_data_cached(movie.get("title"), movie.get("year"))

    with ThreadPoolExecutor(max_workers=5) as ex:
        tmdb_movies = [d for d in ex.map(_enrich_d, results[:5]) if d]

    print(f"🎯 Discover: {len(results)} filmes filtrados, {len(tmdb_movies)} posters")
    return {"text": response_text, "movies": tmdb_movies}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """Endpoint principal — recebe a conversa e devolve resposta com RAG."""
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(400, "Nenhuma mensagem do utilizador.")
    last_user = user_messages[-1].content

    print(f"\n📝 Pergunta: {last_user}")
    print(f"💬 Histórico: {len(req.messages)} mensagens")

    # 1. Classificar com histórico
    messages_dict = [m.model_dump() for m in req.messages]
    decision = classify_query(messages_dict)
    tool = decision.get("tool", "none")
    argument = decision.get("argument", "")
    print(f"🧠 Decisão: tool={tool}, argument='{argument}'")

    final_messages = list(messages_dict)
    context_movies = []
    response_override = None
    direct_tmdb_movies = None

    # 2. Tratamento por tipo de ferramenta
    if tool == "followup":
        print("🔁 Followup — vou usar o histórico como contexto.")
        followup_system = (
            "És um assistente especializado em filmes. O utilizador está a fazer uma "
            "pergunta sobre filmes JÁ MENCIONADOS no histórico desta conversa. "
            "Responde em português de Portugal usando APENAS os filmes que já foram "
            "discutidos. Não inventes filmes novos nem inventes detalhes."
        )
        final_messages = [{"role": "system", "content": followup_system}] + messages_dict

    elif tool == "franchise" and argument:
        print(f"🎬 Franchise: buscando saga '{argument}'")
        franchise_movies = search_tmdb_franchise(argument)
        if franchise_movies:
            direct_tmdb_movies = franchise_movies
            count = len(franchise_movies)
            response_override = (
                f"Encontrei {count} filmes da saga **{argument}** no TMDB! "
                f"Podes ver todos na galeria ao lado."
            )
        else:
            print(f"⚠️  Franchise não encontrada — fallback para search: '{argument}'")
            api_data = call_movie_api_cached("search", f"{argument} film")
            if api_data:
                results = api_data.get("results", [])
                if results:
                    context_movies = results[:5]
                    context = format_context(api_data, "search")
                    final_messages = [{"role": "user", "content": build_final_prompt(last_user, context)}]
                else:
                    response_override = f"Não encontrei nenhuma saga nem filmes relacionados com '{argument}' no catálogo."

    elif tool == "more_like" and argument:
        print(f"🎬 More-like: buscando filmes parecidos com '{argument}'")
        api_data = find_movie_and_search_similar(argument)
        if api_data and api_data.get("results"):
            context_movies = api_data.get("results", [])[:5]
            ref_title = api_data.get("reference", {}).get("title", argument)
            context = format_context(api_data, "more_like")
            print(f"📚 Contexto recuperado ({len(context)} chars)")
            enriched_prompt = f"""És um assistente especializado em filmes. Respondes em português de Portugal.

O utilizador quer mais filmes parecidos com **{ref_title}**. Encontrei estes no catálogo:

{context}

REGRAS:
1. Recomenda 2-3 filmes da lista que sejam genuinamente parecidos com {ref_title}.
2. NÃO recomendes o {ref_title} (já foi mencionado).
3. Para cada filme: título, ano, realizador, e o que tem em comum com {ref_title}.
4. Sê natural e conversacional.

PERGUNTA: {last_user}

RESPOSTA:"""
            final_messages = [{"role": "user", "content": enriched_prompt}]
        else:
            print(f"⚠️  Não foi possível encontrar referência para '{argument}'")

    elif tool == "filter_combined" and argument:
        print(f"🎯 Filter combined: {argument}")
        api_data = call_movie_api_cached(tool, argument)
        if api_data:
            results = api_data.get("results", [])
            context_movies = results[:5]
            print(f"🎬 Filmes encontrados: {api_data.get('count', 0)}")
            if not results:
                final_messages = [{"role": "user", "content": (
                    f"O utilizador perguntou: '{last_user}'. "
                    f"Pesquisei no catálogo com os filtros pedidos mas não encontrei filmes. "
                    f"Responde em português de Portugal de forma natural, sem inventar filmes."
                )}]
            else:
                lines = [f"FILMES ENCONTRADOS NO CATÁLOGO ({len(results)} filmes):\n"]
                for i, m in enumerate(results, 1):
                    line = f"{i}. {m.get('title')} ({m.get('year')}) — Realizador: {m.get('director')}"
                    if m.get("genres"):  line += f" | Géneros: {m['genres']}"
                    if m.get("cast"):    line += f" | Cast: {m['cast']}"
                    if m.get("vote_average"): line += f" | Rating: {m['vote_average']}"
                    lines.append(line)
                context = "\n".join(lines)
                print(f"📚 Contexto recuperado ({len(context)} chars)")
                try:
                    filters_dict = json.loads(argument) if isinstance(argument, str) else argument
                except json.JSONDecodeError:
                    filters_dict = {}
                final_messages = [{"role": "user", "content": build_filter_prompt(last_user, context, filters_dict)}]

    elif tool != "none" and argument:
        api_data = call_movie_api_cached(tool, argument)
        if tool == "by_actor" and (not api_data or not api_data.get("results")):
            dir_data = call_movie_api_cached("by_director", argument)
            if dir_data and dir_data.get("results"):
                api_data = dir_data
                tool = "by_director"
                print(f"🔄 Fallback by_actor→by_director para '{argument}'")
        if api_data:
            results = api_data.get("results", [])
            if not results:
                response_override = (
                    "Não encontrei nenhum filme no catálogo que corresponda ao que pediste. "
                    "Tenta reformular a pesquisa — por exemplo, pesquisa pelo tema sem o ator."
                )
            else:
                context_movies = results[:5]
                context = format_context(api_data, tool)
                print(f"📚 Contexto recuperado ({len(context)} chars)")
                final_messages = [{"role": "user", "content": build_final_prompt(last_user, context)}]

    # 3. Preparar TMDB em paralelo (cached) — acontece antes do streaming para que os posters
    #    estejam prontos logo que a Llama termine de gerar
    def _fetch_one(movie):
        data = get_tmdb_data_cached(movie.get("title"), movie.get("year"))
        if data:
            data["director"] = movie.get("director")
            data["genres"]   = movie.get("genres")
            return data
        return None

    if direct_tmdb_movies is not None:
        tmdb_movies = direct_tmdb_movies
    else:
        with ThreadPoolExecutor(max_workers=5) as ex:
            tmdb_movies = [r for r in ex.map(_fetch_one, context_movies) if r is not None]
    if tmdb_movies:
        print(f"🎬 TMDB: {len(tmdb_movies)} posters prontos")

    retrieved = [
        {"title": m.get("title"), "year": m.get("year"),
         "director": m.get("director"), "genres": m.get("genres")}
        for m in context_movies
    ]

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # 4a. Modo streaming — tokens chegam ao browser em tempo real
    if req.stream:
        _tmdb  = tmdb_movies
        _retr  = retrieved
        _final = final_messages
        _ovr   = response_override

        def event_stream():
            cid = completion_id
            ts  = created

            def _chunk(content):
                return f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':ts,'model':'movie-bot','choices':[{'index':0,'delta':{'content':content},'finish_reason':None}]})}\n\n"

            # Primeiro chunk: anuncia role
            yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':ts,'model':'movie-bot','choices':[{'index':0,'delta':{'role':'assistant','content':''},'finish_reason':None}]})}\n\n"

            if _ovr:
                yield _chunk(_ovr)
            else:
                for token in _stream_llama(_final):
                    yield _chunk(token)

            # Chunk final (finish_reason)
            yield f"data: {json.dumps({'id':cid,'object':'chat.completion.chunk','created':ts,'model':'movie-bot','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"

            # Evento especial com filmes + retrieved (o frontend detecta por "type")
            yield f"data: {json.dumps({'type':'movies','movies':_tmdb,'retrieved':_retr})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # 4b. Modo normal (não-streaming)
    if response_override:
        response_text = response_override
        print("⚠️  Resposta directa (sem Llama)")
    else:
        response_text = call_llama(final_messages)
        print(f"✅ Resposta gerada ({len(response_text)} chars)")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": "movie-bot",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "movies": tmdb_movies,
        "retrieved": retrieved,
    }
