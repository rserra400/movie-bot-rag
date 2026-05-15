"""
Wrapper OpenAI-compatible para o Movie Bot.
Recebe pedidos de chat, decide se usa RAG, e devolve resposta da Llama via llama-cpp.
Suporta histórico de conversa para perguntas de followup.
"""

import json
import os
import time
import uuid
import requests
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
# Lógica de decisão de ferramentas (com histórico)
# ---------------------------------------------------------------------------
ROUTER_PROMPT = """Decide a ferramenta para responder sobre filmes. Responde APENAS com JSON: {"tool": "...", "argument": "..."}

Ferramentas:
- search: pesquisa semântica (recomendações, descrições, temas, combinações ator+tema)
- by_actor: filmes de um ator — argumento deve ser APENAS o nome do ator, nada mais
- by_director: filmes de um realizador — argumento deve ser APENAS o nome do realizador
- by_title: informação sobre um filme específico
- filter_combined: múltiplos filtros em JSON {"director","actor","genre","year_from","year_to"}
- franchise: TODOS os filmes de uma saga/franquia — argumento é o nome da saga em inglês
- more_like: filmes parecidos com um já mencionado no histórico
- followup: APENAS quando referencia EXPLICITAMENTE filmes já mencionados ("esse", "aquele", "o primeiro", "dos que recomendaste")
- none: não é sobre filmes

REGRAS CRÍTICAS:
1. Se a pergunta combina ator/realizador com tema, género ou característica (ex: "Tom Cruise com drogas", "Spielberg de terror") → usa SEMPRE "search" com ator+tema juntos.
2. by_actor e by_director: argumento é APENAS o nome da pessoa, nunca inclui tema, género ou descrição.
3. Se a pergunta introduz NOVO tema — mesmo com histórico — usa "search" ou outro. "followup" só para filmes específicos já citados.
4. QUERY EXPANSION — Se o utilizador referencia um jogo, livro, série, música ou outro conteúdo NÃO cinematográfico, converte o argumento para o tema/género cinematográfico equivalente. Nunca uses o nome do jogo/livro como argumento de pesquisa.

Géneros em inglês: Action, Comedy, Drama, Horror, Thriller, Science Fiction, Animation, Fantasy, Crime, Adventure, Romance, Mystery.

Exemplos:
"Recomenda filmes de aventura" → {"tool":"search","argument":"adventure films"}
"Quero filmes sobre drogas" → {"tool":"search","argument":"films about drugs addiction narcotics"}
"Filmes do Tom Cruise" → {"tool":"by_actor","argument":"Tom Cruise"}
"Tom Cruise com drogas" → {"tool":"search","argument":"Tom Cruise drug film"}
"Filmes do Nolan" → {"tool":"by_director","argument":"Christopher Nolan"}
"Spielberg filmes de terror" → {"tool":"search","argument":"Spielberg horror film"}
"Filmes de terror dos anos 80" → {"tool":"filter_combined","argument":"{\\"genre\\":\\"Horror\\",\\"year_from\\":1980,\\"year_to\\":1989}"}
"Mais parecidos com o Inception" → {"tool":"more_like","argument":"Inception"}
"Qual o mais popular dos que recomendaste?" → {"tool":"followup","argument":""}
"Filmes tipo Forza Horizon" → {"tool":"search","argument":"car racing driving speed automotive action films"}
"Filmes como Need for Speed" → {"tool":"search","argument":"street racing illegal car chase action films"}
"Filmes parecidos com GTA" → {"tool":"search","argument":"crime gangster heist urban open world action films"}
"Filmes tipo Minecraft" → {"tool":"search","argument":"adventure survival fantasy exploration films"}
"Filmes tipo Harry Potter livro" → {"tool":"search","argument":"magic wizard school fantasy adventure films"}
"Todos os filmes do Velocidade Furiosa" → {"tool":"franchise","argument":"Fast and Furious"}
"Saga completa do Star Wars" → {"tool":"franchise","argument":"Star Wars"}
"Todos os filmes do Harry Potter" → {"tool":"franchise","argument":"Harry Potter"}
"Filmes todos da saga Marvel Vingadores" → {"tool":"franchise","argument":"Avengers"}

CONTEXTO ATUAL:
"""


def classify_query(messages: list[dict]) -> dict:
    """Pede à Llama para decidir que ferramenta usar, considerando o histórico."""
    history_lines = []
    for m in messages[:-1]:
        role = m["role"]
        content = m["content"]
        if role == "assistant" and len(content) > 300:
            content = content[:300] + "..."
        history_lines.append(f"{role.capitalize()}: {content}")

    history = "\n".join(history_lines) if history_lines else "(conversa nova, sem histórico)"
    last_question = messages[-1]["content"]

    full_prompt = ROUTER_PROMPT + f"""
Histórico:
{history}

Pergunta atual: {last_question}
Resposta:"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 80,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(LLAMA_CHAT_URL, json=payload, timeout=90)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"tool": "none", "argument": ""}


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
            r = requests.post(
                f"{MOVIE_API}/search",
                json={"query": argument, "top_k": 8},
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
        "max_tokens": 450,
    }
    r = requests.post(LLAMA_CHAT_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Endpoints OpenAI-compatible
# ---------------------------------------------------------------------------

@app.get("/movies/popular")
def popular_movies():
    """Filmes populares enriquecidos com posters TMDB."""
    try:
        r = requests.get(f"{MOVIE_API}/movies/popular?limit=12", timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        enriched = []
        for movie in results:
            tmdb = get_tmdb_data(movie.get("title"), movie.get("year"))
            enriched.append({
                "title": movie.get("title"),
                "year": movie.get("year"),
                "director": movie.get("director"),
                "genres": movie.get("genres"),
                "vote_average": movie.get("vote_average"),
                "poster_url": tmdb.get("poster_url") if tmdb else None,
                "tmdb_rating": tmdb.get("tmdb_rating") if tmdb else None,
                "tmdb_id": tmdb.get("tmdb_id") if tmdb else None,
                "overview": tmdb.get("overview") if tmdb else None,
            })
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

    tmdb_movies = []
    for movie in results[:5]:
        data = get_tmdb_data(movie.get("title"), movie.get("year"))
        if data:
            tmdb_movies.append(data)

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
            # Fallback: pesquisa semântica com o nome como tema
            print(f"⚠️  Franchise não encontrada — fallback para search: '{argument}'")
            api_data = call_movie_api("search", f"{argument} film")
            if api_data:
                results = api_data.get("results", [])
                if results:
                    context_movies = results[:5]
                    context = format_context(api_data, "search")
                    print(f"📚 Fallback context: {len(context)} chars")
                    final_messages = [{
                        "role": "user",
                        "content": build_final_prompt(last_user, context),
                    }]
                else:
                    response_override = (
                        f"Não encontrei nenhuma saga nem filmes relacionados com '{argument}' "
                        f"no catálogo."
                    )

    elif tool == "more_like" and argument:
        print(f"🎬 More-like: buscando filmes parecidos com '{argument}'")
        api_data = find_movie_and_search_similar(argument)
        if api_data and api_data.get("results"):
            context_movies = api_data.get("results", [])[:5]
            ref = api_data.get("reference", {})
            ref_title = ref.get("title", argument)
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
        api_data = call_movie_api(tool, argument)
        if api_data:
            results = api_data.get("results", [])
            context_movies = results[:5]
            print(f"🎬 Filmes encontrados: {api_data.get('count', 0)} (mostrando até 10)")
            if not results:
                # Sem resultados — informar a Llama
                final_messages = [{
                    "role": "user",
                    "content": (
                        f"O utilizador perguntou: '{last_user}'. "
                        f"Pesquisei no catálogo com os filtros pedidos mas não encontrei "
                        f"filmes que correspondessem. Responde-lhe a explicar isso em "
                        f"português de Portugal de forma natural, sem inventar filmes."
                    ),
                }]
            else:
                # Construir contexto detalhado com TODOS os filmes encontrados (até 10)
                lines = [f"FILMES ENCONTRADOS NO CATÁLOGO ({len(results)} filmes):\n"]
                for i, m in enumerate(results, 1):
                    line = (
                        f"{i}. {m.get('title')} ({m.get('year')}) "
                        f"— Realizador: {m.get('director')}"
                    )
                    if m.get("genres"):
                        line += f" | Géneros: {m['genres']}"
                    if m.get("cast"):
                        line += f" | Cast: {m['cast']}"
                    if m.get("vote_average"):
                        line += f" | Rating: {m['vote_average']}"
                    lines.append(line)
                context = "\n".join(lines)
                print(f"📚 Contexto recuperado ({len(context)} chars)")

                try:
                    filters_dict = json.loads(argument) if isinstance(argument, str) else argument
                except json.JSONDecodeError:
                    filters_dict = {}

                final_messages = [{
                    "role": "user",
                    "content": build_filter_prompt(last_user, context, filters_dict),
                }]

    elif tool != "none" and argument:
        api_data = call_movie_api(tool, argument)
        if api_data:
            results = api_data.get("results", [])
            if not results:
                response_override = (
                    "Não encontrei nenhum filme no catálogo que corresponda ao que pediste. "
                    "Tenta reformular a pesquisa — por exemplo, pesquisa pelo tema sem o ator, "
                    "ou pelo ator sem o tema."
                )
            else:
                context_movies = results[:5]
                context = format_context(api_data, tool)
                print(f"📚 Contexto recuperado ({len(context)} chars)")
                final_messages = [{
                    "role": "user",
                    "content": build_final_prompt(last_user, context),
                }]

    # 3. Chamar a Llama (ou usar resposta directa se não há resultados)
    if response_override:
        response_text = response_override
        print(f"⚠️  Sem resultados — resposta directa (sem chamar Llama)")
    else:
        response_text = call_llama(final_messages)
        print(f"✅ Resposta gerada ({len(response_text)} chars)")

    # 4. Enriquecer com dados TMDB (ou usar franchise directamente)
    if direct_tmdb_movies is not None:
        tmdb_movies = direct_tmdb_movies
    else:
        tmdb_movies = []
        for movie in context_movies:
            data = get_tmdb_data(movie.get("title"), movie.get("year"))
            if data:
                data["director"] = movie.get("director")
                data["genres"]   = movie.get("genres")
                tmdb_movies.append(data)
    if tmdb_movies:
        print(f"🎬 TMDB: {len(tmdb_movies)} posters encontrados")

    retrieved = [
        {"title": m.get("title"), "year": m.get("year"),
         "director": m.get("director"), "genres": m.get("genres")}
        for m in context_movies
    ]

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # 4a. Modo streaming
    if req.stream:
        def event_stream():
            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "movie-bot",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(first)}\n\n"

            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "movie-bot",
                "choices": [{
                    "index": 0,
                    "delta": {"content": response_text},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"

            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "movie-bot",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # 4b. Modo normal
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
