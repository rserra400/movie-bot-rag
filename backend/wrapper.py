"""
Wrapper OpenAI-compatible para o LibreChat.
Recebe pedidos de chat, decide se usa RAG, e devolve resposta da Llama.
Suporta histórico de conversa para perguntas de followup.
"""

import json
import time
import uuid
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.1:8b"
MOVIE_API = "http://localhost:8000"

app = FastAPI(title="Movie Bot Wrapper", version="0.3.0")


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


# ---------------------------------------------------------------------------
# Lógica de decisão de ferramentas (com histórico)
# ---------------------------------------------------------------------------
ROUTER_PROMPT = """És um assistente que decide qual ferramenta usar para responder a uma pergunta sobre filmes, e prepara o argumento de pesquisa. Tens acesso ao HISTÓRICO da conversa para resolver referências contextuais.

Ferramentas disponíveis:
- search: pesquisa semântica. Usa para "recomenda-me", "filme parecido com X (NOME NOVO)", "algo sobre Y"
- by_actor: lista filmes de um ator (sem outros filtros).
- by_director: lista filmes de um realizador (sem outros filtros).
- by_title: procura informação sobre um filme específico pelo nome.
- filter_combined: COMBINA múltiplos filtros (realizador, ator, género, ano). Usa quando o utilizador junta dois ou mais critérios.
- more_like: NOVA pesquisa baseada num filme JÁ MENCIONADO no histórico.
- followup: pergunta sobre filmes JÁ MENCIONADOS, sem nova pesquisa.
- none: pergunta não é sobre filmes.

REGRAS:
- Se a pergunta combina dois ou mais critérios (realizador+género, ator+ano, etc.), usa "filter_combined".
- Para filter_combined, o argument é um JSON com os filtros disponíveis: director, actor, genre, year_from, year_to.
- Géneros em INGLÊS: Action, Comedy, Drama, Horror, Romance, Science Fiction, Thriller, Animation, Fantasy, Crime, Mystery, Adventure, etc.

Responde APENAS com JSON (sem markdown):
{"tool": "nome_ferramenta", "argument": "argumento_ou_JSON"}

EXEMPLOS:

Histórico: (vazio)
Pergunta: "Recomenda-me um filme com robôs"
Resposta: {"tool": "search", "argument": "movie about robots artificial intelligence androids"}

Histórico: (vazio)
Pergunta: "Filmes do Spielberg"
Resposta: {"tool": "by_director", "argument": "Steven Spielberg"}

Histórico: (vazio)
Pergunta: "Filmes do Nolan que sejam ficção científica"
Resposta: {"tool": "filter_combined", "argument": "{\\"director\\": \\"Christopher Nolan\\", \\"genre\\": \\"Science Fiction\\"}"}

Histórico: (vazio)
Pergunta: "Filmes de ação dos anos 80"
Resposta: {"tool": "filter_combined", "argument": "{\\"genre\\": \\"Action\\", \\"year_from\\": 1980, \\"year_to\\": 1989}"}

Histórico: (vazio)
Pergunta: "Comédias com a Sandra Bullock"
Resposta: {"tool": "filter_combined", "argument": "{\\"actor\\": \\"Sandra Bullock\\", \\"genre\\": \\"Comedy\\"}"}

Histórico: (vazio)
Pergunta: "Filmes de Spielberg dos anos 90"
Resposta: {"tool": "filter_combined", "argument": "{\\"director\\": \\"Steven Spielberg\\", \\"year_from\\": 1990, \\"year_to\\": 1999}"}

Histórico: (vazio)
Pergunta: "Filmes de terror dos anos 2000"
Resposta: {"tool": "filter_combined", "argument": "{\\"genre\\": \\"Horror\\", \\"year_from\\": 2000, \\"year_to\\": 2009}"}

Histórico:
User: Filmes do Nolan
Assistant: [recomendou Interstellar, Inception, Dark Knight]
Pergunta: "Qual é o mais cerebral?"
Resposta: {"tool": "followup", "argument": ""}

Histórico:
User: Filmes do Nolan
Assistant: [recomendou Interstellar, Inception, Dark Knight]
Pergunta: "Quero mais filmes parecidos com o Inception"
Resposta: {"tool": "more_like", "argument": "Inception"}

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
        "options": {"temperature": 0.0},
        "format": "json",
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["message"]["content"]
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
    return f"""És um assistente especializado em filmes. Respondes em português de Portugal.

REGRAS CRÍTICAS:
1. SÓ podes mencionar filmes que aparecem na lista abaixo. NÃO inventes filmes.
2. Recomenda os 2-3 filmes da lista que melhor respondem à pergunta. Se houver menos relevantes, recomenda menos.
3. Para cada filme inclui: título, ano, realizador, e uma frase a explicar porque é relevante.
4. Se nenhum filme da lista for relevante, diz-o honestamente: "Os filmes que encontrei no catálogo não correspondem bem ao que pediste."
5. Sê natural e conversacional.

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

    return f"""És um assistente especializado em filmes. Respondes em português de Portugal.

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


def call_llama(messages: list[dict]) -> str:
    """Chama a Llama via Ollama e devolve a resposta de texto."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7},
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Endpoints OpenAI-compatible
# ---------------------------------------------------------------------------

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

    elif tool == "more_like" and argument:
        print(f"🎬 More-like: buscando filmes parecidos com '{argument}'")
        api_data = find_movie_and_search_similar(argument)
        if api_data and api_data.get("results"):
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
            final_messages[-1] = {"role": "user", "content": enriched_prompt}
        else:
            print(f"⚠️  Não foi possível encontrar referência para '{argument}'")

    elif tool == "filter_combined" and argument:
        print(f"🎯 Filter combined: {argument}")
        api_data = call_movie_api(tool, argument)
        if api_data:
            results = api_data.get("results", [])
            print(f"🎬 Filmes encontrados: {api_data.get('count', 0)} (mostrando até 10)")
            if not results:
                # Sem resultados — informar a Llama
                final_messages[-1] = {
                    "role": "user",
                    "content": (
                        f"O utilizador perguntou: '{last_user}'. "
                        f"Pesquisei no catálogo com os filtros pedidos mas não encontrei "
                        f"filmes que correspondessem. Responde-lhe a explicar isso em "
                        f"português de Portugal de forma natural, sem inventar filmes."
                    ),
                }
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

                final_messages[-1] = {
                    "role": "user",
                    "content": build_filter_prompt(last_user, context, filters_dict),
                }

    elif tool != "none" and argument:
        api_data = call_movie_api(tool, argument)
        if api_data:
            context = format_context(api_data, tool)
            print(f"📚 Contexto recuperado ({len(context)} chars)")
            final_messages[-1] = {
                "role": "user",
                "content": build_final_prompt(last_user, context),
            }

    # 3. Chamar a Llama
    response_text = call_llama(final_messages)
    print(f"✅ Resposta gerada ({len(response_text)} chars)")

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
    }
