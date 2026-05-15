# MovieBot RAG

Sistema de recomendação de filmes com IA, usando RAG (*Retrieval-Augmented Generation*).
Faz perguntas em português e recebe recomendações geradas por um LLM local, com posters e ratings.

## Tecnologias

- **LLM local** — Meta-Llama 3.1 8B (via llama-cpp-python, corre em CPU)
- **Vector store** — ChromaDB com 4803 filmes do dataset TMDB 5000
- **Embeddings** — sentence-transformers/all-MiniLM-L6-v2
- **Backend** — FastAPI (Python)
- **Frontend** — HTML/CSS/JS vanilla com streaming SSE

## Setup

### 1. Requisitos

- Python 3.10+
- ~6GB de espaço em disco (modelo LLM)
- 8GB+ RAM recomendado
- Chave da [TMDB API](https://www.themoviedb.org/settings/api) (gratuita)

### 2. Instalar dependências

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar a chave TMDB

Cria o ficheiro `backend/.env`:

```
TMDB_API_KEY=a_tua_chave_aqui
```

### 4. Descarregar o modelo LLM (~4.6GB)

```bash
./download_model.sh
```

### 5. Construir o índice de filmes (só na primeira vez)

```bash
cd backend && source venv/bin/activate
python build_index.py
```

### 6. Arrancar

```bash
./start.sh
```

Abre automaticamente `http://localhost:9000/test.html`.

```bash
./stop.sh   # para parar
```

## Funcionalidades

- Recomendações por tema, género, ator, realizador
- Filtros por época (ex: "filmes de terror dos anos 80")
- Sagas completas (ex: "saga Star Wars")
- "Mais parecidos com X"
- Histórico de conversa contextual
- Posters e ratings via TMDB API
- Watchlist local (localStorage)
- Streaming token-a-token

## Estrutura

```
backend/
  api.py          # API de filmes (ChromaDB)
  wrapper.py      # Wrapper principal (routing, RAG, LLM, TMDB)
  build_index.py  # Constrói o índice ChromaDB
test.html         # Frontend
start.sh          # Arranca todos os serviços
stop.sh           # Para todos os serviços
download_model.sh # Download do modelo LLM
```
