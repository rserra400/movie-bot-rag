#!/bin/bash
# ============================================================
# Movie Bot - Script de arranque
# ============================================================

set -e

PROJECT_DIR="$HOME/projetos/movie-bot"
BACKEND_DIR="$PROJECT_DIR/backend"
MODELS_DIR="$PROJECT_DIR/models"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "Movie Bot — Script de arranque"
echo "=================================="
echo ""

# ------------------------------------------------------------
# 1. Arrancar llama-cpp server (porta 8080)
# ------------------------------------------------------------
echo "[1/3] A verificar llama-cpp server..."
if curl -s http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "   OK — llama-cpp server ja esta a correr (porta 8080)"
else
    MODEL_FILE="$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    if [ ! -f "$MODEL_FILE" ]; then
        echo "   ERRO — modelo nao encontrado: $MODEL_FILE"
        echo "   Faz o download primeiro: ./download_model.sh"
        exit 1
    fi
    echo "   A arrancar llama-cpp server..."
    cd "$BACKEND_DIR"
    source venv/bin/activate
    nohup python -m llama_cpp.server \
        --model "$MODEL_FILE" \
        --host 0.0.0.0 \
        --port 8080 \
        --n_ctx 2048 \
        --n_threads 8 \
        --n_batch 512 \
        --chat_format llama-3 \
        > "$LOG_DIR/llama.log" 2>&1 &
    echo $! > "$LOG_DIR/llama.pid"
    echo "   A aguardar llama-cpp ficar pronto (pode demorar 30-60s)..."
    for i in $(seq 1 24); do
        sleep 5
        if curl -s http://localhost:8080/v1/models > /dev/null 2>&1; then
            echo "   OK — llama-cpp server pronto (porta 8080)"
            break
        fi
        echo "   ... a aguardar ($((i*5))s)"
        if [ $i -eq 24 ]; then
            echo "   ERRO — llama-cpp nao arrancou. Ve $LOG_DIR/llama.log"
            exit 1
        fi
    done
fi

# ------------------------------------------------------------
# 2. Arrancar Movie API (porta 8000)
# ------------------------------------------------------------
echo ""
echo "[2/3] A verificar Movie API..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "   OK — Movie API ja esta a correr (porta 8000)"
else
    echo "   A arrancar Movie API (a carregar modelo de embeddings)..."
    cd "$BACKEND_DIR"
    source venv/bin/activate
    nohup uvicorn api:app --host 0.0.0.0 --port 8000 \
        > "$LOG_DIR/api.log" 2>&1 &
    echo $! > "$LOG_DIR/api.pid"
    for i in $(seq 1 8); do
        sleep 5
        if curl -s http://localhost:8000/ > /dev/null 2>&1; then
            echo "   OK — Movie API pronta (porta 8000)"
            break
        fi
        secs=$((i*5))
        echo "   ... a aguardar (${secs}s)"
        if [ $i -eq 8 ]; then
            echo "   ERRO — Movie API falhou. Ve $LOG_DIR/api.log"
            exit 1
        fi
    done
fi

# ------------------------------------------------------------
# 3. Arrancar Wrapper (porta 8001)
# ------------------------------------------------------------
echo ""
echo "[3/3] A verificar Wrapper..."
if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "   OK — Wrapper ja esta a correr (porta 8001)"
else
    echo "   A arrancar Wrapper..."
    cd "$BACKEND_DIR"
    source venv/bin/activate
    nohup uvicorn wrapper:app --host 0.0.0.0 --port 8001 \
        > "$LOG_DIR/wrapper.log" 2>&1 &
    echo $! > "$LOG_DIR/wrapper.pid"
    sleep 5
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "   OK — Wrapper pronto (porta 8001)"
    else
        echo "   ERRO — Wrapper falhou. Ve $LOG_DIR/wrapper.log"
        exit 1
    fi
fi

# ------------------------------------------------------------
# 4. Arrancar servidor HTTP para o site de teste (porta 9000)
# ------------------------------------------------------------
echo ""
echo "[4/4] A verificar servidor HTTP (porta 9000)..."
if curl -s http://localhost:9000/test.html > /dev/null 2>&1; then
    echo "   OK — servidor HTTP ja esta a correr (porta 9000)"
else
    echo "   A arrancar servidor HTTP..."
    cd "$PROJECT_DIR"
    nohup python3 -m http.server 9000 \
        > "$LOG_DIR/http.log" 2>&1 &
    echo $! > "$LOG_DIR/http.pid"
    sleep 2
    if curl -s http://localhost:9000/test.html > /dev/null 2>&1; then
        echo "   OK — servidor HTTP pronto (porta 9000)"
    else
        echo "   AVISO — servidor HTTP falhou. Ve $LOG_DIR/http.log"
    fi
fi

# ------------------------------------------------------------
# Abrir o Movie Bot no browser
# ------------------------------------------------------------
echo ""
echo "   A abrir o Movie Bot no browser..."
xdg-open http://localhost:9000/test.html > /dev/null 2>&1 &

# ------------------------------------------------------------
# Status final
# ------------------------------------------------------------
echo ""
echo "Tudo pronto!"
echo "=================================="
echo "Movie Bot:  http://localhost:9000/test.html"
echo "Movie API:  http://localhost:8000/docs"
echo "Wrapper:    http://localhost:8001/v1/models"
echo "LLM:        http://localhost:8080/v1/models"
echo ""
echo "Logs em $LOG_DIR/"
echo "Para parar tudo: ./stop.sh"
