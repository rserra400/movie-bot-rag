#!/bin/bash
# ============================================================
# Movie Bot - Script de paragem
# ============================================================

LOG_DIR="$HOME/projetos/movie-bot/logs"

echo "Movie Bot — A parar servicos..."
echo "=================================="

# 1. Parar Wrapper
echo "A parar Wrapper..."
if [ -f "$LOG_DIR/wrapper.pid" ]; then
    kill $(cat "$LOG_DIR/wrapper.pid") 2>/dev/null && echo "   OK — Wrapper parado" || echo "   Wrapper ja estava parado"
    rm "$LOG_DIR/wrapper.pid"
else
    pkill -f "uvicorn wrapper:app" 2>/dev/null && echo "   OK — Wrapper parado (via pkill)" || echo "   Wrapper nao encontrado"
fi

# 2. Parar Movie API
echo "A parar Movie API..."
if [ -f "$LOG_DIR/api.pid" ]; then
    kill $(cat "$LOG_DIR/api.pid") 2>/dev/null && echo "   OK — Movie API parada" || echo "   Movie API ja estava parada"
    rm "$LOG_DIR/api.pid"
else
    pkill -f "uvicorn api:app" 2>/dev/null && echo "   OK — Movie API parada (via pkill)" || echo "   Movie API nao encontrada"
fi

# 3. Parar llama-cpp server
echo "A parar llama-cpp server..."
if [ -f "$LOG_DIR/llama.pid" ]; then
    kill $(cat "$LOG_DIR/llama.pid") 2>/dev/null && echo "   OK — llama-cpp parado" || echo "   llama-cpp ja estava parado"
    rm "$LOG_DIR/llama.pid"
else
    pkill -f "llama_cpp.server" 2>/dev/null && echo "   OK — llama-cpp parado (via pkill)" || echo "   llama-cpp nao encontrado"
fi

# 4. Parar servidor HTTP
echo "A parar servidor HTTP..."
if [ -f "$LOG_DIR/http.pid" ]; then
    kill $(cat "$LOG_DIR/http.pid") 2>/dev/null && echo "   OK — servidor HTTP parado" || echo "   servidor HTTP ja estava parado"
    rm "$LOG_DIR/http.pid"
else
    pkill -f "http.server 9000" 2>/dev/null && echo "   OK — servidor HTTP parado (via pkill)" || echo "   servidor HTTP nao encontrado"
fi

echo ""
echo "Tudo parado."
