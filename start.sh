#!/bin/bash
# ============================================================
# Movie Bot - Script de arranque
# Inicia todos os serviços necessários
# ============================================================

set -e  # parar em caso de erro

PROJECT_DIR="$HOME/projetos/movie-bot"
BACKEND_DIR="$PROJECT_DIR/backend"
LIBRECHAT_DIR="$PROJECT_DIR/LibreChat"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "🎬 Movie Bot — Script de arranque"
echo "=================================="
echo ""

# ------------------------------------------------------------
# 1. Verificar Ollama
# ------------------------------------------------------------
echo "🔍 [1/4] A verificar Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ⚠️  Ollama não está a responder. A tentar arrancar..."
    sudo systemctl start ollama
    sleep 3
fi

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "   ✅ Ollama OK (porta 11434)"
else
    echo "   ❌ Ollama falhou. Arranca manualmente: sudo systemctl start ollama"
    exit 1
fi

# ------------------------------------------------------------
# 2. Arrancar Movie API (porta 8000)
# ------------------------------------------------------------
echo ""
echo "🔍 [2/4] A verificar Movie API..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "   ✅ Movie API já está a correr (porta 8000)"
else
    echo "   🚀 A arrancar Movie API..."
    cd "$BACKEND_DIR"
    source venv/bin/activate
    nohup uvicorn api:app --host 0.0.0.0 --port 8000 \
        > "$LOG_DIR/api.log" 2>&1 &
    echo $! > "$LOG_DIR/api.pid"
    sleep 5

    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "   ✅ Movie API OK (porta 8000) — log em $LOG_DIR/api.log"
    else
        echo "   ❌ Movie API falhou. Vê $LOG_DIR/api.log"
        exit 1
    fi
fi

# ------------------------------------------------------------
# 3. Arrancar Wrapper (porta 8001)
# ------------------------------------------------------------
echo ""
echo "🔍 [3/4] A verificar Wrapper..."
if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "   ✅ Wrapper já está a correr (porta 8001)"
else
    echo "   🚀 A arrancar Wrapper..."
    cd "$BACKEND_DIR"
    source venv/bin/activate
    nohup uvicorn wrapper:app --host 0.0.0.0 --port 8001 \
        > "$LOG_DIR/wrapper.log" 2>&1 &
    echo $! > "$LOG_DIR/wrapper.pid"
    sleep 5

    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "   ✅ Wrapper OK (porta 8001) — log em $LOG_DIR/wrapper.log"
    else
        echo "   ❌ Wrapper falhou. Vê $LOG_DIR/wrapper.log"
        exit 1
    fi
fi

# ------------------------------------------------------------
# 4. Arrancar LibreChat (Docker)
# ------------------------------------------------------------
echo ""
echo "🔍 [4/4] A verificar LibreChat..."
cd "$LIBRECHAT_DIR"
if docker compose ps --format json 2>/dev/null | grep -q "LibreChat.*running"; then
    echo "   ✅ LibreChat já está a correr (porta 3080)"
else
    echo "   🚀 A arrancar LibreChat..."
    docker compose up -d > "$LOG_DIR/librechat-up.log" 2>&1
    echo "   ⏳ A aguardar 15s para LibreChat ficar pronto..."
    sleep 15
    echo "   ✅ LibreChat iniciado (porta 3080)"
fi

# ------------------------------------------------------------
# Status final
# ------------------------------------------------------------
echo ""
echo "🎉 Tudo pronto!"
echo "=================================="
echo "📍 LibreChat:  http://localhost:3080"
echo "📍 Movie API:  http://localhost:8000/docs"
echo "📍 Wrapper:    http://localhost:8001/v1/models"
echo ""
echo "📝 Logs em $LOG_DIR/"
echo "🛑 Para parar tudo: ./stop.sh"
