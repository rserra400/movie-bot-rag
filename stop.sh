#!/bin/bash
# ============================================================
# Movie Bot - Script de paragem
# Pára todos os serviços
# ============================================================

PROJECT_DIR="$HOME/projetos/movie-bot"
LIBRECHAT_DIR="$PROJECT_DIR/LibreChat"
LOG_DIR="$PROJECT_DIR/logs"

echo "🛑 Movie Bot — A parar serviços..."
echo "=================================="

# 1. Parar LibreChat (Docker)
echo "🐳 A parar LibreChat..."
cd "$LIBRECHAT_DIR"
docker compose down > /dev/null 2>&1
echo "   ✅ LibreChat parado"

# 2. Parar Wrapper
echo "🔌 A parar Wrapper..."
if [ -f "$LOG_DIR/wrapper.pid" ]; then
    kill $(cat "$LOG_DIR/wrapper.pid") 2>/dev/null && echo "   ✅ Wrapper parado" || echo "   ⚠️  Wrapper já estava parado"
    rm "$LOG_DIR/wrapper.pid"
else
    # fallback: matar pelo nome
    pkill -f "uvicorn wrapper:app" 2>/dev/null && echo "   ✅ Wrapper parado (via pkill)" || echo "   ⚠️  Wrapper não encontrado"
fi

# 3. Parar Movie API
echo "🎬 A parar Movie API..."
if [ -f "$LOG_DIR/api.pid" ]; then
    kill $(cat "$LOG_DIR/api.pid") 2>/dev/null && echo "   ✅ Movie API parada" || echo "   ⚠️  Movie API já estava parada"
    rm "$LOG_DIR/api.pid"
else
    pkill -f "uvicorn api:app" 2>/dev/null && echo "   ✅ Movie API parada (via pkill)" || echo "   ⚠️  Movie API não encontrada"
fi

echo ""
echo "✅ Tudo parado."
echo "💡 Ollama continua a correr como serviço do sistema."
echo "   Para parar: sudo systemctl stop ollama"
