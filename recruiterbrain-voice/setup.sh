#!/bin/bash

# ============================================
# RecruiterBrain Voice Interview Setup Script
# ============================================

set -e  # Exit on error

echo "========================================================================"
echo "RecruiterBrain Voice Interview System - Setup"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================
# Check Prerequisites
# ============================================

echo "📋 Checking prerequisites..."
echo ""

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3.10+ is required"
    exit 1
fi

# PostgreSQL
if command -v psql &> /dev/null; then
    PG_VERSION=$(psql --version | cut -d' ' -f3)
    echo -e "${GREEN}✓${NC} PostgreSQL: $PG_VERSION"
else
    echo -e "${YELLOW}⚠${NC} PostgreSQL not found (install: sudo apt install postgresql)"
fi

# pip
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} pip: $(pip3 --version | cut -d' ' -f2)"
else
    echo -e "${RED}✗${NC} pip is required"
    exit 1
fi

echo ""

# ============================================
# Install Python Dependencies
# ============================================

echo "📦 Installing Python dependencies..."
echo ""

pip3 install -r requirements.txt

echo ""
echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# ============================================
# Environment Configuration
# ============================================

if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env file"
    echo -e "${YELLOW}⚠${NC}  Please edit .env with your credentials before continuing"
    echo ""
    echo "Required credentials:"
    echo "  - PostgreSQL (POSTGRES_USER, POSTGRES_PASSWORD)"
    echo "  - LiveKit (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)"
    echo "  - Deepgram (DEEPGRAM_API_KEY)"
    echo "  - Google Cloud (GOOGLE_APPLICATION_CREDENTIALS)"
    echo "  - OpenAI (OPENAI_API_KEY)"
    echo ""
    
    read -p "Press Enter when you've updated .env..."
else
    echo -e "${GREEN}✓${NC} .env file exists"
fi

echo ""

# ============================================
# Database Setup
# ============================================

echo "💾 Setting up databases..."
echo ""

# Check if PostgreSQL is accessible
if command -v psql &> /dev/null; then
    echo "Running PostgreSQL migrations..."
    
    # Source .env for database credentials
    export $(grep -v '^#' .env | xargs)
    
    # Run migrations
    if [ -f migrations/002_add_missing_columns.sql ]; then
        psql -U $POSTGRES_USER -d $POSTGRES_DB -f migrations/002_add_missing_columns.sql
        echo -e "${GREEN}✓${NC} PostgreSQL migrations complete"
    else
        echo -e "${YELLOW}⚠${NC} PostgreSQL migration file not found"
    fi
else
    echo -e "${YELLOW}⚠${NC} PostgreSQL not accessible, skipping migrations"
    echo "   Run manually: psql -U user -d db -f migrations/002_add_missing_columns.sql"
fi

echo ""

# Milvus setup
echo "Setting up Milvus collection..."
python3 migrations/setup_milvus_v2.py

echo ""

# ============================================
# Verify Setup
# ============================================

echo "🔍 Verifying setup..."
echo ""

# Test database connection
python3 -c "
from app.services.database import initialize_database_connections
import asyncio

async def test():
    success = await initialize_database_connections()
    if success:
        print('${GREEN}✓${NC} Database connections working')
    else:
        print('${RED}✗${NC} Database connection failed')
        exit(1)

asyncio.run(test())
" && echo -e "${GREEN}✓${NC} Database verification passed" || echo -e "${RED}✗${NC} Database verification failed"

echo ""

# ============================================
# Summary
# ============================================

echo "========================================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start LiveKit Agent:"
echo "   python app/livekit_agent/worker.py"
echo ""
echo "2. Start API Server (in another terminal):"
echo "   python app/api/main.py"
echo ""
echo "3. Test the API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "4. View API docs:"
echo "   http://localhost:8000/docs"
echo ""
echo "========================================================================"
echo ""
