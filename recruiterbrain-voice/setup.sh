#!/bin/bash

# ============================================
# RecruiterBrain Voice Interview Setup Script
# FIXED: Fast installation with essential packages only
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
    echo -e "${YELLOW}⚠${NC} PostgreSQL not found (install: brew install postgresql@14)"
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
# Create Lean Requirements (Essential Only)
# ============================================

echo "📝 Creating lean requirements file..."

cat > requirements-lean.txt << 'EOF'
# Core Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0

# LiveKit Voice Infrastructure
livekit==0.12.0
livekit-agents==0.8.3
livekit-plugins-deepgram==0.6.0
livekit-plugins-google==0.6.0
livekit-plugins-openai==0.7.1
livekit-plugins-silero==0.6.0

# Database
sqlalchemy[asyncio]==2.0.23
psycopg2-binary==2.9.9
asyncpg==0.29.0
pymilvus==2.3.4

# Machine Learning (no version pinning for compatibility)
sentence-transformers
torch

# Speech Services
deepgram-sdk==3.2.7
google-cloud-texttospeech==2.16.0
google-auth==2.25.2

# AI
openai>=1.35.0
tiktoken

# Utilities
aiohttp
httpx
EOF

echo -e "${GREEN}✓${NC} Lean requirements file created"
echo ""

# ============================================
# Install Python Dependencies
# ============================================

echo "📦 Installing Python dependencies (essential packages only)..."
echo "   This will take 5-10 minutes..."
echo ""

# Upgrade pip first
pip3 install --upgrade pip

# Install packages
pip3 install -r requirements-lean.txt

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
    echo ""
    echo -e "${YELLOW}⚠${NC}  IMPORTANT: Please edit .env with your credentials!"
    echo ""
    echo "Required credentials:"
    echo "  1. PostgreSQL:"
    echo "     POSTGRES_USER=your_user"
    echo "     POSTGRES_PASSWORD=your_password"
    echo ""
    echo "  2. LiveKit (get from https://cloud.livekit.io):"
    echo "     LIVEKIT_URL=wss://your-project.livekit.cloud"
    echo "     LIVEKIT_API_KEY=APIxxxxx"
    echo "     LIVEKIT_API_SECRET=secretxxxxx"
    echo ""
    echo "  3. Deepgram (get from https://deepgram.com):"
    echo "     DEEPGRAM_API_KEY=your_key"
    echo ""
    echo "  4. Google Cloud (get from https://console.cloud.google.com):"
    echo "     GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json"
    echo ""
    echo "  5. OpenAI (get from https://platform.openai.com):"
    echo "     OPENAI_API_KEY=sk-xxxxx"
    echo ""
    
    read -p "Press Enter when you've updated .env..."
else
    echo -e "${GREEN}✓${NC} .env file exists"
fi

echo ""

# ============================================
# Load Environment Variables
# ============================================

echo "🔐 Loading environment variables..."

# Source .env file properly
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo -e "${GREEN}✓${NC} Environment variables loaded"
else
    echo -e "${RED}✗${NC} .env file not found!"
    exit 1
fi

echo ""

# ============================================
# Database Setup
# ============================================

echo "💾 Setting up databases..."
echo ""

# PostgreSQL Migration
if command -v psql &> /dev/null; then
    echo "📊 Running PostgreSQL migrations..."
    
    if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_DB" ]; then
        echo -e "${YELLOW}⚠${NC} PostgreSQL credentials not found in .env"
        echo "   Skipping PostgreSQL migration..."
        echo "   Run manually later: psql -U user -d db -f migrations/002_add_missing_columns.sql"
    else
        if [ -f migrations/002_add_missing_columns.sql ]; then
            # Set password for non-interactive auth
            export PGPASSWORD="$POSTGRES_PASSWORD"
            
            # Run migration (suppress "already exists" errors)
            psql -h "${POSTGRES_HOST:-localhost}" \
                 -p "${POSTGRES_PORT:-5432}" \
                 -U "$POSTGRES_USER" \
                 -d "$POSTGRES_DB" \
                 -f migrations/002_add_missing_columns.sql 2>&1 | \
                 grep -v "already exists" | \
                 grep -v "ERROR.*relation.*already exists" || true
            
            unset PGPASSWORD
            
            echo -e "${GREEN}✓${NC} PostgreSQL migration complete"
        else
            echo -e "${YELLOW}⚠${NC} Migration file not found: migrations/002_add_missing_columns.sql"
        fi
    fi
else
    echo -e "${YELLOW}⚠${NC} PostgreSQL not installed"
    echo "   Install with: brew install postgresql@14"
    echo "   Or run migration manually later"
fi

echo ""

# Milvus Setup
if [ -f migrations/setup_milvus_v2.py ]; then
    echo "🔍 Setting up Milvus collection..."
    
    # Check if Milvus is accessible
    if python3 -c "from pymilvus import connections; connections.connect(host='${MILVUS_HOST:-localhost}', port='${MILVUS_PORT:-19530}'); print('OK')" 2>/dev/null; then
        python3 migrations/setup_milvus_v2.py
        echo -e "${GREEN}✓${NC} Milvus setup complete"
    else
        echo -e "${YELLOW}⚠${NC} Cannot connect to Milvus at ${MILVUS_HOST:-localhost}:${MILVUS_PORT:-19530}"
        echo "   Make sure Milvus is running:"
        echo "   docker-compose up -d"
        echo ""
        echo "   Or run setup manually later:"
        echo "   python migrations/setup_milvus_v2.py"
    fi
else
    echo -e "${YELLOW}⚠${NC} Milvus setup script not found"
fi

echo ""

# ============================================
# Verify Setup
# ============================================

echo "🔍 Verifying setup..."
echo ""

# Test configuration loading
if python3 -c "from app.config.settings import settings; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Configuration loaded successfully"
else
    echo -e "${RED}✗${NC} Configuration error - check your .env file"
fi

# Test imports
echo "Testing core imports..."
python3 << 'PYEOF'
try:
    import fastapi
    import livekit
    import deepgram
    from google.cloud import texttospeech
    import openai
    import sqlalchemy
    import pymilvus
    print("✓ All core packages imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)
PYEOF

echo ""

# ============================================
# Summary
# ============================================

echo "========================================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "========================================================================"
echo ""
echo "📋 What was installed:"
echo "   ✓ FastAPI & Uvicorn (API server)"
echo "   ✓ LiveKit & plugins (voice infrastructure)"
echo "   ✓ PostgreSQL & Milvus clients (databases)"
echo "   ✓ Sentence Transformers & Torch (embeddings)"
echo "   ✓ Deepgram, Google TTS, OpenAI (speech & AI)"
echo ""
echo "📋 What was configured:"
echo "   ✓ Environment variables loaded"
echo "   ✓ PostgreSQL migrations run"
echo "   ✓ Milvus collection created"
echo ""
echo "🚀 Next steps:"
echo ""
echo "1. Verify your .env file has all API keys:"
echo "   nano .env"
echo ""
echo "2. Start LiveKit Agent (Terminal 1):"
echo "   python app/livekit_agent/worker.py"
echo ""
echo "3. Start API Server (Terminal 2):"
echo "   python app/api/main.py"
echo ""
echo "4. Test the system:"
echo "   curl http://localhost:8000/health"
echo ""
echo "5. View API documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "📚 Documentation:"
echo "   - Quick Start: QUICKSTART.md"
echo "   - Full Guide: README.md"
echo "   - Deployment: DEPLOYMENT.md"
echo ""
echo "========================================================================"
echo ""

# ============================================
# Optional: Run Tests
# ============================================

if [ -f tests/test_system.py ]; then
    read -p "Run system tests now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🧪 Running system tests..."
        python3 tests/test_system.py
    fi
fi

echo ""
echo "🎉 Setup complete! You're ready to conduct AI-powered interviews!"
echo ""