#!/usr/bin/env python3
"""Debug script to check all environment variables and connections."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_env():
    """Check all required environment variables."""
    print("=" * 60)
    print("ENVIRONMENT VARIABLES CHECK")
    print("=" * 60)
    
    required = {
        "BEY_API_KEY": "Beyond Presence API",
        "OPENAI_API_KEY": "OpenAI API for LLM extraction",
        "HF_TOKEN": "HuggingFace for embeddings (prevents rate limiting)",
        "DATABASE_URL": "PostgreSQL connection",
        "MILVUS_HOST": "Milvus vector database host",
        "MILVUS_URI": "Milvus URI (alternative to host)",
        "GCS_BUCKET": "GCS bucket for call history",
    }
    
    optional = {
        "MILVUS_TOKEN": "Milvus authentication token",
        "MILVUS_PORT": "Milvus port (default 19530)",
        "MILVUS_COLLECTION": "Milvus collection name",
        "ADMIN_API_KEY": "Admin API authentication",
    }
    
    print("\n📋 REQUIRED:")
    all_set = True
    for key, desc in required.items():
        value = os.getenv(key)
        if value:
            # Mask sensitive values
            if "KEY" in key or "TOKEN" in key or "URL" in key or "PASSWORD" in key:
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            else:
                masked = value
            print(f"  ✅ {key}: {masked}")
        else:
            print(f"  ❌ {key}: NOT SET - {desc}")
            all_set = False
    
    print("\n📋 OPTIONAL:")
    for key, desc in optional.items():
        value = os.getenv(key)
        if value:
            if "KEY" in key or "TOKEN" in key:
                masked = value[:8] + "..." if len(value) > 8 else "***"
            else:
                masked = value
            print(f"  ✅ {key}: {masked}")
        else:
            print(f"  ⚪ {key}: not set - {desc}")
    
    return all_set


def check_database():
    """Test database connection."""
    print("\n" + "=" * 60)
    print("DATABASE CONNECTION TEST")
    print("=" * 60)
    
    try:
        import psycopg
        from psycopg.rows import dict_row
        
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            print("  ❌ DATABASE_URL not set")
            return False
        
        print(f"  Connecting to database...")
        with psycopg.connect(db_url, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # Test connection
                cur.execute("SELECT 1")
                print("  ✅ Database connection OK")
                
                # Check tables
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name
                """)
                tables = [r["table_name"] for r in cur.fetchall()]
                print(f"  📊 Tables found: {tables}")
                
                # Check call_transcripts
                if "call_transcripts" in tables:
                    cur.execute("SELECT COUNT(*), status FROM call_transcripts GROUP BY status")
                    stats = cur.fetchall()
                    print(f"  📞 call_transcripts status breakdown:")
                    for row in stats:
                        print(f"      {row['status']}: {row['count']}")
                
                # Check job_requirements or jd_requirements
                job_table = "job_requirements" if "job_requirements" in tables else "jd_requirements" if "jd_requirements" in tables else None
                if job_table:
                    cur.execute(f"SELECT COUNT(*) as cnt FROM {job_table}")
                    cnt = cur.fetchone()["cnt"]
                    print(f"  💼 {job_table}: {cnt} jobs")
                    
                    cur.execute(f"SELECT COUNT(*) as cnt FROM {job_table} WHERE milvus_synced = FALSE")
                    unsynced = cur.fetchone()["cnt"]
                    print(f"      Unsynced to Milvus: {unsynced}")
                else:
                    print("  ⚠️  No job_requirements or jd_requirements table found")
                
                # Check processing_logs
                if "processing_logs" in tables:
                    cur.execute("""
                        SELECT stage, level, COUNT(*) as cnt 
                        FROM processing_logs 
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                        GROUP BY stage, level
                        ORDER BY cnt DESC
                        LIMIT 10
                    """)
                    logs = cur.fetchall()
                    print(f"  📝 Recent processing_logs (last 24h):")
                    for row in logs:
                        icon = "❌" if row["level"] == "error" else "⚠️" if row["level"] == "warning" else "✅"
                        print(f"      {icon} {row['stage']}: {row['cnt']}")
                
        return True
    except ImportError:
        print("  ❌ psycopg not installed")
        return False
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False


def check_huggingface():
    """Test HuggingFace token and embedding model."""
    print("\n" + "=" * 60)
    print("HUGGINGFACE / EMBEDDINGS TEST")
    print("=" * 60)
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("  ❌ HF_TOKEN not set - will get rate limited!")
        print("     Get a token from: https://huggingface.co/settings/tokens")
        return False
    
    print(f"  ✅ HF_TOKEN is set: {hf_token[:8]}...")
    
    # Set the token for HuggingFace
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_HUB_TOKEN"] = hf_token
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
        print(f"  Loading model: {model_name}...")
        
        model = SentenceTransformer(model_name, device="cpu")
        print("  ✅ Model loaded successfully")
        
        # Test embedding
        test_text = "Senior Java Developer with Spring Boot experience"
        embedding = model.encode(test_text)
        print(f"  ✅ Test embedding generated: dim={len(embedding)}")
        
        return True
    except Exception as e:
        print(f"  ❌ Embedding error: {e}")
        return False


def check_milvus():
    """Test Milvus connection."""
    print("\n" + "=" * 60)
    print("MILVUS CONNECTION TEST")
    print("=" * 60)
    
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_host = os.getenv("MILVUS_HOST")
    
    if not milvus_uri and not milvus_host:
        print("  ❌ Neither MILVUS_URI nor MILVUS_HOST set")
        return False
    
    try:
        from pymilvus import connections, utility
        
        if milvus_uri:
            print(f"  Connecting via URI: {milvus_uri[:30]}...")
            connections.connect(
                alias="test",
                uri=milvus_uri,
                token=os.getenv("MILVUS_TOKEN"),
            )
        else:
            port = os.getenv("MILVUS_PORT", "19530")
            print(f"  Connecting to {milvus_host}:{port}...")
            connections.connect(
                alias="test",
                host=milvus_host,
                port=port,
            )
        
        print("  ✅ Milvus connection OK")
        
        # List collections
        collections = utility.list_collections(using="test")
        print(f"  📊 Collections: {collections}")
        
        collection_name = os.getenv("MILVUS_COLLECTION", "job_postings")
        if collection_name in collections:
            print(f"  ✅ Collection '{collection_name}' exists")
        else:
            print(f"  ⚠️  Collection '{collection_name}' not found")
        
        connections.disconnect("test")
        return True
    except Exception as e:
        print(f"  ❌ Milvus error: {e}")
        return False


def check_bey_api():
    """Test Bey API connection."""
    print("\n" + "=" * 60)
    print("BEY API TEST")
    print("=" * 60)
    
    api_key = os.getenv("BEY_API_KEY")
    if not api_key:
        print("  ❌ BEY_API_KEY not set")
        return False
    
    print(f"  ✅ BEY_API_KEY is set: {api_key[:8]}...")
    
    try:
        import requests
        
        # Just test if the API responds (don't create agent)
        response = requests.get(
            "https://api.bey.dev/v1/calls",
            headers={"x-api-key": api_key},
            timeout=10,
        )
        
        if response.status_code == 200:
            print("  ✅ Bey API connection OK")
            return True
        else:
            print(f"  ⚠️  Bey API returned: {response.status_code}")
            return response.status_code != 401  # 401 = bad key
    except Exception as e:
        print(f"  ❌ Bey API error: {e}")
        return False


def check_openai():
    """Test OpenAI API."""
    print("\n" + "=" * 60)
    print("OPENAI API TEST")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ❌ OPENAI_API_KEY not set")
        return False
    
    print(f"  ✅ OPENAI_API_KEY is set: {api_key[:8]}...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        # Quick test with minimal tokens
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'OK'"}],
            max_tokens=5,
        )
        
        result = response.choices[0].message.content
        print(f"  ✅ OpenAI API OK: '{result}'")
        return True
    except Exception as e:
        print(f"  ❌ OpenAI error: {e}")
        return False


def main():
    print("\n🔍 RCRUTR AI SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    results = {}
    
    results["env"] = check_env()
    results["database"] = check_database()
    results["huggingface"] = check_huggingface()
    results["milvus"] = check_milvus()
    results["bey_api"] = check_bey_api()
    results["openai"] = check_openai()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False
    
    print()
    if all_ok:
        print("🎉 All systems operational!")
    else:
        print("⚠️  Some issues need attention. See details above.")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())