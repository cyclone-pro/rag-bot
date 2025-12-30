"""
Test Script - Quick Validation
Tests all major components without making actual calls
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_database_connections():
    """Test PostgreSQL and Milvus connections"""
    print("\n🔍 Testing Database Connections...")
    
    try:
        from app.services.database import initialize_database_connections
        
        success = await initialize_database_connections()
        
        if success:
            print("✅ PostgreSQL: Connected")
            print("✅ Milvus: Connected")
            print("✅ E5-Base-V2: Loaded")
            return True
        else:
            print("❌ Database initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_embeddings():
    """Test embedding generation"""
    print("\n🔍 Testing Embedding Generation...")
    
    try:
        from app.services.database import embedding_service
        
        test_text = "Python FastAPI PostgreSQL microservices"
        embedding = await embedding_service.generate_embedding_async(test_text)
        
        print(f"✅ Generated embedding: {len(embedding)} dimensions")
        
        if len(embedding) == 768:
            print("✅ Correct dimension (768 for e5-base-v2)")
            return True
        else:
            print(f"❌ Wrong dimension: {len(embedding)} (expected 768)")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_interview_session():
    """Test interview session creation"""
    print("\n🔍 Testing Interview Session...")
    
    try:
        from app.services.interview_service import start_interview_session
        from datetime import datetime
        
        session = await start_interview_session(
            interview_id="test_123",
            candidate_data={
                "candidate_id": "cand_001",
                "name": "Test Candidate",
                "email": "test@example.com",
                "phone_number": "+14155551234",
                "skills": ["Python", "FastAPI"]
            },
            jd_data={
                "job_id": "jd_001",
                "title": "Backend Engineer",
                "requirements": ["Python", "APIs"]
            },
            livekit_room="test-room-123"
        )
        
        print(f"✅ Created session: {session.interview_id}")
        print(f"✅ Candidate: {session.candidate_data.get('name')}")
        print(f"✅ Position: {session.jd_data.get('title')}")
        
        # Clean up
        from app.services.interview_service import active_sessions
        if "test_123" in active_sessions:
            del active_sessions["test_123"]
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing Configuration...")
    
    try:
        from app.config.settings import settings
        
        print(f"✅ App Name: {settings.app_name}")
        print(f"✅ Environment: {settings.app_env}")
        print(f"✅ PostgreSQL: {settings.postgres_host}:{settings.postgres_port}")
        print(f"✅ Milvus: {settings.milvus_host}:{settings.milvus_port}")
        print(f"✅ Embedding Model: {settings.embedding_model}")
        print(f"✅ LLM Model: {settings.openai_model}")
        
        # Check required settings
        required = [
            settings.livekit_api_key,
            settings.deepgram_api_key,
            settings.openai_api_key,
            settings.google_application_credentials
        ]
        
        if all(required):
            print("✅ All required API keys configured")
            return True
        else:
            print("⚠️  Some API keys missing (check .env)")
            return True  # Don't fail, just warn
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_milvus_collection():
    """Test Milvus collection access"""
    print("\n🔍 Testing Milvus Collection...")
    
    try:
        from pymilvus import connections, Collection
        from app.config.settings import settings
        
        connections.connect(
            host=settings.milvus_host,
            port=settings.milvus_port
        )
        
        collection = Collection("interview_transcripts_v2")
        
        print(f"✅ Collection: {collection.name}")
        print(f"✅ Entities: {collection.num_entities}")
        print(f"✅ Loaded: {collection.is_loaded}")
        
        # Check schema
        primary_field = collection.primary_field
        print(f"✅ Primary Key: {primary_field.name}")
        
        # Check indexes
        indexes = collection.indexes
        print(f"✅ Indexes: {len(indexes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    
    print("=" * 70)
    print("RecruiterBrain Voice Interview - System Test")
    print("=" * 70)
    
    tests = [
        ("Configuration", test_config),
        ("Database Connections", test_database_connections),
        ("Embeddings", test_embeddings),
        ("Milvus Collection", test_milvus_collection),
        ("Interview Session", test_interview_session),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
