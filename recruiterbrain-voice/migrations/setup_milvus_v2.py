"""
Milvus Setup Script - Delete v1 and Create Optimized v2
- Deletes interview_transcripts_v1 (if exists)
- Creates interview_transcripts_v2 with optimized schema
- 768-dimensional embeddings (e5-base-v2)
- Proper scalar indexes for fast filtering
- One record per interview (not per Q&A)
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)


def setup_milvus_collection():
    """
    Complete Milvus setup:
    1. Delete old v1 collection
    2. Create optimized v2 collection
    3. Create all indexes
    """
    
    print("=" * 70)
    print("Milvus Collection Setup - Interview Transcripts v2")
    print("=" * 70)
    print()
    
    # ==========================================
    # STEP 1: Connect to Milvus
    # ==========================================
    
    print("📡 Connecting to Milvus...")
    
    # Update these if your Milvus is on different host/port
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        print(f"   Make sure Milvus is running on {MILVUS_HOST}:{MILVUS_PORT}")
        return False
    
    print()
    
    # ==========================================
    # STEP 2: Delete Old Collections
    # ==========================================
    
    print("🗑️  Checking for old collections...")
    
    # Delete v1 if exists
    if utility.has_collection("interview_transcripts_v1"):
        print("   Found: interview_transcripts_v1")
        print("   Deleting interview_transcripts_v1...")
        utility.drop_collection("interview_transcripts_v1")
        print("   ✅ Deleted interview_transcripts_v1")
    else:
        print("   interview_transcripts_v1 does not exist (ok)")
    
    # Delete v2 if exists (for clean setup)
    if utility.has_collection("interview_transcripts_v2"):
        print("   Found: interview_transcripts_v2")
        print("   Deleting interview_transcripts_v2...")
        utility.drop_collection("interview_transcripts_v2")
        print("   ✅ Deleted interview_transcripts_v2")
    else:
        print("   interview_transcripts_v2 does not exist (ok)")
    
    print()
    
    # ==========================================
    # STEP 3: Define Optimized Schema
    # ==========================================
    
    print("📐 Creating optimized schema...")
    
    fields = [
        # ========== PRIMARY KEY ==========
        # One record per INTERVIEW (not per Q&A)
        FieldSchema(
            name="interview_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            description="Unique interview identifier (PRIMARY KEY)"
        ),
        
        # ========== FOREIGN KEYS ==========
        FieldSchema(
            name="candidate_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Candidate identifier"
        ),
        
        FieldSchema(
            name="job_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Job description identifier"
        ),
        
        # ========== VECTOR EMBEDDING ==========
        # Full interview embedding (e5-base-v2: 768 dimensions)
        # This is the COMBINED embedding of all Q&A pairs
        FieldSchema(
            name="interview_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=768,
            description="Full interview transcript embedding (e5-base-v2)"
        ),
        
        # ========== SEARCHABLE METADATA ==========
        FieldSchema(
            name="job_title",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Job position title"
        ),
        
        FieldSchema(
            name="interview_date",
            dtype=DataType.INT64,
            description="Interview date (Unix timestamp)"
        ),
        
        FieldSchema(
            name="evaluation_score",
            dtype=DataType.FLOAT,
            description="Overall evaluation score (0.0 to 1.0)"
        ),
        
        # ========== DISPLAY TEXT ==========
        # For showing context in search results
        FieldSchema(
            name="full_transcript_text",
            dtype=DataType.VARCHAR,
            max_length=8192,
            description="Full interview text (for context in results)"
        ),
        
        FieldSchema(
            name="skills_discussed",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Comma-separated skills mentioned"
        ),
        
        FieldSchema(
            name="interview_summary",
            dtype=DataType.VARCHAR,
            max_length=1024,
            description="AI-generated interview summary"
        ),
    ]
    
    # Create schema
    schema = CollectionSchema(
        fields=fields,
        description="Interview transcripts with full-interview embeddings (optimized for batch insert)",
        enable_dynamic_field=False  # Strict schema
    )
    
    print("✅ Schema created with 10 fields")
    
    print()
    
    # ==========================================
    # STEP 4: Create Collection
    # ==========================================
    
    collection_name = "interview_transcripts_v2"
    
    print(f"📦 Creating collection: {collection_name}")
    
    collection = Collection(
        name=collection_name,
        schema=schema,
        using="default"
    )
    
    print(f"✅ Collection created: {collection_name}")
    
    print()
    
    # ==========================================
    # STEP 5: Create Vector Index (HNSW)
    # ==========================================
    
    print("🔍 Creating vector index (HNSW)...")
    
    # HNSW parameters:
    # - M: Number of connections (16-64, higher = better accuracy, slower build)
    # - efConstruction: Build quality (100-500, higher = better quality, slower build)
    
    collection.create_index(
        field_name="interview_embedding",
        index_params={
            "metric_type": "COSINE",      # Best for e5-base-v2 embeddings
            "index_type": "HNSW",         # Fast and accurate
            "params": {
                "M": 16,                  # Good balance of speed/accuracy
                "efConstruction": 200     # High quality index
            }
        },
        index_name="idx_interview_embedding"
    )
    
    print("✅ Vector index created (HNSW, COSINE, M=16, efConstruction=200)")
    
    print()
    
    # ==========================================
    # STEP 6: Create Scalar Indexes
    # ==========================================
    
    print("📑 Creating scalar indexes for fast filtering...")
    
    # Index on candidate_id (for "show me all interviews for this candidate")
    collection.create_index(
        field_name="candidate_id",
        index_name="idx_candidate_id"
    )
    print("✅ Index created: candidate_id")
    
    # Index on job_id (for "show me all interviews for this position")
    collection.create_index(
        field_name="job_id",
        index_name="idx_job_id"
    )
    print("✅ Index created: job_id")
    
    # Index on interview_date (for time-range queries)
    collection.create_index(
        field_name="interview_date",
        index_name="idx_interview_date"
    )
    print("✅ Index created: interview_date")
    
    # Index on evaluation_score (for "show me top-scored interviews")
    collection.create_index(
        field_name="evaluation_score",
        index_name="idx_evaluation_score"
    )
    print("✅ Index created: evaluation_score")
    
    print()
    
    # ==========================================
    # STEP 7: Load Collection into Memory
    # ==========================================
    
    print("💾 Loading collection into memory...")
    
    collection.load()
    
    print("✅ Collection loaded and ready for use")
    
    print()
    
    # ==========================================
    # STEP 8: Show Summary
    # ==========================================
    
    print("=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print()
    print("📊 Collection Summary:")
    print(f"   Name: {collection_name}")
    print(f"   Status: Loaded")
    print(f"   Entities: {collection.num_entities}")
    print(f"   Vector Dimension: 768 (e5-base-v2)")
    print(f"   Primary Key: interview_id")
    print()
    print("🔍 Indexes Created:")
    print("   Vector Index:")
    print("     - Field: interview_embedding")
    print("     - Type: HNSW (COSINE)")
    print("     - Params: M=16, efConstruction=200")
    print()
    print("   Scalar Indexes:")
    print("     - candidate_id")
    print("     - job_id")
    print("     - interview_date")
    print("     - evaluation_score")
    print()
    print("📝 Schema:")
    for field in schema.fields:
        field_type = str(field.dtype).split('.')[-1]
        if field.dtype == DataType.FLOAT_VECTOR:
            field_type = f"FloatVector({field.params['dim']})"
        print(f"   - {field.name}: {field_type}")
    print()
    print("=" * 70)
    print()
    print("🎉 Ready to use!")
    print()
    print("Next steps:")
    print("1. Update your code to use 'interview_transcripts_v2'")
    print("2. Start inserting interview data")
    print("3. Run semantic searches")
    print()
    print("Example search:")
    print("   from pymilvus import Collection")
    print("   collection = Collection('interview_transcripts_v2')")
    print("   results = collection.search(...)")
    print()
    
    return True


def verify_setup():
    """Verify the collection was created correctly"""
    
    print("🔍 Verifying setup...")
    print()
    
    try:
        collection = Collection("interview_transcripts_v2")
        
        # Check schema
        print("Schema Fields:")
        for field in collection.schema.fields:
            print(f"  ✓ {field.name}")
        
        print()
        
        # Check indexes
        print("Indexes:")
        for index in collection.indexes:
            print(f"  ✓ {index.field_name}: {index.params.get('index_type', 'Scalar')}")
        
        print()
        
        # Check status
        print(f"Loaded: {'Yes' if collection.is_loaded else 'No'}")
        print(f"Entities: {collection.num_entities}")
        
        print()
        print("✅ Verification complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    # Run setup
    success = setup_milvus_collection()
    
    if success:
        print()
        print("-" * 70)
        print()
        
        # Verify
        verify_setup()
        
        print()
        print("=" * 70)
        print("🚀 All done! Your Milvus collection is ready for production.")
        print("=" * 70)
    else:
        print()
        print("❌ Setup failed. Please check the error messages above.")
        print()
