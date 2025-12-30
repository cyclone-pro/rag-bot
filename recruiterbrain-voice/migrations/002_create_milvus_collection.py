"""
Milvus Collection Setup for Interview Transcripts
Optimized for batch inserts and e5-base-v2 embeddings (768 dimensions)
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)


def create_interview_collection():
    """
    Create optimized Milvus collection for interview transcripts
    - 768-dim embeddings (e5-base-v2)
    - Batch insert optimized
    - Scalar indexes for filtering
    """
    
    # Connect to Milvus
    connections.connect(
        alias="default",
        host="localhost",  # Update with your Milvus host
        port="19530"
    )
    
    collection_name = "interview_transcripts_v2"
    
    # Drop old collection if exists
    if utility.has_collection(collection_name):
        print(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)
    
    # Define schema
    fields = [
        # Primary key (interview_id - one embedding per interview)
        FieldSchema(
            name="interview_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            description="Unique interview identifier"
        ),
        
        # Foreign keys
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
        
        # Full interview embedding (e5-base-v2: 768 dimensions)
        # This is the COMBINED embedding of all Q&A pairs
        FieldSchema(
            name="interview_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=768,
            description="Full interview transcript embedding (e5-base-v2)"
        ),
        
        # Searchable metadata
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
        
        # Concatenated text for display in search results
        FieldSchema(
            name="full_transcript_text",
            dtype=DataType.VARCHAR,
            max_length=8192,
            description="Full interview text (for context in results)"
        ),
        
        # Skills discussed (for filtering)
        FieldSchema(
            name="skills_discussed",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Comma-separated skills mentioned"
        ),
        
        # Summary (LLM-generated)
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
        description="Interview transcripts with full-interview embeddings (batch insert optimized)"
    )
    
    # Create collection
    print(f"Creating collection: {collection_name}")
    collection = Collection(
        name=collection_name,
        schema=schema,
        using="default"
    )
    
    print(f"✅ Collection created: {collection_name}")
    
    # Create vector index (HNSW for speed and accuracy)
    print("Creating vector index (HNSW)...")
    collection.create_index(
        field_name="interview_embedding",
        index_params={
            "metric_type": "COSINE",  # Best for e5-base-v2
            "index_type": "HNSW",
            "params": {
                "M": 16,              # Number of connections (16-64, higher = better accuracy, slower build)
                "efConstruction": 200 # Build quality (100-500, higher = better quality, slower build)
            }
        },
        index_name="idx_interview_embedding"
    )
    print("✅ Vector index created")
    
    # Create scalar indexes for filtering
    print("Creating scalar indexes...")
    
    # Candidate ID index (for "show me all interviews for this candidate")
    collection.create_index(
        field_name="candidate_id",
        index_name="idx_candidate_id"
    )
    
    # JD ID index (for "show me all interviews for this position")
    collection.create_index(
        field_name="job_id",
        index_name="idx_job_id"
    )
    
    # Date index (for time-range queries)
    collection.create_index(
        field_name="interview_date",
        index_name="idx_interview_date"
    )
    
    # Score index (for "show me top-scored interviews")
    collection.create_index(
        field_name="evaluation_score",
        index_name="idx_evaluation_score"
    )
    
    print("✅ All scalar indexes created")
    
    # Load collection into memory for searching
    collection.load()
    print("✅ Collection loaded into memory")
    
    # Show collection stats
    stats = collection.num_entities
    print(f"\n📊 Collection Stats:")
    print(f"   Name: {collection_name}")
    print(f"   Entities: {stats}")
    print(f"   Vector Dimension: 768 (e5-base-v2)")
    print(f"   Index Type: HNSW (COSINE)")
    
    return collection


def verify_collection():
    """Verify collection was created correctly"""
    
    connections.connect(alias="default", host="localhost", port="19530")
    
    collection_name = "interview_transcripts_v2"
    
    if not utility.has_collection(collection_name):
        print(f"❌ Collection {collection_name} does not exist!")
        return False
    
    collection = Collection(collection_name)
    
    print(f"\n✅ Collection verified: {collection_name}")
    print(f"   Schema: {collection.schema}")
    print(f"   Indexes: {collection.indexes}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Milvus Interview Collection Setup")
    print("=" * 60)
    print()
    
    # Create collection
    collection = create_interview_collection()
    
    print()
    print("=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Update your app to use 'interview_transcripts_v2' collection")
    print("2. Use e5-base-v2 model for embeddings (768 dimensions)")
    print("3. Batch insert interviews after completion")
    print()
    
    # Verify
    verify_collection()
