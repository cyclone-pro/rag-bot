"""
Simple script to create the interview_transcripts_v1 collection in Milvus.

Run this once to set up the collection for storing interview embeddings.
"""

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "34.55.41.188")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = "interview_transcripts_v1"
EMBEDDING_DIM = 768  # e5-base-v2 dimension

def create_collection():
    """Create interview transcript collection."""
    
    # Connect to Milvus
    if MILVUS_TOKEN:
        connections.connect(
            alias="default",
            uri=f"{MILVUS_HOST}:{MILVUS_PORT}",
            token=MILVUS_TOKEN
        )
    else:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
    print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    
    # Check if collection exists
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️  Collection '{COLLECTION_NAME}' already exists!")
        response = input("Do you want to drop and recreate it? (yes/no): ")
        if response.lower() == "yes":
            utility.drop_collection(COLLECTION_NAME)
            print(f"✅ Dropped existing collection")
        else:
            print("Aborted. Keeping existing collection.")
            return
    
    # Define schema
    print(f"Creating collection: {COLLECTION_NAME}")
    
    fields = [
        FieldSchema(name="transcript_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="interview_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="candidate_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="question_index", dtype=DataType.INT16),
        FieldSchema(name="question_text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="answer_text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="qa_embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="job_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="interview_date", dtype=DataType.INT64),
        FieldSchema(name="keywords_extracted", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="sentiment", dtype=DataType.VARCHAR, max_length=32),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Interview transcript embeddings for semantic search"
    )
    
    # Create collection
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using='default',
        consistency_level="Strong"
    )
    print("✅ Collection created")
    
    # Create index
    print("Creating HNSW index on qa_embedding...")
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    
    collection.create_index(
        field_name="qa_embedding",
        index_params=index_params
    )
    print("✅ Index created")
    
    # Load collection
    collection.load()
    print("✅ Collection loaded to memory")
    
    print("\n" + "="*60)
    print(f"✅ SUCCESS! Collection '{COLLECTION_NAME}' is ready")
    print("="*60)
    print(f"Fields: {[f.name for f in collection.schema.fields]}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Index type: HNSW (COSINE)")
    print("\nYou can now start storing interview transcripts!")

if __name__ == "__main__":
    try:
        create_collection()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()