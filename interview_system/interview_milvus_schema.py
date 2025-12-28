"""
Milvus collection schema for interview transcripts.

Stores embeddings of Q&A pairs for semantic search across all interviews.
"""

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility
)
import os
import logging

logger = logging.getLogger(__name__)

# Collection name
INTERVIEW_COLLECTION = "interview_transcripts_v1"

# Embedding dimension (matching your model: intfloat/e5-base-v2)
EMBEDDING_DIM = 768  # e5-base-v2 produces 768-dimensional embeddings


def create_interview_transcript_collection(
    collection_name: str = INTERVIEW_COLLECTION,
    drop_existing: bool = False
) -> Collection:
    """
    Create Milvus collection for interview transcript embeddings.
    
    Schema:
    - transcript_id (PK): Links to PostgreSQL
    - interview_id: Links to interview
    - candidate_id: Links to candidate
    - question_index: Which question (1-6)
    - question_text: The question asked
    - answer_text: Candidate's answer
    - qa_embedding: Vector embedding of "Q: ... A: ..."
    - job_title: For filtering
    - created_at: Timestamp
    
    Args:
        collection_name: Name of collection to create
        drop_existing: If True, drop existing collection first
        
    Returns:
        Collection object
    """
    # Connect to Milvus
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )
    
    logger.info(f"Creating interview transcript collection: {collection_name}")
    
    # Drop existing if requested
    if drop_existing and utility.has_collection(collection_name):
        logger.warning(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)
    
    # Check if already exists
    if utility.has_collection(collection_name):
        logger.info(f"Collection {collection_name} already exists, loading...")
        return Collection(collection_name)
    
    # Define schema
    fields = [
        # Primary key
        FieldSchema(
            name="transcript_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
            description="Primary key linking to PostgreSQL interview_transcripts table"
        ),
        
        # Foreign keys
        FieldSchema(
            name="interview_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Interview ID"
        ),
        
        FieldSchema(
            name="candidate_id",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="Candidate ID"
        ),
        
        # Q&A metadata
        FieldSchema(
            name="question_index",
            dtype=DataType.INT16,
            description="Question number (1-6)"
        ),
        
        FieldSchema(
            name="question_text",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Question asked"
        ),
        
        FieldSchema(
            name="answer_text",
            dtype=DataType.VARCHAR,
            max_length=8192,
            description="Candidate's answer"
        ),
        
        # Embedding vector
        FieldSchema(
            name="qa_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM,
            description="Embedding of concatenated question + answer"
        ),
        
        # Filterable fields
        FieldSchema(
            name="job_title",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Job title for filtering"
        ),
        
        FieldSchema(
            name="interview_date",
            dtype=DataType.INT64,
            description="Timestamp (Unix epoch) for time-based filtering"
        ),
        
        # Skills mentioned (for filtering)
        FieldSchema(
            name="keywords_extracted",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Comma-separated keywords/skills mentioned"
        ),
        
        # Sentiment
        FieldSchema(
            name="sentiment",
            dtype=DataType.VARCHAR,
            max_length=32,
            description="Sentiment: positive, negative, neutral"
        ),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Interview transcript embeddings for semantic search"
    )
    
    # Create collection
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        consistency_level="Strong"
    )
    
    logger.info(f"✅ Collection created: {collection_name}")
    
    # Create index for vector search
    index_params = {
        "metric_type": "COSINE",  # Using COSINE for e5-base-v2
        "index_type": "HNSW",     # Fast and accurate
        "params": {
            "M": 16,               # Number of connections
            "efConstruction": 200  # Build-time accuracy
        }
    }
    
    logger.info("Creating HNSW index on qa_embedding...")
    collection.create_index(
        field_name="qa_embedding",
        index_params=index_params
    )
    
    # Load collection to memory
    collection.load()
    logger.info(f"✅ Collection loaded and ready: {collection_name}")
    
    return collection


def get_interview_collection(collection_name: str = INTERVIEW_COLLECTION) -> Collection:
    """
    Get existing interview transcript collection.
    
    Args:
        collection_name: Name of collection
        
    Returns:
        Collection object
        
    Raises:
        ValueError: If collection doesn't exist
    """
    # Connect to Milvus
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )
    
    if not utility.has_collection(collection_name):
        raise ValueError(
            f"Collection {collection_name} does not exist. "
            f"Create it first with create_interview_transcript_collection()"
        )
    
    collection = Collection(collection_name)
    
    # Load if not loaded
    if not collection.is_loaded:
        collection.load()
    
    return collection


def insert_interview_transcript(
    collection: Collection,
    transcript_id: str,
    interview_id: str,
    candidate_id: str,
    question_index: int,
    question_text: str,
    answer_text: str,
    qa_embedding: list,
    job_title: str = "",
    interview_date: int = 0,
    keywords_extracted: list = None,
    sentiment: str = "neutral"
) -> None:
    """
    Insert interview transcript with embedding into Milvus.
    
    Args:
        collection: Milvus collection
        transcript_id: Unique transcript ID
        interview_id: Interview ID
        candidate_id: Candidate ID
        question_index: Question number (1-6)
        question_text: Question asked
        answer_text: Candidate's answer
        qa_embedding: Vector embedding (768-dim for e5-base-v2)
        job_title: Job title for filtering
        interview_date: Unix timestamp
        keywords_extracted: List of skills/keywords
        sentiment: positive/negative/neutral
    """
    # Prepare keywords
    keywords_str = ", ".join(keywords_extracted) if keywords_extracted else ""
    
    # Prepare data
    data = [{
        "transcript_id": transcript_id,
        "interview_id": interview_id,
        "candidate_id": candidate_id,
        "question_index": question_index,
        "question_text": question_text[:2048],  # Truncate if needed
        "answer_text": answer_text[:8192],      # Truncate if needed
        "qa_embedding": qa_embedding,
        "job_title": job_title[:256],
        "interview_date": interview_date,
        "keywords_extracted": keywords_str[:2048],
        "sentiment": sentiment
    }]
    
    # Insert
    collection.insert(data)
    
    # Flush to persist
    collection.flush()
    
    logger.info(f"✅ Inserted transcript: {transcript_id} (Q{question_index})")


def search_interview_transcripts(
    collection: Collection,
    query_embedding: list,
    top_k: int = 10,
    candidate_id: str = None,
    job_title: str = None,
    min_date: int = None
) -> list:
    """
    Search interview transcripts by semantic similarity.
    
    Args:
        collection: Milvus collection
        query_embedding: Query vector (768-dim)
        top_k: Number of results
        candidate_id: Filter by candidate
        job_title: Filter by job title
        min_date: Filter by minimum date (Unix timestamp)
        
    Returns:
        List of search results
    """
    # Build filter expression
    filters = []
    
    if candidate_id:
        filters.append(f'candidate_id == "{candidate_id}"')
    
    if job_title:
        filters.append(f'job_title == "{job_title}"')
    
    if min_date:
        filters.append(f'interview_date >= {min_date}')
    
    filter_expr = " && ".join(filters) if filters else None
    
    # Search parameters
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 100}  # Search-time accuracy
    }
    
    # Execute search
    results = collection.search(
        data=[query_embedding],
        anns_field="qa_embedding",
        param=search_params,
        limit=top_k,
        expr=filter_expr,
        output_fields=[
            "transcript_id", "interview_id", "candidate_id",
            "question_index", "question_text", "answer_text",
            "job_title", "keywords_extracted", "sentiment"
        ]
    )
    
    return results[0] if results else []


# Example usage script
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python interview_milvus_schema.py create    # Create collection")
        print("  python interview_milvus_schema.py drop      # Drop collection")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        collection = create_interview_transcript_collection()
        print(f"✅ Collection created: {INTERVIEW_COLLECTION}")
        print(f"   Fields: {[field.name for field in collection.schema.fields]}")
        print(f"   Dimension: {EMBEDDING_DIM}")
        
    elif command == "drop":
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530")
        )
        
        if utility.has_collection(INTERVIEW_COLLECTION):
            utility.drop_collection(INTERVIEW_COLLECTION)
            print(f"✅ Collection dropped: {INTERVIEW_COLLECTION}")
        else:
            print(f"⚠️  Collection {INTERVIEW_COLLECTION} does not exist")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)