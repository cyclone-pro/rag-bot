"""
Milvus operations for semantic search
"""

import logging
from typing import List, Dict, Any
from pymilvus import (
    connections, Collection, FieldSchema, 
    CollectionSchema, DataType, utility
)
from app.config.settings import settings

logger = logging.getLogger(__name__)


class MilvusService:
    """Handle Milvus operations"""
    
    def __init__(self):
        self.collection_name = settings.milvus_qa_collection
        self.collection = None
        self._connect()
        self._ensure_collection()
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            logger.info(f"✅ Connected to Milvus at {settings.milvus_host}:{settings.milvus_port}")
        except Exception as e:
            logger.error(f"❌ Milvus connection failed: {e}")
            raise
    
    def _ensure_collection(self):
        """Create collection if not exists"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"✅ Using existing collection: {self.collection_name}")
        else:
            self._create_collection()
    
    def _create_collection(self):
        """Create new collection"""
        fields = [
            FieldSchema(
                name="id", 
                dtype=DataType.VARCHAR, 
                max_length=100, 
                is_primary=True
            ),
            FieldSchema(
                name="interview_id", 
                dtype=DataType.VARCHAR, 
                max_length=64
            ),
            FieldSchema(
                name="candidate_id", 
                dtype=DataType.VARCHAR, 
                max_length=64
            ),
            FieldSchema(
                name="job_id", 
                dtype=DataType.VARCHAR, 
                max_length=64
            ),
            FieldSchema(
                name="job_title",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="job_description",
                dtype=DataType.VARCHAR,
                max_length=2048
            ),
            FieldSchema(
                name="question_index", 
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="answer_snippet", 
                dtype=DataType.VARCHAR, 
                max_length=500
            ),
            FieldSchema(
                name="interview_date",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="embedding", 
                dtype=DataType.FLOAT_VECTOR, 
                dim=768
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Interview Q&A embeddings for semantic search (v2)"
        )
        
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        self.collection.create_index(
            field_name="interview_id",
            index_params={"index_type": "INVERTED"}
        )
        self.collection.create_index(
            field_name="candidate_id",
            index_params={"index_type": "INVERTED"}
        )
        self.collection.create_index(
            field_name="job_id",
            index_params={"index_type": "INVERTED"}
        )
        
        logger.info(f"✅ Created collection: {self.collection_name}")
    
    def insert_embeddings(
        self, 
        embeddings_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Insert embeddings to Milvus
        
        Args:
            embeddings_data: List of dicts with:
                - id: Unique ID
                - interview_id
                - candidate_id
                - job_id
                - job_title
                - job_description
                - question_index
                - answer_snippet
                - interview_date
                - embedding (768-dim list)
        
        Returns:
            List of inserted IDs
        """
        if not embeddings_data:
            return []
        
        # Prepare data
        data = [
            [d["id"] for d in embeddings_data],
            [d["interview_id"] for d in embeddings_data],
            [d["candidate_id"] for d in embeddings_data],
            [d.get("job_id", "") for d in embeddings_data],
            [d.get("job_title", "") for d in embeddings_data],
            [d.get("job_description", "") for d in embeddings_data],
            [d["question_index"] for d in embeddings_data],
            [d["answer_snippet"] for d in embeddings_data],
            [d.get("interview_date", 0) for d in embeddings_data],
            [d["embedding"] for d in embeddings_data]
        ]
        
        # Insert
        mr = self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"✅ Inserted {len(embeddings_data)} embeddings to Milvus")
        
        return mr.primary_keys
    
    def search(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        filter_expr: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar answers
        
        Args:
            query_embedding: 768-dim query vector
            limit: Number of results
            filter_expr: Optional Milvus filter expression (e.g., "interview_id == 'xxx'")
        
        Returns:
            List of results with scores
        """
        # Load collection to memory
        self.collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=[
                "interview_id", 
                "candidate_id", 
                "job_id",
                "job_title",
                "job_description",
                "question_index",
                "interview_date",
                "answer_snippet"
            ],
            expr=filter_expr
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "milvus_id": hit.id,
                    "interview_id": hit.entity.get("interview_id"),
                    "candidate_id": hit.entity.get("candidate_id"),
                    "job_id": hit.entity.get("job_id"),
                    "job_title": hit.entity.get("job_title"),
                    "job_description": hit.entity.get("job_description"),
                    "question_index": hit.entity.get("question_index"),
                    "interview_date": hit.entity.get("interview_date"),
                    "answer_snippet": hit.entity.get("answer_snippet"),
                    "similarity_score": hit.score
                })
        
        return formatted_results


# Global instance
_milvus_service = None

def get_milvus_service() -> MilvusService:
    """Get singleton Milvus service"""
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = MilvusService()
    return _milvus_service
