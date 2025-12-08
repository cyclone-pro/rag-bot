from pymilvus import connections, Collection
from datetime import datetime, timezone
import os
from typing import List, Dict, Any

# ==================== CONFIGURATION ====================
MILVUS_HOST = os.getenv("MILVUS_HOST", "34.55.41.188")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
EMBEDDING_DIM = 768
JOB_COLLECTION_NAME = "job_postings"

# ==================== MILVUS CONNECTION ====================
def connect_milvus():
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


# ==================== EMBEDDING STUB ====================
def generate_embedding(text: str) -> List[float]:
    """
    Replace this with your real embedding call (OpenAI, Cohere, local model, etc.)
    It MUST return a list[float] of length EMBEDDING_DIM.
    """
    # TODO: integrate your actual embedding model here.
    # Example:
    #   response = your_model.embed(text)
    #   return response["vector"]
    return [0.0] * EMBEDDING_DIM  # placeholder so code runs


# ==================== INSERT FUNCTION ====================
def insert_jobs(job_list: List[Dict[str, Any]]):
    """
    job_list: list of dicts with keys:
        job_id, title, company, location, posted_ts (int or ISO str),
        status, jd_text
    """
    connect_milvus()

    # Open existing collection
    job_col = Collection(JOB_COLLECTION_NAME)

    job_ids = []
    titles = []
    companies = []
    locations = []
    posted_ts_list = []
    statuses = []
    jd_texts = []
    jd_embeddings = []

    for job in job_list:
        job_id = str(job["job_id"])
        title = job["title"]
        company = job.get("company", "")
        location = job.get("location", "")
        status = job.get("status", "active")
        jd_text = job["jd_text"]

        # Handle posted_ts: can be int timestamp or ISO string
        posted_ts_raw = job.get("posted_ts", None)
        if isinstance(posted_ts_raw, int):
            posted_ts = posted_ts_raw
        elif isinstance(posted_ts_raw, str):
            # Example: "2025-01-15T10:30:00"
            dt = datetime.fromisoformat(posted_ts_raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            posted_ts = int(dt.timestamp())
        else:
            # default to "now"
            posted_ts = int(datetime.now(tz=timezone.utc).timestamp())

        # Generate embedding for jd_text
        emb = generate_embedding(jd_text)
        if len(emb) != EMBEDDING_DIM:
            raise ValueError(f"Embedding length {len(emb)} != {EMBEDDING_DIM}")

        job_ids.append(job_id)
        titles.append(title)
        companies.append(company)
        locations.append(location)
        posted_ts_list.append(posted_ts)
        statuses.append(status)
        jd_texts.append(jd_text)
        jd_embeddings.append(emb)

    # Order MUST match your schema:
    # [job_id, title, company, location, posted_ts, status, jd_text, jd_embedding]
    entities = [
        job_ids,
        titles,
        companies,
        locations,
        posted_ts_list,
        statuses,
        jd_texts,
        jd_embeddings,
    ]

    insert_result = job_col.insert(entities)
    job_col.flush()

    print(f"✅ Inserted {len(job_list)} jobs into '{JOB_COLLECTION_NAME}'")
    print("Primary keys:", insert_result.primary_keys)


# ==================== EXAMPLE: YOUR 3 JDs ====================
if __name__ == "__main__":
    # 🔁 Replace the placeholders below with your 3 real JDs
    jobs = [
        {
            "job_id": "JOB-001",
            "title": "Senior Python Backend Engineer",
            "company": "Acme Corp",
            "location": "Remote - USA",
            "posted_ts": "2025-01-15T10:30:00",
            "status": "active",
            "jd_text": """
We are looking for a Senior Python Backend Engineer with experience in Django,
REST APIs, PostgreSQL, Docker, and cloud platforms like AWS or GCP...
            """,
        },
        {
            "job_id": "JOB-002",
            "title": "DevOps Engineer",
            "company": "BetaTech",
            "location": "Chicago, IL",
            "posted_ts": "2025-01-17T09:00:00",
            "status": "active",
            "jd_text": """
Seeking a DevOps Engineer with strong knowledge of Kubernetes, CI/CD pipelines,
Terraform, monitoring (Prometheus, Grafana), and Linux administration...
            """,
        },
        {
            "job_id": "JOB-003",
            "title": "Data Engineer",
            "company": "Gamma Analytics",
            "location": "New York, NY",
            "posted_ts": "2025-01-20T14:15:00",
            "status": "active",
            "jd_text": """
Data Engineer role focusing on building ETL pipelines with Python, Spark,
Kafka, and data warehousing on Snowflake/BigQuery...
            """,
        },
    ]

    insert_jobs(jobs)
