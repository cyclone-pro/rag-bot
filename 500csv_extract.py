# pip install "pymilvus==2.5.0" "numpy<2" "torch==2.2.2" --extra-index-url https://download.pytorch.org/whl/cpu
# pip install "transformers<4.45" "sentence-transformers==2.7.0"

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import csv

MILVUS_URI = "http://34.135.232.156:19530"   # or http://<YOUR_HOST>:9091
COLLECTION = "resume_chunks"

client = MilvusClient(uri=MILVUS_URI)
model  = SentenceTransformer("intfloat/e5-small-v2")  # 384-d

"""
def batched(x, n=256):
    for i in range(0, len(x), n):
        yield x[i:i+n]

rows = list(csv.DictReader(open("resume_chunks_500_with_names.csv", newline="", encoding="utf-8")))

for chunk in batched(rows, 256):
    texts = [f"passage: {r['text']}" for r in chunk]  # E5 doc-side prefix
    vecs  = model.encode(texts, batch_size=128, normalize_embeddings=True).tolist()  # 384-d

    payload = []
    for r, v in zip(chunk, vecs):
        payload.append({
            # DO NOT send "id" because AutoID is ON
            "embeddings": v,
            "first_name": r["first_name"],
            "last_name": r["last_name"],
            "gmail": r["gmail"],
            "candidate_id": int(r["candidate_id"]),
            "job_id": int(r["job_id"]),
            "application_id": int(r["application_id"]),
            "chunk_id": int(r["chunk_id"]),
            "text": r["text"],
        })

    client.insert(collection_name=COLLECTION, data=payload)

print("Inserted all rows to Milvus.")
"""