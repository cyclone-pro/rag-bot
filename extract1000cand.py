# ingest_1000_csv.py
# -----------------------------------------------------------
# Ingests candidate_pool_1000.csv with local embeddings (intfloat/e5-base-v2).
# pip install "pymilvus>=2.4.4" "sentence-transformers>=2.7.0" torch --extra-index-url https://download.pytorch.org/whl/cpu pandas numpy
# -----------------------------------------------------------

import ast
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

MILVUS_URI = "http://34.135.232.156:19530"
COLLECTION = "candidate_pool"
CSV_PATH = "candidate_pool_1000.csv"

def to_list(s):
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else []
        return v if isinstance(v, (list, tuple)) else [str(s)]
    except Exception:
        return [str(s)] if isinstance(s, str) else []

def build_summary_text(row):
    chunks = [
        row.get("semantic_summary", ""),
        " ".join(to_list(row.get("skills_extracted", "[]"))),
        " ".join(to_list(row.get("tools_and_technologies", "[]"))),
        " ".join(to_list(row.get("domains_of_expertise", "[]"))),
        row.get("keywords_summary","")
    ]
    return "passage: " + " ".join([c for c in chunks if isinstance(c, str)])

def build_skills_text(row):
    return "passage: " + " ".join([
        " ".join(to_list(row.get("skills_extracted","[]"))),
        " ".join(to_list(row.get("tools_and_technologies","[]")))
    ])

def embed_texts(model, texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        out.append(model.encode(texts[i:i+batch_size], normalize_embeddings=True))
    return np.vstack(out).astype(np.float32)

def main():
    client = MilvusClient(uri=MILVUS_URI, token="")
    df = pd.read_csv(CSV_PATH).fillna("")

    model = SentenceTransformer("intfloat/e5-base-v2")

    summary_texts = [build_summary_text(r) for _, r in df.iterrows()]
    skills_texts  = [build_skills_text(r)  for _, r in df.iterrows()]

    summary_vecs = embed_texts(model, summary_texts)  # (N,768)
    skills_vecs  = embed_texts(model, skills_texts)   # (N,768)

    df["summary_embedding"] = [v.tolist() for v in summary_vecs]
    df["skills_embedding"]  = [v.tolist() for v in skills_vecs]

    records = df.to_dict(orient="records")
    client.insert(collection_name=COLLECTION, data=records)

    print(f"Inserted {len(records)} rows into `{COLLECTION}`.")

if __name__ == "__main__":
    main()
