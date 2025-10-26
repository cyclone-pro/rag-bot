from pymilvus import MilvusClient

COLL = "resume_chunks"
client = MilvusClient(uri="http://34.135.232.156:19530")

# 1) Drop current index (AUTOINDEX) on the vector field
client.drop_index(collection_name=COLL, field_name="embeddings")

# 2) Create HNSW index
idx = client.prepare_index_params()
idx.add_index(
    field_name="embeddings",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 32, "efConstruction": 200},
)
client.create_index(collection_name=COLL, index_params=idx)

# 3) Load collection
client.load_collection(collection_name=COLL)
print("Switched to HNSW.")
