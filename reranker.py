from sentence_transformers import SentenceTransformer, util
import torch

# ✅ Models to compare
models = {
    "BAAI/bge-large-en-v1.5": SentenceTransformer("BAAI/bge-large-en-v1.5"),
    "thenlper/gte-large": SentenceTransformer("thenlper/gte-large"),
    "intfloat/e5-large-v2": SentenceTransformer("intfloat/e5-large-v2", device="cpu"),
    "sentence-transformers/all-mpnet-base-v2": SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
}

# 🧾 Example resume and job description
resume = """Python developer with 3 years of experience in Django, REST APIs, and AWS.
Worked on cloud deployments, containerized applications, and data pipelines."""
job_desc = """We are hiring a backend engineer experienced with Django, Python, and AWS.
The candidate should be familiar with REST API development and CI/CD pipelines."""

# 🧠 Compute similarities
embeddings = {}
for name, model in models.items():
    emb_resume = model.encode(resume, convert_to_tensor=True)
    emb_jd = model.encode(job_desc, convert_to_tensor=True)
    score = util.cos_sim(emb_resume, emb_jd)
    embeddings[name] = float(score.item())

# 📊 Print results
print("\n🔍 Resume ↔ JD Semantic Similarity Comparison:\n")
for name, score in embeddings.items():
    print(f"{name:<45}  →  {score:.4f}")
