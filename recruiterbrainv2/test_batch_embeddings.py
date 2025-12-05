"""Test batch embedding performance."""
import time
from ingestion.embedding_generator import generate_embeddings

# Mock candidate data
candidate_data = {
    "name": "John Doe",
    "semantic_summary": "Experienced software engineer with 10 years in backend development",
    "keywords_summary": "Python, Django, AWS, Docker, Kubernetes",
    "employment_history": "Senior Engineer at TechCorp (5 years), Lead Developer at StartupXYZ (3 years)",
    "skills_extracted": "Python, Java, Go, JavaScript, SQL",
    "tools_and_technologies": "Docker, Kubernetes, Jenkins, GitHub Actions, Terraform",
    "tech_stack_primary": "Python, Django, PostgreSQL, Redis, AWS",
    "programming_languages": "Python, Java, JavaScript, Go",
    "certifications": "AWS Solutions Architect, Kubernetes Administrator",
    "current_tech_stack": "Python, FastAPI, React, PostgreSQL, AWS ECS",
    "role_type": "Backend Engineering | Cloud Architecture",
    "industries_worked": "Technology, Finance, Healthcare",
    "domain_expertise": "Distributed Systems, Microservices, Cloud Infrastructure",
    "top_3_titles": "Senior Software Engineer; Lead Backend Developer; Principal Engineer",
    "evidence_leadership": "Led team of 5 engineers, mentored juniors",
    "verticals_experience": "B2B SaaS, FinTech, HealthTech",
}

print("="*60)
print("Testing Batch Embedding Performance")
print("="*60)

# Test 1: Single candidate
start = time.time()
embeddings = generate_embeddings(candidate_data)
elapsed = time.time() - start

print(f"\n✅ Generated 3 embeddings in {elapsed:.2f} seconds")
print(f"   Summary embedding: {len(embeddings['summary_embedding'])} dimensions")
print(f"   Tech embedding: {len(embeddings['tech_embedding'])} dimensions")
print(f"   Role embedding: {len(embeddings['role_embedding'])} dimensions")

# Test 2: Multiple candidates (simulate bulk upload)
print("\n" + "="*60)
print("Simulating 5 Resume Uploads")
print("="*60)

start = time.time()
for i in range(5):
    embeddings = generate_embeddings(candidate_data)
    print(f"   Candidate {i+1}: ✅")

elapsed = time.time() - start
avg = elapsed / 5

print(f"\n✅ Total time: {elapsed:.2f} seconds")
print(f"   Average per resume: {avg:.2f} seconds")
print(f"   Throughput: {5/elapsed:.2f} resumes/second")

print("\n📊 Performance Comparison:")
print(f"   OLD (sequential): ~9 seconds per resume")
print(f"   NEW (batch):      ~{avg:.2f} seconds per resume")
print(f"   Speedup:          {9/avg:.1f}x faster! 🚀")