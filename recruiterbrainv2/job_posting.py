from pymilvus import connections, Collection
import time

EMBEDDING_DIM = 768
FAKE_VECTOR = [0.0] * EMBEDDING_DIM

def insert_sample_jobs():
    connections.connect(host="34.55.41.188", port="19530")
    col = Collection("job_postings")

    job_ids   = ["JOB-1001", "JOB-1002", "JOB-1003", "JOB-1004", "JOB-1005"]
    titles    = [
        "Senior Python Engineer",
        "Frontend React Developer",
        "Data Engineer (Contract)",
        "DevOps Engineer (C2C)",
        "ML Engineer Intern",
    ]
    companies = ["Google", "Meta", "Netflix", "Amazon", "OpenAI"]
    departments = ["Engineering", "Product", "Data", "Cloud", "AI Research"]
    locations = ["Remote - US", "NYC, NY", "Los Angeles, CA", "Seattle, WA", "San Francisco, CA"]

    # employment_type: full_time / contract / internship
    employment_types = ["full_time", "full_time", "contract", "contract", "internship"]

    # tax_term: w2 / c2c / 1099 / na
    tax_terms = ["w2", "w2", "1099", "c2c", "na"]

    salary_ranges = ["150k–180k", "130k–160k", "70–90/hr", "80–100/hr", "40–50/hr"]
    req_ids = ["REQ-101", "REQ-102", "REQ-103", "REQ-104", "REQ-105"]
    status = ["active"] * 5
    posted_ts = [int(time.time())] * 5

    jd_texts = [
        "Senior Python Engineer working on backend services with Django, REST, and GCP.",
        "React developer building modern web UIs with TypeScript and GraphQL.",
        "Contract Data Engineer building ETL pipelines on Spark and BigQuery.",
        "C2C DevOps Engineer focused on Kubernetes, Terraform, and AWS.",
        "ML Engineer intern supporting LLM fine-tuning, evaluation, and data pipelines.",
    ]
    embeddings = [FAKE_VECTOR for _ in range(5)]

    entities = [
        job_ids,
        titles,
        companies,
        departments,
        locations,
        employment_types,
        tax_terms,
        salary_ranges,
        req_ids,
        status,
        posted_ts,
        jd_texts,
        embeddings,
    ]

    res = col.insert(entities)
    col.flush()
    print("✅ Inserted sample jobs:", res.primary_keys)

if __name__ == "__main__":
    insert_sample_jobs()
