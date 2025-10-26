# search_cli.py
# -----------------------------------------------------------
# Interactive “type-a-query, get results” CLI for Milvus (HNSW/COSINE)
# - Loads local intfloat/e5-base-v2 once
# - Loops until you type: exit / quit / q
# - Commands:
#     /field <summary_embedding|skills_embedding>
#     /industry <IndustryName or ->  (use '-' to clear filter)
#     /topk <int>
#     /help
# -----------------------------------------------------------
# pip install "pymilvus>=2.4.4" "sentence-transformers>=2.7.0" torch --extra-index-url https://download.pytorch.org/whl/cpu

import sys
import shlex
from typing import Optional

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = "http://34.135.232.156:19530"
COLLECTION = "candidate_pool"

def run_search(
    client: MilvusClient,
    model: SentenceTransformer,
    query: str,
    top_k: int = 5,
    anns_field: str = "summary_embedding",
    industry: Optional[str] = None,
):
    qvec = model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()

    filter_expr = None
    if industry:
        val = industry.replace('"', '\\"')
        filter_expr = f'primary_industry == "{val}"'

    res = client.search(
        collection_name=COLLECTION,
        data=[qvec],
        anns_field=anns_field,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        output_fields=[
            "candidate_id","name","primary_industry",
            "skills_extracted","tools_and_technologies","semantic_summary"
        ],
        filter=filter_expr
    )
    return res

def pretty_print_results(res):
    if not res or not res[0]:
        print("No results.")
        return
    for i, hit in enumerate(res[0], 1):
        entity = hit["entity"]
        dist = hit.get("distance", 0.0)
        print(f"{i}. {entity.get('candidate_id')} | {entity.get('name')} | dist={dist:.4f}")
        print(f"   Industry: {entity.get('primary_industry')}")
        print(f"   Skills:   {entity.get('skills_extracted')}")
        print(f"   Tools:    {entity.get('tools_and_technologies')}")
        summary = entity.get("semantic_summary") or ""
        print(f"   Summary:  {summary[:200]}{'...' if len(summary)>200 else ''}")
        print("-" * 80)

def print_help():
    print("""Commands:
  /field <summary_embedding|skills_embedding>   Set which vector field to search (default: summary_embedding)
  /industry <IndustryName or ->                Set scalar filter on primary_industry (use '-' to clear)
  /topk <int>                                  Set number of results (default: 5)
  /help                                        Show this help
  /exit | exit | quit | q                      Quit the CLI

Examples:
  /field skills_embedding
  /industry Healthcare
  /topk 10
  FHIR and GenAI for clinical NLP
""")

def main():
    print("Connecting to Milvus and loading local embedding model...")
    client = MilvusClient(uri=MILVUS_URI, token="")
    model = SentenceTransformer("intfloat/e5-base-v2")

    anns_field = "summary_embedding"
    industry = None
    top_k = 5

    print("Ready. Type your query (or /help). Type 'exit' to quit.\n")

    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        low = line.lower()
        if low in ("exit", "quit", "q", "/exit"):
            print("Bye.")
            break

        if line.startswith("/"):
            # handle command
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = line.split()

            cmd = parts[0].lower()
            if cmd in ("/help",):
                print_help()

            elif cmd in ("/field",) and len(parts) >= 2:
                val = parts[1]
                if val not in ("summary_embedding", "skills_embedding"):
                    print("Invalid field. Use: summary_embedding | skills_embedding")
                else:
                    anns_field = val
                    print(f"Search field set to: {anns_field}")

            elif cmd in ("/industry",) and len(parts) >= 2:
                val = parts[1]
                if val == "-" or val.lower() in ("none","null"):
                    industry = None
                    print("Industry filter cleared.")
                else:
                    industry = val
                    print(f"Industry filter set to: {industry}")

            elif cmd in ("/topk",) and len(parts) >= 2:
                try:
                    top_k = max(1, int(parts[1]))
                    print(f"Top-K set to: {top_k}")
                except ValueError:
                    print("Top-K must be an integer.")

            else:
                print("Unknown command. Type /help for options.")
            continue

        # regular query
        try:
            res = run_search(client, model, line, top_k=top_k, anns_field=anns_field, industry=industry)
            pretty_print_results(res)
        except Exception as e:
            print(f"Search error: {e}")

if __name__ == "__main__":
    main()
