# search_lite.py
# -----------------------------------------------------------
# Minimal result output for Milvus semantic candidate search
# Modes:
#   /mode name   -> prints "Name | 93.4%"
#   /mode why    -> prints "Name | 93.4%  —  <~50 words why-fit>"
# Other commands:
#   /field <summary_embedding|skills_embedding>
#   /topk <int>
#   /help
# Quit with: exit / quit / q
# -----------------------------------------------------------

import re
import shlex
from typing import List, Dict, Any

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI = "http://34.135.232.156:19530"
COLLECTION = "candidate_pool"

def encode_query(model: SentenceTransformer, text: str):
    return model.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()

def percent(sim: float) -> str:
    # COSINE metric in Milvus returns similarity (0..1), higher is better
    return f"{sim * 100:.1f}%"

def search_milvus(client: MilvusClient, qvec, top_k: int, anns_field: str):
    # Only fetch the minimum we need for both modes
    return client.search(
        collection_name=COLLECTION,
        data=[qvec],
        anns_field=anns_field,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        output_fields=[
            "name",
            "skills_extracted",
            "tools_and_technologies",
            "domains_of_expertise",
            "primary_industry",
            "semantic_summary",
            "keywords_summary"
        ],
        filter=None,
    )

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\+\-_.#]+", (text or "").lower())

def build_why_fit(query: str, entity: Dict[str, Any], target_words: int = 50) -> str:
    """
    Heuristic 50-word explanation:
    - Match query tokens against skills/tools/domains/keywords
    - Emphasize overlaps and key strengths
    """
    q_tokens = set(tokenize(query))
    skills = str(entity.get("skills_extracted", ""))
    tools  = str(entity.get("tools_and_technologies", ""))
    domains = str(entity.get("domains_of_expertise", ""))
    industry = str(entity.get("primary_industry", ""))
    summary  = str(entity.get("semantic_summary", ""))
    keywords = str(entity.get("keywords_summary", ""))

    bag_text = " ".join([skills, tools, domains, keywords, summary])
    cand_tokens = set(tokenize(bag_text))

    overlaps = sorted(q_tokens.intersection(cand_tokens))
    top_overlap = ", ".join(overlaps[:6]) if overlaps else ""

    phrases = []
    if industry:
        phrases.append(f"Industry: {industry}")
    if top_overlap:
        phrases.append(f"Relevant: {top_overlap}")
    if domains:
        phrases.append(f"Domains: {domains[:80]}")
    if skills:
        phrases.append(f"Skills: {skills[:120]}")
    if tools:
        phrases.append(f"Tools: {tools[:100]}")

    text = ". ".join(phrases)
    # Trim to ~50 words
    words = text.split()
    if len(words) > target_words:
        text = " ".join(words[:target_words]) + "..."
    return text

def print_results(query: str, res, mode: str):
    hits = res[0] if res else []
    if not hits:
        print("No results.")
        return
    for h in hits:
        sim = float(h["distance"])  # COSINE similarity
        entity = h["entity"]
        name = entity.get("name") or "unknown"
        if mode == "name":
            print(f"{name} | {percent(sim)}")
        else:  # mode == "why"
            why = build_why_fit(query, entity, target_words=50)
            print(f"{name} | {percent(sim)} — {why}")

def print_help():
    print("""Commands:
  /mode <name|why>                         Output mode (default: name)
  /field <summary_embedding|skills_embedding>
                                           Vector field to search (default: summary_embedding)
  /topk <int>                              Number of results (default: 5)
  /help                                    Show help
  exit | quit | q                          Quit
Examples:
  /mode why
  /field skills_embedding
  /topk 10
  FHIR and GenAI for clinical NLP
""")

def main():
    print("Connecting & loading model...")
    client = MilvusClient(uri=MILVUS_URI, token="")
    model = SentenceTransformer("intfloat/e5-base-v2")

    mode = "name"  # or "why"
    anns_field = "summary_embedding"
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
        if low in ("exit", "quit", "q"):
            print("Bye.")
            break

        if line.startswith("/"):
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = line.split()
            cmd = parts[0].lower()

            if cmd == "/help":
                print_help()
            elif cmd == "/mode" and len(parts) >= 2:
                val = parts[1].lower()
                if val in ("name", "why"):
                    mode = val
                    print(f"Mode set to: {mode}")
                else:
                    print("Use: /mode name  or  /mode why")
            elif cmd == "/field" and len(parts) >= 2:
                val = parts[1]
                if val not in ("summary_embedding", "skills_embedding"):
                    print("Use: /field summary_embedding | skills_embedding")
                else:
                    anns_field = val
                    print(f"Field set to: {anns_field}")
            elif cmd == "/topk" and len(parts) >= 2:
                try:
                    top_k = max(1, int(parts[1]))
                    print(f"Top-K set to: {top_k}")
                except ValueError:
                    print("Top-K must be an integer.")
            else:
                print("Unknown command. /help for options.")
            continue

        # Regular query
        try:
            qvec = encode_query(model, line)
            res = search_milvus(client, qvec, top_k=top_k, anns_field=anns_field)
            print_results(line, res, mode)
        except Exception as e:
            print(f"Search error: {e}")

if __name__ == "__main__":
    main()
