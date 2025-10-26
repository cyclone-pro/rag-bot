#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Terminal Recruiter — Milvus + 768d e5 embeddings

What it does:
- Connects to Milvus (env-configurable)
- Ensures `client_memory` exists (scalar INVERTED indexes on company/phone/email)
- Asks once for client company + phone, upserts into `client_memory`
- Enters a chat-style loop:
   * Type a natural-language requirement (JD or search query)
   * We vector-search `candidate_pool.summary_embedding` (HNSW/COSINE) with ef=96
   * Optional scalar filters (country/state/city) kept as session state
   * Shows top-K candidates with why-fields
- Commands:
    :help               -> show help
    :filters            -> show current filters
    :set country=USA    -> set filter_country (also state/city)
    :clear              -> clear filters
    :topk 5             -> set TOP_K
    :ef 128             -> set HNSW ef
    :quit               -> exit

Prereqs:
  pip install pymilvus sentence-transformers torch python-dotenv

Env:
  MILVUS_URI="http://localhost:19530"
  MILVUS_DB="default"
  CANDIDATES_COLLECTION="candidate_pool"
  CLIENT_MEMORY_COLLECTION="client_memory"
  PRIMARY_VEC_FIELD="summary_embedding"     # 768-d
  SECONDARY_VEC_FIELD="skills_embedding"    # optional (unused by default)
  EF_SEARCH="96"
  TOP_K="8"
"""

import os, sys, json, uuid, time, re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# ------------- Config -------------
MILVUS_URI   = os.getenv("MILVUS_URI", "http://34.135.232.156:19530")
MILVUS_DB    = os.getenv("MILVUS_DB", "default")
CAND_COL     = os.getenv("CANDIDATES_COLLECTION", "candidate_pool")
MEM_COL      = os.getenv("CLIENT_MEMORY_COLLECTION", "client_memory")

PRIMARY_VEC_FIELD   = os.getenv("PRIMARY_VEC_FIELD", "summary_embedding")  # 768-d
SECONDARY_VEC_FIELD = os.getenv("SECONDARY_VEC_FIELD", "skills_embedding") # 768-d (optional)
SEARCH_EF = int(os.getenv("EF_SEARCH", "96"))
TOP_K     = int(os.getenv("TOP_K", "8"))

# ------------- Milvus -------------
from pymilvus import connections, db, utility, FieldSchema, CollectionSchema, DataType, Collection

def connect_milvus():
    connections.connect("default", uri=MILVUS_URI)
    try:
        db.using_database(MILVUS_DB)
    except Exception:
        if MILVUS_DB not in db.list_database():
            db.create_database(MILVUS_DB)
        db.using_database(MILVUS_DB)

def ensure_client_memory() -> Collection:
    if utility.has_collection(MEM_COL):
        coll = Collection(MEM_COL)
    else:
        schema = CollectionSchema(fields=[
            FieldSchema("client_id", DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            FieldSchema("client_company", DataType.VARCHAR, max_length=256),
            FieldSchema("phone", DataType.VARCHAR, max_length=48),
            FieldSchema("email", DataType.VARCHAR, max_length=256),
            FieldSchema("contact_name", DataType.VARCHAR, max_length=128),
            FieldSchema("notes", DataType.VARCHAR, max_length=4096),
            FieldSchema("preferences", DataType.VARCHAR, max_length=8192),
            FieldSchema("last_queries", DataType.VARCHAR, max_length=8192),
            FieldSchema("created_at", DataType.VARCHAR, max_length=64),
            FieldSchema("updated_at", DataType.VARCHAR, max_length=64),
            FieldSchema("searches_count", DataType.INT64),
            FieldSchema("desired_headcount", DataType.INT64),
            FieldSchema("status", DataType.VARCHAR, max_length=32), # Active/OnHold/Closed
        ], description="Client contact & preferences")
        coll = Collection(MEM_COL, schema=schema, shards_num=1, consistency_level="Bounded")
        for f in ["client_company", "phone", "email", "status"]:
            try:
                coll.create_index(field_name=f, index_params={"index_type": "INVERTED"})
            except Exception as e:
                print(f"[warn] could not index {f}: {e}")
    coll.load()
    return coll

def get_candidate_collection() -> Collection:
    if not utility.has_collection(CAND_COL):
        raise RuntimeError(f"Collection '{CAND_COL}' not found. Create it first.")
    coll = Collection(CAND_COL)
    coll.load()
    return coll

# ------------- Embeddings (query) -------------
_EMBEDDER = None
def embed_query_e5(text: str) -> List[float]:
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("intfloat/e5-base-v2")
    # e5 requires "query: " prefix and cosine normalization
    vec = _EMBEDDER.encode([f"query: {text}"], normalize_embeddings=True, show_progress_bar=False)[0]
    return vec.tolist()

# ------------- Client memory upsert -------------
def upsert_client(mem: Collection, company: str, phone: str, email: Optional[str], contact_name: Optional[str]) -> str:
    try:
        rows = mem.query(expr=f'client_company == "{company}" and phone == "{phone}"',
                         output_fields=["client_id"], limit=1)
    except Exception:
        rows = []
    if rows:
        cid = rows[0]["client_id"]
        mem.update(expr=f'client_id == "{cid}"', field_name="updated_at", value=datetime.utcnow().isoformat())
        return cid

    cid = str(uuid.uuid4())
    row = {
        "client_id": cid,
        "client_company": company,
        "phone": phone,
        "email": email or "",
        "contact_name": contact_name or "",
        "notes": "",
        "preferences": "",
        "last_queries": "",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "searches_count": 0,
        "desired_headcount": 0,
        "status": "Active",
    }
    mem.insert([list(row.values())])
    mem.flush()
    return cid

# ------------- Filters / Session state -------------
@dataclass
class Session:
    client_company: Optional[str] = None
    client_phone: Optional[str] = None
    client_email: Optional[str] = None
    contact_name: Optional[str] = None
    client_id: Optional[str] = None

    filter_country: Optional[str] = None
    filter_state: Optional[str] = None
    filter_city: Optional[str] = None

    top_k: int = TOP_K
    ef: int = SEARCH_EF

    def expr(self) -> str:
        clauses = ['candidate_id != ""']
        if self.filter_country: clauses.append(f'location_country == "{self.filter_country}"')
        if self.filter_state:   clauses.append(f'location_state == "{self.filter_state}"')
        if self.filter_city:    clauses.append(f'location_city == "{self.filter_city}"')
        return " and ".join(clauses)

# ------------- Candidate fields to return -------------
CAND_OUTPUT = [
    "candidate_id","name","email","phone","linkedin_url",
    "location_city","location_state","location_country",
    "skills_extracted","tools_and_technologies","certifications",
    "top_titles_mentioned","domains_of_expertise",
    "employment_history","semantic_summary","keywords_summary",
    "total_experience_years","career_stage",
    "source_channel","hiring_manager_notes","interview_feedback",
    "offer_status","assigned_recruiter","last_updated"
]

# ------------- Ranking helpers -------------
def lexical_rank(rows: List[Dict[str,Any]], query_text: str) -> List[Tuple[float, Dict[str,Any]]]:
    # very light lexical bonus on top of vector distance
    terms = [w.lower() for w in re.findall(r"[a-zA-Z0-9#\+\.-]{3,}", query_text)][:12]
    out = []
    for r in rows:
        blob = " ".join([
            str(r.get("skills_extracted","")),
            str(r.get("tools_and_technologies","")),
            str(r.get("semantic_summary","")),
            str(r.get("employment_history","")),
            str(r.get("keywords_summary",""))
        ]).lower()
        bonus = sum(1.0 for t in terms if t in blob)
        base = 0.0
        try:
            # smaller distance => closer match; invert to score
            base = 1.0 - float(r.get("_distance", 1.0))
        except Exception:
            pass
        out.append((base + 0.05*bonus, r))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def format_candidate(r: Dict[str,Any]) -> str:
    name = r.get("name","N/A")
    loc = ", ".join([x for x in [r.get("location_city",""), r.get("location_state",""), r.get("location_country","")] if x])
    titles = r.get("top_titles_mentioned","")
    years = r.get("total_experience_years")
    yrs = f"{years:.1f}y" if isinstance(years, (int,float)) else ""
    why = []
    if r.get("skills_extracted"): why.append("skills ✓")
    if r.get("tools_and_technologies"): why.append("tools ✓")
    return f"- {name} {f'({yrs})' if yrs else ''}\n  {loc} | {titles}\n  {r.get('semantic_summary','')[:180]}{'...' if len(str(r.get('semantic_summary',''))) > 180 else ''}\n"

# ------------- Search -------------
def vector_search(cand: Collection, sess: Session, query: str, top_k: int) -> List[Dict[str,Any]]:
    qvec = embed_query_e5(query)
    hits = cand.search(
        data=[qvec],
        anns_field=PRIMARY_VEC_FIELD,
        param={"metric_type":"COSINE","params":{"ef": sess.ef}},
        limit=max(top_k*3, 24),
        expr=sess.expr(),
        output_fields=CAND_OUTPUT
    )[0]
    out = []
    for h in hits:
        row = {f: h.entity.get(f) for f in CAND_OUTPUT}
        row["_distance"] = float(h.distance)
        out.append(row)
    return out

# ------------- CLI helpers -------------
HELP_TEXT = """
Commands:
  :help                 Show this help
  :filters              Show current filters
  :set country=USA      Set filter (also supports state=IL, city=Chicago)
  :clear                Clear all filters
  :topk 5               Set number of results to show
  :ef 128               Set HNSW ef search parameter (speed/recall tradeoff)
  :quit                 Exit

Just type any requirement (e.g., "GenAI in healthcare with FHIR, remote OK")
and I'll retrieve candidates from candidate_pool.
"""

def print_filters(sess: Session):
    print(f"Filters -> country={sess.filter_country or '-'} | state={sess.filter_state or '-'} | city={sess.filter_city or '-'} | topK={sess.top_k} | ef={sess.ef}")

def handle_command(sess: Session, line: str) -> bool:
    # returns True if handled as command
    if not line.startswith(":"): return False
    cmd = line.strip()
    if cmd == ":help":
        print(HELP_TEXT)
    elif cmd == ":filters":
        print_filters(sess)
    elif cmd == ":clear":
        sess.filter_country = sess.filter_state = sess.filter_city = None
        print("Filters cleared.")
    elif cmd.startswith(":set "):
        try:
            k, v = cmd[len(":set "):].split("=", 1)
            k, v = k.strip().lower(), v.strip()
            if k == "country": sess.filter_country = v
            elif k == "state": sess.filter_state = v
            elif k == "city": sess.filter_city = v
            else: print(f"Unknown key '{k}'. Use country/state/city.")
            print_filters(sess)
        except Exception:
            print("Usage: :set country=USA  (also state=IL, city=Chicago)")
    elif cmd.startswith(":topk "):
        try:
            sess.top_k = max(1, int(cmd.split()[1]))
            print(f"TOP_K -> {sess.top_k}")
        except Exception:
            print("Usage: :topk 8")
    elif cmd.startswith(":ef "):
        try:
            sess.ef = max(8, int(cmd.split()[1]))
            print(f"ef -> {sess.ef}")
        except Exception:
            print("Usage: :ef 96")
    elif cmd == ":quit":
        print("bye!")
        sys.exit(0)
    else:
        print("Unknown command. Type :help")
    return True

# ------------- Main loop -------------
def main():
    print("Connecting to Milvus ...")
    connect_milvus()
    mem = ensure_client_memory()
    cand = get_candidate_collection()
    print(f"Connected. DB={MILVUS_DB} candidates={CAND_COL} clients={MEM_COL}")
    print()

    sess = Session()

    # Begin: ask for client info once
    print("Let's start by saving your client details (so we can greet you properly next time).")
    sess.client_company = input("Client company name: ").strip() or None
    sess.client_phone   = input("Client phone (digits or E.164): ").strip() or None
    sess.client_email   = input("Client email (optional): ").strip() or None
    sess.contact_name   = input("Your name (optional): ").strip() or None

    if sess.client_company and sess.client_phone:
        sess.client_id = upsert_client(mem, sess.client_company, sess.client_phone, sess.client_email, sess.contact_name)
        print(f"Saved client_id={sess.client_id}\n")
    else:
        print("Skipped saving client (missing company or phone). You can still search.\n")

    print("Type :help for commands. Enter a requirement to search.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not line:
            continue
        if handle_command(sess, line):
            continue

        # Vector search
        try:
            rows = vector_search(cand, sess, line, sess.top_k)
        except Exception as e:
            print(f"[error] vector search failed: {e}")
            continue

        ranked = lexical_rank(rows, line)
        top = [r for _, r in ranked][:sess.top_k]

        if not top:
            print("no matches (try relaxing filters or increasing :ef)\n")
            continue

        print(f"\nTop {len(top)} candidates:")
        for r in top:
            print(format_candidate(r))
        print("-" * 60)

if __name__ == "__main__":
    main()
