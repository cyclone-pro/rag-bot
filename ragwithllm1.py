# llm_milvus_assistant.py
# -----------------------------------------------------------------------------
# Natural-language QA over your Milvus candidate_pool using OpenAI as a router.
# - Understands questions like: "total how many candidates ... construction ... CAD and Python"
# - LLM parses intent → structured plan (JSON)
# - Executes Milvus vector search + scalar/keyword filters
# - Returns count, short list, or ~50-word "why fit" depending on intent
#
# Setup:
#   pip install "pymilvus>=2.4.4" "sentence-transformers>=2.7.0" torch \
#       --extra-index-url https://download.pytorch.org/whl/cpu openai>=1.51.0
#   export OPENAI_API_KEY="sk-..."
#
# Run:
#   python llm_milvus_assistant.py
#   > ask your question (type 'exit' to quit)
# -----------------------------------------------------------------------------

import os
import json
import re
import shlex
import ast
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# ----------- CONFIG -----------
MILVUS_URI = os.environ.get("MILVUS_URI")
COLLECTION = "candidate_pool"
EMBED_FIELD_DEFAULT = "summary_embedding"  # or "skills_embedding"
EMBED_MODEL_NAME = "intfloat/e5-base-v2"
TOPK_DEFAULT = 1000  # large so we can count after post-filtering
SHOW_TOP = 20        # when listing names
# -----------------------------

# ---------- OpenAI client ----------
from openai import OpenAI
OPENAI_MODEL = "gpt-4o-mini"   # cheap+fast; change to your preferred model
client_oa = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------- Embedding model (local) ----------
model_embed = SentenceTransformer(EMBED_MODEL_NAME)

# ---------- Milvus client ----------
client = MilvusClient(uri=MILVUS_URI, token="")

# ---------- Helpers ----------
NETWORK_PAT = re.compile(
    r"\b(network|networking|subnet|vpc|tcp|udp|lan|wan|bgp|ospf|sonic|router|switch|kubernetes|docker)\b",
    re.I
)

def encode_query(text: str) -> List[float]:
    return model_embed.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()

def contains_any(text: Any, needles: List[str]) -> bool:
    s = (text or "")
    if not isinstance(s, str):
        s = str(s)
    low = s.lower()
    return any(n.lower() in low for n in needles)

def bag_from_entity(e: Dict[str, Any]) -> str:
    return " ".join([
        str(e.get("name","")),
        str(e.get("career_stage","")),
        str(e.get("skills_extracted","")),
        str(e.get("tools_and_technologies","")),
        str(e.get("domains_of_expertise","")),
        str(e.get("keywords_summary","")),
        str(e.get("semantic_summary","")),
    ])

def to_pct(sim: float) -> str:
    return f"{sim * 100:.1f}%"

def parse_list_like(value: Any) -> List[str]:
    """Normalize list-ish fields that might arrive as JSON, repr, or CSV text."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Try safe literal evaluation first (handles single-quoted reprs)
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        # Try JSON decode as a fallback
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        # Fall back to comma separation
        if "," in text:
            return [seg.strip() for seg in text.split(",") if seg.strip()]
        return [text]
    return [str(value)]

def join_head(values: List[str], limit: int = 8) -> str:
    if not values:
        return ""
    trimmed = values[:limit]
    suffix = "" if len(values) <= limit else ", ..."
    return ", ".join(trimmed) + suffix

def render_candidate(e: Dict[str, Any], sim: float, detailed: bool) -> str:
    """Craft a conversational summary for a candidate hit."""
    name = e.get("name", "Unknown candidate")
    pct = to_pct(sim)
    context_bits = [bit for bit in (e.get("career_stage"), e.get("primary_industry")) if bit]
    header = f"{name} ({pct} match"
    if context_bits:
        header += f", {' · '.join(context_bits)}"
    header += ")."

    summary = (e.get("semantic_summary") or e.get("keywords_summary") or "").strip()
    if summary.endswith("."):
        summary = summary[:-1]
    skills = join_head(parse_list_like(e.get("skills_extracted")), limit=8)
    tools = join_head(parse_list_like(e.get("tools_and_technologies")), limit=6)
    domains = join_head(parse_list_like(e.get("domains_of_expertise")), limit=5)

    sentences = [header]
    if summary:
        sentences.append(summary + ".")

    if detailed:
        if skills:
            sentences.append(f"Core skills: {skills}.")
        if tools:
            sentences.append(f"Tools & tech: {tools}.")
        if domains:
            sentences.append(f"Focus areas: {domains}.")
    else:
        highlights = []
        if skills:
            highlights.append(f"skills like {skills}")
        if domains:
            highlights.append(f"experience across {domains}")
        if tools and not highlights:
            highlights.append(f"tools such as {tools}")
        if highlights:
            sentences.append("Highlights include " + " and ".join(highlights) + ".")

    return " ".join(sentences)

def llm_plan(question: str) -> Dict[str, Any]:
    """
    Ask OpenAI to convert the natural-language question into a retrieval plan.
    The plan controls which vector field, filters, and output style we use.
    """
    system = (
        "You parse recruiting analytics questions into a JSON plan for Milvus retrieval over a resume collection. "
        "Never add fields not in schema. Output ONLY JSON. Keys:\n"
        "{\n"
        '  "intent": "count|list|why",\n'
        '  "vector_field": "summary_embedding|skills_embedding",\n'
        '  "must_have_keywords": ["keyword", ...],  // lower-case tokens to find in resume text bag\n'
        '  "industry_equals": "string or null",     // exact match for primary_industry\n'
        '  "require_domains": ["Healthcare IT","Construction","CAD","NLP","GenAI", ...],\n'
        '  "require_career_stage": "Entry|Mid|Senior|Lead/Manager|Director+|Any",\n'
        '  "networking_required": true|false,       // if question implies basic networking knowledge\n'
        '  "top_k": 1000,\n'
        '  "return_top": 20                         // for list/why\n'
        "}\n"
        "If the user asks for 'total how many', set intent='count'. If they want names, 'list'. "
        "If they want a brief justification, 'why'. Use 'summary_embedding' by default."
    )
    user = f"Question: {question}"
    resp = client_oa.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except Exception:
        # Fallback minimal plan
        return {
            "intent": "count",
            "vector_field": EMBED_FIELD_DEFAULT,
            "must_have_keywords": [],
            "industry_equals": None,
            "require_domains": [],
            "require_career_stage": "Any",
            "networking_required": False,
            "top_k": TOPK_DEFAULT,
            "return_top": SHOW_TOP,
        }

def post_filter(
    hits: List[Dict[str, Any]],
    must_have_keywords: List[str],
    industry_equals: Optional[str],
    require_domains: List[str],
    require_career_stage: str,
    networking_required: bool
) -> List[Tuple[Dict[str, Any], float]]:
    results = []
    for h in hits:
        e = h["entity"]
        sim = float(h["distance"])

        # industry equality
        if industry_equals:
            if (e.get("primary_industry") or "") != industry_equals:
                continue

        # domains containment (string list stored as text)
        doms = str(e.get("domains_of_expertise",""))
        if require_domains:
            if not any(d in doms for d in require_domains):
                continue

        # career stage
        if require_career_stage and require_career_stage != "Any":
            if (e.get("career_stage") or "") != require_career_stage:
                continue

        # must-have keywords in bag
        bag = bag_from_entity(e).lower()
        if must_have_keywords:
            if not all(k.lower() in bag for k in must_have_keywords):
                continue

        # networking heuristic
        if networking_required:
            if not NETWORK_PAT.search(bag):
                continue

        results.append((e, sim))

    return results

def search_and_answer(question: str) -> str:
    plan = llm_plan(question)

    vector_field = plan.get("vector_field") or EMBED_FIELD_DEFAULT
    top_k = int(plan.get("top_k") or TOPK_DEFAULT)
    return_top = int(plan.get("return_top") or SHOW_TOP)
    intent = plan.get("intent") or "count"

    qvec = encode_query(question)

    # Build server-side scalar filter if industry_equals provided
    filter_expr = None
    industry_equals = plan.get("industry_equals")
    if industry_equals:
        safe = industry_equals.replace('"','\\"')
        filter_expr = f'primary_industry == "{safe}"'

    res = client.search(
        collection_name=COLLECTION,
        data=[qvec],
        anns_field=vector_field,
        search_params={"metric_type":"COSINE","params":{"ef":128}},
        limit=top_k,
        output_fields=[
            "candidate_id","name","career_stage","primary_industry",
            "skills_extracted","tools_and_technologies",
            "domains_of_expertise","semantic_summary","keywords_summary"
        ],
        filter=filter_expr
    )
    hits = res[0] if res else []

    filtered = post_filter(
        hits,
        must_have_keywords=plan.get("must_have_keywords", []),
        industry_equals=industry_equals,
        require_domains=plan.get("require_domains", []),
        require_career_stage=plan.get("require_career_stage") or "Any",
        networking_required=bool(plan.get("networking_required"))
    )
    top_filtered = filtered[:return_top]

    if intent == "count":
        return f"Total matched candidates: {len(filtered)}"
    elif intent == "list":
        lines = [
            f"{idx}. {render_candidate(e, sim, detailed=False)}"
            for idx, (e, sim) in enumerate(top_filtered, start=1)
        ]
        return f"Total matched: {len(filtered)}\n\n" + "\n".join(lines) if lines else f"Total matched: {len(filtered)}"
    else:  # intent == "why"
        blocks = [render_candidate(e, sim, detailed=True) for e, sim in top_filtered]
        if not blocks:
            return f"Total matched: {len(filtered)}"
        return f"Total matched: {len(filtered)}\n\n" + "\n\n".join(blocks)

def print_help():
    print("""Commands:
  /help                     Show this help
  /exit | exit | quit | q   Quit
Type natural language questions, e.g.:
  "total how many candidates have experience in construction, know CAD and Python"
  "list top 10 names for Django + HIPAA + networking"
  "why these 5 fit FHIR + GenAI + NLP"
""")

def main():
    print("LLM+Milvus assistant ready. Ask questions (type /help).")
    while True:
        try:
            line = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        low = line.lower()
        if low in ("exit","/exit","quit","q"):
            print("Bye.")
            break
        if low == "/help":
            print_help()
            continue
        try:
            answer = search_and_answer(line)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set; set it to enable LLM parsing.")
    main()
