"""User workflow + LLM planning glue for recruiter brain."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import re  
from recruiterbrain.core_retrieval import ann_search
from recruiterbrain.shared_config import (
    INSIGHT_DEFAULT_K,
    LABEL_ORDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ROUTER_CONFIDENCE_THRESHOLD,
    RETURN_TOP,
    TOP_K,
    VECTOR_FIELD_DEFAULT,
    get_openai_client,
)
from recruiterbrain.shared_utils import (
    _norm, 
    brief_why,
    coverage,
    data_quality_check,
    evidence_snippets,
    extract_latest_title,
    extract_overlaps,
    format_row,
    normalize_tools,
    notes_label,
    percentiles_min_rank,
    render_candidate,
    render_position,
    select_industries,
    tier_label,
)

logger = logging.getLogger(__name__)

CONTACT_PHRASES = [
    "contacts",
    "contact",
    "give their info",
    "give me emails",
    "emails",
    "email ids",
    "gmail",
    "phone",
    "phone number",
    "phone numbers",
    "linkedin",
    "linked in",
    "send email",
    "send them email",
    "share phone numbers",
    "share contacts",
    "give info",
]
JD_TOOL_HINTS = {
    "python": ["python"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "mlflow": ["mlflow"],
    "kubeflow": ["kubeflow"],
    "sagemaker": ["sagemaker"],
    "airflow": ["airflow"],
    "kafka": ["kafka"],
    "redis": ["redis"],
    "kubernetes": ["kubernetes"],
    "docker": ["docker"],
    "aws": ["aws"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "gen ai": ["gen ai", "genai"],
    "llm": ["llm", "large language model"],
    "transformers": ["transformers"],
    "spark": ["spark"],
    "bigquery": ["bigquery"],
    "vertex ai": ["vertex ai"],
}


def extract_tools_from_jd(jd_text: str) -> List[str]:
    """
    Very simple heuristic to pull tool names out of a JD.
    You can later replace this with an LLM call that fills a structured JD object.
    """
    if not jd_text:
        return []

    text = jd_text.lower()
    found: List[str] = []

    for canonical, variants in JD_TOOL_HINTS.items():
        if any(v in text for v in variants):
            found.append(canonical)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in found:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

def _default_plan(question: str) -> Dict[str, Any]:
    q = (question or "").lower()

    # If user explicitly asks "how many / total", treat as a count query.
    if "how many" in q or "total" in q or "count" in q:
        intent = "count"
    else:
        intent = "list"
    

    return {
        "intent": intent,
        "vector_field": VECTOR_FIELD_DEFAULT,
        "must_have_keywords": [],
        "industry_equals": [],
        "require_domains": [],
        "require_career_stage": "Any",
        "networking_required": False,
        "top_k": TOP_K,
        "return_top": RETURN_TOP,
        "question": question,
    }



DEFAULT_REQUIRED_TOOLS = ["milvus", "dbt", "aws", "vertex ai"]
INSIGHT_KEYWORDS = {
    "compare",
    "comparison",
    "rank",
    "ranking",
    "best",
    "notes",
    "missing",
    "tiers",
    "tier",
    "perfect",
    "good",
    "partial",
    "scorecard",
    "stack rank",
    "stackrank",
}
CONTACT_TRIGGERS = ["give their info", "linkedin", "gmail", "email", "contact"]

LAST_INSIGHT_RESULT: Optional[Dict[str, Any]] = None

# For stripping out contact/meta phrases before we create embeddings or keyword filters
def strip_contact_meta_phrases(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for phrase in CONTACT_PHRASES:
        # simple case-insensitive removal
        cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE)
    # normalize extra spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


JD_MARKERS = [
    "job title:",
    "role:",
    "roles and responsibilities:",
    "responsibilities:",
    "responsibility:",
    "qualifications:",
    "requirements:",
    "requirement:",
    "skills required:",
    "skills:",
    "experience:",
]

GREETING_MARKERS = [
    "hi",
    "hello",
    "dear",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "How are you",
    "How's it going",
    "How's everything",
    "How's everything going",
    "How's everything going",

]

SIGNATURE_MARKERS = [
    "regards",
    "thanks",
    "thank you",
    "sincerely",
    "best regards",
    "warm regards",
]

INDUSTRY_ALIASES = {
    "healthcare": "Healthcare",
    "healthcare it": "Healthcare",
    "health care": "Healthcare",
    "logistics": "Logistics",
    "logistic": "Logistics",
    "finance": "Finance",
    "banking": "Finance",
    "education": "Education",
    "manufacturing": "Manufacturing",
    "retail": "Retail",
    "energy": "Energy",
    "government": "Government",
}

def canonicalize_industry(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    return INDUSTRY_ALIASES.get(v)

def is_jd_mode(text: str) -> bool:
    """
    Heuristic JD detector (RULE #2, RULE #8):
    - length > 300 chars
    - OR standard JD markers
    - OR obvious recruiter email with bullets/signature
    """
    if not text:
        return False

    t = text.lower()

    if len(t) > 300:
        return True

    if any(marker in t for marker in JD_MARKERS):
        return True

    # many bullet points
    bullet_count = len(re.findall(r"(^[-*•]\s+)", text, flags=re.MULTILINE))
    if bullet_count >= 5:
        return True

    # recruiter signature
    if any(sig in t for sig in SIGNATURE_MARKERS):
        return True

    if "this is a jd" in t or "job description" in t:
        return True

    return False


def clean_jd_text(raw_text: str) -> str:
    """
    Remove greetings, signatures, boilerplate and enforce max size (RULE #2, RULE #10).
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # strip greeting on first line
    lines = text.splitlines()
    cleaned_lines = []
    for i, line in enumerate(lines):
        l = line.strip().lower()
        if i == 0 and any(l.startswith(g) for g in GREETING_MARKERS):
            # drop greeting line
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # remove signature block
    lowered = text.lower()
    for sig in SIGNATURE_MARKERS:
        if sig in lowered:
            idx = lowered.index(sig)
            text = text[:idx].strip()
            break

    # remove some obvious boilerplate phrases
    boilerplate_phrases = [
        "if you are interested, please reply",
        "kindly reply",
        "feel free to reach out",
        "share your updated resume",
        "looking forward to hearing from you",
        "have a great day",
    ]
    for bp in boilerplate_phrases:
        text = re.sub(re.escape(bp), "", text, flags=re.IGNORECASE)

    # length control: we accept up to 8000 but we may later summarize
    if len(text) > 8000:
        # TODO: call your summarizer here; stub = truncate
        text = text[:3000]

    return text.strip()


def build_jd_semantic_query(cleaned_jd: str) -> str:
    """
    Compact JD into 300–500 char semantic query for embeddings (RULE #10).
    For now, just take first ~500 chars; you can replace with LLM summary later.
    """
    text = cleaned_jd.strip()
    # very basic compression: drop extra whitespace and cut
    text = re.sub(r"\s+", " ", text)
    return text[:500]
def _normalize_text_list(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    normalized: List[str] = []
    for value in values:
        text = str(value).strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _derive_required_tools(plan: Dict[str, Any]) -> List[str]:
    collected: List[str] = []
    for key in ("required_tools", "must_have_keywords"):
        for token in _normalize_text_list(plan.get(key)):
            if token not in collected:
                collected.append(token)
    return collected


def _route_insight(question: str) -> tuple[bool, float]:
    text = (question or "").lower()
    matches = sum(1 for keyword in INSIGHT_KEYWORDS if keyword in text)
    if "insight" in text:
        matches += 2
    is_insight = matches > 0
    if not is_insight:
        logger.debug("Router did not classify question as insight: %s", question)
        return False, 0.35
    confidence = 0.5 + 0.08 * min(matches, 5)
    if "tier" in text or "perfect" in text:
        confidence += 0.05
    return True, min(0.95, confidence)


def _prettify_tool(token: str) -> str:
    if not token:
        return token
    if token.isupper():
        return token
    return " ".join(word.capitalize() for word in token.split())


def _clarifier_prompt(tools: List[str]) -> str:
    sample = tools[:4] or DEFAULT_REQUIRED_TOOLS
    pretty = ", ".join(_prettify_tool(tool) for tool in sample)
    return f"Did you want an insight ranking on tools {pretty}?"


def _tools_match_line(
    required: List[str],
    normalized_tools: set[str],
    weak_hits: Dict[str, List[str]],
    missing: List[str],
) -> str:
    """
    Render a per-tool status line like:
      ✅ OpenAI API, ⚠️ Airflow (~Prefect), ❌ GCP (Score: 3/4; Missing: GCP)
    """
    # simple local normalization so we don't depend on _norm here
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    missing_norm = {norm(m) for m in missing}
    norm_tools = {norm(t) for t in normalized_tools}
    parts: List[str] = []

    for tool in required:
        pretty = _prettify_tool(tool)
        t_norm = norm(tool)

        if t_norm in missing_norm:
            parts.append(f"❌ {pretty}")
        elif weak_hits.get(tool):
            # weak hits are “cousins” like AWS vs Amazon Web Services, GCP vs Google Cloud Platform, etc.
            alts = "/".join(_prettify_tool(t) for t in weak_hits[tool])
            parts.append(f"⚠️ {pretty} (~{alts})")
        elif any(
            t_norm == nt or t_norm in nt or nt in t_norm
            for nt in norm_tools
        ):
            # strong match in candidate tools
            parts.append(f"✅ {pretty}")
        else:
            # fallback – should be rare if missing is correctly computed
            parts.append(f"❌ {pretty}")

    total = len(required)
    covered_count = total - len(missing)
    score = f"{covered_count}/{total}" if total else "0/0"

    if missing:
        missing_pretty = ", ".join(_prettify_tool(m) for m in missing)
        return f"{', '.join(parts)} (Score: {score}; Missing: {missing_pretty})"

    return f"{', '.join(parts)} (Score: {score})"

def _is_greeting(text: str) -> bool:
    """
    Only treat *short* one-line messages as greetings.
    We do NOT want to accidentally classify full JDs or long emails.
    """
    if not text:
        return False

    t = text.strip().lower()

    # If it's long or has multiple words, it's probably not just "hi" / "hello".
    if len(t) > 60:
        return False
    if "\n" in t:
        return False

    words = t.split()
    if len(words) > 5:
        return False

    base_greetings = {"hi", "hello", "hey", "yo", "yoo", "buddy", "bro","how","doing","yo yo"}
    if t in base_greetings:
        return True

    # Short patterns like "hi team", "hello john", etc.
    if words and words[0] in {"hi", "hello", "hey"}:
        return True

    return False

def _infer_industry_equals(question: str, plan: Dict[str, Any]) -> None:
    """
    Best-effort rule to set industry_equals based on natural language,
    but only if the LLM planner did not already set it.
    """
    # If LLM already decided, don't override.
    if plan.get("industry_equals"):
        return

    text = (question or "").lower()

    # Healthcare
    if "healthcare" in text and "candidate" in text:
        plan["industry_equals"] = "Healthcare"
        logger.info("Heuristic set industry_equals='Healthcare' based on question: %s", question)
        return

    # Finance / Banking
    if ("finance" in text or "banking" in text) and "candidate" in text:
        plan["industry_equals"] = "Finance"
        logger.info("Heuristic set industry_equals='Finance' based on question: %s", question)
        return

    if "construction" in text and "candidate" in text:
        plan["industry_equals"] = "Construction"
        logger.info("Heuristic set industry_equals='Construction' based on question: %s", question)
        return
    if "manufacturing" in text and "candidate" in text:
        plan["industry_equals"] = "Manufacturing"
        logger.info("Heuristic set industry_equals='Manufacturing' based on question: %s", question)
        return
    if "logistics" in text and "candidate" in text:
        plan["industry_equals"] = "Logistics"
        logger.info("Heuristic set industry_equals='Logistics' based on question: %s", question)
        return
    if "government" in text and "candidate" in text:
        plan["industry_equals"] = "Government"
        logger.info("Heuristic set industry_equals='Government' based on question: %s", question)
        return
    if "education" in text and "candidate" in text:
        plan["industry_equals"] = "Education"
        logger.info("Heuristic set industry_equals='Education' based on question: %s", question)
        return
    if "energy" in text and "candidate" in text:
        plan["industry_equals"] = "Energy"
        logger.info("Heuristic set industry_equals='Energy' based on question: %s", question)
        return
    if "retail" in text and "candidate" in text:
        plan["industry_equals"] = "Retail"
        logger.info("Heuristic set industry_equals='Retail' based on question: %s", question)
        return

def _augment_plan_with_heuristics(original_question: str, plan: Dict[str, Any]) -> None:
    """
    Simple deterministic tweaks on top of the LLM plan so that
    obvious phrases like 'django', 'spring boot', 'spring', 'healthcare'
    actually show up in must_have_keywords / require_domains.
    """
    text = (original_question or "").lower()

    # ---- Skills / tools keywords ----
    must = _normalize_text_list(plan.get("must_have_keywords", []))

    def add_kw(token: str):
        t = (token or "").lower().strip()
        if t and t not in must:
            must.append(t)

    if "django" in text:
        add_kw("django")

    # Spring / Spring Boot variants
    if "spring boot" in text or "springboot" in text:
        # CVs might say 'spring boot', 'springboot', or just 'spring'
        add_kw("spring boot")
        add_kw("springboot")
        add_kw("spring")
    elif " spring " in f" {text} " or text.strip().startswith("spring "):
        # plain 'spring' query
        add_kw("spring")

    plan["must_have_keywords"] = must

    # ---- Domain / industry-like phrases ----
    domains = _normalize_text_list(plan.get("require_domains", []))

    def add_domain(token: str):
        t = (token or "").lower().strip()
        if t and t not in domains:
            domains.append(t)

    if "healthcare" in text:
        add_domain("healthcare")
    if "logistics" in text:
        add_domain("logistics")
    if "construction" in text:
        add_domain("construction")
    if "manufacturing" in text:
        add_domain("manufacturing")
    if "retail" in text:
        add_domain("retail")
    if "government" in text or "public sector" in text:
        add_domain("government")

    plan["require_domains"] = domains

"""
def llm_plan(question: str) -> Dict[str, Any]:
   
    TEMP: bypass OpenAI planner and use a simple default plan
    so we can debug Milvus + filtering first.
    
    plan = _default_plan(question)
    # make sure question is included so embedding uses it
    plan["question"] = question
    logger.info("Using default plan (LLM disabled) for question='%s': %s", question, plan)
    return plan

"""

def llm_plan(question: str) -> Dict[str, Any]:
    client = get_openai_client()
    logger.info("get_openai_client returned: %r", client)

    plan: Dict[str, Any]

    if not client:
        logger.debug("LLM planner unavailable; using default plan")
        plan = _default_plan(question)
    else:
        system_prompt = (
            "You parse recruiting analytics questions into a JSON plan for Milvus retrieval over a resume collection. "
            "Never add fields not in schema. Output ONLY JSON. Keys:\n"
            "{\n"
            '  "intent": "count|list|why",\n'
            '  "vector_field": "summary_embedding|skills_embedding",\n'
            '  "must_have_keywords": ["keyword", ...],\n'
            '  "industry_equals": "string or null",\n'
            '  "require_domains": ["Healthcare IT","Construction","CAD","NLP","GenAI", ...],\n'
            '  "require_career_stage": "Entry|Mid|Senior|Lead/Manager|Director+|Any",\n'
            '  "networking_required": true|false,\n'
            '  "top_k": 1000,\n'
            '  "return_top": 20\n'
            "}\n"
            "If the user asks for 'total how many', set intent='count'. If they want names, 'list'. "
            "If they want a brief justification, 'why'. Use 'summary_embedding' by default."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.1,
                messages=messages,
            )
            raw = resp.choices[0].message.content.strip()
            plan = json.loads(raw)
        except Exception as exc:
            logger.warning("LLM planner failed, falling back to heuristic: %s", exc)
            plan = _default_plan(question)

    # --- Normalize + defaults ---
    plan = plan or {}
    plan.setdefault("vector_field", VECTOR_FIELD_DEFAULT)
    plan.setdefault("top_k", TOP_K)
    plan.setdefault("return_top", RETURN_TOP)
    plan.setdefault("intent", "count")
    plan.setdefault("must_have_keywords", [])
    plan.setdefault("require_domains", [])
    plan.setdefault("require_career_stage", "Any")
    plan.setdefault("networking_required", False)
    plan["question"] = question

    # --- NEW: heuristic industry inference (e.g. 'healthcare candidates') ---
    _infer_industry_equals(question, plan)
    plan["industry_equals"] = canonicalize_industry(plan.get("industry_equals"))


    # --- Insight routing ---
    is_insight, router_conf = _route_insight(question)
    if is_insight:
        plan["intent"] = "insight"
        plan["k"] = int(plan.get("k") or INSIGHT_DEFAULT_K)
        logger.info("Router promoted question to insight intent (confidence=%.2f)", router_conf)
    plan["_router_confidence"] = router_conf

    # --- Required tools derivation ---
    normalized_tools = _derive_required_tools(plan)
    plan["required_tools"] = normalized_tools

    # If lots of tools, treat as insight even if LLM didn't say so.
    if plan.get("intent") != "insight" and len(normalized_tools) >= 3:
        plan["intent"] = "insight"
        plan["k"] = int(plan.get("k") or INSIGHT_DEFAULT_K)
        router_conf = max(router_conf, ROUTER_CONFIDENCE_THRESHOLD + 0.05)
        plan["_router_confidence"] = router_conf

    # --- Clarifier for low-confidence insight ---
    if plan.get("intent") == "insight":
        plan["k"] = int(plan.get("k") or INSIGHT_DEFAULT_K)
        if router_conf < ROUTER_CONFIDENCE_THRESHOLD:
            logger.info(
                "Router confidence %.2f below threshold; requesting clarification",
                router_conf,
            )
            plan["clarify"] = _clarifier_prompt(normalized_tools)
            return plan

    return plan


def _should_show_contacts(user_last_message: str) -> bool:
    text = (user_last_message or "").lower()
    return any(trigger in text for trigger in CONTACT_TRIGGERS)


def _scarcity_message(tier_counts: Dict[str, int], missing_counter: Dict[str, int]) -> Optional[str]:
    if tier_counts.get("Perfect", 0) == 0 and tier_counts.get("Good", 0) == 0:
        if not missing_counter:
            return "Few strong matches. Consider allowing 2/4 or accepting Pinecone/Weaviate as equivalents."
        top_missing = max(missing_counter.items(), key=lambda item: item[1])[0]
        top_missing_pretty = _prettify_tool(top_missing)
        return (
            f"Most candidates missing {top_missing_pretty}. Consider allowing 2/4 or "
            "accepting Pinecone/Weaviate as equivalents."
        )
    return None



def answer_question(question: str, plan_override: Optional[Dict[str, Any]] = None) -> str:
    global LAST_INSIGHT_RESULT

    original_question = question or ""

    if _is_greeting(original_question):
        # Short friendly intro instead of hammering Milvus
        return 'Hey John how can i help you with! Ask me things like "compare Milvus, dbt, AWS, Vertex AI" or "list top 10 healthcare candidates".'

    # --- Build or reuse a plan (LLM planner) ---
    # For override, we respect the caller's plan but still sanitize later.
    if plan_override is not None:
        plan = dict(plan_override)
    else:
        # For planning we already prefer the version without contact/meta noise.
        cleaned_for_plan = strip_contact_meta_phrases(original_question)
        plan = llm_plan(cleaned_for_plan)

    if not isinstance(plan, dict):
        return str(plan)

    # --- JD detection & embedding_query construction (RULE #2, RULE #9, RULE #10) ---
    jd_mode = is_jd_mode(original_question)
    plan["_jd_mode"] = jd_mode

    if jd_mode:
        cleaned_jd = clean_jd_text(original_question)
        semantic_query = build_jd_semantic_query(cleaned_jd)
        plan["_jd_raw"] = cleaned_jd
        # This is what will be embedded
        plan["embedding_query"] = semantic_query
        # Also make sure 'question' is this compact semantic JD string,
        # so any downstream uses of question stay relatively clean.
        plan["question"] = semantic_query
        # sensible defaults for JD flow
        plan.setdefault("intent", "insight")
        plan.setdefault("vector_field", VECTOR_FIELD_DEFAULT)
        plan.setdefault("top_k", TOP_K)
        plan.setdefault("return_top", RETURN_TOP)
            # ✅ Derive required tools from the JD text
        jd_tools = extract_tools_from_jd(cleaned_jd)
        if jd_tools:
          plan["required_tools"] = jd_tools

    else:
        # Normal mode: remove contact/meta words BEFORE embedding.
        cleaned_for_embed = strip_contact_meta_phrases(original_question)
        plan["embedding_query"] = cleaned_for_embed
        plan["question"] = cleaned_for_embed
    
    _augment_plan_with_heuristics(original_question, plan)
   # --- Remove contact terms from filters as well (RULE #1 & #7) ---
    contact_set = {c.lower() for c in CONTACT_PHRASES}

    def _clean_keyword_list(values: Any) -> List[str]:
        items = _normalize_text_list(values)
        return [v for v in items if all(t not in v for t in contact_set)]

    plan["must_have_keywords"] = _clean_keyword_list(plan.get("must_have_keywords", []))
    plan["required_tools"] = _clean_keyword_list(plan.get("required_tools", []))

    # --- LAST-RESORT SAFETY: ensure embedding_query/question are set ---
    embed_text = plan.get("embedding_query") or plan.get("question")
    if not embed_text:
        # Rebuild from the original user text (contact-stripped) so ANN never breaks
        cleaned_for_embed = strip_contact_meta_phrases(original_question)
        fallback = cleaned_for_embed or original_question
        plan["embedding_query"] = fallback
        plan["question"] = fallback
        logger.warning(
            "Plan missing embedding_query/question; rebuilt from original_question (len=%d)",
            len(fallback or ""),
        )

    # --- Core retrieval ---
        # --- LAST-RESORT SAFETY: ensure embedding_query/question are set ---
    embed_text = plan.get("embedding_query") or plan.get("question")
    if not embed_text:
        cleaned_for_embed = strip_contact_meta_phrases(original_question)
        fallback = cleaned_for_embed or original_question
        plan["embedding_query"] = fallback
        plan["question"] = fallback
        logger.warning(
            "Plan missing embedding_query/question; rebuilt from original_question (len=%d)",
            len(fallback or ""),
        )

    logger.info("Final plan for question '%s': %s", original_question, plan)

    
    try:
        paired_hits, total_matches = ann_search(plan)
    except Exception as exc:
        # Log full traceback server-side
        logger.exception("Error during ann_search for question: %s", original_question)
        # TEMP: return error message to UI so we see what is breaking
        return f"ANN search error: {exc}"

    intent = (plan.get("intent") or "count").lower()


    # ==========================
    # NON-INSIGHT BRANCH
    # ==========================
    if intent != "insight":
        LAST_INSIGHT_RESULT = None
        logger.debug("Handling %s intent for question", intent)
        return_top = int(plan.get("return_top") or RETURN_TOP)
        top_hits = paired_hits[:return_top]

        # detect if user wants contacts (use original text!)
        show_contacts = _should_show_contacts(original_question)

        if intent == "count":
            return f"Total matched candidates: {total_matches}"

        if intent == "list":
            lines: List[str] = []
            for idx, (entity, sim) in enumerate(top_hits, start=1):
                base = render_candidate(entity, sim, detailed=False)

                if show_contacts:
                    contacts_bits: List[str] = []
                    linkedin = entity.get("linkedin_url")
                    email = entity.get("email")
                    phone = entity.get("phone")

                    # Optional: mask LinkedIn as markdown link
                    if linkedin:
                        name = entity.get("name") or f"Candidate {idx}"
                        contacts_bits.append(f"LinkedIn: [{name}]({linkedin})")
                    if email:
                        contacts_bits.append(f"Email: {email}")
                    if phone:
                        contacts_bits.append(f"Phone: {phone}")

                    if contacts_bits:
                        base = base + " | " + " | ".join(contacts_bits)

                lines.append(f"{idx}. {base}")

            body = "\n".join(lines)
            return (
                f"Total matched: {total_matches}\n\n{body}"
                if body
                else f"Total matched: {total_matches}"
            )

        # Fallback: detailed blocks if some other intent (e.g. 'why')
        blocks = [render_candidate(entity, sim, detailed=True) for entity, sim in top_hits]
        if not blocks:
            return f"Total matched: {total_matches}"
        return f"Total matched: {total_matches}\n\n" + "\n\n".join(blocks)

    # ==========================
    # INSIGHT BRANCH
    # ==========================
    return_top = int(plan.get("k") or plan.get("return_top") or INSIGHT_DEFAULT_K)
    top_hits = paired_hits[:return_top]
    logger.info("Generated %d insight rows (total matches: %d)", len(top_hits), total_matches)

    required = [tool.strip().lower() for tool in plan.get("required_tools", []) if tool]
    if not required and not plan.get("_jd_mode") :
        required = list(DEFAULT_REQUIRED_TOOLS)

    tier_buckets: Dict[str, List[Dict[str, Any]]] = {label: [] for label in LABEL_ORDER}
    missing_counter: Dict[str, int] = {}

    for entity, sim in top_hits:
        tools, _ctx = normalize_tools(entity)
        covered, missing, weak_hits = coverage(required, tools)
        for miss in missing:
            missing_counter[miss] = missing_counter.get(miss, 0) + 1
        label = tier_label(covered)
        tier_buckets.setdefault(label, []).append(
            {
                "entity": entity,
                "sim": float(sim),
                "covered": covered,
                "missing": missing,
                "weak_hits": weak_hits,
                "tools": tools,
            }
        )

    def _bucket_sort(item: Dict[str, Any]) -> tuple[float, float]:
        years = float(item["entity"].get("total_experience_years") or 0.0)
        return (-item["sim"], -years)

    for label in LABEL_ORDER:
        tier_buckets.setdefault(label, [])
        tier_buckets[label].sort(key=_bucket_sort)

    final_entries: List[Dict[str, Any]] = []
    for label in LABEL_ORDER:
        bucket = tier_buckets[label]
        for idx, entry in enumerate(bucket):
            entry_copy = dict(entry)
            entry_copy["tier"] = label
            entry_copy["rank_in_perfect"] = idx if label == "Perfect" else None
            final_entries.append(entry_copy)

    sims_final = [entry["sim"] for entry in final_entries]
    percentiles = percentiles_min_rank(sims_final) if sims_final else []
    show_contacts = _should_show_contacts(original_question)

    formatted_rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(final_entries):
        entity = entry["entity"]
        title = extract_latest_title(entity.get("employment_history"), entity.get("top_titles_mentioned"))
        position = render_position(entity.get("career_stage", ""), title)
        primary, secondary = select_industries(entity.get("primary_industry"), entity.get("sub_industries"))
        overlaps = extract_overlaps(required, entry["tools"])
        why = brief_why(entity, overlaps, max_len=120)
        notes = notes_label(entry.get("rank_in_perfect"), entry["covered"], entry["missing"], entry["weak_hits"])
        percentile_value = percentiles[idx] if percentiles else 100
        tools_line = _tools_match_line(required, entry["tools"], entry["weak_hits"], entry["missing"])
        row = format_row(
             entity,
             percentile_value,
             entry["covered"],
             len(required),
             position,
             primary,
             secondary,
             why,
             notes,
             show_contacts,
             candidate_name=entity.get("name"),
             tools_match=tools_line,
        )
        evidence = evidence_snippets(entity)
        if evidence:
            row["evidence_preview"] = evidence[0]
            row["evidence_popover"] = evidence
        row["tier"] = entry["tier"]
        formatted_rows.append(row)

    tier_counts = {label: len(tier_buckets.get(label, [])) for label in LABEL_ORDER}
    scarcity_msg = _scarcity_message(tier_counts, missing_counter)
    dq_banner = data_quality_check([entry["entity"] for entry in final_entries])

    LAST_INSIGHT_RESULT = {
        "rows": formatted_rows,
        "scarcity_message": scarcity_msg,
        "data_quality_banner": dq_banner,
        "total_matched": total_matches,
        "required_tools": required,
        "tier_counts": tier_counts,
    }
    logger.debug("Cached insight result with %d rows", len(formatted_rows))

    if formatted_rows:
        table_lines = ["Candidate\tTools Match\tNotes"]
        for row in formatted_rows[:return_top]:
            table_lines.append(
                f"{row.get('candidate','Unknown')}\t{row.get('tools_match','')}\t{row['notes']}"
            )
        body = "\n".join(table_lines)
        if show_contacts:
            contact_lines = ["", "Contacts:"]
            for row in formatted_rows[:return_top]:
                contacts = row.get("contacts") or {}
                if not contacts.get("linkedin_url") and not contacts.get("email"):
                    continue
                name = row.get("candidate", "Candidate")
                parts = [name]
                if contacts.get("linkedin_url"):
                    parts.append(f"LinkedIn: {contacts['linkedin_url']}")
                if contacts.get("email"):
                    parts.append(f"Email: {contacts['email']}")
                contact_lines.append(" - ".join(parts))
            if len(contact_lines) > 2:
                body += "\n" + "\n".join(contact_lines)
    else:
        body = "No ranked candidates qualified under the current criteria."

    header = f"Total matched: {total_matches}"
    extras = [msg for msg in (scarcity_msg, dq_banner) if msg]
    preface = ("\n".join(extras) + "\n\n") if extras else ""
    return f"{header}\n\n{preface}{body}"



def get_last_insight_result() -> Optional[Dict[str, Any]]:
    return LAST_INSIGHT_RESULT


def print_help() -> None:
    print(
        """Commands:
  /help                     Show this help
  /exit | exit | quit | q   Quit
Type natural language questions, e.g.:
  "total how many candidates have experience in construction, know CAD and Python"
  "list top 10 names for Django + HIPAA + networking"
  "why these 5 fit FHIR + GenAI + NLP"
"""
    )


def run_cli() -> None:
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set; default heuristic planner will be used.")
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
        if low in {"exit", "/exit", "quit", "q"}:
            print("Bye.")
            break
        if low == "/help":
            print_help()
            continue
        try:
            answer = answer_question(line)
            print(answer)
        except Exception as exc:  # pragma: no cover - best-effort logging on CLI
            print(f"Error: {exc}")


__all__ = ["answer_question", "get_last_insight_result", "llm_plan", "print_help", "run_cli"]
