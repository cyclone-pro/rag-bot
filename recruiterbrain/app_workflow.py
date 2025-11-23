"""User workflow + LLM planning glue for recruiter brain."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence
import re  
from recruiterbrain.env_loader import load_env
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

load_env()

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

JD_START_PATTERNS = [
    r"\btitle\s*:",               # Title: Python with Gen AI...
    r"\bjd\s*:",                  # JD:
    r"\bjob\s+description\b",     # Job Description
    r"\bmandatory\s+skills\s*:",  # Mandatory Skills :
    r"\brequirement\s*:",         # Requirement:
]

JD_END_PATTERNS = [
    r"\bplease note\b",
    r"\bregards\b",
    r"\bbest regards\b",
    r"\bthanks\b",
    r"\bthank you\b",
]
SKILL_SECTION_HINTS = [
    "mandatory skills",
    "requirements",
    "requirement:",
    "what we need to see",
    "what you'll need",
    "must have",
    "nice to have",
    "good to have",
    "skills:",
    "technical skills",
]

_SKILL_STOPWORDS = {
    "strong", "background", "experience", "experiences", "knowledge", "skills",
    "and", "or", "with", "using", "in", "of", "to", "for", "on", "the", "a", "an",
    "good", "great", "solid", "hands-on", "handson", "ability", "abilities",
    "etc", "etc.", "including", "include", "includes",
}

# Tokens we NEVER treat as skills (visa types, countries, raw numbers, etc.)
_SKILL_BLOCKLIST = {
    "h1b", "h4", "gc", "ead", "us", "ca",
    "30", "60", "6", "6+", "months", "month",
    "sme",
}

# Phrases that usually start vague, non-technical fragments
_NON_SKILL_PREFIXES = {
    "define", "develop", "drives", "drive", "downtimes", "downtime",
    "growth", "plans", "plan", "problem", "service", "services",
    "solution", "solutions", "compatibility", "expertise", "experience",
    "validate", "prepare", "produce", "performs", "build", "builds",
    "quarterly", "optimization", "optimize", "manage", "manages",
}


def extract_jd_block(text: str) -> str:
    """
    Given a long recruiter email, extract the JD-ish block:
    from Title/JD/Mandatory Skills onward, and stop at Regards/Please Note/etc.
    If we can't find markers, return the original text.
    """
    if not text:
        return ""

    full = text.strip()
    lower = full.lower()

    # --- Find JD start ---
    start_idx = None
    for pat in JD_START_PATTERNS:
        m = re.search(pat, lower)
        if m:
            if start_idx is None or m.start() < start_idx:
                start_idx = m.start()

    if start_idx is None:
        # No clear JD marker; just use the whole text.
        start_idx = 0

    # --- Find JD end (optional) ---
    end_idx = len(full)
    for pat in JD_END_PATTERNS:
        m = re.search(pat, lower)
        if m and m.start() > start_idx:
            end_idx = min(end_idx, m.start())

    jd_block = full[start_idx:end_idx].strip()
    return jd_block or full
"""
def extract_tools_from_jd(jd_text: str) -> List[str]:
    
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
"""
def find_new_skills_for_catalog(
    jd_skills: List[str],
    known_skills: Sequence[str],
) -> List[str]:
    """
    Return skills that appear in the JD but are not in your existing catalog.
    You can then persist these however you like (DB, JSON, etc.).
    """
    known = {s.strip().lower() for s in known_skills if s}
    new: List[str] = []
    for s in jd_skills:
        key = s.strip().lower()
        if key and key not in known and key not in new:
            new.append(key)
    return new

def _clean_skill_phrase(raw: str) -> str:
    # Normalize spacing and lower
    txt = re.sub(r"\s+", " ", raw).strip().lower()
    # Strip leading/trailing punctuation
    txt = txt.strip(",.;:-/\\()[]{}")
    if not txt:
        return ""

    # Pure numbers or tokens with no letters -> drop
    if not any(ch.isalpha() for ch in txt):
        return ""

    # Blocklisted "skills" (visa types, raw nums, etc.)
    if txt in _SKILL_BLOCKLIST:
        return ""

    # Split into words
    words = txt.split()

    # Too long → likely a sentence, not a skill
    if len(words) > 6:
        return ""

    # If phrase starts with a verb/generic word, treat as non-skill
    if words[0] in _NON_SKILL_PREFIXES:
        return ""

    # Remove trailing generic words
    while words and words[-1] in _SKILL_STOPWORDS:
        words.pop()
    if not words:
        return ""

    cleaned = " ".join(words)

    # Still blocklist / stopword after trimming?
    if cleaned in _SKILL_STOPWORDS or cleaned in _SKILL_BLOCKLIST:
        return ""

    return cleaned



def _extract_skill_candidates_from_sections(text: str) -> List[str]:
    """
    Pulls likely skills from JD sections that talk about 'Mandatory Skills',
    'Requirements', 'What we need to see', etc.
    """
    skills: List[str] = []
    lines = text.splitlines()

    for line in lines:
        lower = line.lower()
        if any(hint in lower for hint in SKILL_SECTION_HINTS):
            # Take part after ':' if present, else whole line
            after = line.split(":", 1)[-1] if ":" in line else line
            # Split on common separators
            chunks = re.split(r"[,/•;]| and | AND ", after)
            for chunk in chunks:
                skill = _clean_skill_phrase(chunk)
                if skill:
                    skills.append(skill)
    return skills


def _extract_inline_acronyms_and_tools(text: str) -> List[str]:
    """
    Grab things like OSPF, BGP, MPLS, VXLAN, DMVPN, SR-TE, etc. anywhere in the JD.
    """
    skills = set()

    # Acronyms / hyphenated tech tokens
    for match in re.finditer(r"\b[A-Z0-9]{2,}(?:-[A-Z0-9]+)?\b", text):
        token = match.group(0).strip()
        # Avoid useless generic acronyms
        if token in {"JD", "ADA"}:
            continue
        skills.add(token.lower())

    # Known trigger words followed by something
    # e.g. "experience with VMware", "knowledge of Kubernetes"
    for m in re.finditer(
        r"(experience|knowledge|skills)\s+(?:in|with|of)\s+([A-Za-z0-9+_.\- ]{2,40})",
        text,
        flags=re.IGNORECASE,
    ):
        chunk = m.group(2)
        skill = _clean_skill_phrase(chunk)
        if skill:
            skills.add(skill)

    return list(skills)


def extract_tools_from_jd(jd_text: str) -> List[str]:
    """
    Extract *all* reasonable skills from a JD:
    - Start with dynamic extraction from sections ('Mandatory Skills', 'Requirements', etc.)
    - Add inline acronyms/tools (OSPF, BGP, MPLS, VXLAN, VMware, etc.)
    - Optionally also include hits from JD_TOOL_HINTS (if defined).
    Result is a de-duplicated, ordered list and is NOT artificially limited to 5–7.
    """
    if not jd_text:
        return []

    # Keep original for regex that cares about case, but also a lowercase copy
    raw = jd_text.strip()
    lower = raw.lower()

    found: List[str] = []

    # 1) Dynamic section-based extraction
    found.extend(_extract_skill_candidates_from_sections(raw))

    # 2) Inline acronyms / technologies anywhere in the text
    found.extend(_extract_inline_acronyms_and_tools(raw))

    # 3) Optional: still leverage existing JD_TOOL_HINTS if you have it
    #    (so synonyms like 'k8s' → 'kubernetes' still normalize)
    try:
        from recruiterbrain.shared_config import JD_TOOL_HINTS  # type: ignore[attr-defined]
    except Exception:
        JD_TOOL_HINTS = {}

    for canonical, variants in getattr(JD_TOOL_HINTS, "items", lambda: [])():
        if any(v in lower for v in variants):
            found.append(canonical.lower())

    # 4) Deduplicate while preserving order; drop very generic words
    seen = set()
    unique: List[str] = []
    for t in found:
        skill = _clean_skill_phrase(t)
        if not skill:
            continue
        if skill in _SKILL_STOPWORDS:
            continue
        if skill not in seen:
            seen.add(skill)
            unique.append(skill)

    return unique

def canonicalize_jd_tools(plan: Dict[str, Any]) -> None:
    """
    Normalize the JD tools list so that scoring is stable across runs and phrasings.
    - use union of must_have_keywords and required_tools
    - lowercase
    - strip
    - deduplicate
    - sort
    Only applies when _jd_mode is True.
    """
    if not plan.get("_jd_mode"):
        return

    raw_tools: List[str] = []

    req = plan.get("required_tools") or []
    if isinstance(req, str):
        req = [t.strip() for t in req.split(",") if t.strip()]
    raw_tools.extend(req)

    kws = plan.get("must_have_keywords") or []
    if isinstance(kws, str):
        kws = [k.strip() for k in kws.split(",") if k.strip()]
    raw_tools.extend(kws)

    canonical: List[str] = []
    seen = set()

    for t in raw_tools:
        norm = t.strip().lower()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        canonical.append(norm)

    canonical.sort()
    plan["required_tools"] = canonical

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

STAGE_RANK = {
    "Entry": 1,
    "Mid": 2,
    "Senior": 3,
    "Lead/Manager": 4,
    "Director+": 5,
}

def _infer_stage_from_years_text(text: str) -> str | None:
    """
    Parse phrases like:
      '2 years', '3-5 years', 'at least 5 years', '5+ years', 'minimum of 5 yrs'
    and map to Entry / Mid / Senior.
    """
    q = text.lower()

    # patterns: 5+ years, 5+ yrs, 5 plus years
    m = re.search(r"(\d+)\s*\+\s*(?:years|yrs|yr)", q)
    if m:
        years = int(m.group(1))
        return _stage_for_years(years)

    # at least 5 years, minimum of 5 years, min 5 years
    m = re.search(r"(?:at\s+least|min(?:imum)?\s+of|min)\s+(\d+)\s*(?:years|yrs|yr)", q)
    if m:
        years = int(m.group(1))
        return _stage_for_years(years)

    # range: 3-5 years
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*(?:years|yrs|yr)", q)
    if m:
        low = int(m.group(1))
        high = int(m.group(2))
        # use the lower bound as requirement
        return _stage_for_years(low)

    # fallback: plain '5 years'
    m = re.search(r"(\d+)\s*(?:years|yrs|yr)", q)
    if m:
        years = int(m.group(1))
        return _stage_for_years(years)

    return None

def _stage_for_years(years: int) -> str:
    """
    Map years to Entry / Mid / Senior based on your bands:
      - 0–3 -> Entry
      - >3–5 -> Mid
      - >5   -> Senior
    """
    if years <= 3:
        return "Entry"
    if years <= 5:
        return "Mid"
    return "Senior"

def _better_stage(current: str | None, candidate: str | None) -> str | None:
    """Pick the stricter stage between current and candidate based on STAGE_RANK."""
    if not candidate:
        return current
    if not current or current == "Any":
        return candidate
    cur_rank = STAGE_RANK.get(current, 0)
    cand_rank = STAGE_RANK.get(candidate, 0)
    return candidate if cand_rank > cur_rank else current


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
    normalized_tools: Sequence[str],
    weak_hits: Dict[str, List[str]],
    missing: Sequence[str],
    total: int,
) -> str:
    """
    Render a compact per-candidate skill summary:

    - present/total and % match
    - up to ~10 skills total (mix of present + missing)
    - avoids huge ugly lines
    """
    norm_set = {t.lower().strip() for t in normalized_tools}
    missing_set = {m.lower().strip() for m in (missing or [])}

    # Total JD skills used for scoring
    total_skills = total or len(required) or 1

    # Missing skills (pretty)
    missing_pretty = [_prettify_tool(m) for m in missing]
    covered = total_skills - len(missing_pretty)
    pct = round(covered / total_skills * 100)

    # Partition required into present vs absent (using coverage's "missing")
    present_tokens: List[str] = []
    absent_tokens: List[str] = []

    for token in required:
        key = token.lower().strip()
        pretty = _prettify_tool(token)

        if key in missing_set:
            # JD says required, coverage says missing
            absent_tokens.append(pretty)
        else:
            # Treat as covered (strong or weak)
            present_tokens.append(pretty)

    # Limit how many skills we show (for readability)
    max_total_display = 10
    max_present = max_total_display // 2  # e.g. 5
    max_absent = max_total_display - max_present  # e.g. 5

    present_display = present_tokens[:max_present]
    absent_display = absent_tokens[:max_absent]

    present_str = ", ".join(f"✅ {p}" for p in present_display) if present_display else "None"
    absent_str = ", ".join(f"❌ {a}" for a in absent_display) if absent_display else "None"

    line = (
        f"{covered}/{total_skills} skills matched ({pct}% match). "
        f"Present: {present_str}. "
        f"Missing: {absent_str}"
    )

    return line


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

    original = (question or "").strip()

    # --- JD detection using your heuristic ---
    jd_mode = is_jd_mode(original)

    if jd_mode:
        # Clean the JD email down to just the JD-ish body
        cleaned_jd = clean_jd_text(original)
        semantic_jd = build_jd_semantic_query(cleaned_jd)
        planner_question = semantic_jd
    else:
        # For non-JD, just strip obvious contact/meta noise
        cleaned_for_plan = strip_contact_meta_phrases(original)
        planner_question = cleaned_for_plan
        cleaned_jd = ""
        semantic_jd = ""

    if not client:
        logger.debug("LLM planner unavailable; using default plan")
        plan = _default_plan(planner_question)
    else:
        system_prompt = (
            "You parse recruiting and sourcing questions into a JSON plan for Milvus retrieval over a resume collection. "
            "Never add fields not in this schema. Output ONLY JSON (no prose). Keys:\n"
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
            "INTERPRET YEARS OF EXPERIENCE AS CAREER STAGE USING THESE RULES:\n"
            "- 0–3 years, or phrases like 'junior', 'entry level', 'new grad', 'fresher' -> require_career_stage = 'Entry'.\n"
            "- 3–5 years, or phrases like 'mid level', 'intermediate' -> require_career_stage = 'Mid'.\n"
            "- 5+ years, or phrases like 'senior', '8+ years', '10 years', 'principal', 'staff', "
            "  or any explicit 'minimum of 5 years' -> require_career_stage = 'Senior'.\n"
            "- If the user explicitly asks for a management level (e.g. 'team lead', 'engineering manager', "
            "  'director'), use 'Lead/Manager' or 'Director+' instead of 'Senior'.\n"
            "- If the question does NOT clearly specify experience or level, set require_career_stage = 'Any'.\n"
            "OTHER RULES:\n"
            "- If the user asks for 'total how many', set intent = 'count'.\n"
            "- If they want a list of candidates, set intent = 'list'.\n"
            "- If they want an explanation or justification ('why', 'explain'), set intent = 'why'.\n"
            "- Use 'summary_embedding' by default for vector_field.\n"
            "- Only include 'industry_equals' or 'require_domains' when the question clearly specifies an industry or domain.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {planner_question}"},
        ]

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.0,  # deterministic planner
                messages=messages,
            )
            raw = resp.choices[0].message.content.strip()
            plan = json.loads(raw)
        except Exception as exc:
            logger.warning("LLM planner failed, falling back to heuristic: %s", exc)
            plan = _default_plan(planner_question)

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

    # Original raw question stays here for logging / explanations
    plan["question"] = original

    # --- Heuristic industry inference based on the original question ---
    _infer_industry_equals(original, plan)
    plan["industry_equals"] = canonicalize_industry(plan.get("industry_equals"))

    # --- Insight routing based on original question ---
    is_insight, router_conf = _route_insight(original)
    if is_insight:
        plan["intent"] = "insight"
        plan["k"] = int(plan.get("k") or INSIGHT_DEFAULT_K)
        logger.info("Router promoted question to insight intent (confidence=%.2f)", router_conf)
    plan["_router_confidence"] = router_conf

    # --- Required tools from LLM plan ---
    normalized_tools = _derive_required_tools(plan)

    if jd_mode:
        # Flag JD mode
        plan["_jd_mode"] = True
        plan["_jd_raw"] = cleaned_jd

        # This is what will be embedded for ANN
        plan["embedding_query"] = semantic_jd or cleaned_jd or original

        # Extract tools from JD using your heuristic hints
        jd_tools = extract_tools_from_jd(cleaned_jd)

        # Merge: LLM tools + JD heuristic tools + must_have_keywords
        merged: List[str] = []
        seen = set()

        for lst in (normalized_tools, jd_tools, plan.get("must_have_keywords") or []):
            if isinstance(lst, str):
                lst = [lst]
            for t in lst:
                norm = str(t).strip().lower()
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                merged.append(norm)

        merged.sort()
        plan["required_tools"] = merged
        try:
            from recruiterbrain.shared_config import SKILL_CATALOG  # e.g. a list of known skills
        except Exception:
            SKILL_CATALOG = []

        new_skills = find_new_skills_for_catalog(jd_tools, SKILL_CATALOG)
        if new_skills:
            logger.info("New JD skills to add to catalog: %s", ", ".join(new_skills))
            # TODO: append to your DB / JSON / Notion / wherever you store skills

        # For JD flows, make sure it's insight with a sane top N
        plan["intent"] = "insight"
        if "return_top" not in plan:
            plan["return_top"] = 10
        plan["k"] = int(plan.get("k") or plan["return_top"])
    else:
        # Non-JD: use the LLM-derived tools, and clean embedding_query for ANN
        cleaned_for_embed = strip_contact_meta_phrases(original)
        plan["embedding_query"] = cleaned_for_embed or original
        plan["required_tools"] = normalized_tools

    # If lots of tools, treat as insight even if LLM didn't say so.
    if plan.get("intent") != "insight" and len(plan.get("required_tools") or []) >= 3:
        plan["intent"] = "insight"
        plan["k"] = int(plan.get("k") or INSIGHT_DEFAULT_K)
        router_conf = max(router_conf, ROUTER_CONFIDENCE_THRESHOLD + 0.05)
        plan["_router_confidence"] = router_conf

    # Clarifier for low-confidence insight
    if plan.get("intent") == "insight":
        plan["k"] = int(plan.get("k") or INSIGHT_DEFAULT_K)
        if router_conf < ROUTER_CONFIDENCE_THRESHOLD:
            logger.info(
                "Router confidence %.2f below threshold; requesting clarification",
                router_conf,
            )
            plan["clarify"] = _clarifier_prompt(plan.get("required_tools") or [])

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
        return (
            'Hey John how can i help you with! Ask me things like '
            '"compare Milvus, dbt, AWS, Vertex AI" or "list top 10 healthcare candidates".'
        )

    # --- Build or reuse a plan (LLM planner) ---
    if plan_override is not None:
        plan = dict(plan_override)
    else:
        # For planning we already prefer the version without contact/meta noise.
        cleaned_for_plan = strip_contact_meta_phrases(original_question)
        plan = llm_plan(cleaned_for_plan)

    if not isinstance(plan, dict):
        return str(plan)

    # -------------------------------
    # JD detection & embedding_query
    # -------------------------------
    jd_mode = is_jd_mode(original_question)
    plan["_jd_mode"] = jd_mode

    if jd_mode:
        # 1) Clean the JD text (strip greetings/signature/boilerplate)
        cleaned_jd = clean_jd_text(original_question)

        # 2) Build compact semantic query (first ~500 chars)
        semantic_query = build_jd_semantic_query(cleaned_jd)
        plan["_jd_raw"] = cleaned_jd
        plan["embedding_query"] = semantic_query
        plan["question"] = semantic_query

        # 3) Sensible defaults for JD flow
        plan.setdefault("intent", "insight")
        plan.setdefault("vector_field", VECTOR_FIELD_DEFAULT)
        plan.setdefault("top_k", TOP_K)
        plan.setdefault("return_top", RETURN_TOP)

        # 4) Derive required tools from the JD text and merge with any planner tools
        jd_tools = extract_tools_from_jd(cleaned_jd)  # e.g. ["python", "mlops", "pytorch", ...]
        if jd_tools:
            existing = [t for t in plan.get("required_tools", []) if t]
            merged: List[str] = []
            seen = set()
            for t in list(existing) + list(jd_tools):
                t_norm = t.strip().lower()
                if t_norm and t_norm not in seen:
                    seen.add(t_norm)
                    merged.append(t_norm)
            plan["required_tools"] = merged
            # OPTIONAL: discover new skills from this JD and log them
            try:
                 from recruiterbrain.shared_config import SKILL_CATALOG
            except Exception:
              SKILL_CATALOG = []

            new_skills = find_new_skills_for_catalog(jd_tools, SKILL_CATALOG)
            if new_skills:
              logger.info("New JD skills to add to catalog: %s", ", ".join(new_skills))
            # TODO: persist new_skills somewhere durable

    else:
        # Normal question: remove contact/meta words BEFORE embedding.
        cleaned_for_embed = strip_contact_meta_phrases(original_question)
        plan["embedding_query"] = cleaned_for_embed
        plan["question"] = cleaned_for_embed

    # Heuristic booster (years-of-exp -> stage, etc.)
    _augment_plan_with_heuristics(original_question, plan)

    # --- Respect explicit "N candidates" / "top N" in the question ---
    text_lower = (original_question or "").lower()
    explicit_n: Optional[int] = None

    # Patterns like "top 10" or "10 django candidates"
    m = re.search(r"\btop\s+(\d+)\b", text_lower)
    if not m:
        m = re.search(r"\b(\d+)\s+(?:[a-z]+\s+)?(candidates|profiles|people)\b", text_lower)

    if m:
        try:
            explicit_n = max(1, min(int(m.group(1)), 50))
        except ValueError:
            explicit_n = None

    if explicit_n is not None:
        plan["return_top"] = explicit_n
        if (plan.get("intent") or "").lower() == "insight":
            plan["k"] = max(int(plan.get("k") or 0), explicit_n)
    elif plan.get("_jd_mode"):
        # Long JD emails with no explicit count -> default top 10
        current = int(plan.get("return_top") or RETURN_TOP)
        plan["return_top"] = min(current, 10)
        if (plan.get("intent") or "").lower() == "insight":
            plan["k"] = max(int(plan.get("k") or 0), plan["return_top"])

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

    logger.info("Final plan for question '%s': %s", original_question, plan)

    # --- Core retrieval ---
    try:
        paired_hits, total_matches = ann_search(plan)
    except Exception as exc:
        logger.exception("Error during ann_search for question: %s", original_question)
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
    if not required and not plan.get("_jd_mode"):
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
                "covered": covered,      # usually a set of matched tools
                "missing": missing,      # list of required tools not present
                "weak_hits": weak_hits,  # dict: required -> [equivalents]
                "tools": tools,          # normalized tools for the candidate
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

    # Should we include LinkedIn / email in the insight output?
    show_contacts = _should_show_contacts(original_question)

    total_required = len(required) or 1
    formatted_rows: List[Dict[str, Any]] = []

    for idx, entry in enumerate(final_entries):
        entity = entry["entity"]
        tools = entry["tools"]
        missing_tokens = entry.get("missing") or []
        weak_hits = entry.get("weak_hits") or {}

        # how many JD skills this candidate effectively covers
        covered_count = total_required - len(missing_tokens)

        title = extract_latest_title(
            entity.get("employment_history"),
            entity.get("top_titles_mentioned"),
        )
        position = render_position(entity.get("career_stage", ""), title)
        primary, secondary = select_industries(
            entity.get("primary_industry"),
            entity.get("sub_industries"),
        )
        overlaps = extract_overlaps(required, tools)
        why = brief_why(entity, overlaps, max_len=120)

        notes = notes_label(
            entry.get("rank_in_perfect"),
            covered_count,
            [_prettify_tool(m) for m in missing_tokens],
            weak_hits,
        )

        percentile_value = percentiles[idx] if percentiles else 100

        tools_line = _tools_match_line(
            required=required,
            normalized_tools=tools,
            weak_hits=weak_hits,
            missing=missing_tokens,
            total=total_required,
        )

        row = format_row(
            entity,
            percentile_value,
            covered_count,
            total_required,
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
             f"{row.get('candidate','Unknown')}\t"
             f"{row.get('position','')}\t"
             f"{row.get('tools_match','')}\t"
             f"{row['notes']}"
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
