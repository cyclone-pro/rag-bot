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
    WEAK_EQUIVALENTS, 
    BRAND_TO_CATEGORIES,
    
)
try:
    from rapidfuzz import process, fuzz
except ImportError:  # pragma: no cover
    process = None
    fuzz = None
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
    classify_primary_gap,
    experience_gap_comment,
    build_gap_explanation
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
    "get me their contacts",
    "get me their contact",
    "get their contacts",
    "get their contact",
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
     "ci/cd": ["ci/cd", "ci-cd", "cicd"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "power platform": ["power platform", "powerapps", "power apps"],
}
# ----------------- Fuzzy skill & role catalogs (canonical mapping) ----------------- #

# Canonical skills -> common variants / misspellings / abbreviations
CANONICAL_SKILLS: Dict[str, List[str]] = {
    # Languages
    "python": ["python", "pyhton", "pyton"],
    "java": ["java", "jaav", "jave"],
    "c": ["c"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp", "csharp"],
    "javascript": ["javascript", "js", "javscript", "javasript"],
    "typescript": ["typescript", "ts", "typesript"],
    "go": ["go", "golang"],
    "ruby": ["ruby", "rubby"],
    "ruby on rails": ["ruby on rails", "ror", "rails"],
    "php": ["php"],

    # Backend frameworks
    "django": ["django", "djanog"],
    "flask": ["flask"],
    "fastapi": ["fastapi", "fast api"],
    "spring": ["spring"],
    "spring boot": ["spring boot", "springboot", "spring-boot"],
    "node.js": ["node.js", "nodejs", "node js"],
    "express": ["express", "expressjs", "express js"],

    # Frontend
    "react": ["react", "reactjs", "react js"],
    "angular": ["angular", "angularjs", "angular js"],
    "vue": ["vue", "vuejs", "vue js"],

    # Clouds
    "aws": ["aws", "amazon web services", "amazone web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform", "gogle cloud"],

    # DevOps / infra
    "docker": ["docker", "dockers"],
    "kubernetes": ["kubernetes", "k8s", "kuberenetes", "kubernets"],
    "terraform": ["terraform", "terrafrom", "terrafom"],
    "ansible": ["ansible", "ansibl"],
    "jenkins": ["jenkins", "jenknis"],
    "github actions": ["github actions", "git hub actions", "gitactions"],
    "gitlab ci": ["gitlab ci", "gitlab-ci"],
    "circleci": ["circleci"],
    "argo cd": ["argo cd", "argo-cd"],
    "helm": ["helm", "helm chart"],

    # Data / DB / big data
    "sql": ["sql", "sequel"],
    "postgresql": ["postgresql", "postgres", "postgre", "postgress"],
    "mysql": ["mysql", "my sql"],
    "mongodb": ["mongodb", "mongo db"],
    "redis": ["redis"],
    "snowflake": ["snowflake", "snowflak"],
    "databricks": ["databricks", "data bricks"],
    "kafka": ["kafka", "kafak"],
    "spark": ["spark", "pyspark"],
    "airflow": ["airflow", "apache airflow"],

    # Networking / security
    "cisco": ["cisco", "ciscco"],
    "cisco ios": ["cisco ios", "ios xe", "ios-xe"],
    "cisco nx-os": ["cisco nx-os", "nx-os", "nexus os"],
    "juniper junos": ["juniper junos", "junos"],
    "arista eos": ["arista eos", "eos"],
    "sonic": ["sonic", "mellanox sonic"],
    "bgp": ["bgp"],
    "ospf": ["ospf"],
    "mpls": ["mpls"],
    "vxlan": ["vxlan"],
    "palo alto": ["palo alto", "paloalto", "pan-os"],
    "fortinet fortigate": ["fortinet", "fortigate"],
    "checkpoint": ["checkpoint", "check point"],
    "f5": ["f5", "f5 ltm", "f5 gtm"],

    # Systems / virtualization / infra
    "vmware": ["vmware", "v mware"],
    "vsphere": ["vsphere", "v sphere"],
    "esxi": ["esxi"],
    "nutanix": ["nutanix", "nutanix ahv"],
    "olvm": ["olvm"],
    "rhel": ["rhel", "red hat", "redhat"],
    "linux": ["linux"],
    "windows server": ["windows server", "win server"],
    "active directory": ["active directory", "ad", "microsoft ad"],
    "sccm": ["sccm", "mecm", "configmgr"],
    "intune": ["intune"],
    "dns": ["dns"],
    "dhcp": ["dhcp"],

    # Misc scripting
    "powershell": ["powershell"],
    "bash": ["bash", "shell scripting"],
}

# Canonical role families -> variants (titles)
CANONICAL_ROLES: Dict[str, List[str]] = {
    # SWE / SDE
    "software engineer": [
        "software engineer", "swe", "swe i", "swe ii", "swe iii",
        "sde", "sde 1", "sde 2", "sde 3", "software developer",
    ],
    "backend engineer": [
        "backend engineer", "back end engineer", "backend developer",
        "server side engineer",
    ],
    "frontend engineer": [
        "frontend engineer", "front end engineer", "frontend developer",
        "ui engineer", "react developer", "reactjs developer",
    ],
    "fullstack engineer": [
        "fullstack engineer", "full stack engineer", "fullstack developer",
        "full stack developer",
    ],

    # QA / SDET
    "sdet": ["sdet", "software development engineer in test"],
    "qa engineer": ["qa engineer", "qa analyst", "quality engineer"],

    # DevOps / platform
    "devops engineer": [
        "devops engineer", "dev ops engineer",
        "site reliability engineer", "sre", "platform engineer",
    ],

    # Network / security
    "network engineer": [
        "network engineer", "network eng", "network specialist",
        "network administrator",
    ],
    "network security engineer": [
        "network security engineer", "security network engineer",
        "firewall engineer",
    ],

    # Systems / infra
    "systems engineer": [
        "systems engineer", "system engineer",
        "systems administrator", "system administrator",
    ],
}

# Flatten catalogs → lists for rapidfuzz
ALL_SKILL_VARIANTS: List[str] = []
VARIANT_TO_CANON_SKILL: Dict[str, str] = {}
for canon, variants in CANONICAL_SKILLS.items():
    for v in variants:
        v_low = v.lower()
        ALL_SKILL_VARIANTS.append(v_low)
        VARIANT_TO_CANON_SKILL[v_low] = canon

ALL_ROLE_VARIANTS: List[str] = []
VARIANT_TO_CANON_ROLE: Dict[str, str] = {}
for canon, variants in CANONICAL_ROLES.items():
    for v in variants:
        v_low = v.lower()
        ALL_ROLE_VARIANTS.append(v_low)
        VARIANT_TO_CANON_ROLE[v_low] = canon

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
    "etc", "etc.", "including", "include", "includes", "get", "gets", "getting", "got",
    "have", "has", "having", "had",
    "make", "made", "making",
    "take", "taking", "taken",
    "do", "does", "did",
    "also",
    "me", "my", "our", "your", "his", "her", "its",
    "they", "them", "their", "theirs",
    "we", "you","cloud", 
}
TECH_HINTS = {
    # clouds
    "aws", "azure", "gcp", "google cloud", "google-cloud",
    # data / ml / platform
    "kafka", "spark", "dbt", "snowflake", "databricks",
    "airflow", "kubernetes", "k8s", "docker", "terraform", "ansible", "jenkins",
    "github", "gitlab", "git", "gitactions", "github actions", "gitlab ci",
    # languages
    "python", "java", "javascript", "typescript", "go", "golang", "c", "c++",
    "csharp", "c#", "ruby", "rails", "ruby on rails", "perl", "shell",
    # app frameworks
    "django", "flask", "fastapi", "spring", "spring boot", "nodejs", "node",
    "react", "angular", "vue",
    # networking / infra
    "cisco", "aci", "nxos", "ios", "arista", "juniper",
    "palo", "palo alto", "fortinet", "checkpoint", "f5",
    "bgp", "ospf", "isis", "mpls", "vxlan", "evpn",
    "vmware", "vsphere", "esxi", "nutanix", "ahv", "olvm",
    "rhel", "active directory", "sccm", "intune", "dns", "dhcp",
}

# Tokens we NEVER treat as skills (visa types, countries, junky words, etc.)
_SKILL_BLOCKLIST = {
    # visa / work auth
    "h1b", "h4", "gc", "ead", "c2c", "usc", "us citizen",
    # generic junk / tiny tokens
    "us", "ca", "il", "nj", "ny", "tx", "no", "it", "na",
    "bs", "ms", "bs/ms", "b.s.", "m.s.", "degree",
    "ad", "ip", "id",
    "year", "years", "month", "months",
    "remote", "onsite", "hybrid", "contract", "rate", "hr",
    "location",
    "more",
    # generic roles we don't want as "skills" by themselves
    "engineer", "developer", "analyst", "architect", "consultant",
    "manager", "lead", "owner",
    # misc
    "fth", "fte",
     "est", "pst", "cst", "mst", "ist", "gmt", "utc",
}

# Phrases that usually start vague, non-technical fragments
_NON_SKILL_PREFIXES = {
    "according",
    "as",
    "define", "develop", "drives", "drive",
    "downtimes", "downtime", "growth", "plans", "plan",
    "problem", "service", "services", "solution", "solutions",
    "compatibility", "expertise", "validate", "prepare", "produce",
    "performs", "perform", "build", "builds", "design", "designing",
    "maintain", "maintains", "maintaining", "release", "releasing",
    "optimization", "optimize", "manage", "manages", "managing",
    "ability", "attitude", "autonomy",
    "excellent", "strong", "solid",
    "participate", "participates", "collaborate", "collaboration",
    "create", "creating", "responsible", "works", "work",
    "ensure", "ensures", "must", "should", "will",
    "analyze", "analysing", "analyzing",
}

US_STATE_CODES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia","ks","ky","la",
    "me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok",
    "or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv","wi","wy"
}
NON_SKILL_PATTERNS = [
    r"\byears?\b",
    r"\brate\b",
    r"\bper\s+hour\b",
    r"\b(w2|c2c)\b",
    r"\bremote\b",
    r"\bonsite\b",
    r"\bhybrid\b",
    r"\blocat(?:ed|ion)\b",
    r"\bvisa\b",
    r"\bany\s+work\s+auth\b",
    r"\baccording to\b",
    r"\bclient\b",
    r"\b12\s*month(?:s)?\b",
]


def _clean_skill_phrase(raw: str) -> str:
    # Normalize spacing
    txt = re.sub(r"\s+", " ", (raw or "")).strip()
    if not txt:
        return ""

    lower = txt.lower()
        # Drop query/instruction artefacts
    if "give me" in lower or "their contact" in lower or "contyacty" in lower:
        return ""

    # Drop D2D labels from "D2D: Day to Day" sections
    if lower in {"d2d", "d2d:"}:
        return ""

    # Drop pure US-style state abbreviations etc.
    # Keep short language names like "go", "c", "r" explicitly.
    SHORT_ALLOWED = {"go", "c", "r"}
    if len(txt) <= 2 and lower not in SHORT_ALLOWED:
        return ""


    # Kill obvious html-entity junk like client&#39;
    if "&#" in lower:
        return ""

    # If this looks like a sentence fragment (". " inside), skip it
    if ". " in txt:
        return ""

    # Pure number / no letters -> drop
    if not any(ch.isalpha() for ch in txt):
        return ""

    # Full-phrase patterns we never want as skills
    if "years of experience" in lower or "year of experience" in lower:
        return ""
    if "month(s)" in lower or "months contract" in lower:
        return ""
    if "rate" in lower and ("$" in lower or "hr" in lower or "/hr" in lower):
        return ""
    if "location" in lower:
        return ""

    # Collapse things like "1 year in Gen AI" into "gen ai"
    m = re.search(r"\b\d+\s+year[s]?\s+in\s+(gen ai|genai|ai/ml|nlp)\b", lower)
    if m:
        return m.group(1).strip()

    # Normalize punctuation at edges
    txt = txt.strip(",.;:-/\\()[]{}")
    lower = txt.lower()
    for pat in NON_SKILL_PATTERNS:
     if re.search(pat, lower):
         return ""


    if not txt:
        return ""

    # Blocklisted tokens as-is
    if lower in _SKILL_BLOCKLIST:
        return ""

    # Split into words
    words = lower.split()

    # Remove tokens that are just bullets/punctuation (e.g. "•")
    words = [w for w in words if any(ch.isalnum() for ch in w)]
    if not words:
        return ""
    if len(words) > 3:
         return ""

    # Strip leading stopwords like "the", "a", "an"
    while words and words[0] in _SKILL_STOPWORDS:
        words.pop(0)
    if not words:
        return ""
    

    # If starts with a generic verb / vague prefix -> treat as non-skill
    if words[0] in _NON_SKILL_PREFIXES:
        return ""

    # If it looks like "3rd party tools X" -> keep only the tech at the end
    generic_prefix = {
        "3rd", "third", "party", "tools", "tool", "the", "for",
        "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "year", "years", "month", "months", "total", "of", "in", "+", "plus",
    }
    if len(words) >= 3 and all(w in generic_prefix for w in words[:-2]):
        words = words[-2:]

    # Too long -> likely a sentence or description, not a skill
    if len(words) > 6:
        return ""
    if "give me" in lower and "contact" in lower:
         return ""
    if lower in {"d2d", "d2d:", "day to day"}:
         return ""
    if words[-1] == "solutions":
        return ""

    # Drop location-style tokens
    # 'nyc', 'nj', 'ny', state codes, etc.
        # At this point we have a short phrase. If it’s 2–3 words and none of them
    # look like a tech keyword, it’s probably fluff like "excellent working".
    if 2 <= len(words) <= 3:
        if not any(w in TECH_HINTS for w in words):
            return ""

    tok = lower.replace(".", "")
    if tok in US_STATE_CODES or tok in {"nyc", "nj", "ny"}:
         return ""

    # Drop SOW-related tokens
    if tok in {"sow", "statement of work"}:
         return ""


    # Remove trailing generic words
    while words and words[-1] in _SKILL_STOPWORDS:
        words.pop()
    if not words:
        return ""

    # --- NEW: drop clearly soft / managerial phrases with no tech in them ---
    SOFT_MID_WORDS = {
        "aspiration", "aspirations", "goals", "goal",
        "resources", "resource",
        "strategy", "strategies",
        "framework", "frameworks",
        "policy", "policies", "governance",
        "mentorship", "mentor", "coaching", "guidance",
        "culture", "environment",
        "stakeholders", "stakeholder",
        "team", "teams", "users", "business",
        "support", "supporting", "collaboration", "collaborate",
        "manage", "managing", "allocate", "allocating",
        "monitor", "monitoring", "evaluate", "evaluation",
        "optimize", "optimizing", "tune", "tuning",
    }

    

    if any(w in SOFT_MID_WORDS for w in words) and not any(
        w in TECH_HINTS for w in words
    ):
        return ""

    # --- Salesforce phrase collapsing: prefer concrete clouds over generic SFDC ---
    if "salesforce" in words:
        if "service" in words and "cloud" in words:
            return "service cloud"
        if "health" in words and "cloud" in words:
            return "health cloud"
        if "experience" in words and "cloud" in words:
            return "experience cloud"
        return "salesforce"

    cleaned = " ".join(words)

    # Final guards
    if cleaned in _SKILL_STOPWORDS or cleaned in _SKILL_BLOCKLIST:
        return ""

    # Filter out common soft-phrases we never want as skills
    SOFT_SNIPPETS = [
        "take business requirements",
        "problem solving",
        "communication skills",
        "dynamism and flexibility",
        "knowledge of english",
        "as a senior developer",
        "as a senior",
    ]
    for snippet in SOFT_SNIPPETS:
        if snippet in cleaned:
            return ""

    return cleaned

def _map_skill_to_canonical(skill: str) -> str:
    """
    Map a cleaned skill phrase to a canonical skill using fuzzy matching.
    Returns the canonical skill or the original string if no good match.
    """
    s = (skill or "").lower().strip()
    if not s:
        return ""
    # If rapidfuzz isn't available, just return cleaned skill

    if process is None or fuzz is None:
        return skill

    # Exact / near-exact hit
    if s in VARIANT_TO_CANON_SKILL:
        return VARIANT_TO_CANON_SKILL[s]

    if not ALL_SKILL_VARIANTS:
        return skill

    match = process.extractOne(
        s,
        ALL_SKILL_VARIANTS,
        scorer=fuzz.WRatio,
        score_cutoff=80,  # ignore weak matches
    )
    if not match:
        return skill

    best_variant, score, _ = match
    canon = VARIANT_TO_CANON_SKILL.get(best_variant, skill)
    return canon


def _map_role_to_canonical(title: str) -> str:
    """
    Map a job title to a canonical role family (SWE, DevOps, Network, Systems, SDET...).
    """
    t = (title or "").lower().strip()
    if not t:
        return ""

    if process is None or fuzz is None:
        return ""

    if t in VARIANT_TO_CANON_ROLE:
        return VARIANT_TO_CANON_ROLE[t]

    if not ALL_ROLE_VARIANTS:
        return ""

    match = process.extractOne(
        t,
        ALL_ROLE_VARIANTS,
        scorer=fuzz.WRatio,
        score_cutoff=78,
    )
    if not match:
        return ""

    best_variant, score, _ = match
    return VARIANT_TO_CANON_ROLE.get(best_variant, "")

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



def _extract_skill_candidates_from_sections(text: str) -> List[str]:
    """
    Pulls likely skills from JD sections that talk about 'Mandatory Skills',
    'Requirements', 'What we need to see', etc.

    We ONLY treat a line as a "skills" line if:
      - One of SKILL_SECTION_HINTS appears near the beginning, AND
      - There's a ':' after that hint (e.g. 'Requirements: ...').
    This avoids matching random sentences that just mention 'requirements' later.
    """
    skills: List[str] = []
    lines = text.splitlines()

    for line in lines:
        lower = line.lower()

        # Decide if this line is actually a skills header
        is_skill_line = False
        for hint in SKILL_SECTION_HINTS:
            idx = lower.find(hint)
            if idx == -1:
                continue
            # require the hint to be relatively near the start
            if idx > 40:
                continue
            # and require a ':' after the hint
            colon_slice = lower[idx:]
            if ":" not in colon_slice:
                continue
            is_skill_line = True
            break

        if not is_skill_line:
            continue

        # Take part after ':' as the "values" region
        after = line.split(":", 1)[-1]
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
    for m in re.finditer(
        r"(?:such as|like)\s+([A-Za-z0-9+_.\- ,]{2,80})",
        text,
        flags=re.IGNORECASE,
    ):
        chunk = m.group(1)
        # split on commas
    for piece in re.split(r"[,/]| and ", chunk):
            skill = _clean_skill_phrase(piece)
            if skill:
                 skills.add(skill)
    for m in re.finditer(r"(such as|like)\s+([A-Za-z0-9+_.\-/ ,]+)", text, flags=re.I):
      items = re.split(r"[,/]| and ", m.group(2))
      for piece in items:
          skill = _clean_skill_phrase(piece)
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
from recruiterbrain.shared_config import WEAK_EQUIVALENTS, BRAND_TO_CATEGORIES
# WEAK_EQUIVALENTS: Dict[str, Set[str]]
# BRAND_TO_CATEGORIES: Dict[str, Set[str]]  (you can build this in shared_config from CATEGORY_EQUIVALENTS)


def _score_candidate_tools(
    core_required: list[str],
    nice_to_have: list[str],
    candidate_tools: list[str],
) -> dict[str, Any]:
    """
    Compute a nuanced match score between JD tools and candidate tools.

    - core_required: MUST-have tools (canonicalized already)
    - nice_to_have: NICE-to-have tools (canonicalized already)
    - candidate_tools: normalized candidate skills/tools

    For each core:
      exact match      -> 1.0
      weak equivalent  -> 0.4
      category match   -> 0.2
      no match         -> 0.0

    For each nice-to-have:
      exact match      -> 0.6
      weak equivalent  -> 0.3
      category match   -> 0.1

    Final score is normalized to 0–100%.
    """

    core: list[str] = list(dict.fromkeys(core_required or []))  # dedupe, keep order
    nice: list[str] = [t for t in dict.fromkeys(nice_to_have or []) if t not in core]

    cand: set[str] = set((candidate_tools or []))

    # Precompute candidate categories for cheap category matches
    cand_categories: set[str] = set()
    for c in cand:
        for cat in BRAND_TO_CATEGORIES.get(c, set()):
            cand_categories.add(cat)

    core_scores: dict[str, float] = {}
    nice_scores: dict[str, float] = {}

    core_present: list[str] = []
    core_missing: list[str] = []
    nice_present: list[str] = []
    nice_missing: list[str] = []

    def _weak_match(jd_tool: str) -> bool:
        # candidate has something that is a weak-equivalent to jd_tool, or vice versa
        weak_set = WEAK_EQUIVALENTS.get(jd_tool, set())
        if weak_set & cand:
            return True
        # also check reverse: candidate tools whose weak equivalents contain jd_tool
        for c in cand:
            if jd_tool in WEAK_EQUIVALENTS.get(c, set()):
                return True
        return False

    def _category_match(jd_tool: str) -> bool:
        jd_cats = BRAND_TO_CATEGORIES.get(jd_tool, set())
        if not jd_cats or not cand_categories:
            return False
        return bool(jd_cats & cand_categories)

    # Score core (must-have)
    for t in core:
        if t in cand:
            core_scores[t] = 1.0
            core_present.append(t)
        elif _weak_match(t):
            core_scores[t] = 0.4
            core_missing.append(t)
        elif _category_match(t):
            core_scores[t] = 0.2
            core_missing.append(t)
        else:
            core_scores[t] = 0.0
            core_missing.append(t)

    # Score nice-to-have
    for t in nice:
        if t in cand:
            nice_scores[t] = 0.6
            nice_present.append(t)
        elif _weak_match(t):
            nice_scores[t] = 0.3
            nice_missing.append(t)
        elif _category_match(t):
            nice_scores[t] = 0.1
            nice_missing.append(t)
        else:
            nice_scores[t] = 0.0
            nice_missing.append(t)

    # Aggregate
    core_total_weight = float(len(core)) * 1.0
    nice_total_weight = float(len(nice)) * 0.6

    achieved = sum(core_scores.values()) + sum(nice_scores.values())
    ideal = max(core_total_weight + nice_total_weight, 1e-9)

    score_percent = round((achieved / ideal) * 100.0)

    return {
        "score_percent": score_percent,
        "core_total": len(core),
        "core_exact_matches": len([t for t, s in core_scores.items() if s >= 1.0]),
        "core_nonzero_matches": len([t for t, s in core_scores.items() if s > 0.0]),
        "core_present": core_present,
        "core_missing": core_missing,
        "nice_total": len(nice),
        "nice_present": nice_present,
        "nice_missing": nice_missing,
        "core_scores": core_scores,
        "nice_scores": nice_scores,
    }

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
    cleaned = re.sub(r"\bget\s+me\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btheir\s+contacts?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bget me their contacts?\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bget me\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\btheir contacts?\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\btheir\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bresponsible for\b", "", cleaned, flags=re.I)

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
    """
def _canonicalize_required_tools(required: List[str]) -> List[str]:
    
   " Final cleanup for required_tools: "
   " - run through _clean_skill_phrase "
    "- dedupe"
   " - drop anything that doesn't survive cleaning"
   
    cleaned: List[str] = []
    seen = set()
    for t in required or []:
        c = _clean_skill_phrase(t)
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        cleaned.append(c)
    return cleaned

"""
def _canonicalize_required_tools(required: List[str]) -> List[str]:
    """
    Final cleanup for required_tools:

    - run through _clean_skill_phrase
    - fuzzy-map to canonical skills (python, spring boot, kubernetes, cisco ios, etc.)
    - dedupe
    - limit to a sane number for scoring
    """
    cleaned: List[str] = []
    seen: set[str] = set()

    for t in required or []:
        # First do the heavy JD junk filtering
        c = _clean_skill_phrase(t)
        if not c:
            continue

        # Then fuzzy-map "springboot" -> "spring boot", "pyhton" -> "python", etc.
        canon = _map_skill_to_canonical(c)
        key = canon or c

        if key in seen:
            continue
        seen.add(key)
        cleaned.append(key)

    # Cap the number of core skills used for scoring so % match stays meaningful
    MAX_CORE_TOOLS = 25
    return cleaned[:MAX_CORE_TOOLS]

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
    if "backend developer" in text or "back-end developer" in text:
         add_kw("backend")


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
    "\n"
    "GENERAL BEHAVIOR:\n"
    "- The user message may be either a short natural-language query (e.g. 'top 10 java + kafka developers') "
    "  OR a full job description / JD email with responsibilities and required skills.\n"
    "- Your job is ONLY to produce a retrieval plan, not to rewrite the JD.\n"
    "- Always output STRICT JSON, no comments and no extra keys.\n"
    "- NEVER treat meta-requests like 'and also give me their contact', 'explain why', "
    "  'give me their LinkedIn', or 'why these candidates' as skills. Those requests affect only what the user wants, "
    "  not the JD. Do NOT put any part of such sentences into 'must_have_keywords' or 'require_domains'.\n"
    "\n"
    "VECTOR FIELD CHOICE:\n"
    "- Use 'summary_embedding' for most queries, especially:\n"
    "  - When the text is a full JD / long description.\n"
    "  - When the query describes responsibilities, domain, seniority.\n"
    "- Use 'skills_embedding' ONLY when the question is mostly a list of explicit skills/technologies "
    "  (e.g. 'python, kafka, kubernetes, terraform devops engineers').\n"
    "\n"
    "INTERPRET YEARS OF EXPERIENCE AS CAREER STAGE USING THESE RULES:\n"
    "- 0–3 years, or phrases like 'junior', 'entry level', 'new grad', 'fresher' -> require_career_stage = 'Entry'.\n"
    "- 3–5 years, or phrases like 'mid level', 'intermediate' -> require_career_stage = 'Mid'.\n"
    "- 5+ years, or phrases like 'senior', '8+ years', '10 years', 'principal', 'staff', "
    "  or any explicit 'minimum of 5 years' -> require_career_stage = 'Senior'.\n"
    "- If the user explicitly asks for a management level (e.g. 'team lead', 'engineering manager', "
    "  'director'), use 'Lead/Manager' or 'Director+' instead of 'Senior'.\n"
    "- If the question does NOT clearly specify experience or level, set require_career_stage = 'Any'.\n"
   "- The JSON schema does NOT encode exact year thresholds; do NOT try to filter out candidates based "
"on exact years of experience. Treat '8+ years' the same as 'Senior' and let downstream scoring "
"decide if a candidate with 6–7 years is acceptable (with a note such as 'missing 2 years of experience'). "
"Even for '5+ years' JDs, do not hard-filter out 3–4 year candidates; simply mark them as having an experience gap.\n"

    "\n"
    "INTENT:\n"
    "- If the user asks for 'total how many', 'how many', 'count', or 'total', set intent = 'count'.\n"
    "- If they want a list of candidates (e.g. 'list', 'show', 'top 10', 'give me candidates'), set intent = 'list'.\n"
    "- If they want an explanation or justification ('why', 'explain', 'why these candidates'), set intent = 'why'.\n"
    "- If the user combines a JD with a meta request like 'and also give me their contact', "
    "  still treat this as intent = 'list' (or 'count'/'why' as appropriate) and ignore that meta sentence "
    "  when building 'must_have_keywords'.\n"
    "\n"
    "RULES FOR 'must_have_keywords':\n"
    "- Treat 'must_have_keywords' as the core REQUIRED skills/technologies/platforms/tools for the search.\n"
    "- Each keyword must be:\n"
    "  - SHORT (1–3 words),\n"
    "  - A concrete technology, platform, framework, language, tool, database, cloud service, or certification.\n"
    "  - Lowercase (e.g. 'python', 'aws glue', 'aws cdk', 'spark', 'hadoop', 'azure open ai', "
    "    'salesforce service cloud', 'palo alto', 'bgp', 'ospf').\n"
    "- DO NOT include full sentences or responsibility phrases (e.g. 'analyze the customer needs', "
    "  'strong communication skills', 'work with stakeholders', 'encouraging the exploration of new approaches').\n"
    "- DO NOT include:\n"
    "  - Locations (city, state, country), including state abbreviations like 'nv', 'il', 'ia', 'la', etc.\n"
    " DO NOT include locations, even if written as abbreviations or city names:"
    "Examples to ignore: 'nyc', 'nj', 'new york', '30 rockefeller plaza', 'onsite', '3 days/week'."
    "-DO NOT include SOW-related words like 'sow', 'statement of work', 'backfill', 'competition'."

    "  - Company names or client names.\n"
    "  - Rates, compensation, '60/hr', '$55', 'c2c', 'w2', '1099'.\n"
    "  - Contract length or duration ('12 months', '2-year contract').\n"
    "  - Visa / work authorization terms ('h1b', 'gc', 'ead', 'usc').\n"
    "  - Generic role nouns by themselves ('developer', 'engineer', 'architect', 'analyst') "
    "    unless paired with a specific platform (e.g. 'salesforce engineer' is okay but 'engineer' alone is not).\n"
    "  - Soft skills ('communication', 'team player', 'problem solving', 'self starter', 'fast paced environment').\n"
    "  - Generic business phrases ('according to a client', 'this role will be responsible for', 'd2d', 'day to day').\n"
    "  - Meta/instruction phrases from the user ('also give me their contact', 'and send me emails').\n"
    "- DO NOT include brand/TV/network names like 'nbc', 'msnbc', 'cnbc' as skills.\n"
    "- DO NOT include scheduling or workstyle terms like 'remote', 'onsite', 'hybrid', '2 days onsite'.\n"
    "- DO NOT include very generic conceptual phrases like 'big data technologies', 'cloud-based ai', "
    "  'development', 'implementation', 'aspirations', 'strategy', 'governance policies', 'frameworks and best practices'. "
    "  Instead, extract the **concrete tools** mentioned near them (e.g. 'spark', 'hadoop', 'aws', 'azure open ai').\n"
    "- Deduplicate keywords. If the JD says 'java and spring boot' and 'core java', you can use 'java' and 'spring boot'.\n"
    "- Prefer canonical names (e.g. 'active directory' instead of 'ad', 'identity and access management' or 'iam').\n"
    "- For an AWS Glue / AWS CDK JD like:\n"
    "  'Extensive Python, AWS Glue, AWS CDK, Infrastructure as Code, automated release management'\n"
    "  a good 'must_have_keywords' list would be:\n"
    "  ['python', 'aws glue', 'aws cdk', 'aws']\n"
    "  (You may omit 'infrastructure as code' and 'automated release management' as separate keywords, "
    "  unless a specific IaC tool like 'terraform' or 'cloudformation' is named.)\n"
    "- For a GenAI Lead JD like:\n"
    "  '8+ years, 2+ years AI/ML, 1+ year GenAI/NLP, big data (Hadoop/Spark), Azure Open AI/AWS, Python/R, LLMs'\n"
    "  good 'must_have_keywords' would be:\n"
    "  ['python', 'ai/ml', 'gen ai', 'nlp', 'llm', 'spark', 'hadoop', 'azure open ai', 'aws'].\n"
    "\n"
    "- Each 'must_have_keywords' item must be at most 3 words long."
    "- Never output more than 15 distinct 'must_have_keywords' items, even for long JDs."
    "- If you find longer phrases ('architectural principles develop high quality', 'access control financial services cloud'),"
    "break them into the concrete technologies instead (e.g. 'salesforce', 'service cloud', 'access control') or drop them"
    "if they are not concrete tools."

    "RULES FOR 'require_domains' AND 'industry_equals':\n"
    "- Use 'industry_equals' ONLY when the question clearly specifies a primary industry "
    "  (e.g. 'healthcare', 'finance', 'logistics', 'education', 'manufacturing', 'retail', 'government').\n"
    "- Use 'require_domains' for specific domain expertise or subdomains (e.g. 'Healthcare IT', 'NLP', 'GenAI', "
    "  'IAM', 'cybersecurity', 'salesforce service cloud', 'palo alto', 'rpa uipath', 'mlops').\n"
    "- DO NOT repeat the same tokens in both 'must_have_keywords' and 'require_domains' unless clearly necessary.\n"
    "- DO NOT put locations, rates, visa types, or company names into 'require_domains'.\n"
    "- Ignore section labels like 'Must Haves:', 'Requirements:', 'Responsibilities:', 'D2D:', "
    "  'Day to Day:', 'What you will do:' when building keywords and domains.\n"
    "\n"
    "JD VS. SHORT QUERY HANDLING:\n"
    "- If the text looks like a full JD (mentions 'Responsibilities', 'What you will do', 'Must have', 'Required', "
    "  or long bullet lists), treat it as a JOB DESCRIPTION.\n"
    "  - In that case, extract the main technologies/tools/platforms from the JD into 'must_have_keywords'.\n"
    "  - Do NOT put responsibilities sentences, legal text, or boilerplate into 'must_have_keywords'.\n"
    "- If the text is a short query, just map explicit skill mentions (e.g. 'java', 'kafka', 'spring boot', 'aws', "
    "  'terraform', 'ansible', 'uipath', 'forgerock', 'saviynt', 'salesforce') into 'must_have_keywords'.\n"
    "- If the short query also says something like 'and give me their contact', ignore that phrase entirely "
    "  when building keywords or domains.\n"
    "\n"
    "OTHER RULES:\n"
    "- Use 'summary_embedding' by default for vector_field unless the query is almost purely a skill list, "
    "  then you may use 'skills_embedding'.\n"
    "- Only include 'industry_equals' or 'require_domains' when the question clearly specifies an industry or domain.\n"
    "- If the question does not talk about networking or introductions at all, set 'networking_required' = false.\n"
    "- Always set numeric fields as integers: 'top_k' (default 1000) and 'return_top' (default 20) unless the query "
    "  strongly implies a different number (e.g. 'top 5' -> return_top = 5).\n"
    "- Never add extra JSON keys beyond the schema. Never output prose or explanations.\n"
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


    plan["required_tools"] = _canonicalize_required_tools(plan.get("required_tools") or [])
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

        # Filter out junk/non-skill keys before choosing "top missing"
        filtered: Dict[str, int] = {}
        for key, count in missing_counter.items():
            clean = _clean_skill_phrase(key)
            if not clean:
                continue
            filtered[clean] = filtered.get(clean, 0) + count

        if not filtered:
            return "Few strong matches. Consider relaxing strict must-have requirements."

        top_missing = max(filtered.items(), key=lambda item: item[1])[0]
        top_missing_pretty = _prettify_tool(top_missing)
        VECTOR_DB_FAMILY = {"milvus", "pinecone", "weaviate", "qdrant", "chroma", "faiss"}
        if top_missing_pretty.lower() in VECTOR_DB_FAMILY:
             return (
                f"Most candidates missing {top_missing_pretty}. "
                "Consider allowing 2/4 or accepting Pinecone/Weaviate as equivalents."
            )

        return f"Most candidates missing {top_missing_pretty}. Consider relaxing that requirement or treating it as a nice-to-have."
    return None



def answer_question(question: str, plan_override: Optional[Dict[str, Any]] = None) -> str:
    global LAST_INSIGHT_RESULT

    original_question = question or ""

    if _is_greeting(original_question):
        # Short friendly intro instead of hammering Milvus
        return (
            'Hey John how can i help you with! Ask me things like '
            '"get me 10 django candidates with Milvus, dbt, AWS, Vertex AI" or "list top 10 healthcare candidates".'
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
                    contact_block: List[str] = []
                    linkedin = entity.get("linkedin_url")
                    email = entity.get("email")
                    phone = entity.get("phone")

                    if linkedin: contact_block.append(f"LinkedIn: {linkedin}")
                    if email:    contact_block.append(f"Email: {email}")
                    if phone:    contact_block.append(f"Phone: {phone}")

                    if contact_block:
                      base = base + "\n    " + "\n    ".join(contact_block)

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
        tool_match = _score_candidate_tools(
            core_required=required,
            nice_to_have=plan.get("nice_to_have_tools", []),
            candidate_tools=tools,
        )
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
                "tools": tools,   # stash full scoring details 
                "tool_match": tool_match,                    # normalized tools for the candidate
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

    
    formatted_rows: List[Dict[str, Any]] = []

    for idx, entry in enumerate(final_entries):
        entity = entry["entity"]
        tools = entry["tools"]
        weak_hits = entry.get("weak_hits") or {}

        # Use the precomputed fuzzy scoring (fallback if somehow missing)
        tool_match = entry.get("tool_match") or _score_candidate_tools(
            core_required=required,
            nice_to_have=plan.get("nice_to_have_tools", []),
            candidate_tools=tools,
        )

        total_required = tool_match["core_total"] or len(required) or 1
        missing_tokens = tool_match["core_missing"]
        covered_count = tool_match["core_exact_matches"]

        jd_text = plan.get("_jd_raw") or plan.get("question") or ""


        exp_comment = experience_gap_comment(jd_text, entity)
        has_experience_gap = exp_comment is not None

        # Domain match: JD require_domains vs candidate domains_of_expertise
        jd_domains = set(plan.get("require_domains") or [])
        cand_domains = set(entity.get("domains_of_expertise") or [])
        # If JD doesn't specify domains, treat as match (we don't penalize)
        has_domain_match = True if not jd_domains else bool(jd_domains & cand_domains)

        primary_gap = classify_primary_gap(
            skills_covered=covered_count,
            skills_total=total_required,
            has_experience_gap=has_experience_gap,
            has_domain_match=has_domain_match,
        )
        gap_explanation = build_gap_explanation(
            entity=entity,
            jd_text=jd_text,
            required_tools=required,
            missing_tools=missing_tokens,
            primary_gap=primary_gap,
)

        title = extract_latest_title(
            entity.get("employment_history"),
            entity.get("top_titles_mentioned"),
        )
        canonical_role = _map_role_to_canonical(title)
        entity["role_family_canonical"] = canonical_role


        position = render_position(entity.get("career_stage", ""), title)
        primary, secondary = select_industries(
            entity.get("primary_industry"),
            entity.get("sub_industries"),
        )
        overlaps = extract_overlaps(required, tools)
        why = brief_why(entity, overlaps, max_len=120)

        base_notes = notes_label(
            entry.get("rank_in_perfect"),
            covered_count,
            [_prettify_tool(m) for m in missing_tokens],
            weak_hits,
        )

        # Attach experience + primary gap info to notes
        extra_bits: List[str] = []
        if exp_comment:
            extra_bits.append(exp_comment)  # e.g. "JD requires 8+ years; candidate has 6.1 (short by ~1.9 years)."
        if primary_gap and primary_gap != "none":
            extra_bits.append(f"Primary gap: {primary_gap}")
        if gap_explanation:
          extra_bits.append(gap_explanation)


        notes = base_notes
        if extra_bits:
            notes = base_notes + " " + " ".join(extra_bits)


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
        row["experience_comment"] = exp_comment
        row["primary_gap"] = primary_gap
        row["gap_explanation"] = gap_explanation


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
        table_lines = ["Candidate\tPosition\tTools Match\tNotes"]
        for row in formatted_rows[:return_top]:
            # Base candidate line
            line = (
                f"{row.get('candidate', 'Unknown')}\t"
                f"{row.get('position', '')}\t"
                f"{row.get('tools_match', '')}\t"
                f"{row.get('notes', '')}"
            )

            # If user asked for contacts, append them directly beneath this candidate
            if show_contacts:
                contacts = row.get("contacts") or {}
                contact_bits = []
                if contacts.get("linkedin_url"):
                    contact_bits.append(f"LinkedIn: {contacts['linkedin_url']}")
                if contacts.get("email"):
                    contact_bits.append(f"Email: {contacts['email']}")
                if contacts.get("phone"):
                    contact_bits.append(f"Phone: {contacts['phone']}")

                if contact_bits:
                    # Add a newline and indent contacts under the candidate
                    line += "\n    " + " | ".join(contact_bits)

            # Separator line after each candidate
            line += "\n" + "-" * 60
            table_lines.append(line)

        body = "\n".join(table_lines)
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
