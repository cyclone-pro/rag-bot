"""Utility helpers shared across recruiter brain modules."""
from __future__ import annotations

import ast
import json
import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple , Set

from recruiterbrain.env_loader import load_env
from recruiterbrain.shared_config import ALIAS_MAP, DOMAIN_SYNONYMS, WEAK_EQUIVALENTS
import ast

load_env()

logger = logging.getLogger(__name__)

NETWORK_PAT = re.compile(
    r"\b(network|networking|subnet|vpc|tcp|udp|lan|wan|bgp|ospf|sonic|router|switch|kubernetes|docker)\b",
    re.I,
)


def apply_model_prefix(text: str, model_name: str, *, is_query: bool) -> str:
    #since our main models are embedded models we need prefix ,to get better recall we use prefix
    # else models like openAI,cohere doesnt need prefix and they are good to go 
    """Apply standard prefixes for models like E5/BGE so recall stays high."""
    lower = (model_name or "").lower()
    prefix = ""
    if "e5" in lower:
        prefix = "query: " if is_query else "passage: "
    elif "bge" in lower:
        prefix = "query: " if is_query else "passage: "
    elif "gte" in lower:
        prefix = "query: " if is_query else "passage: "
    return f"{prefix}{text}" if prefix else text


def bag_from_entity(entity: Dict[str, Any]) -> str:
    return " ".join(
        str(entity.get(key, ""))
        for key in (
            "name",
            "career_stage",
            "skills_extracted",
            "primary_industry",      
            "sub_industries",  
            "tools_and_technologies",
            "domains_of_expertise",
            "keywords_summary",
            "semantic_summary",
        )
    )


def to_pct(sim: float) -> str:
    return f"{sim * 100:.1f}%"


def parse_list_like(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        if "," in text:
            return [seg.strip() for seg in text.split(",") if seg.strip()]
        return [text]
    return [str(value)]


def join_head(values: Sequence[str], limit: int = 8) -> str:
    if not values:
        return ""
    trimmed = list(values[:limit])
    suffix = "" if len(values) <= limit else ", ..."
    return ", ".join(trimmed) + suffix


def render_candidate(entity: Dict[str, Any], sim: float, detailed: bool) -> str:
    name = entity.get("name", "Unknown candidate")
    pct = to_pct(sim)
    context_bits = [bit for bit in (entity.get("career_stage"), entity.get("primary_industry")) if bit]
    header = f"{name} ({pct} match"
    if context_bits:
        header += f", {' · '.join(context_bits)}"
    header += ")."

    summary = (entity.get("semantic_summary") or entity.get("keywords_summary") or "").strip()
    if summary.endswith("."):
        summary = summary[:-1]

    skills = join_head(parse_list_like(entity.get("skills_extracted")), limit=8)
    tools = join_head(parse_list_like(entity.get("tools_and_technologies")), limit=6)
    domains = join_head(parse_list_like(entity.get("domains_of_expertise")), limit=5)

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


def attach_sim_scores(rows: Iterable[Dict[str, Any]], hits: Sequence[Any], *, id_field: str = "candidate_id") -> List[Tuple[Dict[str, Any], float]]:
    """Pair hydrated rows with their similarity from the ANN response."""
    sim_by_id = {}
    for hit in hits:
        candidate_id = None
        if isinstance(hit, dict):
            entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
            candidate_id = entity.get(id_field) or hit.get(id_field) or hit.get("id")
            distance = hit.get("distance") or hit.get("score")
        else:
            candidate_id = getattr(hit, id_field, None) or getattr(hit, "id", None)
            distance = getattr(hit, "distance", None) or getattr(hit, "score", None)
        if candidate_id is None or distance is None:
            continue
        sim_by_id[str(candidate_id)] = float(distance)

    results: List[Tuple[Dict[str, Any], float]] = []
    for row in rows:
        cid = str(row.get(id_field))
        sim = sim_by_id.get(cid, 0.0)
        results.append((row, sim))
    return results


# === Insight helpers === #

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_latest_title(employment_history: Any, top_titles_mentioned: Any) -> str:
    """Return the latest role title from history or fall back to top titles."""
    title = ""
    try:
        if isinstance(employment_history, list) and employment_history:
            latest = employment_history[-1]
            if isinstance(latest, dict):
                for key in ("title", "role", "position"):
                    if latest.get(key):
                        title = str(latest[key]).strip()
                        break
    except Exception:
        pass
    if not title and isinstance(top_titles_mentioned, list) and top_titles_mentioned:
        title = str(top_titles_mentioned[0]).strip()
    return title


def render_position(career_stage: str, title: str) -> str:
    career = (career_stage or "").strip()
    ttl = (title or "").strip()
    if career and ttl:
        return f"{career} · {ttl}"
    return career or ttl or ""


def select_industries(primary_industry: str, sub_industries: Any) -> Tuple[str, str]:
    primary = (primary_industry or "").strip()
    secondary = ""
    if isinstance(sub_industries, list) and sub_industries:
        secondary = str(sub_industries[0]).strip()
    return primary, secondary


def _gather_text_fields(entity: Dict[str, Any], keys: List[str]) -> List[str]:
    """
    Extract text values from entity fields, properly handling both strings and lists.
    """
    values: List[str] = []
    for key in keys:
        value = entity.get(key)
        if not value:
            continue
            
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            # If it's already a list, join with semicolons to preserve structure
            values.append("; ".join(str(item) for item in value if item))
        else:
            values.append(str(value))
    
    return values


def normalize_tools(entity: Dict[str, Any]) -> tuple[set[str], Dict[str, Any]]:
    """
    Collects all skills/tools/domains from key fields and canonicalizes them into a
    normalized set used for JD coverage.
    """
    text_fields = _gather_text_fields(
        entity,
        [
            "tools_and_technologies",
            "skills_extracted",
            "domains_of_expertise",
            "primary_industry",
            "sub_industries",
            "keywords_summary",
            "semantic_summary",
            "certifications",
            "evidence_tools",
            "evidence_skills",
            "employment_history",
        ],
    )

    raw_tokens: List[str] = []
    for text in text_fields:
        if not text:
            continue
        
        # Convert to string if it's not already
        text_str = str(text)
        
        # CRITICAL: Try to parse stringified lists first
        # Your DB has entries like "['apex', 'visualforce', 'lwc']"
        if text_str.startswith("[") and text_str.endswith("]"):
            try:
                parsed = ast.literal_eval(text_str)
                if isinstance(parsed, list):
                    raw_tokens.extend([str(item).strip() for item in parsed if item])
                    continue
            except (ValueError, SyntaxError):
                # If parsing fails, treat as regular string
                pass
        
        # Split on semicolons (your primary separator), commas, slashes
        for piece in re.split(r"[;,/]|\band\b", text_str, flags=re.IGNORECASE):
            piece = piece.strip()
            # Remove any lingering quotes from the piece
            piece = piece.strip("'\"")
            if piece and piece != "[]":
                raw_tokens.append(piece)

    normalized: set[str] = set()

    for token in raw_tokens:
        if not token:
            continue
        
        # Clean up any remaining quote artifacts
        token = token.strip("'\"[]")
        
        s = _norm(token)  # lower + strip whitespace
        if not s:
            continue

        # === Salesforce canonicalization ===
        if "apex" in s:
            normalized.add("apex")
        if "visualforce" in s or "vf page" in s or "vf" == s:
            normalized.add("visualforce")
        if ("lightning" in s and ("web" in s or "component" in s)) or s == "lwc":
            normalized.add("lightning web components")
            normalized.add("lwc")
        if "cpq" in s or "configure price quote" in s:
            normalized.add("salesforce cpq")
            normalized.add("cpq")
        if "data loader" in s or "dataloader" in s:
            normalized.add("data loader")
        if "data migration" in s or "data import" in s or "import wizard" in s:
            normalized.add("data migration")
        if "service cloud" in s:
            normalized.add("service cloud")
        if "sales cloud" in s:
            normalized.add("sales cloud")
        if "marketing cloud" in s:
            normalized.add("marketing cloud")
        if "financial services cloud" in s or "fsc" in s:
            normalized.add("financial services cloud")
        if "salesforce" in s or "sfdc" in s:
            normalized.add("salesforce")
        
        # === Integration/API ===
        if s in {"rest", "rest api", "restful", "rest services"}:
            normalized.add("rest")
        if s in {"soap", "soap api", "soap services"}:
            normalized.add("soap")
        if s in {"integration", "integrations", "api integration"}:
            normalized.add("integration")
        if s in {"agile", "agile methodology", "scrum"}:
            normalized.add("agile")

        # === Cloud canonicalization (existing) ===
        if "aws" in s:
            normalized.add("aws")
        if "azure" in s:
            normalized.add("azure")
        if "gcp" in s or "google cloud" in s:
            normalized.add("gcp")

        # === Other existing canonicalizations ===
        if "hipaa" in s:
            normalized.add("hipaa")
        if "backend" in s and "developer" in s:
            normalized.add("backend")

        # Finally add the full normalized token itself
        normalized.add(s)

    ctx = {
        "raw_tools": text_fields,
        "normalized_tools": list(normalized),
    }
    return normalized, ctx


def jd_tool_covered(
    req_norm: str,
    norm_tools: Set[str],
    resume_text: Optional[str] = None,
) -> bool:
    """
    Decide if a required JD tool (already normalized) is covered by the candidate.

    1) Check against normalized structured tools (norm_tools).
    2) If not found, fall back to a raw-text search across the resume text blob.
    """
    if not req_norm:
        return False

    # ---- strong match: exact or LIMITED substring match ----
    def strong_match(tool: str) -> bool:
        if req_norm == tool:
            return True

        # Only allow substring matching for reasonably long tokens,
        # to avoid "la" matching "flask", etc.
        if len(req_norm) >= 4 and len(tool) >= 4:
            return req_norm in tool or tool in req_norm
        return False

    if any(strong_match(tool) for tool in norm_tools):
        return True

    # ---- raw resume text fallback ----
    if resume_text:
        txt = resume_text.lower()
        # Slightly relaxed length check here
        if len(req_norm) >= 3 and req_norm in txt:
            return True

    return False


def coverage(
    required_tools: List[str],
    normalized_tools: set[str],
    resume_text: Optional[str] = None,
) -> Tuple[int, List[str], Dict[str, List[str]]]:
    """
    Compute how many of the required tools are covered.

    - Strong match: via jd_tool_covered (structured tools + raw resume text).
    - Weak match: no strong match, but one of the WEAK_EQUIVALENTS is present.
      Weak matches are returned in weak_hits and still count toward coverage.
    """
    logger.debug("Computing coverage for required=%s", required_tools)
     # DEBUG: Log what we're comparing
    logger.info("=" * 60)
    logger.info("COVERAGE DEBUG")
    logger.info("Required tools: %s", required_tools)
    logger.info("Normalized candidate tools: %s", list(normalized_tools)[:20])
    logger.info("=" * 60)

    # normalize tools once
    norm_tools: Set[str] = {_norm(t) for t in normalized_tools if t}

    covered_set: set[str] = set()
    missing: List[str] = []
    weak_hits: Dict[str, List[str]] = {}

    for req in required_tools:
        req_norm = _norm(req)
        if not req_norm:
            continue

        # ---- strong match via structured tools OR raw resume text ----
        if jd_tool_covered(req_norm, norm_tools, resume_text):
            covered_set.add(req_norm)
            continue

        # ---- weak match via WEAK_EQUIVALENTS ----
        hits: List[str] = []
        cousins = WEAK_EQUIVALENTS.get(req_norm) or set()
        for cousin in cousins:
            cousin_norm = _norm(cousin)
            if any(
                cousin_norm == tool
                or cousin_norm in tool
                or tool in cousin_norm
                for tool in norm_tools
            ):
                hits.append(cousin)

        if hits:
            weak_hits[req] = hits
            covered_set.add(req_norm)  # count as covered (but weak)
            logger.info("~ WEAK MATCH: %s via %s", req, hits)
            continue

        # ---- truly missing ----
        missing.append(req)
        logger.info("✗ MISSING: %s", req)

    logger.info("Final coverage: %d/%d", len(covered_set), len(required_tools))
    # tier_label expects an int
    return len(covered_set), missing, weak_hits



def extract_overlaps(required_tools: List[str], normalized_tools: set[str]) -> List[str]:
    overlaps = []
    for req in required_tools:
        req_norm = _norm(req)
        if any(req_norm == tool or req_norm in tool for tool in normalized_tools):
            overlaps.append(req)
        if len(overlaps) >= 3:
            break
    return overlaps



def percentiles_min_rank(similarities: List[float]) -> List[int]:
    count = len(similarities)
    if count <= 1:
        return [100 for _ in similarities]
    indexed = [(score, idx) for idx, score in enumerate(similarities)]
    indexed.sort(key=lambda pair: -pair[0])
    ranks = [0] * count
    cursor = 0
    while cursor < count:
        block_end = cursor
        score = indexed[cursor][0]
        while block_end < count and math.isclose(indexed[block_end][0], score, rel_tol=1e-12, abs_tol=1e-12):
            block_end += 1
        for pos in range(cursor, block_end):
            ranks[indexed[pos][1]] = cursor + 1
        cursor = block_end
    percentiles = []
    for rank in ranks:
        pct = int(math.floor(100.0 * (count - rank) / (count - 1))) if count > 1 else 100
        percentiles.append(max(0, min(100, pct)))
    return percentiles


def tier_label(covered: int) -> str:
    if covered >= 4:
        return "Perfect"
    if covered == 3:
        return "Good"
    if covered == 2:
        return "Acceptable"
    return "Partial"


def _append_near_miss(base: str, weak_hits: Dict[str, List[str]]) -> str:
    milvus_hits = weak_hits.get("milvus") or []
    if milvus_hits:
        near = "/".join(alt.upper() for alt in milvus_hits[:2])
        return f"{base}; close to Milvus via {near}"
    return base


def notes_label(
    perfect_index: Optional[int],
    covered: int,
    missing: List[str],
    weak_hits: Dict[str, List[str]],
) -> str:
    """
    Improved JD match interpretation with stable percentile buckets.
    Uses ratio + absolute coverage threshold to avoid false 'perfect' matches.
    """
    total = covered + len(missing)
    if total <= 0:
        return "partial match"

    ratio = covered / total

    # *** PERFECT MATCH ***
    # Require BOTH:
    # - high coverage %
    # - AND at least 6 strong hits (avoids '4/5 perfect' for tiny subsets)
    if ratio >= 0.80 and covered >= 6:
        if perfect_index == 0:
            return "perfect match"
        if perfect_index == 1:
            return "2nd best perfect match"
        if perfect_index == 2:
            return "3rd best perfect match"
        return "perfect match"

    # *** GOOD MATCH ***
    if ratio >= 0.60:
        base = "good match"
        if missing:
            base += f" (missing {missing[0]})"
        return _append_near_miss(base, weak_hits)

    # *** ACCEPTABLE MATCH ***
    if ratio >= 0.40:
        miss = ", ".join(missing[:2]) if missing else ""
        base = f"acceptable (missing {miss})" if miss else "acceptable"
        return _append_near_miss(base, weak_hits)

    # *** PARTIAL MATCH ***
    base = "partial match"
    return _append_near_miss(base, weak_hits)

def classify_primary_gap(
    skills_covered: int,
    skills_total: int,
    has_experience_gap: bool,
    has_domain_match: bool,
) -> str:
    """
    Returns 'skills', 'experience', 'industry', 'experience and skills',
    'experience and industry', 'skills and industry', or 'none'.
    """
    # Skill coverage ratio
    ratio = skills_covered / skills_total if skills_total > 0 else 1.0

    # Decide thresholds
    skill_weak = ratio < 0.5
    exp_weak = has_experience_gap
    industry_weak = not has_domain_match

    # Priority order: experience > skills > industry
    if exp_weak and (not skill_weak) and (not industry_weak):
        return "experience"
    if skill_weak and (not exp_weak) and (not industry_weak):
        return "skills"
    if industry_weak and (not exp_weak) and (not skill_weak):
        return "industry"

    # If multiple are weak, pick the worst combo
    if exp_weak and skill_weak and not industry_weak:
        return "experience and skills"
    if exp_weak and industry_weak and not skill_weak:
        return "experience and industry"
    if skill_weak and industry_weak and not exp_weak:
        return "skills and industry"

    # All three weak → you can choose one; I’d call it "experience and skills"
    if exp_weak and skill_weak and industry_weak:
        return "experience and skills"

    return "none"


def _normalize_field_to_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        # Try to parse "['HL7', 'HIPAA']" style
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        # Fallback: comma-separated string
        return [v.strip() for v in s.split(",") if v.strip()]
    # anything else
    return [str(value)]

def build_gap_explanation(
    entity: Dict[str, Any],
    jd_text: str,
    required_tools: List[str],
    missing_tools: List[str],
    primary_gap: str,
) -> Optional[str]:
    """
    Build a short human-readable explanation for why this candidate is not a strong fit.

    Examples:
    - "Missing core JD tools: Excel, SharePoint, document automation. Candidate tools focus on Python, GraphQL, AWS; domains: NLP, Software Engineering."
    - "Meets skill requirements, but short on required experience (JD requires 8+ years; candidate has 5.1)."
    - "Skills and experience are good, but domain mismatch: JD is Legal/Professional Services, candidate is NLP / CAD / Software Engineering."
    """
    jd_text_lower = (jd_text or "").lower()
    if not required_tools and not missing_tools and primary_gap == "none":
        return None

    # Hard constraints: missing required tools (limit to 3 for readability)
    missing_pretty = [t for t in missing_tools][:3]

    # Candidate context
    cand_tools = _normalize_field_to_list(
     entity.get("skills_extracted") or entity.get("tools_and_technologies")
)
    cand_domains = _normalize_field_to_list(entity.get("domains_of_expertise"))

    cand_tools_str = ", ".join(str(t) for t in cand_tools[:4])
    cand_domains_str = ", ".join(str(d) for d in cand_domains[:3])

    
    cand_primary_industry = entity.get("primary_industry") or ""
    

    parts: List[str] = []

    if "skills" in primary_gap and missing_pretty:
        parts.append(
            "Missing core JD tools: "
            + ", ".join(missing_pretty)
            + "."
        )

    if "experience" in primary_gap:
        years = entity.get("total_experience_years")
        if isinstance(years, (int, float)):
            parts.append(f"Candidate has {years:.1f} years of total experience, which is below the JD's senior expectation.")

    if "industry" in primary_gap:
        # Very simple heuristic: if JD text mentions 'legal' or 'law' and candidate is not,
        # highlight that mismatch.
        if "legal" in jd_text_lower or "law" in jd_text_lower:
            parts.append(
                "JD is for a Legal / professional services role, but candidate domains are: "
                + (cand_domains_str or "not clearly legal-focused")
                + "."
            )
        elif cand_primary_industry:
            parts.append(
                f"JD domain differs from candidate's primary industry ({cand_primary_industry})."
            )
        else:
            parts.append(
                "JD domain differs from candidate's current domains."
            )

    # If we haven't explained anything but we know tools/domains, still add a generic sentence
    if not parts:
        if cand_tools_str or cand_domains_str:
            parts.append(
                f"Candidate tools: {cand_tools_str or 'N/A'}; domains: {cand_domains_str or 'N/A'}."
            )

    if not parts:
        return None

    return " ".join(parts)

def _extract_min_years_from_jd(jd_text: str) -> Optional[int]:
    """
    Very simple heuristic: find patterns like '5+ years', '8+ years', '10 years of experience'.
    Return the highest number we see, or None if not found.
    """
    if not jd_text:
        return None

    years = []
    # matches '5+ years', '8+ years of', etc.
    for m in re.finditer(r"(\d+)\s*\+\s*years", jd_text.lower()):
        years.append(int(m.group(1)))
    # matches '5 years of experience', '10 years experience'
    for m in re.finditer(r"(\d+)\s+years(?:\s+of)?\s+experience", jd_text.lower()):
        years.append(int(m.group(1)))

    return max(years) if years else None


def _compute_total_years_from_history(entity: Dict[str, Any]) -> Optional[float]:
    """
    If you already pre-compute 'years_of_experience' somewhere, use that.
    Otherwise, this can sum employment_history ranges.
    For now we assume you have a field 'years_of_experience' on the entity.
    """
    yrs = entity.get("years_of_experience")
    if isinstance(yrs, (int, float)):
        return float(yrs)
    return None


def experience_gap_comment(jd_text: str, entity: Dict[str, Any]) -> Optional[str]:
    """
    Returns a string like:
    'JD requires 8+ years; candidate has 6.0 (short by ~2 years).'
    or None if we can't compute.
    """
    jd_min = _extract_min_years_from_jd(jd_text)
    cand_yrs = _compute_total_years_from_history(entity)

    if jd_min is None or cand_yrs is None:
        return None

    delta = jd_min - cand_yrs
    if delta <= 0:
        # candidate meets or exceeds years
        return None

    # This is your -2 years tolerance: don't reject, just mention it.
    return f"JD requires {jd_min}+ years; candidate has {cand_yrs:.1f} (short by ~{delta:.1f} years)."
def brief_why(
    entity: Dict[str, Any],
    overlaps: List[str],
    max_len: int = 120,
    max_words: int = 20,
) -> str:
    summary = (entity.get("semantic_summary") or "").strip()
    if not summary:
        if overlaps:
            summary = "Strong overlap in " + ", ".join(overlaps[:3])
        else:
            title = extract_latest_title(
                entity.get("employment_history"),
                entity.get("top_titles_mentioned"),
            )
            primary, _ = select_industries(
                entity.get("primary_industry"),
                entity.get("sub_industries"),
            )
            summary = "Relevant background in " + (primary or "industry")
            if title:
                summary += f" + {title}"

    summary = summary.strip()

    # First enforce word count ~15–20 words
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "…"
    elif len(summary) > max_len:
        summary = summary[: max_len - 1] + "…"

    return summary


def data_quality_check(entities: List[Dict[str, Any]]) -> Optional[str]:
    def missing_keys(entity: Dict[str, Any]) -> bool:
        title = extract_latest_title(entity.get("employment_history"), entity.get("top_titles_mentioned"))
        position = render_position(entity.get("career_stage", ""), title)
        has_position = bool(position)
        has_industry = bool(entity.get("primary_industry"))
        has_tools = any(
            bool(entity.get(key))
            for key in (
                "tools_and_technologies",
                "skills_extracted",
                "domains_of_expertise",
                "evidence_tools",
            )
        )
        return not (has_position and has_industry and has_tools)

    if not entities:
        return None
    missing_count = sum(1 for row in entities if missing_keys(row))
    ratio = missing_count / max(1, len(entities))
    if ratio > 0.30:
        return (
            "Data quality hint: Many profiles lack position/industry/tools. "
            "Add 'tools_and_technologies' and 'semantic_summary' for better matches."
        )
    return None


def format_row(
    entity: Dict[str, Any],
    percentile: int,
    covered: int,
    required_total: int,
    position: str,
    primary: str,
    secondary: str,
    why: str,
    notes: str,
    show_contacts: bool,
    *,
    candidate_name: Optional[str] = None,
    tools_match: Optional[str] = None,
) -> Dict[str, Any]:
    logger.debug(
        "Formatting row for candidate %s (covered=%s/%s)",
        entity.get("candidate_id"),
        covered,
        required_total,
    )
    name = candidate_name or entity.get("name") or "Unknown"
    sec = f", {secondary}" if secondary else ""
    pos = position or "-"
    header = f"{name} ({percentile}th percentile, {pos}, {primary or ''}{sec})"
    if required_total > 0:
         pct = round(covered / required_total * 100)
    else:
         pct = 0
    match_chip = f"{covered}/{required_total} skills match ({pct}%)"

    payload: Dict[str, Any] = {
        "title_line": header,
        "match_chip": match_chip,
        "why": why,
        "notes": notes,
        "evidence_preview": None,
    }
    payload["candidate"] = name
    if tools_match:
        payload["tools_match"] = tools_match
    if show_contacts:
        payload["contacts"] = {
            "linkedin_url": entity.get("linkedin_url"),
            "email": entity.get("email"),
        }
    return payload


def evidence_snippets(entity: Dict[str, Any]) -> List[Dict[str, str]]:
    logger.debug("Collecting evidence snippets for candidate_id=%s", entity.get("candidate_id"))
    sources = []
    ordered = [
        ("employment_history", "Work history"),
        ("evidence_tools", "Tools evidence"),
        ("evidence_skills", "Skills evidence"),
        ("evidence_domains", "Domains evidence"),
        ("semantic_summary", "Summary"),
    ]
    for key, label in ordered:
        value = entity.get(key)
        if not value:
            continue
        if isinstance(value, list):
            for entry in value:
                sources.append((label, str(entry)))
        else:
            sources.append((label, str(value)))
    snippets: List[Dict[str, str]] = []
    for label, text in sources:
        normalized = _norm(text)
        if not normalized:
            continue
        clean = text.strip()
        if len(clean) > 140:
            clean = clean[:139] + "…"
        snippets.append({"source": label, "text": clean})
        if len(snippets) >= 2:
            break
    return snippets


__all__ = [
    "NETWORK_PAT",
    "apply_model_prefix",
    "attach_sim_scores",
    "bag_from_entity",
    "brief_why",
    "coverage",
    "data_quality_check",
    "evidence_snippets",
    "extract_latest_title",
    "extract_overlaps",
    "format_row",
    "join_head",
    "normalize_tools",
    "notes_label",
    "percentiles_min_rank",
    "parse_list_like",
    "render_position",
    "render_candidate",
    "select_industries",
    "tier_label",
    "to_pct",
]
