"""Utility helpers shared across recruiter brain modules."""
from __future__ import annotations

import ast
import json
import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from recruiterbrain.shared_config import ALIAS_MAP, DOMAIN_SYNONYMS, WEAK_EQUIVALENTS

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
    values: List[str] = []
    for key in keys:
        value = entity.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend([str(item) for item in value if item])
    return values

"""
def normalize_tools(entity: Dict[str, Any]) -> Tuple[set[str], Dict[str, bool]]:
    logger.debug("Normalizing tools for candidate_id=%s", entity.get("candidate_id"))
    raw = _gather_text_fields(
        entity,
        [
            "tools_and_technologies",
            "skills_extracted",
            "domains_of_expertise",
            "evidence_tools",
        ],
    )
    normalized = set()
    raw_lower = [_norm(token) for token in raw if token]
    corpus = " ".join(
        _gather_text_fields(
            entity,
            ["employment_history", "semantic_summary", "keywords_summary"],
        )
    )
    corpus_norm = _norm(corpus)
    context_flags = {
        "vector_context": any(
            needle in corpus_norm
            for needle in [
                "embedding",
                "hnsw",
                "ann",
                "approximate nearest",
                "vector search",
                "vector database",
            ]
        )
    }
    for token in raw_lower:
        mapped = ALIAS_MAP.get(token, token)
        if mapped in DOMAIN_SYNONYMS and context_flags["vector_context"]:
            mapped = DOMAIN_SYNONYMS[mapped]
        normalized.add(mapped)
    return normalized, context_flags

"""
def normalize_tools(entity: Dict[str, Any]) -> tuple[set[str], Dict[str, Any]]:
    text_fields = _gather_text_fields(
        entity,
        [
            "tools_and_technologies",
            "skills_extracted",
            "domains_of_expertise",
            "primary_industry",
            "sub_industries",
        ],
    )

    raw_tokens: List[str] = []
    for text in text_fields:
        if not text:
            continue
        # split on commas, slashes, semicolons, and " and "
        for piece in re.split(r"[,/;]|\band\b", str(text), flags=re.IGNORECASE):
            piece = piece.strip()
            if piece:
                raw_tokens.append(piece)

    raw_lower = [_norm(token) for token in raw_tokens]
    normalized = set(raw_lower)

    ctx = {
        "raw_tools": text_fields,
        "normalized_tools": list(normalized),
    }
    return normalized, ctx

def coverage(required_tools: List[str], normalized_tools: set[str]) -> Tuple[int, List[str], Dict[str, List[str]]]:
    logger.debug("Computing coverage for required=%s", required_tools)
    required_norm = [_norm(req) for req in required_tools]
    covered = 0
    missing: List[str] = []
    weak_hits: Dict[str, List[str]] = {}
    for req in required_tools:
     req_norm = _norm(req)
     if not req_norm:
         continue

    # strong match: exact or substring match inside any normalized tool string
     if any(req_norm == tool or req_norm in tool for tool in normalized_tools):
         covered.add(req_norm)
         continue

    # synonym-based weak matches (also use substring logic)
     if DOMAIN_SYNONYMS:
         hits = []
         for key, alts in DOMAIN_SYNONYMS.items():
             if req_norm == key:
                 for alt in alts:
                     if any(alt == tool or alt in tool for tool in normalized_tools):
                         hits.append(alt)
         if hits:
             weak_hits[req] = hits
            # still count as "covered" but as weak
             continue

    missing.append(req)

    return covered, missing, weak_hits
    


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
    if covered >= 4:
        if perfect_index == 0:
            return "perfect match"
        if perfect_index == 1:
            return "2nd best perfect match"
        if perfect_index == 2:
            return "3rd best perfect match"
        return "perfect match"
    if covered == 3:
        base = "good match"
        if missing:
            base = f"good match (missing {missing[0]})"
        return _append_near_miss(base, weak_hits)
    if covered == 2:
        miss = " & ".join(missing[:2])
        base = f"acceptable (missing {miss})" if miss else "acceptable"
        return _append_near_miss(base, weak_hits)
    base = "partial match"
    return _append_near_miss(base, weak_hits)


def brief_why(entity: Dict[str, Any], overlaps: List[str], max_len: int = 120) -> str:
    summary = (entity.get("semantic_summary") or "").strip()
    if not summary:
        if overlaps:
            summary = "Strong overlap in " + ", ".join(overlaps[:3])
        else:
            title = extract_latest_title(entity.get("employment_history"), entity.get("top_titles_mentioned"))
            primary, _ = select_industries(entity.get("primary_industry"), entity.get("sub_industries"))
            summary = "Relevant background in " + (primary or "industry")
            if title:
                summary += f" + {title}"
    summary = summary.strip()
    if len(summary) > max_len:
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
    match_chip = "match: in" if covered >= 4 else f"match: out ({covered}/{required_total})"
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
