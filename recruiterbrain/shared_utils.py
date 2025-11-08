"""Utility helpers shared across recruiter brain modules."""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

NETWORK_PAT = re.compile(
    r"\b(network|networking|subnet|vpc|tcp|udp|lan|wan|bgp|ospf|sonic|router|switch|kubernetes|docker)\b",
    re.I,
)


def apply_model_prefix(text: str, model_name: str, *, is_query: bool) -> str:
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


__all__ = [
    "NETWORK_PAT",
    "apply_model_prefix",
    "attach_sim_scores",
    "bag_from_entity",
    "join_head",
    "parse_list_like",
    "render_candidate",
    "to_pct",
]
