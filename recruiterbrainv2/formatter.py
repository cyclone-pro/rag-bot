"""Format V2 search results for different output formats."""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def format_for_chat(results: Dict[str, Any], show_contacts: bool = False) -> str:
    """Format V2 results as plain text for chat interface."""
    candidates = results.get("candidates", [])
    total = results.get("total_found", 0)
    mode = results.get("search_mode", "vector")
    
    if not candidates:
        return f"No candidates found. (searched {total} records, mode={mode})"
    
    lines = [f"Found {total} candidates (showing top {len(candidates)}, mode={mode})\n"]
    
    for i, cand in enumerate(candidates, 1):
        match = cand.get("match", {})
        
        # Header
        lines.append(
            f"{i}. {cand.get('name', 'Unknown')} "
            f"({match.get('match_percentage', 0)}% match)"
        )
        
        # Details
        lines.append(f"   Career Stage: {cand.get('career_stage', 'Unknown')}")
        lines.append(f"   Industry: {cand.get('primary_industry', 'Unknown')}")
        lines.append(
            f"   Experience: {cand.get('total_experience_years', 0)} years"
        )
        
        # Skills match
        matched = match.get("matched_skills", [])
        missing = match.get("missing_skills", [])
        
        if matched:
            lines.append(f"   ✅ Matched: {', '.join(matched[:5])}")
        if missing:
            lines.append(f"   ❌ Missing: {', '.join(missing[:5])}")
        
        # Contacts (if requested)
        if show_contacts:
            if cand.get("email"):
                lines.append(f"   📧 {cand['email']}")
            if cand.get("phone"):
                lines.append(f"   📞 {cand['phone']}")
            if cand.get("linkedin_url"):
                lines.append(f"   🔗 {cand['linkedin_url']}")
        
        lines.append("")  # Blank line
    
    return "\n".join(lines)


def format_for_insight(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format V2 results for insight/ranking view."""
    candidates = results.get("candidates", [])
    
    rows = []
    for cand in candidates:
        match = cand.get("match", {})
        
        rows.append({
            "candidate": cand.get("name", "Unknown"),
            "candidate_id": cand.get("candidate_id"),
            "position": f"{cand.get('career_stage', '')} · {cand.get('primary_industry', '')}",
            "match_chip": f"{match.get('matched_skills', []).__len__()}/{match.get('total_required', 0)} skills ({match.get('match_percentage', 0)}%)",
            "matched": match.get("matched_skills", []),
            "missing": match.get("missing_skills", []),
            "why": cand.get("summary", "")[:150],
            "notes": _generate_notes(match),
            "contacts": {
                "email": cand.get("email"),
                "phone": cand.get("phone"),
                "linkedin_url": cand.get("linkedin_url"),
            }
        })
    
    return {
        "rows": rows,
        "total_matched": results.get("total_found", 0),
        "scarcity_message": _generate_scarcity_message(results),
        "data_quality_banner": None,
    }


def _generate_notes(match: Dict[str, Any]) -> str:
    """Generate human-readable match notes."""
    pct = match.get("match_percentage", 0)
    
    if pct >= 80:
        return "Perfect match"
    elif pct >= 60:
        missing = match.get("missing_skills", [])
        if missing:
            return f"Good match (missing {missing[0]})"
        return "Good match"
    elif pct >= 40:
        return "Acceptable match"
    else:
        return "Partial match"


def _generate_scarcity_message(results: Dict[str, Any]) -> str:
    """Generate scarcity/recommendation message."""
    candidates = results.get("candidates", [])
    
    if not candidates:
        return "No matches found. Try relaxing skill requirements."
    
    # Count perfect matches
    perfect = sum(
        1 for c in candidates 
        if c.get("match", {}).get("match_percentage", 0) >= 80
    )
    
    if perfect == 0:
        # Find most common missing skill
        all_missing = []
        for c in candidates:
            all_missing.extend(c.get("match", {}).get("missing_skills", []))
        
        if all_missing:
            from collections import Counter
            most_common = Counter(all_missing).most_common(1)[0][0]
            return f"No perfect matches. Most candidates missing: {most_common}"
        
        return "No perfect matches. Consider relaxing requirements."
    
    return None  # No scarcity message needed