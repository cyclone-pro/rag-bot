"""Format V2 search results - optimized for candidates_v3."""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def format_for_chat(results: Dict[str, Any], show_contacts: bool = False) -> str:
    """Format V3 results as plain text for chat interface."""
    
    # Handle errors first
    if results.get("error"):
        return results["error"]
    
    candidates = results.get("candidates", [])
    total = results.get("total_found", 0)
    mode = results.get("search_mode", "vector")
    
    if not candidates:
        return f"No candidates found. (searched {total} records, mode={mode})"
    
    lines = [f"Found {total} candidates (showing top {len(candidates)}, mode={mode})\n"]
    
    for i, cand in enumerate(candidates, 1):
        match = cand.get("match", {})
        
        # Header with match percentage
        lines.append(
            f"{i}. {cand.get('name', 'Unknown')} "
            f"({match.get('match_percentage', 0)}% match)"
        )
        
        # Career & Industry
        career_stage = cand.get('career_stage', 'Unknown')
        industries = cand.get('industries_worked', cand.get('primary_industry', 'Unknown'))
        exp_years = cand.get('total_experience_years', 0)
        
        lines.append(f"   {career_stage} • {industries} • {exp_years} years exp")
        
        # Location
        if cand.get('location'):
            lines.append(f"   📍 {cand['location']}")
        
        # Role type (NEW in V3)
        if cand.get('role_type'):
            lines.append(f"   💼 {cand['role_type']}")
        
        # Skills match
        matched = match.get("matched_skills", [])
        missing = match.get("missing_skills", [])
        
        if matched:
            lines.append(f"   ✅ Has: {', '.join(matched[:5])}")
        if missing:
            lines.append(f"   ❌ Missing: {', '.join(missing[:3])}")
        
        # Current tech stack (NEW in V3)
        if cand.get('current_tech_stack'):
            current = cand['current_tech_stack'].split(',')[:5]
            lines.append(f"   🔧 Currently using: {', '.join(current)}")
        
        # Management experience (NEW in V3)
        mgmt_years = cand.get('management_experience_years', 0)
        if mgmt_years > 0:
            lines.append(f"   👥 {mgmt_years} years managing teams")
        
        # Contacts (if requested)
        if show_contacts:
            if cand.get("email"):
                lines.append(f"   📧 {cand['email']}")
            if cand.get("phone"):
                lines.append(f"   📞 {cand['phone']}")
            if cand.get("linkedin_url"):
                lines.append(f"   🔗 {cand['linkedin_url']}")
            if cand.get("github_url"):
                lines.append(f"   💻 {cand['github_url']}")
        
        lines.append("")  # Blank line
    
    return "\n".join(lines)


def format_for_insight(results: Dict[str, Any]) -> Dict[str, Any]:
    """Format V3 results for insight/ranking view."""
    
    # Handle errors
    if results.get("error"):
        return {
            "rows": [],
            "total_matched": 0,
            "scarcity_message": results["error"],
            "data_quality_banner": None,
        }
    
    candidates = results.get("candidates", [])
    
    rows = []
    for cand in candidates:
        match = cand.get("match", {})
        
        # Build position string with industries
        career_stage = cand.get("career_stage", "Unknown")
        industries = cand.get("industries_worked", cand.get("primary_industry", "Unknown"))
        
        rows.append({
            "candidate": cand.get("name", "Unknown"),
            "candidate_id": cand.get("candidate_id"),
            "position": f"{career_stage} · {industries}",
            "match_chip": f"{len(match.get('matched_skills', []))}/{match.get('total_required', 0)} skills ({match.get('match_percentage', 0)}%)",
            "matched": match.get("matched_skills", []),
            "missing": match.get("missing_skills", []),
            "why": cand.get("summary", "")[:150],
            "notes": _generate_notes(cand, match),
            "contacts": {
                "email": cand.get("email"),
                "phone": cand.get("phone"),
                "linkedin_url": cand.get("linkedin_url"),
                "github_url": cand.get("github_url"),
            }
        })
    
    return {
        "rows": rows,
        "total_matched": results.get("total_found", 0),
        "scarcity_message": _generate_scarcity_message(results),
        "data_quality_banner": None,
    }


def _generate_notes(cand: Dict[str, Any], match: Dict[str, Any]) -> str:
    """Generate human-readable match notes with V3 insights."""
    pct = match.get("match_percentage", 0)
    
    notes = []
    
    # Match quality
    if pct >= 80:
        notes.append("Perfect match")
    elif pct >= 60:
        missing = match.get("missing_skills", [])
        if missing:
            notes.append(f"Good match (missing {missing[0]})")
        else:
            notes.append("Good match")
    elif pct >= 40:
        notes.append("Acceptable match")
    else:
        notes.append("Partial match")
    
    # Management experience
    mgmt_years = cand.get("management_experience_years", 0)
    if mgmt_years > 0:
        notes.append(f"Led teams ({mgmt_years}y)")
    
    # Role type
    if cand.get("role_type"):
        role = cand["role_type"].split("|")[0].strip()
        notes.append(role)
    
    return " • ".join(notes)


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