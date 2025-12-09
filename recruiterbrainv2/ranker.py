"""Hybrid ranking: vector + keyword matching."""
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


def hybrid_rank(
    candidates: List[Dict[str, Any]],
    required_skills: List[str],
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Rank candidates using hybrid scoring.
    
    Args:
        candidates: List with 'vector_score' already set
        required_skills: Required skill strings
        vector_weight: Weight for semantic similarity
        keyword_weight: Weight for keyword matching
    
    Returns:
        List of (candidate, hybrid_score) sorted descending
    """
    if not candidates:
        return []
    
    required_normalized = [s.lower().strip() for s in required_skills]
    
    ranked = []
    for candidate in candidates:
        vector_score = candidate.get('vector_score', 0.0)
        keyword_score = _keyword_score(candidate, required_normalized)
        
        hybrid_score = (
            vector_weight * vector_score + 
            keyword_weight * keyword_score
        )
        
        candidate['keyword_score'] = keyword_score
        candidate['hybrid_score'] = hybrid_score
        
        ranked.append((candidate, hybrid_score))
    
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(
        "Ranked %d candidates (avg_hybrid=%.3f)",
        len(ranked),
        sum(s for _, s in ranked) / len(ranked) if ranked else 0
    )
    
    return ranked


def _keyword_score(candidate: Dict[str, Any], required_skills: List[str]) -> float:
    """Compute keyword matching score (0.0 to 1.0)."""
    if not required_skills:
        return 1.0
    
    # Gather all text
    text = " ".join([
        str(candidate.get("skills_extracted", "")),
        str(candidate.get("tools_and_technologies", "")),
        str(candidate.get("domains_of_expertise", "")),
        str(candidate.get("semantic_summary", "")),
        str(candidate.get("keywords_summary", "")),
    ]).lower()
    
    # Count matches
    matched = sum(1 for skill in required_skills if skill in text)
    
    return matched / len(required_skills)


def compute_match_details(
    candidate: Dict[str, Any],
    required_skills: List[str],
) -> Dict[str, Any]:
    """Generate detailed match explanation."""
    text = " ".join([
        str(candidate.get("skills_extracted", "")),
        str(candidate.get("tools_and_technologies", "")),
    ]).lower()
    
    matched = []
    missing = []
    
    for skill in required_skills:
        if skill.lower() in text:
            matched.append(skill)
        else:
            missing.append(skill)
    
    match_pct = (len(matched) / len(required_skills) * 100) if required_skills else 100
    
    return {
        "match_percentage": round(match_pct),
        "matched_skills": matched,
        "missing_skills": missing,
        "total_required": len(required_skills),
        "vector_score": candidate.get('vector_score', 0),
        "keyword_score": candidate.get('keyword_score', 0),
        "hybrid_score": candidate.get('hybrid_score', 0),
    }
# In ranker.py

def compute_match_details_enhanced(
    candidate: Dict[str, Any],
    required_skills: List[str],
    requirements: Dict[str, Any]  # NEW: Full requirements dict
) -> Dict[str, Any]:
    """
    Enhanced match details with fit analysis (Phase 1).
    
    Args:
        candidate: Candidate data from Milvus
        required_skills: List of required skills
        requirements: Full requirements from extract_requirements()
            {
                "must_have_skills": [...],
                "nice_to_have_skills": [...],
                "seniority_level": "Senior",
                "industry": "Healthcare",
                "years_experience_min": 5,
                "role_type": "Backend Engineering",
                "query_text": "original query"  # IMPORTANT for module detection
            }
    
    Returns:
        Enhanced match details with fit analysis
    """
    
    
    text = " ".join([...])  
    matched = []
    missing = []
    text = " ".join([
        str(candidate.get("skills_extracted", "")),
        str(candidate.get("tools_and_technologies", "")),
    ]).lower()
    for skill in required_skills:
        if skill.lower() in text:
            matched.append(skill)
        else:
            missing.append(skill)
    
    match_pct = (len(matched) / len(required_skills) * 100) if required_skills else 100
    
    # === STEP 2: Check for critical mismatches (NEW) ===
    critical_mismatch = detect_critical_mismatch(candidate, requirements)
    
    if critical_mismatch:
        # Immediate rejection
        return {
            "match_percentage": round(match_pct),
            "matched_skills": matched,
            "missing_skills": missing,
            "total_required": len(required_skills),
            
            # Phase 1 additions
            "fit_level": "not_fit",
            "fit_badge": "❌ NOT A FIT",
            "quick_reason": critical_mismatch["short_reason"],
            "critical_mismatch": critical_mismatch,
            
            # Existing scores
            "vector_score": candidate.get('vector_score', 0),
            "keyword_score": candidate.get('keyword_score', 0),
            "hybrid_score": candidate.get('hybrid_score', 0),
        }
    
    # === STEP 3: Determine fit level (NEW) ===
    fit_level, fit_badge, quick_reason = determine_fit_level(
        match_pct=match_pct,
        matched_skills=matched,
        missing_skills=missing,
        candidate=candidate,
        requirements=requirements
    )
    
    return {
        "match_percentage": round(match_pct),
        "matched_skills": matched,
        "missing_skills": missing,
        "total_required": len(required_skills),
        
        # Phase 1 additions
        "fit_level": fit_level,
        "fit_badge": fit_badge,
        "quick_reason": quick_reason,
        "critical_mismatch": None,
        
        # Existing scores
        "vector_score": candidate.get('vector_score', 0),
        "keyword_score": candidate.get('keyword_score', 0),
        "hybrid_score": candidate.get('hybrid_score', 0),
    }


def detect_critical_mismatch(
    candidate: Dict[str, Any],
    requirements: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Detect deal breakers that immediately disqualify candidate.
    
    Returns None if no critical issues, otherwise returns mismatch details.
    """
    
    # === ORACLE MODULE CHECK ===
    oracle_mismatch = _check_oracle_module_mismatch(candidate, requirements)
    if oracle_mismatch:
        return oracle_mismatch
    
    # === SEVERE EXPERIENCE GAP ===
    exp_mismatch = _check_experience_gap(candidate, requirements)
    if exp_mismatch:
        return exp_mismatch
    
    # === LOCATION INCOMPATIBILITY ===
    location_mismatch = _check_location_mismatch(candidate, requirements)
    if location_mismatch:
        return location_mismatch
    
    return None


def _check_oracle_module_mismatch(candidate, requirements):
    """Check for Oracle HCM vs ERP vs CX mismatch."""
    if not is_module_specific_role(requirements):
        logger.info("Role is Oracle module-agnostic, skipping module mismatch check")
        return None
    query_text = requirements.get("query_text", "").lower()
    
    # Detect required module
    required_module = None
    if any(term in query_text for term in [
        "oracle fusion erp", "oracle erp", "financials", 
        "general ledger", "accounts payable", "supply chain", 
        "scm", "procurement", "inventory management"
    ]):
        required_module = "ERP"
    elif any(term in query_text for term in [
        "oracle fusion hcm", "oracle hcm", "human capital",
        "payroll", "talent management", "core hr", "absence management"
    ]):
        required_module = "HCM"
    elif any(term in query_text for term in [
        "oracle cx", "sales cloud", "service cloud", 
        "marketing cloud", "cpq"
    ]):
        required_module = "CX"
    
    if not required_module:
        return None  
    
    # Detect candidate's module
    candidate_text = " ".join([
        candidate.get("skills_extracted", ""),
        candidate.get("domain_expertise", ""),
        candidate.get("tools_and_technologies", ""),
        candidate.get("tech_stack_primary", ""),
        candidate.get("current_tech_stack", ""),
        candidate.get("semantic_summary", "")[:500]  # First 500 chars
    ]).lower()
    
    candidate_module = None
    hcm_indicators = ["hcm", "human capital", "payroll", "talent", "hr management"]
    erp_indicators = ["erp", "financials", "general ledger", "gl", "ap", "ar", "supply chain"]
    cx_indicators = ["cx", "sales cloud", "service cloud", "cpq"]
    
    hcm_count = sum(1 for term in hcm_indicators if term in candidate_text)
    erp_count = sum(1 for term in erp_indicators if term in candidate_text)
    cx_count = sum(1 for term in cx_indicators if term in candidate_text)
    logger.info(f"Oracle module scores - HCM: {hcm_count}, ERP: {erp_count}")
    if hcm_count == 0 and erp_count == 0:
        return None
    if hcm_count > 0 and erp_count > 0:
        # Check which is dominant
        ratio = max(hcm_count, erp_count) / min(hcm_count, erp_count)
        
        if ratio < 2:  # Fairly balanced experience
            # Check if required module is present
            if required_module == "ERP" and erp_count >= hcm_count:
                return None  # Has ERP experience, no mismatch
            elif required_module == "HCM" and hcm_count >= erp_count:
                return None  # Has HCM experience, no mismatch
            elif required_module == "ERP" and hcm_count > erp_count:
                # HCM dominant but has some ERP
                return {
                    "type": "oracle_module_mismatch",
                    "required": "ERP",
                    "candidate_has": "Mixed (HCM-dominant)",
                    "short_reason": f"Primarily HCM experience, limited ERP background",
                    "severity": "medium",  # Not critical, has some exposure
                    "note": f"Candidate has mixed Oracle experience (HCM: {hcm_count} mentions, ERP: {erp_count} mentions)"
                }
        
        # One module is clearly dominant
        if hcm_count > erp_count * 2:
            candidate_module = "HCM"
        else:
            candidate_module = "ERP"
    
    # SINGLE MODULE EXPERIENCE
    elif hcm_count > erp_count:
        candidate_module = "HCM"
    else:
        candidate_module = "ERP"
    
    # Check for critical mismatch
    if required_module != candidate_module:
        return {
            "type": "oracle_module_mismatch",
            "required": required_module,
            "candidate_has": candidate_module,
            "short_reason": f"Oracle {required_module} required, candidate specializes in {candidate_module}",
            "severity": "critical"
        }
    
    return None  # No 
def is_module_specific_role(requirements: Dict[str, Any]) -> bool:
    """
    Determine if role requires specific Oracle module expertise
    or is module-agnostic.
    """
    query = requirements.get("query_text", "").lower()
    
    # Module-agnostic indicators
    agnostic_indicators = [
        "oracle cloud integration",
        "oracle middleware",
        "oracle database",
        "oracle infrastructure",
        "oracle administration",
        "oracle dba",
        "oracle cloud platform",
        "oci integration"  # Oracle Cloud Infrastructure (platform level)
    ]
    
    if any(indicator in query for indicator in agnostic_indicators):
        return False  # Not module-specific
    
    # Module-specific indicators
    module_specific = [
        "fusion hcm", "fusion erp", "fusion cx",
        "financials", "scm", "supply chain",
        "payroll", "talent management", "core hr",
        "sales cloud", "service cloud"
    ]
    
    if any(indicator in query for indicator in module_specific):
        return True  # Module-specific
    
    # Default: assume module-agnostic if unclear
    return False

def _check_experience_gap(candidate, requirements):
    """Check for severe experience gap (< 50% of required)."""
    
    min_exp = requirements.get("years_experience_min", 0)
    if min_exp == 0:
        return None
    
    candidate_exp = candidate.get("total_experience_years", 0)
    
    # If candidate has less than half required experience
    if candidate_exp < (min_exp * 0.5):
        return {
            "type": "experience_gap",
            "required": f"{min_exp}+ years",
            "candidate_has": f"{candidate_exp} years",
            "short_reason": f"Requires {min_exp}+ years, has {candidate_exp} years",
            "severity": "high"
        }
    
    return None


def _check_location_mismatch(candidate, requirements):
    """Check for location incompatibility (if job has strict location requirement)."""
    
    # Only check if job explicitly requires specific location with no remote option
    if not requirements.get("location_required") or requirements.get("remote_ok"):
        return None
    
    required_location = requirements["location_required"].lower()
    candidate_city = candidate.get("location_city", "").lower()
    candidate_state = candidate.get("location_state", "").lower()
    remote_pref = candidate.get("remote_preference", "").lower()
    relocation = candidate.get("relocation_willingness", "").lower()
    
    # Check if candidate is in required location OR open to remote/relocation
    is_in_location = (required_location in candidate_city or required_location in candidate_state)
    is_flexible = ("remote" in remote_pref or "yes" in relocation or "open" in relocation)
    
    if not is_in_location and not is_flexible:
        return {
            "type": "location_mismatch",
            "required": requirements["location_required"],
            "candidate_location": f"{candidate.get('location_city')}, {candidate.get('location_state')}",
            "short_reason": f"Requires {requirements['location_required']}, candidate in {candidate.get('location_city')} (no remote/relocation)",
            "severity": "medium"
        }
    
    return None


def determine_fit_level(
    match_pct: float,
    matched_skills: List[str],
    missing_skills: List[str],
    candidate: Dict[str, Any],
    requirements: Dict[str, Any]
) -> Tuple[str, str, str]:
    """
    Determine fit level and generate quick reason.
    
    Returns:
        (fit_level, fit_badge, quick_reason)
    """
    
    # Count critical missing (from must_have list)
    must_have = set(s.lower() for s in requirements.get("must_have_skills", []))
    missing_lower = set(s.lower() for s in missing_skills)
    critical_missing = [s for s in missing_skills if s.lower() in must_have]
    
    # === FIT LEVELS ===
    
    if match_pct >= 85 and len(critical_missing) == 0:
        return (
            "excellent",
            "✅ EXCELLENT FIT",
            "Perfect technical match - all critical skills present"
        )
    
    elif match_pct >= 70 and len(critical_missing) <= 1:
        if critical_missing:
            return (
                "good",
                "✓ GOOD FIT",
                f"Strong match, minor gap: {critical_missing[0]}"
            )
        else:
            return (
                "good",
                "✓ GOOD FIT",
                "Strong technical match with good skill coverage"
            )
    
    elif match_pct >= 50 and len(critical_missing) <= 3:
        top_missing = ", ".join(critical_missing[:2])
        return (
            "partial",
            "⚠️ PARTIAL FIT",
            f"Acceptable match, missing {len(critical_missing)} critical: {top_missing}"
        )
    
    elif match_pct >= 30:
        return (
            "poor",
            "⚠️ POOR FIT",
            f"Limited alignment - missing {len(critical_missing)} critical skills"
        )
    
    else:
        return (
            "not_fit",
            "❌ NOT A FIT",
            f"Insufficient match ({int(match_pct)}%) - significant skill gaps"
        )