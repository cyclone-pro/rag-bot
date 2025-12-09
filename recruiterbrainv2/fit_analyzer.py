"""
Deep candidate fit analysis (Phase 2).

Called on-demand when user clicks "Why?" button.
Provides comprehensive analysis beyond quick match percentage.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def analyze_candidate_fit(
    candidate: Dict[str, Any],
    requirements: Dict[str, Any],
    quick_match: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive fit analysis for a candidate.
    
    Args:
        candidate: Full candidate record from Milvus
        requirements: Extracted requirements from query
        quick_match: Results from compute_match_details_enhanced()
    
    Returns:
        Detailed analysis report
    """
    
    # If critical mismatch, generate detailed explanation
    if quick_match.get("critical_mismatch"):
        return generate_critical_mismatch_report(
            candidate,
            requirements,
            quick_match
        )
    
    # Otherwise, generate full fit analysis
    return generate_fit_report(
        candidate,
        requirements,
        quick_match
    )


def generate_critical_mismatch_report(candidate, requirements, quick_match):
    """Generate detailed report for critical mismatches."""
    
    mismatch = quick_match["critical_mismatch"]
    mismatch_type = mismatch["type"]
    
    # === ORACLE MODULE MISMATCH ===
    if mismatch_type == "oracle_module_mismatch":
        required_mod = mismatch["required"]
        candidate_mod = mismatch["candidate_has"]
        
        # Module descriptions
        module_info = {
            "ERP": {
                "full_name": "Enterprise Resource Planning",
                "focus": "Financial & Supply Chain Management",
                "modules": ["Financials", "Procurement", "Supply Chain", "Manufacturing", "Project Management"],
                "processes": ["General Ledger", "Accounts Payable/Receivable", "Inventory", "Order Management"],
                "data": "Financial transactions, vendor/customer data, inventory levels"
            },
            "HCM": {
                "full_name": "Human Capital Management",
                "focus": "HR & Workforce Management",
                "modules": ["Core HR", "Payroll", "Talent Management", "Absence Management", "Recruiting"],
                "processes": ["Employee lifecycle", "Compensation", "Benefits", "Performance reviews"],
                "data": "Employee records, payroll data, skills/competencies"
            },
            "CX": {
                "full_name": "Customer Experience",
                "focus": "Sales & Customer Service",
                "modules": ["Sales Cloud", "Service Cloud", "Marketing", "CPQ"],
                "processes": ["Lead-to-cash", "Case management", "Campaign management"],
                "data": "Customer records, opportunities, service requests"
            }
        }
        
        required_info = module_info.get(required_mod, {})
        candidate_info = module_info.get(candidate_mod, {})
        
        explanation = f"""
**Oracle Module Incompatibility Detected**

This is a fundamental mismatch that cannot be overcome with training or upskilling.

**Job Requires: Oracle Fusion {required_mod}**
- {required_info.get('full_name', required_mod)}
- Focus: {required_info.get('focus', 'N/A')}
- Key Modules: {', '.join(required_info.get('modules', []))}
- Business Processes: {', '.join(required_info.get('processes', []))}

**Candidate Has: Oracle Fusion {candidate_mod}**
- {candidate_info.get('full_name', candidate_mod)}
- Focus: {candidate_info.get('focus', 'N/A')}
- Experience: {candidate.get('total_experience_years', 0)} years in {candidate_mod}

**Why This is a Deal Breaker:**

Oracle Fusion {required_mod} and {candidate_mod} are separate product lines with:

1. **Different Business Domains**
   - {required_mod}: {required_info.get('focus', 'N/A')}
   - {candidate_mod}: {candidate_info.get('focus', 'N/A')}

2. **Different Technical Architecture**
   - Different data models and database schemas
   - Different security frameworks
   - Different integration patterns

3. **Different Functional Knowledge**
   - {required_mod} requires understanding of {required_info.get('focus', 'business processes').lower()}
   - Candidate's {candidate.get('total_experience_years', 0)} years are in {candidate_info.get('focus', 'different domain').lower()}

**What Would Be Required:**
If you hired this candidate anyway, they would need to:
- Learn Oracle {required_mod} modules from scratch (6-12 months)
- Understand new business domain ({required_info.get('focus', 'new processes')})
- Earn new certifications ({required_mod} certifications don't transfer)
- Essentially start over as a junior {required_mod} consultant

**Estimated Impact:**
- Time to productivity: 6-12 months
- Training cost: $15,000-$25,000
- Risk level: HIGH (unfamiliar domain, steep learning curve)
- Likelihood of success: LOW (fundamentally different skill set)
"""
        
        # Still identify strengths (for context)
        strengths = analyze_strengths(candidate, requirements, quick_match)
        weaknesses = [f"Wrong Oracle module ({candidate_mod} instead of {required_mod}) - CRITICAL DEAL BREAKER"]
        weaknesses.extend([f"No {required_mod} experience", f"Domain expertise in {candidate_mod}, not {required_mod}"])
        
        recommendation = f"""
**STRONG RECOMMENDATION: DO NOT PROCEED**

This candidate has {candidate.get('total_experience_years', 0)} years of valuable Oracle {candidate_mod} experience, but it does not apply to Oracle {required_mod} roles.

**Better Approach:**
Search specifically for:
- "Oracle Fusion {required_mod}" or "Oracle {required_mod}"
- "{required_info.get('focus', required_mod + ' specialist')}"
- Candidates with {required_mod}-specific certifications

**If Client is Flexible:**
If the client also needs {candidate_mod} work, this candidate could be excellent for {candidate_mod} implementation projects.
"""
        
        return {
            "fit_level": "not_fit",
            "score": quick_match["match_percentage"],
            "fit_badge": quick_match["fit_badge"],
            "explanation": explanation.strip(),
            "strengths": strengths[:5],  # Show a few strengths for context
            "weaknesses": weaknesses,
            "recommendation": recommendation.strip(),
            "critical_mismatch": mismatch,
            "onboarding_estimate": {
                "time_to_productivity": "6-12 months",
                "training_cost_range": "$15,000-$25,000",
                "risk_level": "HIGH",
                "success_likelihood": "LOW"
            }
        }
    
    # === EXPERIENCE GAP ===
    elif mismatch_type == "experience_gap":
        # Similar detailed explanation for experience gaps
        # ... (implement similar pattern)
        pass
    
    # === LOCATION MISMATCH ===
    elif mismatch_type == "location_mismatch":
        # Similar detailed explanation for location issues
        # ... (implement similar pattern)
        pass


def generate_fit_report(candidate, requirements, quick_match):
    """Generate comprehensive fit report for non-critical cases."""
    
    strengths = analyze_strengths(candidate, requirements, quick_match)
    weaknesses = analyze_weaknesses(candidate, requirements, quick_match)
    recommendation = generate_recommendation(
        candidate, requirements, quick_match, strengths, weaknesses
    )
    onboarding = estimate_onboarding(candidate, requirements, quick_match)
    
    # Generate narrative explanation
    explanation = generate_fit_explanation(
        candidate, requirements, quick_match, strengths, weaknesses
    )
    
    return {
        "fit_level": quick_match["fit_level"],
        "score": quick_match["match_percentage"],
        "fit_badge": quick_match["fit_badge"],
        "explanation": explanation,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendation": recommendation,
        "onboarding_estimate": onboarding,
        "critical_mismatch": None
    }


def analyze_strengths(candidate, requirements, quick_match):
    """
    Identify candidate strengths relevant to the role.
    
    Uses ALL available candidate data.
    """
    strengths = []
    
    # 1. Experience alignment
    candidate_exp = candidate.get("total_experience_years", 0)
    required_exp = requirements.get("years_experience_min", 0)
    
    if candidate_exp >= required_exp * 1.5:
        strengths.append(f"Extensive experience ({int(candidate_exp)} years, exceeds {required_exp} year requirement)")
    elif candidate_exp >= required_exp:
        strengths.append(f"Meets experience requirement ({int(candidate_exp)} years)")
    
    # 2. Skill depth (from top_5_skills_with_years)
    top_skills = candidate.get("top_5_skills_with_years", "")
    if top_skills:
        skill_years = {}
        for item in top_skills.split(","):
            if ":" in item:
                skill, years_str = item.strip().split(":")
                try:
                    skill_years[skill] = int(years_str)
                except:
                    pass
        
        # Find deep expertise in matched skills
        matched_lower = [s.lower() for s in quick_match.get("matched_skills", [])]
        for skill, years in skill_years.items():
            if any(skill.lower() in m or m in skill.lower() for m in matched_lower):
                if years >= 5:
                    strengths.append(f"Deep expertise: {skill} ({years} years)")
                elif years >= 3:
                    strengths.append(f"Solid experience: {skill} ({years} years)")
    
    # 3. Industry alignment
    required_industry = requirements.get("industry", "").lower()
    candidate_industries = candidate.get("industries_worked", "").lower()
    
    if required_industry and required_industry in candidate_industries:
        strengths.append(f"Industry experience: {required_industry.title()}")
    
    # 4. Domain expertise
    candidate_domains = candidate.get("domain_expertise", "").lower()
    if candidate_domains:
        # Check if any required domain matches
        for domain in candidate_domains.split(",")[:3]:
            domain = domain.strip()
            if domain:
                strengths.append(f"Domain expertise: {domain.title()}")
    
    # 5. Management experience (if relevant)
    mgmt_years = candidate.get("management_experience_years", 0)
    if mgmt_years > 0:
        role_type = requirements.get("role_type", "").lower()
        if any(term in role_type for term in ["lead", "manager", "director"]):
            strengths.append(f"Leadership experience ({int(mgmt_years)} years managing teams)")
    
    # 6. Current tech stack (shows recency)
    current_stack = candidate.get("current_tech_stack", "").lower()
    recent_tools = []
    for skill in requirements.get("must_have_skills", [])[:5]:
        if skill.lower() in current_stack:
            recent_tools.append(skill)
    
    if recent_tools:
        strengths.append(f"Currently using: {', '.join(recent_tools[:3])}")
    
    # 7. Certifications
    certs = candidate.get("certifications", "")
    if certs:
        relevant_certs = []
        for skill in requirements.get("must_have_skills", []):
            if skill.lower() in certs.lower():
                relevant_certs.append(skill)
        
        if relevant_certs:
            strengths.append(f"Certified: {', '.join(relevant_certs)}")
    
    # 8. Career stage alignment
    candidate_stage = candidate.get("career_stage", "")
    required_stage = requirements.get("seniority_level", "")
    
    stage_levels = {"Entry": 1, "Mid": 2, "Senior": 3, "Lead/Manager": 4, "Director+": 5}
    candidate_level = stage_levels.get(candidate_stage, 0)
    required_level = stage_levels.get(required_stage, 0)
    
    if candidate_level == required_level:
        strengths.append(f"Career level matches: {candidate_stage}")
    elif candidate_level > required_level and candidate_level - required_level == 1:
        strengths.append(f"Senior to requirements: {candidate_stage} for {required_stage} role")
    
    # 9. Evidence of relevant work
    evidence_projects = candidate.get("evidence_projects", "")
    if evidence_projects:
        projects_lower = evidence_projects.lower()
        relevant_mentions = []
        for skill in requirements.get("must_have_skills", [])[:3]:
            if skill.lower() in projects_lower:
                relevant_mentions.append(skill)
        
        if relevant_mentions:
            strengths.append(f"Proven project experience: {', '.join(relevant_mentions)}")
    
    # Return top strengths
    return strengths[:8] if strengths else ["General technical background"]


def analyze_weaknesses(candidate, requirements, quick_match):
    """Identify weaknesses with context."""
    weaknesses = []
    
    # 1. Missing critical skills
    critical_missing = [
        s for s in quick_match.get("missing_skills", [])
        if s in requirements.get("must_have_skills", [])
    ]
    
    for skill in critical_missing[:5]:
        # Check if trainable
        trainable = is_skill_trainable(skill, candidate)
        if trainable:
            weaknesses.append(f"Missing {skill} (trainable in {trainable})")
        else:
            weaknesses.append(f"No {skill} experience")
    
    # 2. Experience gap
    candidate_exp = candidate.get("total_experience_years", 0)
    required_exp = requirements.get("years_experience_min", 0)
    
    if required_exp > 0 and candidate_exp < required_exp:
        gap = required_exp - candidate_exp
        weaknesses.append(f"Below experience requirement by {int(gap)} years")
    
    # 3. Management gap (if needed)
    if any(term in requirements.get("role_type", "").lower() for term in ["lead", "manager"]):
        mgmt_years = candidate.get("management_experience_years", 0)
        if mgmt_years == 0:
            weaknesses.append("No team management experience")
    
    # 4. Industry gap
    required_industry = requirements.get("industry", "")
    if required_industry:
        candidate_industries = candidate.get("industries_worked", "").lower()
        if required_industry.lower() not in candidate_industries:
            weaknesses.append(f"No {required_industry} industry experience")
    
    # 5. Recency concern
    years_stale = candidate.get("years_since_last_update", 0)
    if years_stale > 2:
        weaknesses.append(f"Resume not recently updated ({int(years_stale)} years old)")
    
    # 6. Remote/location preference mismatch
    if requirements.get("location_required"):
        remote_pref = candidate.get("remote_preference", "").lower()
        if "remote" not in remote_pref and "hybrid" not in remote_pref:
            weaknesses.append("Prefers onsite work (job may require flexibility)")
    
    return weaknesses if weaknesses else []


def is_skill_trainable(skill, candidate):
    """
    Estimate if skill is trainable and how long it takes.
    
    Returns timeframe string or None if not easily trainable.
    """
    skill_lower = skill.lower()
    
    # Check if candidate has related foundation
    candidate_skills = candidate.get("skills_extracted", "").lower()
    
    # Programming languages (if candidate knows similar language)
    lang_families = {
        "python": ["java", "c#", "ruby", "javascript"],
        "java": ["c#", "python", "kotlin", "scala"],
        "javascript": ["typescript", "python", "java"],
    }
    
    for target_lang, related_langs in lang_families.items():
        if target_lang in skill_lower:
            if any(lang in candidate_skills for lang in related_langs):
                return "2-3 months"
    
    # Frameworks (if candidate knows base language)
    if "django" in skill_lower and "python" in candidate_skills:
        return "1-2 months"
    if "react" in skill_lower and "javascript" in candidate_skills:
        return "1-2 months"
    if "spring" in skill_lower and "java" in candidate_skills:
        return "2-3 months"
    
    # Cloud platforms (if candidate knows another cloud)
    clouds = ["aws", "azure", "gcp"]
    if any(cloud in skill_lower for cloud in clouds):
        if any(cloud in candidate_skills for cloud in clouds):
            return "2-4 months (cloud transition)"
    
    # Certifications
    if "certification" in skill_lower or "certified" in skill_lower:
        return "2-3 months (certification course)"
    
    # Domain knowledge (harder to train)
    domains = ["healthcare", "finance", "banking", "insurance"]
    if any(domain in skill_lower for domain in domains):
        return "6+ months (domain expertise)"
    
    # Default: can probably be trained
    return "3-6 months"


def generate_recommendation(candidate, requirements, quick_match, strengths, weaknesses):
    """Generate hiring recommendation."""
    
    fit_level = quick_match["fit_level"]
    match_pct = quick_match["match_percentage"]
    
    if fit_level == "excellent":
        return f"""
**HIGHLY RECOMMENDED - Move to Interview Immediately**

{candidate.get('name', 'This candidate')} is an excellent match for this role with {match_pct}% technical alignment. All critical requirements are met, and the candidate brings valuable experience.

**Next Steps:**
1. Schedule initial phone screen within 24-48 hours
2. Prepare technical assessment focused on {', '.join(quick_match.get('matched_skills', [])[:3])}
3. Fast-track to final round if phone screen goes well

**Risk Level:** LOW - Strong technical and cultural fit indicators
"""
    
    elif fit_level == "good":
        missing_count = len([w for w in weaknesses if "missing" in w.lower()])
        
        return f"""
**RECOMMENDED - Proceed with Interview**

{candidate.get('name', 'This candidate')} is a strong match with {match_pct}% technical alignment. There are {missing_count} minor skill gap(s) that can be addressed.

**Considerations:**
- {weaknesses[0] if weaknesses else 'No major concerns'}
- Gaps are trainable within first 3-6 months
- Strong foundation in core requirements

**Next Steps:**
1. Schedule technical interview
2. Assess learning ability and adaptability
3. Discuss skill development plan during interview

**Risk Level:** LOW-MEDIUM - Minor gaps manageable with training
"""
    
    elif fit_level == "partial":
        return f"""
**PROCEED WITH CAUTION - Conditional Recommendation**

{candidate.get('name', 'This candidate')} shows {match_pct}% technical alignment with notable gaps. Consider only if:
- Candidate demonstrates strong learning ability
- You have training budget and timeline
- Other stronger candidates are unavailable

**Major Concerns:**
{chr(10).join(f'- {w}' for w in weaknesses[:3])}

**If You Proceed:**
1. Detailed technical assessment required
2. Probe for quick learning examples
3. Set clear 90-day performance milestones

**Risk Level:** MEDIUM-HIGH - Significant gaps require investment
"""
    
    else:  # poor or not_fit
        return f"""
**NOT RECOMMENDED - Do Not Proceed**

{candidate.get('name', 'This candidate')} shows only {match_pct}% technical alignment. The skill gaps are too significant for this role.

**Why Not:**
{chr(10).join(f'- {w}' for w in weaknesses[:4])}

**Better Approach:**
Continue searching for candidates with stronger alignment to core requirements: {', '.join(requirements.get('must_have_skills', [])[:3])}

**Risk Level:** HIGH - Low probability of success
"""


def estimate_onboarding(candidate, requirements, quick_match):
    """Estimate time and cost to onboard candidate."""
    
    fit_level = quick_match["fit_level"]
    missing_critical = len([
        s for s in quick_match.get("missing_skills", [])
        if s in requirements.get("must_have_skills", [])
    ])
    
    if fit_level == "excellent":
        return {
            "time_to_productivity": "2-4 weeks",
            "training_cost_range": "$500-$1,000",
            "risk_level": "LOW",
            "notes": "Standard onboarding only"
        }
    
    elif fit_level == "good":
        return {
            "time_to_productivity": "1-2 months",
            "training_cost_range": "$1,000-$3,000",
            "risk_level": "LOW",
            "notes": f"Training needed for {missing_critical} skill(s)"
        }
    
    elif fit_level == "partial":
        return {
            "time_to_productivity": "3-6 months",
            "training_cost_range": "$3,000-$8,000",
            "risk_level": "MEDIUM",
            "notes": f"Significant training required for {missing_critical} critical skills"
        }
    
    else:
        return {
            "time_to_productivity": "6+ months",
            "training_cost_range": "$8,000-$15,000+",
            "risk_level": "HIGH",
            "notes": "Extensive training required, high risk of underperformance"
        }


def generate_fit_explanation(candidate, requirements, quick_match, strengths, weaknesses):
    """Generate narrative explanation of fit."""
    
    fit_level = quick_match["fit_level"]
    match_pct = quick_match["match_percentage"]
    name = candidate.get("name", "This candidate")
    
    # Opening
    if fit_level == "excellent":
        opening = f"{name} is an **excellent match** for this role with {match_pct}% technical alignment."
    elif fit_level == "good":
        opening = f"{name} is a **strong match** for this role with {match_pct}% technical alignment."
    elif fit_level == "partial":
        opening = f"{name} shows **partial alignment** with this role ({match_pct}% match) with notable skill gaps."
    else:
        opening = f"{name} has **limited alignment** with this role ({match_pct}% match) and significant skill gaps."
    
    # Strengths summary
    strength_summary = ""
    if strengths:
        top_strengths = strengths[:3]
        strength_summary = "\n\n**Key Strengths:**\n" + "\n".join(f"- {s}" for s in top_strengths)
    
    # Weaknesses summary
    weakness_summary = ""
    if weaknesses:
        top_weaknesses = weaknesses[:3]
        weakness_summary = "\n\n**Areas of Concern:**\n" + "\n".join(f"- {w}" for w in top_weaknesses)
    
    # Closing assessment
    if fit_level in ["excellent", "good"]:
        closing = f"\n\n**Overall Assessment:** {name} has the core skills and experience needed for success in this role. {' Minor gaps can be addressed through training.' if weaknesses else ''}"
    else:
        closing = f"\n\n**Overall Assessment:** The skill gaps are significant and would require substantial investment in training and development."
    
    return opening + strength_summary + weakness_summary + closing