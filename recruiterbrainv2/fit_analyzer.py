"""
Deep candidate fit analysis (Phase 2).

Called on-demand when user clicks "Why?" button.
Provides comprehensive analysis beyond quick match percentage.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def erp_count(
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
    
    # STEP 1: VALIDATE DATA QUALITY
    is_valid, missing_fields, warnings = validate_candidate_data(candidate)
    
    if not is_valid:
        return {
            "fit_level": "unknown",
            "score": 0,
            "fit_badge": "⚠️ INCOMPLETE DATA",
            "explanation": f"Cannot analyze candidate due to missing data: {', '.join(missing_fields)}. Please update candidate record.",
            "strengths": [],
            "weaknesses": [f"Missing: {', '.join(missing_fields)}"],
            "recommendation": "UPDATE CANDIDATE RECORD - Cannot provide reliable analysis without complete data.",
            "critical_mismatch": None,
            "data_quality_issues": missing_fields,
            "data_warnings": warnings,
            "onboarding_estimate": None
        }
    
    # If we have warnings but can proceed
    if warnings:
        logger.warning(f"Data quality warnings for {candidate.get('candidate_id')}: {warnings}")
    
    # STEP 2: CHECK FOR REQUIREMENT CONFLICTS
    conflicts = detect_requirement_conflicts(requirements)
    conflict_note = ""
    
    if conflicts:
        logger.warning(f"Requirement conflicts detected: {conflicts}")
        conflict_note = "\n\n**Note:** Potential conflicts in job requirements:\n" + \
                       "\n".join(f"- {c}" for c in conflicts)
    
    # STEP 3: HANDLE CRITICAL MISMATCHES
    if quick_match.get("critical_mismatch"):
        analysis = generate_critical_mismatch_report(
            candidate,
            requirements,
            quick_match
        )
        
        # Add warnings if exist
        if warnings:
            analysis["data_warnings"] = warnings
        
        # Add conflicts if exist
        if conflict_note:
            analysis["explanation"] += conflict_note
        
        return analysis
    
    # STEP 4: GENERATE FULL FIT REPORT
    analysis = generate_fit_report(
        candidate,
        requirements,
        quick_match
    )
    
    # Add warnings if exist
    if warnings:
        analysis["data_warnings"] = warnings
    
    # Add conflicts if exist
    if conflict_note:
        analysis["explanation"] += conflict_note
    
    return analysis


def detect_requirement_conflicts(requirements: Dict[str, Any]) -> List[str]:
    """Detect contradictory requirements."""
    
    conflicts = []
    
    # Conflict 1: Seniority vs Experience mismatch
    seniority = requirements.get("seniority_level", "")
    min_exp = requirements.get("years_experience_min", 0)
    
    seniority_exp_map = {
        "Entry": (0, 3),
        "Mid": (3, 7),
        "Senior": (7, 15),
        "Lead/Manager": (8, 20),
        "Director+": (10, 30)
    }
    
    if seniority and seniority in seniority_exp_map:
        expected_range = seniority_exp_map[seniority]
        if min_exp > 0 and (min_exp < expected_range[0] or min_exp > expected_range[1]):
            conflicts.append(
                f"Seniority '{seniority}' typically requires {expected_range[0]}-{expected_range[1]} years, "
                f"but {min_exp} years specified"
            )
    
    # Conflict 2: Must-have skills include mutually exclusive tech
    must_have = [s.lower() for s in requirements.get("must_have_skills", [])]
    
    # Check for framework conflicts
    framework_conflicts = [
        (["react", "angular", "vue"], "frontend framework"),
        (["django", "flask", "fastapi"], "Python framework"),
    ]
    
    for frameworks, category in framework_conflicts:
        matching = [f for f in frameworks if f in must_have]
        if len(matching) >= 3:
            conflicts.append(
                f"Requires expertise in multiple competing {category}s: {', '.join(matching)}"
            )
    
    return conflicts

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
    
    # STEP 1: VALIDATE DATA QUALITY
    is_valid, missing_fields, warnings = validate_candidate_data(candidate)
    
    if not is_valid:
        return {
            "fit_level": "unknown",
            "score": 0,
            "fit_badge": "⚠️ INCOMPLETE DATA",
            "explanation": f"Cannot analyze candidate due to missing data: {', '.join(missing_fields)}. Please update candidate record.",
            "strengths": [],
            "weaknesses": [f"Missing: {', '.join(missing_fields)}"],
            "recommendation": "UPDATE CANDIDATE RECORD - Cannot provide reliable analysis without complete data.",
            "critical_mismatch": None,
            "data_quality_issues": missing_fields,
            "data_warnings": warnings,
            "onboarding_estimate": None
        }
    
    # If we have warnings but can proceed
    if warnings:
        logger.warning(f"Data quality warnings for {candidate.get('candidate_id')}: {warnings}")
    
    # STEP 2: CHECK FOR REQUIREMENT CONFLICTS
    conflicts = detect_requirement_conflicts(requirements)
    conflict_note = ""
    
    if conflicts:
        logger.warning(f"Requirement conflicts detected: {conflicts}")
        conflict_note = "\n\n**Note:** Potential conflicts in job requirements:\n" + \
                       "\n".join(f"- {c}" for c in conflicts)
    
    # STEP 3: HANDLE CRITICAL MISMATCHES
    if quick_match.get("critical_mismatch"):
        analysis = generate_critical_mismatch_report(
            candidate,
            requirements,
            quick_match
        )
        
        # Add warnings if exist
        if warnings:
            analysis["data_warnings"] = warnings
        
        # Add conflicts if exist
        if conflict_note:
            analysis["explanation"] += conflict_note
        
        return analysis
    
    # STEP 4: GENERATE FULL FIT REPORT
    analysis = generate_fit_report(
        candidate,
        requirements,
        quick_match
    )
    
    # Add warnings if exist
    if warnings:
        analysis["data_warnings"] = warnings
    
    # Add conflicts if exist
    if conflict_note:
        analysis["explanation"] += conflict_note
    
    return analysis

def generate_fit_report(
    candidate: Dict[str, Any],
    requirements: Dict[str, Any],
    quick_match: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate comprehensive fit report for non-critical cases."""
    
    # Analyze different aspects
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
def generate_fit_explanation(
    candidate: Dict[str, Any],
    requirements: Dict[str, Any],
    quick_match: Dict[str, Any],
    strengths: List[str],
    weaknesses: List[str]
) -> str:
    """Generate narrative explanation of fit with length limit."""
    
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
    
    explanation = opening + strength_summary + weakness_summary + closing
    
    # Limit length (frontend can handle ~5000 chars comfortably)
    MAX_LENGTH = 4000
    
    if len(explanation) > MAX_LENGTH:
        explanation = explanation[:MAX_LENGTH]
        # Find last complete sentence
        last_period = explanation.rfind('.')
        if last_period > MAX_LENGTH - 200:
            explanation = explanation[:last_period + 1]
        explanation += "\n\n[Analysis truncated for brevity]"
    
    return explanation

def assess_profile_recency(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Assess how current the candidate profile is."""
    
    years_stale = candidate.get("years_since_last_update", 0)
    last_updated = candidate.get("last_updated", "")
    
    if years_stale >= 5:
        return {
            "severity": "high",
            "warning": f"Profile not updated in {int(years_stale)} years",
            "recommendation": "Contact candidate to verify current skills and experience before proceeding"
        }
    elif years_stale >= 3:
        return {
            "severity": "medium",
            "warning": f"Profile {int(years_stale)} years old",
            "recommendation": "Verify current tech stack during interview"
        }
    elif years_stale >= 1:
        return {
            "severity": "low",
            "warning": f"Profile last updated {int(years_stale)} year(s) ago",
            "recommendation": "Check for recent experience updates"
        }
    else:
        return {
            "severity": "none",
            "warning": None,
            "recommendation": None
        }

def assess_overqualification(
    candidate: Dict[str, Any],
    requirements: Dict[str, Any],
    quick_match: Dict[str, Any]
) -> Dict[str, Any]:
    """Check if candidate might be overqualified or profile is suspicious."""
    
    match_pct = quick_match["match_percentage"]
    
    if match_pct == 100:
        # All skills match - could be legitimate or suspicious
        suspicion_score = 0
        notes = []
        
        # 1. Check if profile has generic descriptions
        summary = candidate.get("semantic_summary", "").lower()
        generic_phrases = [
            "highly motivated", "team player", "fast learner",
            "excellent communication", "detail oriented"
        ]
        generic_count = sum(1 for phrase in generic_phrases if phrase in summary)
        
        if generic_count >= 4:
            suspicion_score += 2
            notes.append("Profile contains many generic phrases")
        
        # 2. Check for unrealistic skill breadth
        skills_str = candidate.get("skills_extracted", "")
        total_skills = len([s.strip() for s in skills_str.split(",") if s.strip()])
        
        if total_skills > 30:
            suspicion_score += 2
            notes.append(f"Claims {total_skills} skills (unusually broad)")
        
        # 3. Check experience vs. skills ratio
        exp_years = candidate.get("total_experience_years", 0)
        if exp_years < 5 and total_skills > 15:
            suspicion_score += 3
            notes.append(f"Too many skills ({total_skills}) for experience level ({exp_years}y)")
        
        # 4. Check if all recent technologies
        current_stack = candidate.get("current_tech_stack", "").lower()
        recent_tech = ["react", "kubernetes", "docker", "aws", "microservices", "graphql"]
        recent_count = sum(1 for tech in recent_tech if tech in current_stack)
        
        if recent_count >= 5 and exp_years < 3:
            suspicion_score += 2
            notes.append("Claims expertise in many recent technologies with limited experience")
        
        # Assess
        if suspicion_score >= 5:
            return {
                "is_suspicious": True,
                "suspicion_level": "high",
                "notes": notes,
                "recommendation": "Profile may be exaggerated. Conduct thorough technical screening."
            }
        elif suspicion_score >= 3:
            return {
                "is_suspicious": True,
                "suspicion_level": "medium",
                "notes": notes,
                "recommendation": "Verify skills depth through technical assessment."
            }
    
    return {
        "is_suspicious": False,
        "suspicion_level": "none",
        "notes": [],
        "recommendation": None
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
    top_skills_str = candidate.get("top_5_skills_with_years", "")
    skill_years = parse_skills_with_years(top_skills_str)
    
    if skill_years:
        matched_lower = [s.lower() for s in quick_match.get("matched_skills", [])]
        for skill, years in skill_years.items():
            if any(skill.lower() in m or m in skill.lower() for m in matched_lower):
                if years >= 5:
                    strengths.append(f"Deep expertise: {skill} ({years} years)")
                elif years >= 3:
                    strengths.append(f"Solid experience: {skill} ({years} years)")
    else:
        # Fallback: use total_experience_years
        total_exp = candidate.get("total_experience_years", 0)
        if total_exp > 0:
            strengths.append(f"General experience: {int(total_exp)} years")
    
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


def generate_recommendation(
    candidate: Dict[str, Any],
    requirements: Dict[str, Any],
    quick_match: Dict[str, Any],
    strengths: List[str],
    weaknesses: List[str]
) -> str:
    """Generate hiring recommendation with overqualification check."""
    
    fit_level = quick_match["fit_level"]
    match_pct = quick_match["match_percentage"]
    candidate_name = candidate.get('name', 'This candidate')
    
    # EXCELLENT FIT
    if fit_level == "excellent":
        # Check for suspicious perfect matches
        overqual = assess_overqualification(candidate, requirements, quick_match)
        
        if overqual["is_suspicious"]:
            return f"""**RECOMMENDED WITH VERIFICATION**

{candidate_name} shows perfect technical alignment ({match_pct}%).

⚠️ **Profile Quality Alert:**
{chr(10).join(f'- {note}' for note in overqual['notes'])}

**Recommendation:** {overqual['recommendation']}

**Next Steps:**
1. Conduct thorough technical screening
2. Ask for specific project examples
3. Verify depth in claimed technologies
4. Check references carefully

**Risk Level:** MEDIUM - Profile may be overstated"""
        
        # Normal excellent fit
        return f"""**HIGHLY RECOMMENDED - Move to Interview Immediately**

{candidate_name} is an excellent match for this role with {match_pct}% technical alignment. All critical requirements are met, and the candidate brings valuable experience.

**Next Steps:**
1. Schedule initial phone screen within 24-48 hours
2. Prepare technical assessment focused on {', '.join(quick_match.get('matched_skills', [])[:3])}
3. Fast-track to final round if phone screen goes well

**Risk Level:** LOW - Strong technical and cultural fit indicators"""
    
    # GOOD FIT
    elif fit_level == "good":
        missing_count = len([w for w in weaknesses if "missing" in w.lower()])
        
        return f"""**RECOMMENDED - Proceed with Interview**

{candidate_name} is a strong match with {match_pct}% technical alignment. There are {missing_count} minor skill gap(s) that can be addressed.

**Considerations:**
- {weaknesses[0] if weaknesses else 'No major concerns'}
- Gaps are trainable within first 3-6 months
- Strong foundation in core requirements

**Next Steps:**
1. Schedule technical interview
2. Assess learning ability and adaptability
3. Discuss skill development plan during interview

**Risk Level:** LOW-MEDIUM - Minor gaps manageable with training"""
    
    # PARTIAL FIT
    elif fit_level == "partial":
        return f"""**PROCEED WITH CAUTION - Conditional Recommendation**

{candidate_name} shows {match_pct}% technical alignment with notable gaps. Consider only if:
- Candidate demonstrates strong learning ability
- You have training budget and timeline
- Other stronger candidates are unavailable

**Major Concerns:**
{chr(10).join(f'- {w}' for w in weaknesses[:3])}

**If You Proceed:**
1. Detailed technical assessment required
2. Probe for quick learning examples
3. Set clear 90-day performance milestones

**Risk Level:** MEDIUM-HIGH - Significant gaps require investment"""
    
    # POOR OR NOT FIT
    else:
        return f"""**NOT RECOMMENDED - Do Not Proceed**

{candidate_name} shows only {match_pct}% technical alignment. The skill gaps are too significant for this role.

**Why Not:**
{chr(10).join(f'- {w}' for w in weaknesses[:4])}

**Better Approach:**
Continue searching for candidates with stronger alignment to core requirements: {', '.join(requirements.get('must_have_skills', [])[:3])}

**Risk Level:** HIGH - Low probability of success"""



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
def validate_candidate_data(candidate: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that candidate has minimum required data for analysis.
    
    Returns:
        (is_valid, list_of_missing_critical_fields, list_of_warnings)
    """
    
    missing_fields = []
    warnings = []
    
    # Critical fields (analysis can't proceed without these)
    critical_fields = {
        "candidate_id": "Candidate ID",
        "name": "Candidate name",
    }
    
    for field, display_name in critical_fields.items():
        if not candidate.get(field):
            missing_fields.append(display_name)
    
    # Important fields (analysis can proceed but with warnings)
    important_fields = {
        "skills_extracted": "Skills",
        "total_experience_years": "Experience",
        "semantic_summary": "Profile summary",
    }
    
    for field, display_name in important_fields.items():
        value = candidate.get(field)
        if not value or (isinstance(value, str) and len(value.strip()) < 5):
            warnings.append(f"Missing or limited {display_name}")
    
    return (len(missing_fields) == 0, missing_fields, warnings)

def parse_skills_with_years(skills_str: str) -> Dict[str, int]:
    """
    Safely parse top_5_skills_with_years field.
    
    Handles formats:
    - "Python:7, AWS:5, Java:10"
    - "Python: 7, AWS: 5"
    - "Python - 7 years, AWS - 5 years"
    - Malformed data
    
    Returns:
        Dict of {skill: years}
    """
    if not skills_str or not isinstance(skills_str, str):
        return {}
    
    skill_years = {}
    
    # Try comma-separated format first
    items = skills_str.split(',')
    
    for item in items:
        item = item.strip()
        
        # Try colon separator
        if ':' in item:
            parts = item.split(':', 1)
        # Try dash separator
        elif '-' in item:
            parts = item.split('-', 1)
        else:
            continue
        
        if len(parts) != 2:
            continue
        
        skill = parts[0].strip()
        years_str = parts[1].strip()
        
        # Extract number from years string
        # Handles: "7", "7 years", "7y", etc.
        import re
        match = re.search(r'(\d+)', years_str)
        
        if match:
            try:
                years = int(match.group(1))
                if 0 <= years <= 50:  # Sanity check
                    skill_years[skill] = years
            except (ValueError, AttributeError):
                logger.warning(f"Could not parse years from: {years_str}")
                continue
    
    return skill_years

def erp_count(candidate, requirements, quick_match, strengths, weaknesses):
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
    explanation = opening + strength_summary + weakness_summary + closing
    
    # Limit length (frontend can handle ~5000 chars comfortably)
    MAX_LENGTH = 4000
    
    if len(explanation) > MAX_LENGTH:
        explanation = explanation[:MAX_LENGTH]
        # Find last complete sentence
        last_period = explanation.rfind('.')
        if last_period > MAX_LENGTH - 200:
            explanation = explanation[:last_period + 1]
        explanation += "\n\n[Analysis truncated for brevity]"
    
    return explanation