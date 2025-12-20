"""Insert candidate data into Milvus with duplicate detection."""
import logging
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Tuple, Optional
from ..config import get_milvus_client, COLLECTION

logger = logging.getLogger(__name__)


def generate_candidate_id() -> str:
    """Generate unique 6-character candidate ID."""
    return uuid.uuid4().hex[:6].upper()


def normalize_phone(phone: str) -> str:
    """
    Normalize phone number for comparison.
    
    Removes all non-digit characters.
    Example: "(555) 123-4567" -> "5551234567"
    """
    if not phone:
        return ""
    return ''.join(c for c in phone if c.isdigit())


def normalize_email(email: str) -> str:
    """
    Normalize email for comparison.
    
    Lowercase and strip whitespace.
    """
    if not email:
        return ""
    return email.lower().strip()


def normalize_linkedin(linkedin_url: str) -> str:
    """
    Normalize LinkedIn URL for comparison.
    
    Removes protocol and trailing slashes.
    Example: "https://linkedin.com/in/johndoe/" -> "linkedin.com/in/johndoe"
    """
    if not linkedin_url:
        return ""
    
    url = linkedin_url.lower().strip()
    # Remove protocol
    url = url.replace("https://", "").replace("http://", "")
    # Remove www
    url = url.replace("www.", "")
    # Remove trailing slash
    url = url.rstrip("/")
    
    return url


def check_for_duplicate(
    client,
    email: str = "",
    phone: str = "",
    linkedin_url: str = "",
    github_url: str = ""
) -> Optional[Dict]:
    """
    Check if candidate already exists in Milvus.
    
    Checks in priority order:
    1. LinkedIn URL (most reliable)
    2. Email (very reliable)
    3. Phone (reliable but can change)
    4. GitHub URL (less reliable but useful)
    
    Args:
        client: Milvus client
        email: Candidate email
        phone: Candidate phone
        linkedin_url: LinkedIn profile URL
        github_url: GitHub profile URL
    
    Returns:
        Dict with existing candidate info if found, None otherwise
        {
            "candidate_id": "ABC123",
            "name": "John Doe",
            "email": "john@example.com",
            "last_updated": "2025-12-20T10:30:00Z",
            "match_reason": "linkedin_url"
        }
    """
    # Normalize inputs
    email_norm = normalize_email(email)
    phone_norm = normalize_phone(phone)
    linkedin_norm = normalize_linkedin(linkedin_url)
    github_norm = normalize_linkedin(github_url)  # Same normalization logic
    
    # Track which field matched
    match_checks = []
    
    # Priority 1: LinkedIn URL (most unique)
    if linkedin_norm:
        match_checks.append({
            "filter": f'linkedin_url like "%{linkedin_norm}%"',
            "reason": "linkedin_url",
            "priority": 1
        })
    
    # Priority 2: Email (very reliable)
    if email_norm:
        match_checks.append({
            "filter": f'email == "{email_norm}"',
            "reason": "email",
            "priority": 2
        })
    
    # Priority 3: Phone (reliable but can change)
    if phone_norm and len(phone_norm) >= 10:  # Only check if we have full phone
        match_checks.append({
            "filter": f'phone like "%{phone_norm}%"',
            "reason": "phone",
            "priority": 3
        })
    
    # Priority 4: GitHub URL (less common but useful)
    if github_norm:
        match_checks.append({
            "filter": f'github_url like "%{github_norm}%"',
            "reason": "github_url",
            "priority": 4
        })
    
    if not match_checks:
        logger.debug("No identifiers provided for duplicate check")
        return None
    
    # Sort by priority (LinkedIn first, then email, etc.)
    match_checks.sort(key=lambda x: x["priority"])
    
    # Try each check in priority order
    for check in match_checks:
        try:
            results = client.query(
                collection_name=COLLECTION,
                filter=check["filter"],
                output_fields=["candidate_id", "name", "email", "phone", "linkedin_url", "last_updated"],
                limit=1
            )
            
            if results:
                existing = results[0]
                existing["match_reason"] = check["reason"]
                
                logger.info(
                    f"🔍 Duplicate found via {check['reason']}: "
                    f"{existing['candidate_id']} ({existing.get('name', 'Unknown')})"
                )
                
                return existing
        
        except Exception as e:
            logger.warning(f"Duplicate check failed for {check['reason']}: {e}")
            continue
    
    logger.debug("No duplicate found")
    return None


def get_candidate_version_count(client, base_candidate_id: str) -> int:
    """
    Get the number of versions for a candidate.
    
    Counts all records with candidate_id starting with base_candidate_id.
    """
    try:
        # Query for all versions
        filter_expr = f'candidate_id like "{base_candidate_id}%"'
        
        results = client.query(
            collection_name=COLLECTION,
            filter=filter_expr,
            output_fields=["candidate_id"],
            limit=100  # Should never have this many versions
        )
        
        return len(results)
    
    except Exception as e:
        logger.warning(f"Failed to get version count: {e}")
        return 1


def insert_candidate(
    candidate_data: Dict,
    embeddings: Dict,
    source_channel: str = "Upload",
    check_duplicates: bool = True,
    create_version: bool = False
) -> Tuple[str, bool, Optional[str]]:
    """
    Insert candidate into Milvus with duplicate detection.
    
    Args:
        candidate_data: Extracted candidate data
        embeddings: Generated embeddings
        source_channel: Upload source
        check_duplicates: Whether to check for duplicates (default: True)
        create_version: If duplicate found, create new version instead of rejecting (default: False)
    
    Returns:
        Tuple of (candidate_id, is_new, duplicate_reason)
        - candidate_id: The candidate ID (existing or new)
        - is_new: True if this is a new candidate, False if duplicate/version
        - duplicate_reason: If duplicate, what field matched (email, linkedin_url, etc.)
    """
    client = get_milvus_client()
    
    # ==================== DUPLICATE CHECK ====================
    duplicate_info = None
    
    if check_duplicates:
        logger.info("Checking for duplicates...")
        
        duplicate_info = check_for_duplicate(
            client,
            email=candidate_data.get("email", ""),
            phone=candidate_data.get("phone", ""),
            linkedin_url=candidate_data.get("linkedin_url", ""),
            github_url=candidate_data.get("github_url", "")
        )
        
        if duplicate_info:
            existing_id = duplicate_info["candidate_id"]
            match_reason = duplicate_info["match_reason"]
            
            logger.warning(
                f"⚠️  DUPLICATE DETECTED!\n"
                f"   Existing ID: {existing_id}\n"
                f"   Name: {duplicate_info.get('name', 'Unknown')}\n"
                f"   Matched on: {match_reason}\n"
                f"   Last updated: {duplicate_info.get('last_updated', 'Unknown')}"
            )
            
            if not create_version:
                # Don't insert, return existing ID
                logger.info(f"Returning existing candidate ID: {existing_id}")
                return existing_id, False, match_reason
            
            # Create new version
            logger.info(f"Creating new version for {existing_id}")
            
            version_count = get_candidate_version_count(client, existing_id)
            versioned_id = f"{existing_id}_v{version_count + 1}"
            
            candidate_data["original_candidate_id"] = existing_id
            candidate_data["version_number"] = version_count + 1
            candidate_id = versioned_id
            
            logger.info(f"Created version ID: {candidate_id}")
    
    # ==================== NEW CANDIDATE ====================
    if not duplicate_info:
        candidate_id = candidate_data.get("candidate_id") or generate_candidate_id()
        logger.info(f"Inserting NEW candidate: {candidate_id}")
    
    # ==================== PREPARE DATA FOR MILVUS ====================
    milvus_data = {
        # Identity
        "candidate_id": candidate_id,
        "name": candidate_data.get("name", "Unknown"),
        "email": normalize_email(candidate_data.get("email", "")),
        "phone": candidate_data.get("phone", ""),  # Store original format
        "linkedin_url": candidate_data.get("linkedin_url", ""),
        "portfolio_url": candidate_data.get("portfolio_url", ""),
        "github_url": candidate_data.get("github_url", ""),
        
        # Location
        "location_city": candidate_data.get("location_city", ""),
        "location_state": candidate_data.get("location_state", ""),
        "location_country": candidate_data.get("location_country", ""),
        "relocation_willingness": candidate_data.get("relocation_willingness", "Unknown"),
        "remote_preference": candidate_data.get("remote_preference", "Unknown"),
        "availability_status": candidate_data.get("availability_status", "Unknown"),
        
        # Experience
        "total_experience_years": float(candidate_data.get("total_experience_years", 0.0)),
        "education_level": candidate_data.get("education_level", ""),
        "degrees": candidate_data.get("degrees", ""),
        "institutions": candidate_data.get("institutions", ""),
        "languages_spoken": candidate_data.get("languages_spoken", ""),
        "management_experience_years": float(candidate_data.get("management_experience_years", 0.0)),
        
        # Career
        "career_stage": candidate_data.get("career_stage", "Unknown"),
        "years_in_current_role": float(candidate_data.get("years_in_current_role", 0.0)),
        "top_3_titles": candidate_data.get("top_3_titles", ""),
        "role_type": candidate_data.get("role_type", ""),
        
        # Industry
        "industries_worked": candidate_data.get("industries_worked", ""),
        "domain_expertise": candidate_data.get("domain_expertise", ""),
        "verticals_experience": candidate_data.get("verticals_experience", ""),
        
        # Skills
        "skills_extracted": candidate_data.get("skills_extracted", ""),
        "tools_and_technologies": candidate_data.get("tools_and_technologies", ""),
        "certifications": candidate_data.get("certifications", ""),
        "tech_stack_primary": candidate_data.get("tech_stack_primary", ""),
        "programming_languages": candidate_data.get("programming_languages", ""),
        
        # Evidence
        "employment_history": candidate_data.get("employment_history", ""),
        "semantic_summary": candidate_data.get("semantic_summary", ""),
        "keywords_summary": candidate_data.get("keywords_summary", ""),
        "evidence_skills": candidate_data.get("evidence_skills", ""),
        "evidence_projects": candidate_data.get("evidence_projects", ""),
        "evidence_leadership": candidate_data.get("evidence_leadership", ""),
        
        # Recency
        "current_tech_stack": candidate_data.get("current_tech_stack", ""),
        "years_since_last_update": 0.0,  # Fresh upload
        "top_5_skills_with_years": candidate_data.get("top_5_skills_with_years", ""),
        
        # Metadata
        "source_channel": source_channel,
        "hiring_manager_notes": "",
        "interview_feedback": "",
        "offer_status": "Unknown",
        "assigned_recruiter": "",
        "resume_embedding_version": "e5-base-v2-v3",
        "last_updated": datetime.utcnow().isoformat() + "Z",
        
        # Embeddings
        "summary_embedding": embeddings["summary_embedding"],
        "tech_embedding": embeddings["tech_embedding"],
        "role_embedding": embeddings["role_embedding"],
    }
    
    # ==================== INSERT INTO MILVUS ====================
    try:
        client.insert(
            collection_name=COLLECTION,
            data=[milvus_data]
        )
        
        if duplicate_info:
            logger.info(f"✅ Inserted VERSION: {candidate_id} (version of {duplicate_info['candidate_id']})")
            return candidate_id, False, duplicate_info["match_reason"]
        else:
            logger.info(f"✅ Inserted NEW candidate: {candidate_id}")
            return candidate_id, True, None
        
    except Exception as e:
        logger.exception(f"Failed to insert candidate {candidate_id}")
        raise RuntimeError(f"Milvus insertion failed: {e}")


def update_candidate(
    candidate_id: str,
    updated_data: Dict,
    embeddings: Dict
) -> str:
    """
    Update existing candidate record.
    
    Note: Milvus doesn't support in-place updates, so we delete and re-insert.
    """
    client = get_milvus_client()
    
    try:
        # Delete existing record
        logger.info(f"Updating candidate {candidate_id} (delete + re-insert)")
        
        client.delete(
            collection_name=COLLECTION,
            filter=f'candidate_id == "{candidate_id}"'
        )
        
        # Re-insert with updated data
        updated_data["candidate_id"] = candidate_id
        updated_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Note: We skip duplicate check when updating
        insert_candidate(
            updated_data,
            embeddings,
            source_channel="Update",
            check_duplicates=False
        )
        
        logger.info(f"✅ Updated candidate: {candidate_id}")
        return candidate_id
        
    except Exception as e:
        logger.exception(f"Failed to update candidate {candidate_id}")
        raise RuntimeError(f"Milvus update failed: {e}")