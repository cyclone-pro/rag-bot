"""Remove and restore PII (email, phone, URLs) from resume text."""
import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def scrub_pii(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Remove PII from text before sending to LLM.
    
    Returns:
        (sanitized_text, pii_dict)
    """
    logger.info("   Scrubbing PII from resume text...")
    pii = {}
    sanitized = text
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, sanitized)
    if emails:
        pii['email'] = emails[0]
        sanitized = re.sub(email_pattern, '[EMAIL_REDACTED]', sanitized)
        logger.info(f"   → Email redacted: {pii['email']}")
    
    # Extract phone numbers
    phone_patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, sanitized)
        if phones and 'phone' not in pii:
            pii['phone'] = phones[0]
            sanitized = re.sub(pattern, '[PHONE_REDACTED]', sanitized)
            logger.info(f"   → Phone redacted: {pii['phone']}")
            break
    
    # Extract LinkedIn URL
    linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+'
    linkedin_urls = re.findall(linkedin_pattern, sanitized, re.IGNORECASE)
    if linkedin_urls:
        pii['linkedin_url'] = linkedin_urls[0]
        sanitized = re.sub(linkedin_pattern, '[LINKEDIN_REDACTED]', sanitized, flags=re.IGNORECASE)
        logger.info(f"   → LinkedIn redacted: {pii['linkedin_url']}")
    
    # Extract GitHub URL
    github_pattern = r'https?://(?:www\.)?github\.com/[A-Za-z0-9_-]+'
    github_urls = re.findall(github_pattern, sanitized, re.IGNORECASE)
    if github_urls:
        pii['github_url'] = github_urls[0]
        sanitized = re.sub(github_pattern, '[GITHUB_REDACTED]', sanitized, flags=re.IGNORECASE)
        logger.info(f"   → GitHub redacted: {pii['github_url']}")
    
    # Extract portfolio URLs
    url_pattern = r'https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?'
    urls = re.findall(url_pattern, sanitized)
    if urls and 'portfolio_url' not in pii:
        for url in urls:
            if 'linkedin.com' not in url.lower() and 'github.com' not in url.lower():
                pii['portfolio_url'] = url
                sanitized = re.sub(re.escape(url), '[URL_REDACTED]', sanitized)
                logger.info(f"   → Portfolio URL redacted: {url}")
                break
    
    logger.info(f"   → Total PII fields scrubbed: {len(pii)}")
    
    return sanitized, pii


def merge_pii(extracted_data: Dict, pii: Dict[str, str]) -> Dict:
    """Merge PII back into extracted data."""
    logger.info("   Merging PII back into extracted data...")
    
    extracted_data.update(pii)
    
    # Ensure all PII fields exist
    extracted_data.setdefault('email', '')
    extracted_data.setdefault('phone', '')
    extracted_data.setdefault('linkedin_url', '')
    extracted_data.setdefault('github_url', '')
    extracted_data.setdefault('portfolio_url', '')
    
    logger.info(f"   → Email: {'✓' if extracted_data.get('email') else '✗'}")
    logger.info(f"   → Phone: {'✓' if extracted_data.get('phone') else '✗'}")
    logger.info(f"   → LinkedIn: {'✓' if extracted_data.get('linkedin_url') else '✗'}")
    
    return extracted_data