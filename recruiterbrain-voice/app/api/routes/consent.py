"""
Consent Management API Routes
Handles pre-interview consent requests via email/SMS
Optimized with connection pooling
"""

import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, EmailStr, validator
import httpx

from app.config.settings import settings
from app.services.database import get_db_session
from sqlalchemy import text


logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================
# Request/Response Models
# ============================================

class SendConsentRequest(BaseModel):
    candidate_id: str
    interview_id: str
    candidate_name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    method: str = "email"  # 'email' or 'sms'
    
    @validator('method')
    def validate_method(cls, v, values):
        if v not in ['email', 'sms']:
            raise ValueError('method must be "email" or "sms"')
        if v == 'email' and not values.get('email'):
            raise ValueError('email is required when method is "email"')
        if v == 'sms' and not values.get('phone'):
            raise ValueError('phone is required when method is "sms"')
        return v


class ConsentSubmission(BaseModel):
    token: str
    consents: Dict[str, bool]
    device_info: Dict[str, str]


class ConsentResponse(BaseModel):
    success: bool
    request_id: int
    consent_url: str
    expires_in_days: int
    method: str


# ============================================
# Helper Functions
# ============================================

async def send_email_with_consent_link(
    email: str,
    candidate_name: str,
    consent_url: str,
    interview_id: str
):
    """
    Send consent email using your email service
    TODO: Replace with your actual email provider (SendGrid, AWS SES, etc)
    """
    subject = "Action Required: Consent for AI Interview"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #007bff;">Consent Required for Your Interview</h2>
            
            <p>Hi {candidate_name},</p>
            
            <p>Before we can proceed with your AI-powered voice interview, we need your consent 
            to record and process your interview data.</p>
            
            <p>This is required by privacy regulations (GDPR, CCPA) to protect your rights.</p>
            
            <div style="margin: 30px 0;">
                <a href="{consent_url}" 
                   style="background: #007bff; color: white; padding: 14px 28px; 
                          text-decoration: none; border-radius: 5px; display: inline-block;">
                    Review and Provide Consent
                </a>
            </div>
            
            <p style="font-size: 14px; color: #666;">
                <strong>This link expires in 7 days.</strong><br>
                Interview ID: {interview_id}
            </p>
            
            <p>If you have any questions, please contact us at support@recruiterbrain.com</p>
            
            <p>Best regards,<br>RecruiterBrain Team</p>
            
            <hr style="margin-top: 30px; border: none; border-top: 1px solid #ddd;">
            <p style="font-size: 12px; color: #999;">
                You received this email because you applied for a position that uses 
                our AI interview platform. You can withdraw your consent at any time.
            </p>
        </div>
    </body>
    </html>
    """
    
    # TODO: Implement actual email sending
    # Example with SendGrid:
    # import sendgrid
    # from sendgrid.helpers.mail import Mail
    # 
    # message = Mail(
    #     from_email=settings.email_from,
    #     to_emails=email,
    #     subject=subject,
    #     html_content=html_body
    # )
    # sg = sendgrid.SendGridAPIClient(settings.sendgrid_api_key)
    # response = sg.send(message)
    
    logger.info(f"Consent email sent to {email} (Interview: {interview_id})")
    return True


async def send_sms_with_consent_link(
    phone: str,
    candidate_name: str,
    consent_url: str,
    interview_id: str
):
    """
    Send consent SMS via Twilio
    """
    from twilio.rest import Client
    
    try:
        # Initialize Twilio client
        client = Client(
            settings.twilio_account_sid,
            settings.twilio_auth_token
        )
        
        # Create message
        message_body = (
            f"Hi {candidate_name}, before your AI interview, please provide consent: "
            f"{consent_url} (Expires in 7 days. Interview ID: {interview_id})"
        )
        
        # Send SMS
        message = client.messages.create(
            body=message_body,
            from_=settings.twilio_phone_number,
            to=phone
        )
        
        logger.info(f"SMS sent to {phone}. Message SID: {message.sid}")
        return message.sid
        
    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")
        raise HTTPException(status_code=500, detail=f"SMS send failed: {str(e)}")


# ============================================
# API Endpoints
# ============================================

@router.post("/send-request", response_model=ConsentResponse)
async def send_consent_request(
    request: SendConsentRequest,
    background_tasks: BackgroundTasks
):
    """
    Step 1: Send consent request to candidate via email or SMS
    
    Creates record in consent_requests table and sends link
    """
    
    try:
        # Generate secure token
        token = secrets.token_urlsafe(32)
        
        # Generate consent form URL
        consent_url = f"{settings.app_base_url}/consent/{token}"
        
        # Consent text (legal language)
        consent_text = """
By consenting, you agree to the following:

1. INTERVIEW PARTICIPATION: You consent to participate in an AI-powered voice interview.

2. CALL RECORDING: Your interview call will be recorded for quality assurance and evaluation purposes.

3. DATA PROCESSING: Your voice data, responses, and evaluation results will be processed by our AI systems.

4. DATA RETENTION: Your data will be stored securely for 90 days, after which it will be automatically deleted.

5. AI EVALUATION: Your responses will be analyzed by artificial intelligence to assess your qualifications.

6. YOUR RIGHTS: You have the right to:
   - Withdraw consent at any time
   - Request deletion of your data
   - Request a copy of your data
   - Object to automated decision-making

For more information, see our Privacy Policy at {app_base_url}/privacy
        """.format(app_base_url=settings.app_base_url)
        
        # Use connection pooling properly - single session for all queries
        async with get_db_session() as session:
            # Insert into consent_requests table
            result = await session.execute(
                text("""
                    INSERT INTO consent_requests (
                        candidate_id,
                        interview_id,
                        request_type,
                        recipient_contact,
                        consent_form_url,
                        consent_token,
                        consent_text,
                        consent_version,
                        consents_requested,
                        status,
                        expires_at,
                        created_at,
                        updated_at
                    ) VALUES (
                        :candidate_id,
                        :interview_id,
                        :request_type,
                        :recipient_contact,
                        :consent_form_url,
                        :consent_token,
                        :consent_text,
                        :consent_version,
                        :consents_requested,
                        :status,
                        :expires_at,
                        NOW(),
                        NOW()
                    ) RETURNING request_id
                """),
                {
                    "candidate_id": request.candidate_id,
                    "interview_id": request.interview_id,
                    "request_type": request.method,
                    "recipient_contact": request.email if request.method == "email" else request.phone,
                    "consent_form_url": consent_url,
                    "consent_token": token,
                    "consent_text": consent_text,
                    "consent_version": "v1.0",
                    "consents_requested": '["interview_participation", "call_recording", "data_processing", "ai_evaluation"]',
                    "status": "pending",
                    "expires_at": datetime.utcnow() + timedelta(days=7)
                }
            )
            
            row = result.fetchone()
            request_id = row[0]
            
            # Commit to get the request_id
            await session.commit()
        
        # Send email/SMS in background (non-blocking)
        if request.method == "email":
            background_tasks.add_task(
                send_email_with_consent_link,
                request.email,
                request.candidate_name,
                consent_url,
                request.interview_id
            )
            message_id = None
        else:
            # SMS needs to be synchronous to get message_id
            message_id = await send_sms_with_consent_link(
                request.phone,
                request.candidate_name,
                consent_url,
                request.interview_id
            )
        
        # Update status to 'sent' and store message_id
        async with get_db_session() as session:
            await session.execute(
                text("""
                    UPDATE consent_requests
                    SET 
                        status = 'sent',
                        sent_at = NOW(),
                        sms_message_id = :message_id,
                        delivery_status = 'delivered'
                    WHERE request_id = :request_id
                """),
                {
                    "request_id": request_id,
                    "message_id": message_id
                }
            )
            await session.commit()
        
        logger.info(f"Consent request sent via {request.method} to {request.candidate_id}")
        
        return ConsentResponse(
            success=True,
            request_id=request_id,
            consent_url=consent_url,
            expires_in_days=7,
            method=request.method
        )
        
    except Exception as e:
        logger.error(f"Error sending consent request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/verify/{token}")
async def verify_consent_token(token: str):
    """
    Step 2: Verify token when user opens consent page
    Returns consent form details
    """
    
    try:
        async with get_db_session() as session:
            result = await session.execute(
                text("""
                    SELECT 
                        cr.request_id,
                        cr.candidate_id,
                        cr.interview_id,
                        cr.consent_text,
                        cr.consent_version,
                        cr.consents_requested,
                        cr.expires_at,
                        cr.status,
                        c.name as candidate_name,
                        i.job_title
                    FROM consent_requests cr
                    LEFT JOIN candidates c ON cr.candidate_id = c.candidate_id
                    LEFT JOIN interviews i ON cr.interview_id = i.interview_id
                    WHERE cr.consent_token = :token
                """),
                {"token": token}
            )
            
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Invalid consent token")
            
            # Check if expired
            if row.expires_at < datetime.utcnow():
                # Update status to expired
                await session.execute(
                    text("UPDATE consent_requests SET status = 'expired' WHERE consent_token = :token"),
                    {"token": token}
                )
                await session.commit()
                raise HTTPException(status_code=400, detail="Consent link has expired")
            
            # Check if already completed
            if row.status == 'completed':
                raise HTTPException(status_code=400, detail="Consent already submitted")
            
            # Update status to 'viewed'
            await session.execute(
                text("UPDATE consent_requests SET status = 'viewed', viewed_at = NOW() WHERE consent_token = :token"),
                {"token": token}
            )
            await session.commit()
        
        return {
            "valid": True,
            "candidate_name": row.candidate_name or "Candidate",
            "job_title": row.job_title or "Position",
            "consent_text": row.consent_text,
            "consents_requested": row.consents_requested,
            "expires_at": row.expires_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        raise HTTPException(status_code=500, detail="Verification failed")


@router.post("/submit")
async def submit_consent(submission: ConsentSubmission, request: Request):
    """
    Step 3: Save consent after user clicks 'I Agree'
    Saves to candidate_consents table (ONE connection for ALL inserts)
    """
    
    try:
        # Get IP address from request
        ip_address = request.client.host
        
        # Use SINGLE database session for ALL operations
        async with get_db_session() as session:
            # Step 3a: Get request details
            result = await session.execute(
                text("""
                    SELECT 
                        request_id,
                        candidate_id,
                        interview_id,
                        consent_text,
                        consent_version,
                        expires_at,
                        status
                    FROM consent_requests
                    WHERE consent_token = :token
                """),
                {"token": submission.token}
            )
            
            row = result.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Invalid token")
            
            if row.status == 'completed':
                raise HTTPException(status_code=400, detail="Consent already submitted")
            
            if row.expires_at < datetime.utcnow():
                raise HTTPException(status_code=400, detail="Consent link expired")
            
            # Extract values
            request_id = row.request_id
            candidate_id = row.candidate_id
            interview_id = row.interview_id
            consent_text = row.consent_text
            consent_version = row.consent_version
            
            # Generate device fingerprint
            device_fingerprint = hashlib.md5(
                submission.device_info.get('user_agent', '').encode()
            ).hexdigest()
            
            # Geo location JSONB
            geo_location = {
                'timezone': submission.device_info.get('timezone'),
                'language': submission.device_info.get('language'),
                'screen_resolution': submission.device_info.get('screen_resolution')
            }
            
            # Step 3b: Insert ALL consents in SINGLE transaction
            consent_ids = []
            
            for consent_type, consented in submission.consents.items():
                result = await session.execute(
                    text("""
                        INSERT INTO candidate_consents (
                            candidate_id,
                            interview_id,
                            request_id,
                            consent_type,
                            consented,
                            consent_method,
                            consent_timestamp,
                            consent_text,
                            consent_version,
                            consent_language,
                            ip_address,
                            user_agent,
                            device_fingerprint,
                            geo_location,
                            data_retention_days,
                            active,
                            revoked,
                            gdpr_compliant,
                            created_at,
                            updated_at
                        ) VALUES (
                            :candidate_id, :interview_id, :request_id,
                            :consent_type, :consented, :consent_method,
                            :consent_timestamp, :consent_text, :consent_version,
                            :consent_language, :ip_address, :user_agent,
                            :device_fingerprint, :geo_location::jsonb,
                            :data_retention_days, :active, :revoked,
                            :gdpr_compliant, NOW(), NOW()
                        ) RETURNING consent_id
                    """),
                    {
                        "candidate_id": candidate_id,
                        "interview_id": interview_id,
                        "request_id": request_id,
                        "consent_type": consent_type,
                        "consented": consented,
                        "consent_method": "email_link",
                        "consent_timestamp": datetime.utcnow(),
                        "consent_text": consent_text,
                        "consent_version": consent_version,
                        "consent_language": "en",
                        "ip_address": ip_address,
                        "user_agent": submission.device_info.get('user_agent', ''),
                        "device_fingerprint": device_fingerprint,
                        "geo_location": str(geo_location),
                        "data_retention_days": 90,
                        "active": True,
                        "revoked": False,
                        "gdpr_compliant": True
                    }
                )
                
                consent_id = result.fetchone()[0]
                consent_ids.append(consent_id)
            
            # Step 3c: Update consent_requests to 'completed'
            await session.execute(
                text("""
                    UPDATE consent_requests
                    SET status = 'completed', completed_at = NOW()
                    WHERE consent_token = :token
                """),
                {"token": submission.token}
            )
            
            # Step 3d: Log to audit_log
            await session.execute(
                text("""
                    INSERT INTO audit_log (
                        event_type,
                        entity_type,
                        entity_id,
                        consent_id,
                        actor_type,
                        actor_id,
                        ip_address,
                        user_agent,
                        action_details,
                        gdpr_relevant,
                        timestamp_utc
                    ) VALUES (
                        'consent_given',
                        'consent',
                        :entity_id,
                        :consent_id,
                        'candidate',
                        :actor_id,
                        :ip_address,
                        :user_agent,
                        :action_details::jsonb,
                        true,
                        NOW()
                    )
                """),
                {
                    "entity_id": consent_ids[0],
                    "consent_id": consent_ids[0],
                    "actor_id": candidate_id,
                    "ip_address": ip_address,
                    "user_agent": submission.device_info.get('user_agent', ''),
                    "action_details": str({
                        'consents_given': list(submission.consents.keys()),
                        'interview_id': interview_id,
                        'all_consents_count': len(consent_ids)
                    })
                }
            )
            
            # Commit ALL changes in single transaction
            await session.commit()
        
        logger.info(f"Consent recorded for candidate {candidate_id}, interview {interview_id}")
        
        return {
            "success": True,
            "message": "Consent recorded successfully",
            "consent_ids": consent_ids,
            "interview_id": interview_id,
            "candidate_id": candidate_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting consent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save consent")


@router.get("/status/{interview_id}")
async def check_consent_status(interview_id: str):
    """
    Check if candidate has provided all required consents
    """
    
    try:
        async with get_db_session() as session:
            result = await session.execute(
                text("""
                    SELECT 
                        consent_type,
                        consented,
                        consent_timestamp,
                        consent_method
                    FROM candidate_consents
                    WHERE interview_id = :interview_id
                      AND active = true
                      AND revoked = false
                    ORDER BY consent_timestamp DESC
                """),
                {"interview_id": interview_id}
            )
            
            rows = result.fetchall()
            
            consents = {
                row.consent_type: {
                    "consented": row.consented,
                    "timestamp": row.consent_timestamp.isoformat(),
                    "method": row.consent_method
                }
                for row in rows
            }
            
            # Required consents
            required = ["interview_participation", "call_recording", "data_processing"]
            all_consented = all(
                consents.get(req, {}).get("consented", False) 
                for req in required
            )
            
            return {
                "interview_id": interview_id,
                "consents": consents,
                "all_required_consents_given": all_consented,
                "can_proceed_with_interview": all_consented
            }
            
    except Exception as e:
        logger.error(f"Error checking consent status: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")