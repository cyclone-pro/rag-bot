#!/usr/bin/env python3
"""
Email notifications for RCRUTR Interviews.

Sends interview invitations, reminders, and results to candidates.

Supports:
- SendGrid (recommended for production)
- SMTP (Gmail, etc.)
- Console output (for testing)
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("rcrutr_interviews_email")

# Configuration
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "console")  # "sendgrid", "smtp", "console"
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "interviews@rcrutr.com")
FROM_NAME = os.getenv("FROM_NAME", "RCRUTR AI Interviews")


def _log_event(level: str, message: str, **fields: Any) -> None:
    """Structured logging."""
    payload = {"message": message, **fields}
    getattr(logger, level if level in ("warning", "error") else "info")(json.dumps(payload))


@dataclass
class EmailResult:
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


def send_email_sendgrid(
    to_email: str,
    to_name: str,
    subject: str,
    html_content: str,
    text_content: str = None,
) -> EmailResult:
    """Send email using SendGrid API."""
    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail, Email, To, Content
        
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        
        message = Mail(
            from_email=Email(FROM_EMAIL, FROM_NAME),
            to_emails=To(to_email, to_name),
            subject=subject,
            html_content=html_content,
        )
        
        if text_content:
            message.add_content(Content("text/plain", text_content))
        
        response = sg.send(message)
        
        _log_event("info", "email_sent_sendgrid", 
                   to=to_email, subject=subject, status=response.status_code)
        
        return EmailResult(
            success=response.status_code in (200, 201, 202),
            message_id=response.headers.get("X-Message-Id"),
        )
        
    except Exception as e:
        _log_event("error", "email_sendgrid_failed", to=to_email, error=str(e))
        return EmailResult(success=False, error=str(e))


def send_email_smtp(
    to_email: str,
    to_name: str,
    subject: str,
    html_content: str,
    text_content: str = None,
) -> EmailResult:
    """Send email using SMTP."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
        msg["To"] = f"{to_name} <{to_email}>"
        
        if text_content:
            msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, to_email, msg.as_string())
        
        _log_event("info", "email_sent_smtp", to=to_email, subject=subject)
        return EmailResult(success=True)
        
    except Exception as e:
        _log_event("error", "email_smtp_failed", to=to_email, error=str(e))
        return EmailResult(success=False, error=str(e))


def send_email_console(
    to_email: str,
    to_name: str,
    subject: str,
    html_content: str,
    text_content: str = None,
) -> EmailResult:
    """Print email to console (for testing)."""
    print()
    print("=" * 60)
    print("📧 EMAIL (Console Mode)")
    print("=" * 60)
    print(f"To: {to_name} <{to_email}>")
    print(f"From: {FROM_NAME} <{FROM_EMAIL}>")
    print(f"Subject: {subject}")
    print("-" * 60)
    print(text_content or html_content)
    print("=" * 60)
    print()
    
    _log_event("info", "email_sent_console", to=to_email, subject=subject)
    return EmailResult(success=True, message_id="console")


def send_email(
    to_email: str,
    to_name: str,
    subject: str,
    html_content: str,
    text_content: str = None,
) -> EmailResult:
    """Send email using configured provider."""
    
    if EMAIL_PROVIDER == "sendgrid" and SENDGRID_API_KEY:
        return send_email_sendgrid(to_email, to_name, subject, html_content, text_content)
    elif EMAIL_PROVIDER == "smtp" and SMTP_USER:
        return send_email_smtp(to_email, to_name, subject, html_content, text_content)
    else:
        return send_email_console(to_email, to_name, subject, html_content, text_content)


# =============================================================================
# EMAIL TEMPLATES
# =============================================================================

def build_interview_invitation_email(
    candidate_name: str,
    job_title: str,
    company: str,
    scheduled_time: datetime,
    meeting_url: str,
    meeting_passcode: str,
    avatar_name: str = "Zara",
) -> Dict[str, str]:
    """Build interview invitation email content."""
    
    # Format date/time nicely
    date_str = scheduled_time.strftime("%A, %B %d, %Y")
    time_str = scheduled_time.strftime("%I:%M %p UTC")
    
    subject = f"Interview Invitation: {job_title} at {company}"
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #4F46E5; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9fafb; padding: 20px; border: 1px solid #e5e7eb; }}
        .meeting-box {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; border: 2px solid #4F46E5; }}
        .btn {{ display: inline-block; background: #4F46E5; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; }}
        .footer {{ text-align: center; padding: 20px; color: #6b7280; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Interview Invitation</h1>
        </div>
        <div class="content">
            <p>Hi {candidate_name},</p>
            
            <p>You've been invited to an AI-powered screening interview for the <strong>{job_title}</strong> position at <strong>{company}</strong>.</p>
            
            <div class="meeting-box">
                <h3>📅 Interview Details</h3>
                <p><strong>Date:</strong> {date_str}</p>
                <p><strong>Time:</strong> {time_str}</p>
                <p><strong>Duration:</strong> ~15-20 minutes</p>
                <p><strong>Interviewer:</strong> {avatar_name} (AI Interview Assistant)</p>
                
                <h3>🔗 Join Meeting</h3>
                <p><a href="{meeting_url}" class="btn">Join Interview</a></p>
                <p style="font-size: 12px; color: #6b7280;">
                    Meeting URL: {meeting_url}<br>
                    Passcode: <strong>{meeting_passcode}</strong>
                </p>
            </div>
            
            <h3>💡 Tips for Success</h3>
            <ul>
                <li>Join from a quiet location with stable internet</li>
                <li>Test your camera and microphone beforehand</li>
                <li>Have your resume handy for reference</li>
                <li>Speak clearly and take your time answering</li>
            </ul>
            
            <p>If you have any questions or need to reschedule, please reply to this email.</p>
            
            <p>Good luck! 🍀</p>
        </div>
        <div class="footer">
            <p>Powered by RCRUTR AI Interview Platform</p>
        </div>
    </div>
</body>
</html>
"""
    
    text_content = f"""
Interview Invitation: {job_title} at {company}

Hi {candidate_name},

You've been invited to an AI-powered screening interview for the {job_title} position at {company}.

INTERVIEW DETAILS
-----------------
Date: {date_str}
Time: {time_str}
Duration: ~15-20 minutes
Interviewer: {avatar_name} (AI Interview Assistant)

JOIN MEETING
------------
URL: {meeting_url}
Passcode: {meeting_passcode}

TIPS FOR SUCCESS
----------------
- Join from a quiet location with stable internet
- Test your camera and microphone beforehand
- Have your resume handy for reference
- Speak clearly and take your time answering

If you have any questions or need to reschedule, please reply to this email.

Good luck!

---
Powered by RCRUTR AI Interview Platform
"""
    
    return {
        "subject": subject,
        "html_content": html_content,
        "text_content": text_content,
    }


def build_interview_reminder_email(
    candidate_name: str,
    job_title: str,
    scheduled_time: datetime,
    meeting_url: str,
    meeting_passcode: str,
    minutes_until: int = 30,
) -> Dict[str, str]:
    """Build interview reminder email content."""
    
    time_str = scheduled_time.strftime("%I:%M %p UTC")
    
    subject = f"⏰ Reminder: Your interview starts in {minutes_until} minutes"
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .alert {{ background: #FEF3C7; border: 2px solid #F59E0B; padding: 20px; border-radius: 8px; text-align: center; }}
        .btn {{ display: inline-block; background: #4F46E5; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; margin-top: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="alert">
            <h2>⏰ Interview Reminder</h2>
            <p>Hi {candidate_name}, your interview for <strong>{job_title}</strong> starts in <strong>{minutes_until} minutes</strong> at {time_str}!</p>
            <p><a href="{meeting_url}" class="btn">Join Now</a></p>
            <p style="font-size: 12px;">Passcode: {meeting_passcode}</p>
        </div>
    </div>
</body>
</html>
"""
    
    text_content = f"""
⏰ Interview Reminder

Hi {candidate_name},

Your interview for {job_title} starts in {minutes_until} minutes at {time_str}!

Join now: {meeting_url}
Passcode: {meeting_passcode}

Good luck!
"""
    
    return {
        "subject": subject,
        "html_content": html_content,
        "text_content": text_content,
    }


# =============================================================================
# HIGH-LEVEL FUNCTIONS
# =============================================================================

def send_interview_invitation(
    candidate_email: str,
    candidate_name: str,
    job_title: str,
    company: str,
    scheduled_time: datetime,
    meeting_url: str,
    meeting_passcode: str,
    avatar_name: str = "Zara",
) -> EmailResult:
    """Send interview invitation to candidate."""
    
    email_content = build_interview_invitation_email(
        candidate_name=candidate_name,
        job_title=job_title,
        company=company,
        scheduled_time=scheduled_time,
        meeting_url=meeting_url,
        meeting_passcode=meeting_passcode,
        avatar_name=avatar_name,
    )
    
    return send_email(
        to_email=candidate_email,
        to_name=candidate_name,
        subject=email_content["subject"],
        html_content=email_content["html_content"],
        text_content=email_content["text_content"],
    )


def send_interview_reminder(
    candidate_email: str,
    candidate_name: str,
    job_title: str,
    scheduled_time: datetime,
    meeting_url: str,
    meeting_passcode: str,
    minutes_until: int = 30,
) -> EmailResult:
    """Send interview reminder to candidate."""
    
    email_content = build_interview_reminder_email(
        candidate_name=candidate_name,
        job_title=job_title,
        scheduled_time=scheduled_time,
        meeting_url=meeting_url,
        meeting_passcode=meeting_passcode,
        minutes_until=minutes_until,
    )
    
    return send_email(
        to_email=candidate_email,
        to_name=candidate_name,
        subject=email_content["subject"],
        html_content=email_content["html_content"],
        text_content=email_content["text_content"],
    )


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    from datetime import timedelta, timezone
    
    parser = argparse.ArgumentParser(description='Test email sending')
    parser.add_argument('--to', type=str, required=True, help='Recipient email')
    parser.add_argument('--name', type=str, default='Test Candidate', help='Recipient name')
    parser.add_argument('--type', type=str, default='invitation', choices=['invitation', 'reminder'])
    
    args = parser.parse_args()
    
    scheduled = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    
    if args.type == 'invitation':
        result = send_interview_invitation(
            candidate_email=args.to,
            candidate_name=args.name,
            job_title="Mid-Level Python Developer",
            company="Elite Solutions",
            scheduled_time=scheduled,
            meeting_url="https://zoom.us/j/123456789",
            meeting_passcode="abc123",
        )
    else:
        result = send_interview_reminder(
            candidate_email=args.to,
            candidate_name=args.name,
            job_title="Mid-Level Python Developer",
            scheduled_time=scheduled,
            meeting_url="https://zoom.us/j/123456789",
            meeting_passcode="abc123",
            minutes_until=30,
        )
    
    print(f"\nResult: {'✅ Success' if result.success else '❌ Failed'}")
    if result.error:
        print(f"Error: {result.error}")
