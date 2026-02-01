"""Build dynamic interview prompts based on candidate and job data."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from models import CandidateData, JobData, InterviewQuestion, InterviewQuestions
from config import INTERVIEW_TOTAL_QUESTIONS, INTERVIEW_MAX_FOLLOWUPS


def build_interview_prompt(
    candidate: CandidateData,
    job: JobData,
    avatar_name: str = "Zara",
    questions: Optional[InterviewQuestions] = None,
) -> Dict[str, str]:
    """
    Build the complete interview prompt for the avatar.
    
    Returns:
        Dict with 'system_prompt' and 'greeting' keys
    """
    
    # Format candidate info
    candidate_name = candidate.name or "the candidate"
    candidate_skills = candidate.top_5_skills_with_years or "Not specified"
    candidate_summary = candidate.semantic_summary or "No summary available"
    candidate_tech_stack = ", ".join(candidate.current_tech_stack) if candidate.current_tech_stack else "Not specified"
    
    # Format employment history
    employment_str = ""
    if candidate.employment_history:
        emp_lines = []
        for emp in candidate.employment_history[:3]:  # Limit to 3 most recent
            company = emp.get("company", "Unknown")
            title = emp.get("title", "Unknown")
            start = emp.get("start_date", "")
            end = emp.get("end_date", "Present")
            emp_lines.append(f"- {title} at {company} ({start} to {end})")
        employment_str = "\n".join(emp_lines)
    
    # Format job info
    job_title = job.title or "the position"
    job_company = job.company or "the company"
    jd_text = job.jd_text or "No job description available"
    
    # Truncate JD if too long (keep prompt under limit)
    if len(jd_text) > 2000:
        jd_text = jd_text[:2000] + "..."
    
    # Build the system prompt
    system_prompt = f"""You are {avatar_name}, a professional AI interviewer conducting a screening interview for RCRUTR AI.

INTERVIEW CONTEXT:
- Position: {job_title} at {job_company}
- Candidate: {candidate_name}
- Candidate's Top Skills: {candidate_skills}
- Candidate's Tech Stack: {candidate_tech_stack}

CANDIDATE BACKGROUND:
{candidate_summary}

RECENT EMPLOYMENT:
{employment_str if employment_str else "Not available"}

JOB DESCRIPTION:
{jd_text}

YOUR ROLE:
You are conducting a professional screening interview. Your goals are:
1. Verify candidate's identity and basic information
2. Confirm work authorization and location
3. Assess technical skills relevant to the job
4. Determine if the candidate is a good fit

INTERVIEW STRUCTURE (8 questions total):

BASIC QUESTIONS (ask first 4):
1. "Hi! Before we begin, could you please confirm your full name?"
2. "What is your current work authorization status?" (e.g., US Citizen, Green Card, H1B, etc.)
3. "Where are you currently located, and are you open to relocation if needed?"
4. "I'll briefly describe the role: {job_title}. [Read 2-3 key requirements]. Does this sound like something you'd be interested in?"

TECHNICAL QUESTIONS (ask 4 based on JD and candidate skills):
- Ask about their experience with technologies mentioned in the JD
- Ask about specific projects related to the role
- Ask about their experience level with key skills
- You may ask UP TO 2 follow-up questions per session based on interesting answers

CONVERSATION GUIDELINES:
- Be warm, professional, and encouraging
- Keep questions concise and clear
- Listen actively and acknowledge their responses
- Ask follow-up questions when answers are interesting or need clarification
- Do NOT exceed 8 main questions (follow-ups don't count toward this limit)
- Keep the interview to about 15-20 minutes

IMPORTANT RULES:
- NEVER make up information about the candidate
- NEVER promise job offers or specific outcomes
- If the candidate asks about salary, say "The recruiter will discuss compensation details with you"
- If technical answers are unclear, ask for clarification
- Stay focused on the interview - redirect off-topic conversations

ENDING THE INTERVIEW:
After all questions, thank the candidate and say:
"Thank you for your time today, {candidate_name}. The recruiting team will review our conversation and reach out with next steps. Do you have any questions for me before we wrap up?"

Answer any reasonable questions briefly, then end with:
"Great! Thank you again. Have a wonderful day!"
"""

    # Build greeting
    greeting = f"Hi {candidate_name}! I'm {avatar_name}, and I'll be conducting your screening interview today for the {job_title} position. This should take about 15-20 minutes. Are you ready to get started?"
    
    return {
        "system_prompt": system_prompt,
        "greeting": greeting,
    }


def generate_interview_questions(
    candidate: CandidateData,
    job: JobData,
) -> InterviewQuestions:
    """
    Generate interview questions based on candidate and job data.
    
    Returns:
        InterviewQuestions object with basic and technical questions
    """
    
    candidate_name = candidate.name or "there"
    job_title = job.title or "the position"
    
    # Parse JD to extract key skills/requirements
    jd_text = job.jd_text or ""
    
    # Basic questions (always the same structure)
    basic_questions = [
        InterviewQuestion(
            index=1,
            question_type="basic",
            question=f"Hi! Before we begin, could you please confirm your full name?",
            expected_keywords=None,
        ),
        InterviewQuestion(
            index=2,
            question_type="basic",
            question="What is your current work authorization status in the US? For example, are you a US Citizen, Green Card holder, or on a work visa?",
            expected_keywords=["citizen", "green card", "h1b", "h4", "ead", "opt", "tn"],
        ),
        InterviewQuestion(
            index=3,
            question_type="basic",
            question="Where are you currently located? And would you be open to relocation if the role requires it?",
            expected_keywords=None,
        ),
        InterviewQuestion(
            index=4,
            question_type="basic",
            question=f"Let me briefly describe the {job_title} role. Based on what I've shared, does this sound like something you'd be interested in pursuing?",
            expected_keywords=["yes", "interested", "excited", "definitely"],
        ),
    ]
    
    # Technical questions (based on JD and candidate skills)
    technical_questions = _generate_technical_questions(candidate, job)
    
    return InterviewQuestions(
        total=INTERVIEW_TOTAL_QUESTIONS,
        basic_questions=basic_questions,
        technical_questions=technical_questions,
    )


def _generate_technical_questions(
    candidate: CandidateData,
    job: JobData,
) -> List[InterviewQuestion]:
    """Generate technical questions based on JD and candidate skills."""
    
    questions = []
    jd_text = (job.jd_text or "").lower()
    candidate_skills = candidate.top_5_skills_with_years or ""
    
    # Parse candidate skills
    skills_list = []
    if candidate_skills:
        # Format: "Python:5, Java:3, AWS:4"
        for skill_entry in candidate_skills.split(","):
            skill_entry = skill_entry.strip()
            if ":" in skill_entry:
                skill_name = skill_entry.split(":")[0].strip()
                skills_list.append(skill_name.lower())
    
    # Common technical question templates
    templates = [
        {
            "keywords": ["python", "django", "flask", "fastapi"],
            "question": "I see you have Python experience. Can you tell me about a significant project where you used Python? What frameworks did you work with?",
            "expected": ["django", "flask", "fastapi", "api", "backend"],
        },
        {
            "keywords": ["java", "spring", "springboot"],
            "question": "Tell me about your Java experience. Have you worked with Spring Boot or microservices architecture?",
            "expected": ["spring", "microservices", "boot", "api"],
        },
        {
            "keywords": ["aws", "azure", "gcp", "cloud"],
            "question": "What cloud platforms have you worked with? Can you describe a project where you deployed and managed cloud infrastructure?",
            "expected": ["aws", "azure", "gcp", "ec2", "s3", "lambda", "kubernetes"],
        },
        {
            "keywords": ["kubernetes", "docker", "container"],
            "question": "Do you have experience with containerization? Tell me about how you've used Docker or Kubernetes in your projects.",
            "expected": ["docker", "kubernetes", "container", "pod", "deployment"],
        },
        {
            "keywords": ["react", "angular", "vue", "frontend", "javascript", "typescript"],
            "question": "What frontend technologies have you worked with? Can you describe a complex UI feature you've built?",
            "expected": ["react", "angular", "vue", "component", "state", "redux"],
        },
        {
            "keywords": ["sql", "database", "postgres", "mysql", "mongodb"],
            "question": "Tell me about your database experience. What types of databases have you worked with, and how do you approach database design?",
            "expected": ["sql", "postgres", "mysql", "mongodb", "query", "schema"],
        },
        {
            "keywords": ["api", "rest", "graphql", "microservice"],
            "question": "Have you designed or built APIs? Can you walk me through your approach to API design and documentation?",
            "expected": ["rest", "graphql", "endpoint", "documentation", "swagger"],
        },
        {
            "keywords": ["devops", "ci/cd", "jenkins", "github actions", "gitlab"],
            "question": "What's your experience with CI/CD pipelines? How do you approach automated testing and deployment?",
            "expected": ["jenkins", "github", "gitlab", "pipeline", "deploy", "test"],
        },
    ]
    
    # Select questions that match either JD or candidate skills
    selected = []
    for template in templates:
        # Check if any keyword matches JD or candidate skills
        jd_match = any(kw in jd_text for kw in template["keywords"])
        skill_match = any(kw in skill for kw in template["keywords"] for skill in skills_list)
        
        if jd_match or skill_match:
            selected.append(template)
    
    # If we don't have enough, add generic questions
    generic_questions = [
        {
            "question": "What would you say is your strongest technical skill, and how have you applied it in your recent work?",
            "expected": [],
        },
        {
            "question": "Can you describe a challenging technical problem you solved recently? What was your approach?",
            "expected": ["debug", "solve", "approach", "solution"],
        },
        {
            "question": "How do you stay current with new technologies and best practices in your field?",
            "expected": ["learn", "course", "read", "practice"],
        },
        {
            "question": "Tell me about a time you had to work with a technology you weren't familiar with. How did you get up to speed?",
            "expected": ["learn", "documentation", "research", "practice"],
        },
    ]
    
    # Combine selected and generic to get 4 technical questions
    while len(selected) < 4 and generic_questions:
        selected.append(generic_questions.pop(0))
    
    # Create InterviewQuestion objects
    for i, q in enumerate(selected[:4]):
        questions.append(InterviewQuestion(
            index=5 + i,  # Start from 5 (after 4 basic questions)
            question_type="technical",
            question=q.get("question", q) if isinstance(q, dict) else q,
            expected_keywords=q.get("expected", []) if isinstance(q, dict) else [],
        ))
    
    return questions


def format_questions_for_db(questions: InterviewQuestions) -> List[Dict[str, Any]]:
    """Format questions for storage in database."""
    result = []
    for q in questions.all_questions():
        result.append({
            "index": q.index,
            "type": q.question_type,
            "question": q.question,
            "expected_keywords": q.expected_keywords or [],
            "is_followup": q.is_followup,
            "parent_index": q.parent_index,
        })
    return result
