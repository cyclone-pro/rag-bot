"""
Build dynamic, conversational interview prompts based on candidate and job data.

The interview style adapts to the role level:
- Junior/Fresher: Simple, foundational questions
- Mid-level: Practical experience, real-world scenarios
- Senior: Architecture, leadership, deep technical knowledge
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from models import CandidateData, JobData, InterviewQuestion, InterviewQuestions
from config import INTERVIEW_TOTAL_QUESTIONS, INTERVIEW_MAX_FOLLOWUPS


def detect_role_level(job_title: str, jd_text: str) -> str:
    """
    Detect the seniority level from job title and description.
    Returns: 'junior', 'mid', or 'senior'
    """
    title_lower = job_title.lower() if job_title else ""
    jd_lower = jd_text.lower() if jd_text else ""
    combined = f"{title_lower} {jd_lower}"
    
    # Senior indicators
    senior_keywords = [
        "senior", "sr.", "lead", "principal", "staff", "architect", 
        "director", "manager", "head of", "vp ", "10+ years", "8+ years",
        "7+ years", "expert", "seasoned"
    ]
    
    # Junior indicators
    junior_keywords = [
        "junior", "jr.", "entry", "fresher", "graduate", "intern",
        "trainee", "associate", "0-2 years", "1-2 years", "beginner",
        "new grad", "entry-level"
    ]
    
    for keyword in senior_keywords:
        if keyword in combined:
            return "senior"
    
    for keyword in junior_keywords:
        if keyword in combined:
            return "junior"
    
    return "mid"


def extract_tech_stack(jd_text: str, candidate_stack: List[str]) -> List[str]:
    """Extract key technologies to focus on during interview."""
    
    # Common tech keywords to look for
    tech_keywords = [
        # Languages
        "python", "javascript", "typescript", "java", "go", "golang", "rust",
        "c++", "c#", "ruby", "php", "scala", "kotlin", "swift",
        # Frameworks
        "fastapi", "django", "flask", "express", "nestjs", "spring", "rails",
        "react", "vue", "angular", "nextjs", "nuxt",
        # Databases
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "dynamodb", "cassandra", "neo4j", "milvus", "pinecone", "pgvector",
        # Cloud
        "aws", "gcp", "azure", "kubernetes", "docker", "terraform",
        # AI/ML
        "machine learning", "deep learning", "tensorflow", "pytorch", "langchain",
        "openai", "llm", "rag", "embeddings", "vector database",
        # Other
        "graphql", "rest api", "microservices", "kafka", "rabbitmq",
    ]
    
    jd_lower = jd_text.lower() if jd_text else ""
    found_in_jd = [tech for tech in tech_keywords if tech in jd_lower]
    
    # Prioritize overlapping skills (candidate has + JD requires)
    candidate_lower = [s.lower() for s in candidate_stack] if candidate_stack else []
    overlapping = [tech for tech in found_in_jd if any(tech in c for c in candidate_lower)]
    
    # Return overlapping first, then JD requirements
    result = overlapping + [t for t in found_in_jd if t not in overlapping]
    return result[:8]  # Limit to 8 key techs


def build_interview_prompt(
    candidate: CandidateData,
    job: JobData,
    avatar_name: str = "Zara",
    questions: Optional[InterviewQuestions] = None,
) -> Dict[str, str]:
    """
    Build a conversational, realistic interview prompt.
    
    Returns:
        Dict with 'system_prompt' and 'greeting' keys
    """
    
    # Extract info
    candidate_name = candidate.name or "there"
    first_name = candidate_name.split()[0] if candidate_name else "there"
    candidate_skills = candidate.top_5_skills_with_years or "Not specified"
    candidate_summary = candidate.semantic_summary or ""
    candidate_tech_stack = candidate.current_tech_stack or []
    
    job_title = job.title or "the position"
    job_company = job.company or "the company"
    jd_text = job.jd_text or ""
    
    # Detect role level and extract tech focus
    role_level = detect_role_level(job_title, jd_text)
    tech_focus = extract_tech_stack(jd_text, candidate_tech_stack)
    
    # Format employment for context
    employment_context = ""
    if candidate.employment_history:
        recent = candidate.employment_history[0] if candidate.employment_history else {}
        if recent:
            employment_context = f"Currently/Recently: {recent.get('title', '')} at {recent.get('company', '')}"
    
    # Truncate JD if needed
    if len(jd_text) > 1500:
        jd_text = jd_text[:1500] + "..."
    
    # Build level-specific instructions
    level_instructions = get_level_instructions(role_level, tech_focus)
    
    system_prompt = f"""You are {avatar_name}, a friendly and experienced technical recruiter having a casual but professional screening conversation. You work for RCRUTR AI and you're talking with {first_name} about the {job_title} role at {job_company}.

## CANDIDATE CONTEXT
- Name: {candidate_name}
- Skills: {candidate_skills}
- Tech Stack: {", ".join(candidate_tech_stack) if candidate_tech_stack else "See resume"}
{f"- Background: {employment_context}" if employment_context else ""}

## ROLE CONTEXT
- Position: {job_title}
- Company: {job_company}
- Level: {role_level.upper()}
- Key Technologies: {", ".join(tech_focus) if tech_focus else "General"}

## JOB REQUIREMENTS
{jd_text}

## YOUR CONVERSATION STYLE
You're NOT reading from a script. You're having a real conversation like an experienced recruiter would:

1. **Be naturally curious** - React to what they say. "Oh interesting, so you used FastAPI... which version? Did you go with the async approach?"

2. **Follow the thread** - If they mention something interesting, dig deeper. "You mentioned you used PostgreSQL - any particular reason you chose that over MongoDB for this project?"

3. **Make connections** - "I see you've worked with vector databases... we're actually using Milvus here. Have you had any experience with that specifically?"

4. **Be conversational** - Use natural transitions like "That makes sense...", "Oh nice!", "Interesting...", "Got it, so..."

5. **Don't interrogate** - Avoid rapid-fire questions. Let them talk, then follow up naturally.

{level_instructions}

## QUESTION FLOW (Natural Conversation)

**Opening (Quick):**
- Confirm their name casually
- Quick work authorization check
- Brief role overview

**Technical Discussion (Main Focus - 10-12 mins):**
Have a real technical conversation based on their background and the role requirements.
- Ask about specific technologies they've used
- Dig into HOW they used them, not just IF
- Ask about decisions they made and why
- Explore challenges and how they solved them
- Connect their experience to what we need

**Example Natural Flow:**
"I see you've got Python and FastAPI on your resume... nice! Which version of FastAPI did you use?... Ah cool, and did you go with sync or async?... What made you choose async?... Makes sense. And for the database, I see PostgreSQL - was there a specific reason you went with Postgres over something like MongoDB?... Oh interesting, so you needed the relational features... Have you worked with any of the PostgreSQL extensions like pgvector?... No worries if not, just curious since we use vector databases here..."

**Closing (2 mins):**
- Ask if they have questions
- Thank them warmly
- Mention next steps

## IMPORTANT RULES
- **NEVER** read questions like a script
- **NEVER** ask "What challenges did you face?" without context - instead: "So with that FastAPI migration, what was the trickiest part?"
- **NEVER** make up information about the candidate
- **NEVER** promise anything about the job
- Keep total conversation to 15-20 minutes
- If they go off-topic, gently redirect: "That's interesting! So going back to your Python experience..."

## ENDING
After a good conversation, wrap up naturally:
"Well {first_name}, this has been a great chat! I've got a good sense of your background. The team will review our conversation and someone will reach out about next steps. Do you have any questions for me about the role or the company?"

Then: "Great questions! Alright, thanks so much for your time today. Talk soon!"
"""

    # Casual but professional greeting
    greeting = f"Hey {first_name}! I'm {avatar_name}. Thanks for jumping on - I'm excited to chat with you about the {job_title} role. This should be pretty casual, just want to learn more about your background and see if it's a good fit. Ready to dive in?"
    
    return {
        "system_prompt": system_prompt,
        "greeting": greeting,
    }


def get_level_instructions(level: str, tech_focus: List[str]) -> str:
    """Get level-specific interview instructions."""
    
    tech_list = ", ".join(tech_focus[:5]) if tech_focus else "relevant technologies"
    
    if level == "junior":
        return f"""
## JUNIOR/FRESHER LEVEL APPROACH
Since this is a junior role, focus on:

**Foundational Knowledge:**
- "So tell me about a project you've worked on... what technologies did you use?"
- "What made you interested in {tech_list}?"
- "How do you typically approach learning a new technology?"

**Basic Technical Understanding:**
- "Can you explain what an API does in simple terms?"
- "What's the difference between SQL and NoSQL databases?"
- "Have you worked with version control like Git?"

**Potential & Attitude:**
- "What kind of projects are you most excited to work on?"
- "How do you handle getting stuck on a problem?"

**Keep it encouraging** - They're early in their career. Focus on potential, not gaps.
"""
    
    elif level == "senior":
        return f"""
## SENIOR LEVEL APPROACH
For senior candidates, dig into architecture and leadership:

**System Design Thinking:**
- "Tell me about a system you architected from scratch... what were the key decisions?"
- "How did you handle scale? What would you do differently now?"
- "When choosing between {tech_list}, what factors do you consider?"

**Technical Depth:**
- "What's your philosophy on database design for high-traffic systems?"
- "How do you approach performance optimization? Walk me through a real example."
- "Tell me about a time you had to debug a really nasty production issue."

**Leadership & Mentorship:**
- "How do you approach code reviews?"
- "Tell me about mentoring junior developers."
- "How do you make technical decisions when the team disagrees?"

**Big Picture:**
- "Where do you see this technology going in the next few years?"
- "What's something you've changed your mind about technically?"

**Be intellectually curious** - Senior engineers enjoy deep technical discussions.
"""
    
    else:  # mid-level
        return f"""
## MID-LEVEL APPROACH
Focus on practical experience and real-world problem solving:

**Hands-on Experience:**
- "I see you've used {tech_list}... tell me about a project where you used [specific tech]"
- "Which version did you use? Did you like it?"
- "What made you choose [tech A] over [tech B] for that project?"

**Problem Solving:**
- "What was the trickiest bug you've dealt with? How did you track it down?"
- "Tell me about a time you had to optimize something for performance"
- "Have you had to integrate different systems? What was that like?"

**Technical Decisions:**
- "When would you use PostgreSQL vs MongoDB?"
- "What's your take on microservices vs monolith?"
- "How do you decide when to use async programming?"

**Collaboration:**
- "How do you handle disagreements about technical approaches?"
- "Tell me about working with a difficult codebase"

**Connect to our stack** - "We use {tech_list} here - what's your experience been with those?"
"""


def generate_interview_questions(
    candidate: CandidateData,
    job: JobData,
) -> InterviewQuestions:
    """
    Generate interview questions based on candidate and job data.
    These serve as a guide for the AI - the actual conversation will be more natural.
    """
    
    candidate_name = candidate.name or "there"
    first_name = candidate_name.split()[0]
    job_title = job.title or "the position"
    jd_text = job.jd_text or ""
    
    role_level = detect_role_level(job_title, jd_text)
    tech_focus = extract_tech_stack(jd_text, candidate.current_tech_stack or [])
    
    # Basic questions (streamlined)
    basic_questions = [
        InterviewQuestion(
            index=1,
            question_type="basic",
            question=f"Quick confirmation - you're {candidate_name}, right? Just making sure I've got the right person!",
            expected_keywords=None,
        ),
        InterviewQuestion(
            index=2,
            question_type="basic", 
            question="And just to check - are you authorized to work in the US? Citizen, green card, visa?",
            expected_keywords=["citizen", "green card", "h1b", "h4", "ead", "opt", "authorized"],
        ),
    ]
    
    # Technical questions based on level
    technical_questions = generate_technical_questions(
        role_level=role_level,
        tech_focus=tech_focus,
        candidate_stack=candidate.current_tech_stack or [],
        job_title=job_title,
    )
    
    total_questions = len(basic_questions) + len(technical_questions)
    return InterviewQuestions(
        total=total_questions,
        basic_questions=basic_questions,
        technical_questions=technical_questions,
    )


def generate_technical_questions(
    role_level: str,
    tech_focus: List[str],
    candidate_stack: List[str],
    job_title: str,
) -> List[InterviewQuestion]:
    """Generate level-appropriate technical questions."""
    
    questions = []
    idx = 3  # Start after basic questions
    
    # Find overlapping technologies
    candidate_lower = [s.lower() for s in candidate_stack]
    overlapping = [t for t in tech_focus if t.lower() in " ".join(candidate_lower).lower()]
    
    primary_tech = overlapping[0] if overlapping else (tech_focus[0] if tech_focus else "your tech stack")
    secondary_tech = overlapping[1] if len(overlapping) > 1 else (tech_focus[1] if len(tech_focus) > 1 else None)
    
    if role_level == "junior":
        questions = [
            InterviewQuestion(
                index=idx,
                question_type="technical",
                question=f"Tell me about a project you've worked on recently - what did you build and what technologies did you use?",
                expected_keywords=tech_focus[:3] if tech_focus else None,
            ),
            InterviewQuestion(
                index=idx + 1,
                question_type="technical",
                question=f"What got you interested in {primary_tech}? How did you learn it?",
                expected_keywords=[primary_tech] if primary_tech else None,
            ),
            InterviewQuestion(
                index=idx + 2,
                question_type="technical",
                question="When you get stuck on a coding problem, what's your process for figuring it out?",
                expected_keywords=None,
            ),
            InterviewQuestion(
                index=idx + 3,
                question_type="technical",
                question="What kind of projects would you be most excited to work on here?",
                expected_keywords=None,
            ),
        ]
    
    elif role_level == "senior":
        questions = [
            InterviewQuestion(
                index=idx,
                question_type="technical",
                question=f"Tell me about a system you architected - what were the key design decisions you made?",
                expected_keywords=["architecture", "design", "scale", "decision"],
            ),
            InterviewQuestion(
                index=idx + 1,
                question_type="technical",
                question=f"I see you've worked with {primary_tech}. How did you approach scaling it? What would you do differently now?",
                expected_keywords=[primary_tech, "scale", "performance"],
            ),
            InterviewQuestion(
                index=idx + 2,
                question_type="technical",
                question="Walk me through debugging a really nasty production issue you've dealt with.",
                expected_keywords=["debug", "production", "logs", "monitoring"],
            ),
            InterviewQuestion(
                index=idx + 3,
                question_type="technical",
                question="How do you approach mentoring junior developers on your team?",
                expected_keywords=["mentor", "code review", "teach", "guide"],
            ),
            InterviewQuestion(
                index=idx + 4,
                question_type="technical",
                question=f"What's your philosophy on choosing between different database technologies for a new project?",
                expected_keywords=["database", "sql", "nosql", "tradeoff"],
            ),
        ]
    
    else:  # mid-level
        db_question = "PostgreSQL" if "postgres" in " ".join(tech_focus).lower() else "your database choice"
        
        questions = [
            InterviewQuestion(
                index=idx,
                question_type="technical",
                question=f"I see {primary_tech} on your resume - tell me about a project where you used it. Which version? What did you build?",
                expected_keywords=[primary_tech],
            ),
            InterviewQuestion(
                index=idx + 1,
                question_type="technical",
                question=f"What made you choose {primary_tech} for that project? Were there other options you considered?",
                expected_keywords=[primary_tech, "chose", "decision", "because"],
            ),
            InterviewQuestion(
                index=idx + 2,
                question_type="technical",
                question=f"For the database, I see you've used {db_question} - any particular reason you went with that over alternatives?",
                expected_keywords=["database", "postgres", "mysql", "mongo", "reason"],
            ),
            InterviewQuestion(
                index=idx + 3,
                question_type="technical",
                question="Tell me about a tricky bug or performance issue you had to solve. How did you approach it?",
                expected_keywords=["debug", "performance", "fix", "solution"],
            ),
            InterviewQuestion(
                index=idx + 4,
                question_type="technical",
                question=f"We use async programming quite a bit here - what's been your experience with async? When do you think it makes sense to use it?",
                expected_keywords=["async", "await", "concurrent", "performance"],
            ),
            InterviewQuestion(
                index=idx + 5,
                question_type="technical",
                question=f"Have you worked with any vector databases or embedding systems? Things like pgvector, Milvus, Pinecone?",
                expected_keywords=["vector", "embedding", "similarity", "search"],
            ),
        ]
    
    return questions


def format_questions_for_db(questions: InterviewQuestions) -> List[Dict[str, Any]]:
    """Format questions for database storage."""
    result = []
    
    for q in questions.basic_questions:
        result.append({
            "index": q.index,
            "type": q.question_type,
            "question": q.question,
            "expected_keywords": q.expected_keywords,
        })
    
    for q in questions.technical_questions:
        result.append({
            "index": q.index,
            "type": q.question_type,
            "question": q.question,
            "expected_keywords": q.expected_keywords,
        })
    
    return result
