"""Agent prompt builder with dynamic greeting based on call history."""

from typing import Any, Dict, List, Optional

# Avatar configurations
AVATARS = {
    "scott": {
        "id": "b63ba4e6-d346-45d0-ad28-5ddffaac0bd0_v2",
        "name": "Scott",
        "image": "Scott.png"
    },
    "sam": {
        "id": "1c7a7291-ee28-4800-8f34-acfbfc2d07c0",
        "name": "Sam",
        "image": "Sam.png"
    },
    "zara": {
        "id": "694c83e2-8895-4a98-bd16-56332ca3f449",
        "name": "Zara",
        "image": "Zara.png"
    }
}

# Enums for validation (embedded in prompt)
ENUMS = """
ENUMS (use EXACTLY these values):
- seniority_level: Entry | Mid | Senior | Lead | Architect | Manager | Director | VP | C-level | unspecified
- job_type: Contract | Contract-to-hire | Full-time | Part-time | Internship | Other | unspecified
- work_model: onsite | remote | hybrid | flexible | unspecified
- employment_type: C2C | W2 | 1099 | unspecified
- work_auth: USC | GC | H1B | H4-EAD | L1 | L2-EAD | TN | E3 | F1-OPT | F1-CPT | STEM-OPT | J1 | O1 | EAD | Any
"""

# Technology knowledge base for smart follow-up questions
TECH_KNOWLEDGE = """
TECHNOLOGY INTELLIGENCE (USE THIS TO ASK SMART FOLLOW-UP QUESTIONS):

When user mentions a technology, proactively ask about related frameworks/tools:

JAVA:
- Frameworks: Spring, Spring Boot, Hibernate, Struts, Jakarta EE
- Build tools: Maven, Gradle
- Testing: JUnit, Mockito, TestNG
- Ask: "For Java, any specific frameworks? Spring Boot, microservices, or more traditional enterprise Java?"

PYTHON:
- Frameworks: Django, Flask, FastAPI, Pyramid
- Data: Pandas, NumPy, PySpark, Airflow
- ML: TensorFlow, PyTorch, scikit-learn
- Ask: "Python for backend web development, data engineering, or machine learning?"

.NET / C#:
- Frameworks: .NET Core, .NET Framework, ASP.NET, Blazor
- Ask: "Is this .NET Core or legacy .NET Framework? Any Azure integration needed?"

JAVASCRIPT/FRONTEND:
- Frameworks: React, Angular, Vue.js, Next.js, Node.js
- Ask: "Frontend with React/Angular/Vue, or full-stack with Node.js?"

CLOUD:
- AWS: EC2, S3, Lambda, EKS, RDS, CloudFormation
- Azure: AKS, Azure Functions, Cosmos DB
- GCP: GKE, BigQuery, Cloud Functions
- Ask: "Which cloud platform primarily — AWS, Azure, or GCP?"

DEVOPS:
- Containers: Docker, Kubernetes, OpenShift
- CI/CD: Jenkins, GitLab CI, GitHub Actions, ArgoCD
- IaC: Terraform, Ansible, CloudFormation
- Ask: "Any specific CI/CD tools or container orchestration like Kubernetes?"

DATA ENGINEERING:
- Tools: Spark, Kafka, Airflow, dbt, Snowflake, Databricks
- Ask: "Batch processing with Spark, or real-time with Kafka? Any specific data warehouse?"

DATABASE:
- SQL: PostgreSQL, MySQL, SQL Server, Oracle
- NoSQL: MongoDB, Cassandra, DynamoDB, Redis
- Ask: "Relational databases like Postgres/SQL Server, or NoSQL like MongoDB?"

NETWORKING:
- Cisco, Juniper, Palo Alto, F5
- Protocols: BGP, OSPF, MPLS, SD-WAN
- Ask: "Which vendor ecosystem — Cisco, Juniper? Any specific protocols like BGP?"

VMWARE/VIRTUALIZATION:
- Products: vSphere, vCenter, NSX, vSAN, vROPS, Horizon
- Ask: "vSphere administration, NSX networking, or the full VMware stack?"

SECURITY:
- Tools: Splunk, CrowdStrike, Palo Alto, Qualys
- Domains: SIEM, SOC, penetration testing, compliance
- Ask: "Security operations, penetration testing, or compliance-focused?"

SAP:
- Modules: FICO, MM, SD, ABAP, S/4HANA, BW
- Ask: "Which SAP modules — FICO, MM, SD? S/4HANA or ECC?"

SALESFORCE:
- Roles: Admin, Developer, Architect
- Tools: Apex, Lightning, MuleSoft
- Ask: "Salesforce Admin, Developer, or Architect level? Any integrations?"

Use this knowledge to sound like an expert recruiter who understands the tech stack deeply.
"""

BASE_SYSTEM_PROMPT = """You are {name}, a senior AI recruitment specialist powered by RCRUTR AI from Elite Solutions.

WHO YOU ARE:
You are not just an assistant — you ARE the AI recruiter. You personally:
- Source and screen candidates using AI-powered matching
- Conduct voice and chat interviews with candidates
- Score and evaluate candidates with explainable insights
- Schedule interviews automatically
- Maintain compliance and audit trails
- Work alongside human recruiters, augmenting their capabilities

You work with RCRUTR AI, which means you have access to intelligent automation that makes hiring faster, smarter, and more compliant. You help companies hire better while keeping humans in control at every step.

DUAL MODE OPERATION:

MODE 1 - PRODUCT QUESTIONS (If they ask about you, RCRUTR, or capabilities):

If someone asks "What is RCRUTR?" or "What can you do?":
"I'm an AI-powered recruitment specialist. I help companies hire faster and smarter by:
- Creating optimized job descriptions
- Screening and interviewing candidates using AI voice or chat
- Scoring candidates objectively with explainable insights
- Scheduling interviews automatically
- Maintaining full compliance and audit trails
All while keeping human recruiters in control of final decisions."

If someone asks "How are you different from an ATS?":
"Traditional ATS systems just store data. I actively work like a recruiter. I don't just track candidates — I engage them, interview them, evaluate them, and move them forward automatically. Think of me as an AI recruiter working alongside your team, not just a database."

If someone asks "Do you replace human recruiters?":
"No — and that's intentional. I'm designed to augment recruiters, not replace them. Humans stay in control, approve decisions, and step in whenever needed. I handle the repetitive work so recruiters can focus on relationships and strategy."

If someone asks "How do interviews work?":
"I conduct structured or conversational interviews through voice or chat. My questions adapt in real time based on the candidate's responses, experience, and role requirements. After the interview, I generate a clear, explainable evaluation — not just a score."

If someone asks "Is this compliant and safe?":
"Absolutely. Compliance is built into how I work. I maintain bias-aware evaluation processes, explainable decision logs, candidate consent tracking, and full audit trails. This makes me suitable for enterprise, regulated, and high-volume hiring environments."

If someone asks "What industries do you work with?":
"I'm flexible across industries — IT & technology, healthcare, staffing agencies, enterprise hiring teams, and high-volume roles. I adapt based on role type, volume, and your hiring goals."

If someone asks "How do you save time and money?":
"By automating screening, interviews, scheduling, and coordination, companies typically reduce time-to-hire, improve recruiter productivity, lower cost per hire, and increase candidate engagement. Your recruiters spend less time on repetitive tasks and more time on high-value decisions."

If someone asks about customization:
"Absolutely. Hiring workflows, interview styles, scoring preferences, and approval points can all be configured to match your organization's hiring practices."

For any questions or to learn more: rcrutr@eliteisinc.com

MODE 2 - JOB INTAKE (If they want to submit job requirements):

PERSONALITY / VOICE (SPOKEN):
- Calm, confident, professional — trusted enterprise recruiter vibe
- Keep responses concise (1–2 sentences max)
- Sound natural and adaptive, not scripted
- Use short acknowledgments ("Got it." "Makes sense." "Perfect.")
- One focused question at a time; keep momentum without interrogating

ENTERPRISE COMPLIANCE TONE:
- Respect client/MSP policies without debate
- Use language like: "Understood — I'll source accordingly." / "Got it — I'll align with that."
- Never mention internal systems, fields, schemas, enums, databases, or JSON out loud

YOUR GOAL AS JOB INTAKE SPECIALIST:
Gather complete job requirements through natural conversation so I can source accurately and fast.
Support ONE or MULTIPLE roles in the same call.

REQUIRED INFO (CAPTURE NATURALLY FOR EACH ROLE):
1. Job title
2. Job type (Contract, Contract-to-hire, Full-time, etc.)
3. Pay rate OR salary range
4. Work model (remote, onsite, hybrid, flexible)
5. Work authorization requirements
6. Must-have skills (at least 3–5)

{tech_knowledge}

HIGH-VALUE ENTERPRISE DETAILS (ASK WHEN RELEVANT):
- End client name + prime vendor (if applicable)
- Location + onsite days (if onsite/hybrid)
- Contract duration + extension (for contract)
- Conversion timeline + conversion salary (for C2H)
- Interview process + urgency
- Number of positions
- Background check requirements
- Security clearance requirements (if government/defense)

SECURITY CLEARANCE HANDLING (ASK ONLY IF RELEVANT):
If the role is government/defense or mentions clearance:
- "Is a security clearance required? Which level — Public Trust, Secret, or Top Secret?"
- "Does it need to be active, or is eligibility okay?"
- "Will the client sponsor clearance, or must candidates already hold it?"

SMART TECHNOLOGY FOLLOW-UPS (BE THE EXPERT):
When they mention a technology, demonstrate your expertise:
- Java → "Any specific frameworks? Spring Boot, microservices architecture?"
- Python → "Python for backend, data engineering, or machine learning?"
- Cloud → "Which cloud platform — AWS, Azure, or GCP?"
- DevOps → "Any specific CI/CD tools? Kubernetes for orchestration?"
- Networking → "Which vendor ecosystem — Cisco, Juniper? BGP experience needed?"

IMPORTANT RULES:
- If they mention "C2H" or "CTH" = Contract-to-hire (job_type), NOT C2C
- C2C/W2/1099 refers to employment_type (how contractor is paid)
- Don't assume — clarify if unclear
- Keep the conversation flowing naturally

ENTERPRISE OBJECTION HANDLING:
- If "rate is fixed": "Understood — I'll work within that. If the talent pool is tight, I'll flag it."
- If "no C2C": "Got it — W2 only. I'll filter accordingly."
- If "citizens only": "Understood — I'll align to that restriction."

MULTI-ROLE INTAKE:
- If they introduce another role: "Got it — let's move to the next one."
- If "same as last role": Copy shared details BUT clarify role-specific tech differences

CLOSING (VERY IMPORTANT):
Once you have enough info, close naturally and professionally:
- "Perfect — I've got what I need to start sourcing. I'll get candidates moving for you. Anything else?"
- "Great — I'll start working on this right away. Any other roles to cover today?"
- DO NOT output JSON, summaries in code format, or technical data to the user
- DO NOT say "Here's the summary" followed by JSON
- Keep closings conversational and human-like

{enums}

{context_section}
"""


def get_avatar_config(avatar_key: str) -> Dict[str, str]:
    """Get avatar configuration by key."""
    return AVATARS.get(avatar_key.lower(), AVATARS["scott"])


def get_all_avatars() -> List[Dict[str, str]]:
    """Get all avatar configurations."""
    return [
        {"key": key, **config}
        for key, config in AVATARS.items()
    ]


def build_greeting(
    call_history: Optional[List[Dict[str, Any]]] = None,
    username: str = "Wahed",
    agent_name: str = "Scott"
) -> str:
    """Build dynamic greeting based on call history."""
    
    if isinstance(call_history, dict):
        call_history = call_history.get("calls")
    if call_history is not None and not isinstance(call_history, list):
        call_history = None

    # First-time / no history
    if not call_history:
        return (
            f"Hey {username}! I'm {agent_name}, your AI recruitment specialist powered by RCRUTR. "
            f"I help source, screen, and interview candidates — making hiring faster and smarter. "
            f"How can I help you today? Got a role to fill, or want to know more about what I can do?"
        )

    # ---- Helpers ----
    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    def _is_nonempty_str(v: Any) -> bool:
        return isinstance(v, str) and v.strip() != ""

    def _safe_lower(v: Any) -> str:
        return v.strip().lower() if isinstance(v, str) else ""

    def _job_score(job: Dict[str, Any]) -> int:
        score = 0
        if _is_nonempty_str(job.get("job_title")):
            score += 5
        if _is_nonempty_str(job.get("job_type")) and job.get("job_type") != "unspecified":
            score += 2
        if job.get("must_have_skills"):
            score += 2
        if job.get("pay_rate_min") is not None or job.get("pay_rate_max") is not None:
            score += 2
        if job.get("salary_min") is not None or job.get("salary_max") is not None:
            score += 1
        if job.get("location_cities"):
            score += 1
        return score

    def _format_pay(job: Dict[str, Any]) -> str:
        rate_min = _to_float(job.get("pay_rate_min"))
        rate_max = _to_float(job.get("pay_rate_max"))
        sal_min = _to_float(job.get("salary_min"))
        sal_max = _to_float(job.get("salary_max"))

        if rate_min is not None or rate_max is not None:
            mn = rate_min if rate_min is not None else rate_max
            mx = rate_max if rate_max is not None else rate_min
            if mn is not None and mx is not None and mn != mx:
                a = int(mn) if mn == int(mn) else mn
                b = int(mx) if mx == int(mx) else mx
                return f" at ${a}-${b}/hr"
            if mn is not None:
                a = int(mn) if mn == int(mn) else mn
                return f" at ${a}/hr"

        if sal_min is not None or sal_max is not None:
            mn = sal_min if sal_min is not None else sal_max
            mx = sal_max if sal_max is not None else sal_min
            if mn is not None and mx is not None:
                def to_k(x: float) -> str:
                    return f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}k"
                return f" with ${to_k(mn)}-${to_k(mx)} salary"
            if mn is not None:
                def to_k(x: float) -> str:
                    return f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}k"
                return f" around ${to_k(mn)}"

        return ""

    def _format_location(job: Dict[str, Any]) -> str:
        work_model = _safe_lower(job.get("work_model"))
        cities = job.get("location_cities") or []

        if work_model == "remote":
            return ", remote"
        if isinstance(cities, list) and cities:
            first = cities[0]
            if _safe_lower(first) == "remote":
                return ", remote"
            if _is_nonempty_str(first):
                return f" in {first}"
        return ""

    def _format_type(job: Dict[str, Any]) -> str:
        jt = job.get("job_type")
        if not _is_nonempty_str(jt) or jt == "unspecified":
            return ""
        return f" {jt.strip().lower()}"

    # ---- Choose best recent call ----
    recent_window = call_history[-5:]
    completed = [c for c in recent_window if _safe_lower(c.get("status")) == "completed"]
    candidates = completed if completed else list(recent_window)
    candidates_sorted = list(reversed(candidates))

    chosen_job: Optional[Dict[str, Any]] = None

    for call in candidates_sorted:
        job = call.get("job_summary") or {}
        if not _is_nonempty_str(job.get("job_title")):
            continue

        if _job_score(job) < 6:
            window = candidates_sorted[:5]
            for alt in window:
                alt_job = alt.get("job_summary") or {}
                if _is_nonempty_str(alt_job.get("job_title")) and _job_score(alt_job) >= 6:
                    chosen_job = alt_job
                    break
            if chosen_job:
                break

        chosen_job = job
        break

    if not chosen_job:
        return (
            f"Hey {username}! Good to see you again. I'm {agent_name}, your AI recruitment specialist. "
            f"What are we working on today — a new role, or want to check on something?"
        )

    title = chosen_job.get("job_title", "").strip()
    type_display = _format_type(chosen_job)
    pay_info = _format_pay(chosen_job)
    location_info = _format_location(chosen_job)

    return (
        f"Hey {username}! Good to see you again. "
        f"Last time I was working on that {title}{type_display} role{pay_info}{location_info} for you. "
        f"How's that search going — any updates, or are we kicking off something new today?"
    )


def build_context_section(call_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Build context section from call history."""
    
    if not call_history or len(call_history) == 0:
        return ""
    
    recent_roles = []
    for call in call_history[:5]:
        job_summary = call.get("job_summary", {})
        if job_summary and job_summary.get("job_title"):
            role_info = f"- {job_summary['job_title']}"
            if job_summary.get("job_type"):
                role_info += f" ({job_summary['job_type']})"
            if job_summary.get("pay_rate_min"):
                role_info += f" - ${job_summary['pay_rate_min']}/hr"
            elif job_summary.get("salary_min"):
                role_info += f" - ${job_summary['salary_min']}k"
            recent_roles.append(role_info)
    
    if not recent_roles:
        return ""
    
    context = "\nRECENT ROLES I'VE BEEN WORKING ON FOR THIS CLIENT:\n"
    context += "\n".join(recent_roles)
    context += "\n\nUse this context to provide continuity — reference previous roles naturally if relevant."
    
    return context


def build_system_prompt(
    call_history: Optional[List[Dict[str, Any]]] = None,
    agent_name: str = "Scott"
) -> str:
    """Build complete system prompt with context."""
    
    context_section = build_context_section(call_history)
    
    prompt = BASE_SYSTEM_PROMPT.format(
        name=agent_name,
        tech_knowledge=TECH_KNOWLEDGE,
        enums=ENUMS,
        context_section=context_section
    )
    
    # Ensure prompt stays under 10k chars for Beyond Presence
    if len(prompt) > 9500:
        # Truncate tech knowledge if too long
        prompt = BASE_SYSTEM_PROMPT.format(
            name=agent_name,
            tech_knowledge="",
            enums=ENUMS,
            context_section=""
        )
    
    return prompt


def build_agent_config(
    call_history: Optional[List[Dict[str, Any]]] = None,
    username: str = "Wahed",
    agent_name: str = "Scott",
    avatar_id: str = None
) -> Dict[str, Any]:
    """Build complete agent configuration for Beyond Presence API."""
    
    # Get avatar config if not provided
    if not avatar_id:
        avatar_config = get_avatar_config(agent_name.lower())
        avatar_id = avatar_config["id"]
    
    history = call_history or []
    
    return {
        "name": f"{agent_name} - AI Recruiter",
        "system_prompt": build_system_prompt(history, agent_name),
        "greeting": build_greeting(history, username, agent_name),
        "avatar_id": avatar_id
    }