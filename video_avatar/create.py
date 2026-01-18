import argparse
import json
import os
from typing import Optional

import requests
from dotenv import load_dotenv

API_URL = "https://api.bey.dev/v1"
EGE_STOCK_AVATAR_ID = "70b1b917-ed16-4531-bb6c-b0bdb79449b4"

# Bey enforces a max system_prompt length of 10,000 chars.
# Keep this compact but strict.
SYSTEM_PROMPT_TEMPLATE = """
You are Ava, a confident, friendly, senior AI recruiter for Elite Solutions.
You are on a live video call with a vendor in a busy environment.

────────────────────────────────
ABSOLUTE SPEECH RULES
────────────────────────────────
• Speak ONLY like a human recruiter.
• NEVER read, explain, describe, or reference JSON, schemas, fields, enums, or systems.
• NEVER say anything technical (no “saving”, “system”, “database”, “fields”).
• The structured output at the end is SILENT and MACHINE-ONLY.
• Do NOT announce or explain the final JSON.
• Do NOT repeat everything the vendor says back to them.
• Do NOT over-summarize during the call.

────────────────────────────────
CONVERSATIONAL STYLE
────────────────────────────────
• Natural, confident, recruiter-like.
• Short acknowledgments only (“Got it.” “Makes sense.” “Okay.”).
• One focused question at a time.
• Keep momentum — no interrogation.
• Sound adaptive, not scripted.
• If something is unclear but non-critical, let it go.

────────────────────────────────
PRIMARY OBJECTIVE
────────────────────────────────
Collect complete and accurate hiring requirements
for ONE OR MORE roles through a natural conversation.

You are NOT reading a checklist.
You are guiding the conversation like an experienced recruiter.
────────────────────────────────
PERSONALITY MODES (RUNTIME CONTROL)
────────────────────────────────
AVA_MODE is one of: fast | detailed | executive

Mode rules:
• fast:
  - Ask only MVP questions.
  - No optional questions unless vendor offers info.
  - Aim for quickest clean intake.

• detailed:
  - Ask MVP + 1 layer deeper on:
    interview process, timeline/urgency, conversion (if CTH), and 1–2 nice-to-haves.
  - If compensation missing, ask once (unless vendor clearly doesn’t know).

• executive:
  - Ask for deal-breakers early:
    “What would make someone an immediate no?”
  - Focus on outcomes, ownership, and evaluation criteria.
  - Minimal back-and-forth; crisp transitions and recap.

────────────────────────────────
INTELLIGENT QUESTIONING LOGIC
────────────────────────────────
Ask questions BASED ON CONTEXT, not blindly.

Examples:
• If role is Full-time → ask about:
  - salary range (if not given)
  - bonus, annual hike, equity, benefits (only if natural)
• If role is Contract → ask about:
  - hourly rate
  - duration
  - extension or conversion
• If hybrid or onsite → clarify location.
• If remote → do NOT ask for city/state.
• If senior role → clarify leadership or ownership expectations.
• If junior/mid → do NOT push management questions.
• If vendor sounds unsure → accept and move on.

Never ask questions just to fill fields.
Only ask when it improves candidate quality.

────────────────────────────────
MULTI-ROLE HANDLING
────────────────────────────────
If the vendor introduces another role, transition smoothly:
• “Alright, let’s move on to the next role.”
• “Got it — let’s talk about the second opening.”

Each role is treated independently unless explicitly shared.

────────────────────────────────
AUTO-DETECT: SHARED DETAILS
────────────────────────────────
If the vendor clearly says:
• “Same for all roles”
• “Applies to both”
• “Everything else is the same”

Apply those details across roles silently.

If they imply sameness (“same location”, “same rate”):
→ Copy ONLY that specific item.

If unclear, ask once:
“Just confirming — is that the same as the previous role?”

────────────────────────────────
CONFIDENCE-BASED FOLLOW-UPS
────────────────────────────────
Ask follow-ups ONLY if:
• Answers are vague or incomplete
• Pay, location, seniority, or work authorization is unclear
• Clarification meaningfully improves submissions

Do NOT ask if:
• Vendor sounds confident
• They say they don’t know
• It’s non-critical

Never invent or assume details.
────────────────────────────────
INTERNAL CONFIDENCE SCORING (SILENT)
────────────────────────────────
Maintain a silent confidence score per role for these categories:
• title_seniority
• job_type_employment
• location_work_model
• compensation
• must_have_skills
• primary_technologies
• work_authorization
• interview_process

Score each category:
high = clear + specific
medium = mostly clear, minor ambiguity
low = vague, unknown, or conflicting

Rules:
• Do NOT speak the score.
• Use the score to decide follow-ups:
  - If a category is low AND it materially affects submissions, ask ONE clarifying question.
  - If low but non-critical (e.g., benefits details), accept and move on.
• If the vendor gives conflicting info, downgrade to low and ask one fix question:
  “Just to confirm — which one should we go with?”
────────────────────────────────
REAL-TIME ROLE COMPLETENESS (SILENT)
────────────────────────────────
Track role completeness continuously.

A role is “complete enough” when all MVP items are captured
AND any job-type-specific compensation item is captured or explicitly unknown.

MVP items:
• job_title
• job_type
• location + work_model
• must_have_skills
• primary_technologies
• work_authorization

Job-type-specific:
• Contract/CTH: pay rate (or explicitly unknown)
• Full-time: salary range (or explicitly unknown)

Behavior:
• Once complete enough, stop asking detail questions.
• Transition to the soft-close question (ask ONCE).
• Then do a short recap and close.

────────────────────────────────
SOFT-CLOSE: CHECK FOR MORE ROLES
────────────────────────────────
Once a role feels complete, ask ONCE:
• “Do you have any other roles we should cover today?”

If no → move to recap and close.
If yes → transition naturally.

────────────────────────────────
ENDING THE CALL (SPOKEN)
────────────────────────────────
• Give a SHORT, HIGH-LEVEL recap (not every detail).
• Confirm alignment (“That all sounds good.”).
• Thank the vendor.
• Close naturally.

Example tone:
“Perfect — I’ve got everything I need. Thanks for walking me through that.”

────────────────────────────────
FINAL OUTPUT RULE (CRITICAL)
────────────────────────────────
After the conversation is complete:
• Output ONLY valid JSON.
• No markdown.
• No commentary.
• No explanations.
• If multiple roles → output a JSON ARRAY directly.
• If one role → output a single JSON object.
• The JSON is NOT spoken aloud, but MUST be included in text.

Use DB-safe defaults where information was not provided.
Never fabricate values.

[JSON SCHEMA FOLLOWS — MACHINE ONLY]
""".strip()


def _load_memory_text(path: str, max_chars: int = 1200) -> str:
    """Load memory JSON, keeping it small to satisfy Bey 10k prompt limit."""
    if not path or not os.path.exists(path):
        return "[]"
    raw_txt = open(path, "r", encoding="utf-8").read()
    try:
        raw = json.loads(raw_txt)
        txt = json.dumps(raw, ensure_ascii=False)
    except Exception:
        txt = raw_txt
    if len(txt) <= max_chars:
        return txt
    return txt[: max_chars - 3] + "..."


def _build_system_prompt(role_name: str, role_description: str, candidate_name: str, memory_path: Optional[str]) -> str:
    memory = _load_memory_text(memory_path or "memory.json", max_chars=1200)

    def render(mem: str) -> str:
        # Don't use .format() due to JSON braces in prompt.
        return (
            SYSTEM_PROMPT_TEMPLATE
            .replace("{memory}", mem)
            .replace("{role_name}", role_name)
            .replace("{role_description}", role_description)
            .replace("{candidate_name}", candidate_name)
        )

    prompt = render(memory)
    if len(prompt) > 9900:
        prompt = render(_load_memory_text(memory_path or "memory.json", max_chars=600))
    if len(prompt) > 10000:
        prompt = render("[]")
    return prompt


def _parse_tags(raw_tags: Optional[str]) -> Optional[dict]:
    if not raw_tags:
        return None
    try:
        tags = json.loads(raw_tags)
    except json.JSONDecodeError as exc:
        raise ValueError("tags must be valid JSON (object)") from exc
    if not isinstance(tags, dict):
        raise ValueError("tags must be a JSON object")
    return tags


def _create_agent(
    api_key: str,
    *,
    role_name: str,
    role_description: str,
    candidate_name: str,
    avatar_id: Optional[str],
    memory_file: Optional[str],
) -> dict:
    system_prompt = _build_system_prompt(role_name, role_description, candidate_name, memory_file)

    resp = requests.post(
        f"{API_URL}/agent",
        headers={"x-api-key": api_key},
        json={
            "name": f"{role_name} Recruiter for {candidate_name}",
            "system_prompt": system_prompt,
            "greeting": (
                f"Hi {candidate_name}! I’m Ava with Elite Solutions. "
                "What new opening(s) are we covering today?"
            ),
            "avatar_id": avatar_id if avatar_id else EGE_STOCK_AVATAR_ID,
        },
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        print(f"Error creating agent: {resp.status_code} - {resp.text}")
        raise SystemExit(1)

    agent = resp.json()
    agent["system_prompt_length"] = len(system_prompt)
    return agent


def _create_call(
    api_key: str,
    *,
    agent_id: str,
    livekit_username: str,
    tags: Optional[dict],
) -> dict:
    payload: dict = {"agent_id": agent_id, "livekit_username": livekit_username}
    if tags:
        payload["tags"] = tags

    resp = requests.post(
        f"{API_URL}/calls",
        headers={"x-api-key": api_key},
        json=payload,
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        print(f"Error creating call: {resp.status_code} - {resp.text}")
        raise SystemExit(1)

    return resp.json()


def _fetch_call_messages(api_key: str, call_id: str) -> dict | list:
    resp = requests.get(
        f"{API_URL}/calls/{call_id}/messages",
        headers={"x-api-key": api_key},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"Error fetching messages: {resp.status_code} - {resp.text}")
        raise SystemExit(1)
    return resp.json()


def main(
    api_key: str,
    role_name: Optional[str],
    role_description: Optional[str],
    candidate_name: Optional[str],
    avatar_id: Optional[str],
    memory_file: Optional[str],
    livekit_username: str,
    tags: Optional[dict],
    fetch_messages: bool,
    call_id: Optional[str],
) -> None:
    if fetch_messages:
        if not call_id:
            raise ValueError("--call-id is required when using --fetch-messages")
        messages = _fetch_call_messages(api_key, call_id)
        print(json.dumps(messages, indent=2))
        return

    if not role_name or not role_description or not candidate_name:
        raise ValueError("--role-name, --role-description, and --candidate-name are required")

    agent = _create_agent(
        api_key,
        role_name=role_name,
        role_description=role_description,
        candidate_name=candidate_name,
        avatar_id=avatar_id,
        memory_file=memory_file,
    )

    agent_id = agent.get("id")
    if not agent_id:
        raise SystemExit("Agent created but no agent id was returned.")

    call_tags = dict(tags or {})
    call_tags.setdefault("agent_id", agent_id)
    call = _create_call(
        api_key,
        agent_id=agent_id,
        livekit_username=livekit_username,
        tags=call_tags,
    )

    payload = {"agent": agent, "call": call}
    static_path = os.path.join(os.path.dirname(__file__), "static", "latest_call.json")
    try:
        with open(static_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        print(f"Warning: failed to write {static_path}: {exc}")

    print(json.dumps(agent, indent=2))
    print(json.dumps(call, indent=2))
    print(f"System prompt length: {agent.get('system_prompt_length')} chars")
    if call.get("id"):
        print(f"\nCall ID: {call['id']}")
        print("Use LiveKit credentials from the call payload to join the session.")
        print("To fetch messages later, run:")
        print(f"  python create.py --fetch-messages --call-id {call['id']}")
        print(f"Saved LiveKit payload to: {static_path}")


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("BEY_API_KEY")
    if not api_key:
        raise ValueError("Please set BEY_API_KEY in .env")

    parser = argparse.ArgumentParser(description="Create Bey Ava Vendor Intake agent (prompt <= 10k)")
    parser.add_argument("--role-name")
    parser.add_argument("--role-description")
    parser.add_argument("--candidate-name")
    parser.add_argument("--avatar-id", default=EGE_STOCK_AVATAR_ID)
    parser.add_argument("--memory-file", default=None)
    parser.add_argument("--livekit-username", default="User")
    parser.add_argument(
        "--tags",
        default=None,
        help='Optional JSON object for call tags, e.g. \'{"session_id":"abc"}\'',
    )
    parser.add_argument(
        "--fetch-messages",
        action="store_true",
        help="Fetch messages for a call id and exit.",
    )
    parser.add_argument("--call-id", default=None, help="Call ID to fetch messages for.")
    args = parser.parse_args()

    main(
        api_key=api_key,
        role_name=args.role_name,
        role_description=args.role_description,
        candidate_name=args.candidate_name,
        avatar_id=args.avatar_id,
        memory_file=args.memory_file,
        livekit_username=args.livekit_username,
        tags=_parse_tags(args.tags),
        fetch_messages=args.fetch_messages,
        call_id=args.call_id,
    )
