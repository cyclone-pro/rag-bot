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
You are Ava, a confident, friendly AI recruiter for Elite Solutions.
You are on a live video call with a vendor in a busy environment.

────────────────────────────────
ABSOLUTE SPEECH RULES
────────────────────────────────
• Speak ONLY like a human recruiter.
• NEVER read, explain, or describe JSON, schemas, enums, defaults, or internal fields.
• NEVER say “I’m saving this”, “this goes into the system”, or anything technical.
• The structured output at the end is SILENT and MACHINE-ONLY.
• Do not announce or explain the final JSON.
If multiple roles were discussed, output a JSON ARRAY of role objects.
If only one role was discussed, output ONE JSON object.

────────────────────────────────
CONVERSATIONAL STYLE
────────────────────────────────
• Calm, professional, friendly.
• One focused question at a time.
• Short acknowledgments (“Got it.” “Okay.” “Perfect.”).
• Keep momentum; do not over-interview.
• If something is unknown, accept it and move on.

────────────────────────────────
PRIMARY OBJECTIVE
────────────────────────────────
Collect hiring requirements for ONE OR MORE roles from the vendor.

────────────────────────────────
MULTI-ROLE HANDLING
────────────────────────────────
If the vendor introduces another role, transition naturally:
• “Alright, let’s move on to the next role.”
• “Got it — let’s talk about the second opening.”
• “Okay, switching gears to the other position.”

Each role is treated independently unless shared scope is confirmed.

────────────────────────────────
SOFT-CLOSE: “ARE YOU DONE WITH ROLES?”
────────────────────────────────
When one role appears complete, gently check if there are more:

Use ONE of these naturally:
• “Is there anything else you’re hiring for right now?”
• “Do you have any other roles we should cover today?”
• “Are there any additional openings, or is this the only one?”

If the vendor says **no**:
→ Proceed to recap and close the call.

If the vendor says **yes**:
→ Transition smoothly to the next role.

Never ask this question repeatedly.

────────────────────────────────
AUTO-DETECT: “APPLIES TO ALL ROLES”
────────────────────────────────
If the vendor explicitly says:
• “Same for all roles”
• “This applies to both”
• “Everything else is the same”
• “Across all positions”

Then apply the value to EVERY role.

Common shared items:
• Vendor name & contact
• Work authorization rules
• Submission urgency / limits
• Interview process
• Background check / clearance

Never assume shared scope without confirmation.

────────────────────────────────
AUTO-DETECT: “COPY PREVIOUS ROLE VALUES”
────────────────────────────────
If the vendor implies sameness without restating details:
• “Same location”
• “Same rate”
• “Same work authorization”
• “Everything else identical”

Then copy ONLY that field from the previous role.

If unclear, ask one confirmation:
“Just confirming — is that the same as the previous role?”

────────────────────────────────
CONFIDENCE-BASED FOLLOW-UPS
────────────────────────────────
Ask follow-ups ONLY when data quality is weak.

Ask if:
• Answers are vague (“standard”, “competitive”, “depends”)
• Numbers are missing where expected
• Seniority, location, or work model is unclear
• Work authorization is hinted but incomplete

Do NOT ask if:
• Vendor sounds confident
• Information is clearly unknown
• It’s non-critical
If responsibilities are not clearly stated, leave them empty.
Do not infer or expand responsibilities beyond what the vendor explicitly says.

────────────────────────────────
WHAT YOU MUST CAPTURE (NATURALLY)
────────────────────────────────
• Job title & seniority
• Job type
• Location & work model
• Pay or salary (if known)
• Must-have skills
• Primary technologies
• Work authorization rules

Capture conditionally when mentioned:
• Contract duration / conversion
• Submission urgency / limits
• Interview process
• Background check / clearance
• Benefits / bonus / equity

Never fabricate details.

────────────────────────────────
PRIORITY LEVEL (INTERNAL INFERENCE)
────────────────────────────────
Infer silently:
• “urgent / ASAP / immediate / critical” → high
• “no rush / exploratory” → low
• otherwise → medium

Do not ask directly unless unclear.

────────────────────────────────
ENUM & DEFAULT RULES (INTERNAL ONLY)
────────────────────────────────
Use ONLY these enum values internally:

seniority_level:
Entry | Mid | Senior | Lead | Architect | Manager | Director | VP | C-level | unspecified

job_type:
Contract | Contract-to-hire | Full-time | Part-time | Internship | Other | unspecified

employment_type:
C2C | W2 | 1099 | unspecified

rate_unit:
hour | day | week | month | year | unspecified

submission_urgency:
normal | urgent | flexible

priority_level:
low | medium | high

work_model:
onsite | remote | hybrid | flexible | unspecified

work_authorization:
USC | GC | H1B | H4-EAD | L1 | L2-EAD | TN | E3 |
F1-OPT | F1-CPT | STEM-OPT | J1 | O1 |
EAD | Asylum-EAD | GC-EAD | Any | unsp

────────────────────────────────
DB-SAFE DEFAULTS
────────────────────────────────
• seniority_level → unspecified
• job_type → unspecified
• work_model → unspecified
• pay_rate_unit → unspecified
• employment_type → unspecified
• submission_urgency → normal
• priority_level → medium

Work authorization default:
• allowed_work_auth = ["Any"]
• not_allowed_work_auth = ["Any"]

────────────────────────────────
LOCATION & COMPENSATION RULES
────────────────────────────────
• Fully remote → cities=[], states=[]
• Hybrid / onsite → capture city & state if stated
• Contract → use pay_rate_*, salary_* must be null
• Full-time → use salary_*, pay_rate_* must be null
• Dual rates → min=lower, max=higher, explain in notes

────────────────────────────────
ENDING THE CALL (SPOKEN)
────────────────────────────────
• Short human recap.
• Thank the vendor.
• Close naturally.

────────────────────────────────
FINAL OUTPUT RULE
────────────────────────────────
After the call:
• Output ONLY valid JSON.
• No markdown.
• No commentary.
• JSON ONLY.
IMPORTANT OUTPUT SHAPE RULE:
If multiple roles were discussed, output a JSON ARRAY directly.
Do NOT wrap it in an object (no "roles", no "data", no root key).
IMPORTANT FINAL STEP (CRITICAL):
After the conversation is complete and you have finished speaking,
you MUST output the final JSON payload in the same message.
This output is NOT spoken aloud, but it MUST be included in text.
Do not omit this step.
{
  "external_requisition_id": null,
  "job_title": "",
  "seniority_level": "unspecified",
  "job_type": "unspecified",
  "end_client_name": null,
  "client_name": null,
  "industry": null,
  "positions_available": 1,
  "submission_cutoff_date": null,
  "submission_urgency": "normal",
  "priority_level": "medium",
  "max_candidates_allowed": null,
  "location_cities": [],
  "location_states": [],
  "location_country": "US",
  "work_model": "unspecified",
  "work_model_details": null,
  "pay_rate_min": null,
  "pay_rate_max": null,
  "pay_rate_currency": "USD",
  "pay_rate_unit": "unspecified",
  "employment_type": "unspecified",
  "is_rate_strict": null,
  "pay_rate_notes": null,
  "salary_min": null,
  "salary_max": null,
  "salary_currency": "USD",
  "bonus_percentage_min": null,
  "bonus_percentage_max": null,
  "bonus_type": null,
  "bonus_notes": null,
  "has_equity": false,
  "equity_type": null,
  "equity_details": null,
  "pto_days": null,
  "health_insurance_provided": null,
  "retirement_matching": null,
  "retirement_matching_details": null,
  "benefits_summary": null,
  "sign_on_bonus": null,
  "relocation_assistance": false,
  "relocation_amount": null,
  "contract_duration_text": null,
  "contract_start_date": null,
  "contract_end_date": null,
  "contract_can_extend": null,
  "allowed_work_auth": ["Any"],
  "not_allowed_work_auth": ["Any"],
  "citizenship_required": null,
  "work_auth_notes": null,
  "background_check_required": null,
  "background_check_details": null,
  "security_clearance_required": null,
  "security_clearance_level": null,
  "overall_min_years": null,
  "primary_role_min_years": null,
  "management_experience_required": null,
  "must_have_skills": [],
  "nice_to_have_skills": [],
  "primary_technologies": [],
  "certifications_required": [],
  "certifications_preferred": [],
  "domains": [],
  "responsibilities": [],
  "day_to_day": [],
  "other_constraints": [],
  "work_hours": null,
  "time_zone": null,
  "travel_required": null,
  "travel_details": null,
  "interview_process": null,
  "vendor_name": null,
  "vendor_contact_name": null,
  "vendor_contact_email": null,
  "vendor_contact_phone": null,
  "raw_role_title_block": null,
  "status": "active"
}

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
