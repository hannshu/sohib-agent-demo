"""
LLM call: free-text user message -> UserProfile fields.
Thin wrapper around the Anthropic API. All scoring logic is in matching.py.
"""
from __future__ import annotations

import json
import os

from .models import UserProfile

_PROMPT_PATH = (
    __import__("pathlib").Path(__file__).parent.parent.parent / "prompts" / "extract_profile.md"
)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text()


def extract_profile_from_text(text: str, current_profile: UserProfile | None = None) -> dict:
    """
    Call the LLM to extract UserProfile fields from free text.
    Returns a dict of extracted fields (may be partial).
    Uses Anthropic claude-sonnet-4-6 by default; falls back to openai if ANTHROPIC_API_KEY not set.
    """
    system_prompt = _load_prompt()
    schema_json = json.dumps(UserProfile.model_json_schema(), indent=2)
    current_json = current_profile.model_dump_json(indent=2) if current_profile else "{}"

    user_content = (
        f"Current profile state:\n{current_json}\n\n"
        f"User message:\n{text}\n\n"
        f"UserProfile JSON schema:\n{schema_json}"
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text.strip()
    except Exception:
        # Fallback to OpenAI if available
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
