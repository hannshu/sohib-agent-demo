"""
LLM call: free-text user message -> UserProfile fields.
Thin wrapper around the Anthropic API. All scoring logic is in matching.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .models import UserProfile

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "extract_profile.md"

# Model preference order — first one available on the key wins.
# Includes both plain names and the us.anthropic.* prefixed names used by gateway keys.
_MODEL_PREFERENCE = [
    "claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-3-5-sonnet-20241022",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
]


def _resolve_model(client) -> str:
    """Return the first model from the preference list that is available on this key."""
    try:
        available = {m.id for m in client.models.list().data}
        for model in _MODEL_PREFERENCE:
            if model in available:
                return model
        # Nothing matched — return the first available Anthropic Claude model as fallback
        claude_models = sorted(m.id for m in client.models.list().data if "claude" in m.id.lower())
        if claude_models:
            return claude_models[-1]  # highest lexicographic = newest
        raise RuntimeError(f"No Claude model found on this key. Available: {sorted(available)}")
    except Exception as e:
        raise RuntimeError(f"Could not list models: {e}") from e


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) >= 2 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def extract_profile_from_text(text: str, current_profile: UserProfile | None = None) -> dict:
    """
    Call the LLM to extract UserProfile fields from free text.
    Returns a dict of extracted fields (may be partial — only changed/new fields).
    Raises RuntimeError with a clear message if the API call fails.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Run: export ANTHROPIC_API_KEY=<your-key>  "
            "or use --no-llm mode: sohib-agent chat --no-llm"
        )

    client = anthropic.Anthropic(api_key=api_key)
    model = _resolve_model(client)

    system_prompt = _PROMPT_PATH.read_text()
    schema_json = json.dumps(UserProfile.model_json_schema(), indent=2)
    current_json = current_profile.model_dump_json(indent=2) if current_profile else "{}"

    user_content = (
        f"Current profile state:\n{current_json}\n\n"
        f"User message:\n{text}\n\n"
        f"UserProfile JSON schema:\n{schema_json}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = _strip_fences(response.content[0].text)

    try:
        parsed = json.loads(raw)
        # If LLM returned a clarifying question, it looks like {"clarify": "..."}
        # Pass it through — caller handles it.
        return parsed
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned invalid JSON for profile extraction.\n"
            f"Raw response: {raw[:500]}\nError: {e}"
        ) from e
