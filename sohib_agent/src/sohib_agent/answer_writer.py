"""
LLM call: RecommendationResult evidence -> prose answer.
Thin wrapper. All scoring logic is complete before this module is called.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .models import RecommendationResult, UserProfile

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "recommendation_answer.md"

_MODEL_PREFERENCE = [
    "claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-3-5-sonnet-20241022",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
]


def _resolve_model(client) -> str:
    try:
        available = {m.id for m in client.models.list().data}
        for model in _MODEL_PREFERENCE:
            if model in available:
                return model
        claude_models = sorted(m.id for m in client.models.list().data if "claude" in m.id.lower())
        if claude_models:
            return claude_models[-1]
        raise RuntimeError(f"No Claude model found. Available: {sorted(available)}")
    except Exception as e:
        raise RuntimeError(f"Could not list models: {e}") from e


def write_answer(profile: UserProfile, result: RecommendationResult) -> str:
    """
    Given a fully-computed RecommendationResult, ask the LLM to narrate it in prose.
    The LLM must not compute or invent scores — it only narrates supplied evidence.
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
    payload = {
        "user_profile": profile.model_dump(),
        "recommendation_result": result.model_dump(),
    }

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    return response.content[0].text.strip()
