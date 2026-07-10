"""
LLM call: RecommendationResult evidence -> prose answer.
Thin wrapper. All scoring logic is complete before this module is called.
"""
from __future__ import annotations

import json
import os

from .models import RecommendationResult, UserProfile

_PROMPT_PATH = (
    __import__("pathlib").Path(__file__).parent.parent.parent / "prompts" / "recommendation_answer.md"
)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text()


def write_answer(
    profile: UserProfile,
    result: RecommendationResult,
) -> str:
    """
    Given a fully-computed RecommendationResult, ask the LLM to narrate it in prose.
    The LLM must not compute or invent scores — it only narrates supplied evidence.
    """
    system_prompt = _load_prompt()
    payload = {
        "user_profile": profile.model_dump(),
        "recommendation_result": result.model_dump(),
    }
    user_content = json.dumps(payload, indent=2)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text.strip()
    except Exception:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content.strip()
