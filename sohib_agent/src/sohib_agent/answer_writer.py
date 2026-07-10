"""
LLM call: RecommendationResult + raw task score tables -> brief prose answer.
The LLM receives the full score tables for matched tasks so it can reason
analytically over the actual data, not just narrate a pre-ranked list.
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


def _build_score_tables(matched_datasets: list[dict], method_scores: dict) -> dict:
    """
    Build a compact score table for each matched task.
    Returns {task_id: {method: {overall, BPC, BER, SPC, DTP, completion_status}}}
    Only includes methods that actually ran (overall is not None).
    """
    tables = {}
    seen_tasks = set()
    for ds in matched_datasets:
        raw_tid = ds.get("task_id")
        task_ids = raw_tid if isinstance(raw_tid, list) else [raw_tid]
        for tid in task_ids:
            if not tid or tid in seen_tasks or tid not in method_scores:
                continue
            seen_tasks.add(tid)
            table = {}
            for method, scores in method_scores[tid].items():
                if scores.get("overall") is not None:
                    table[method] = {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in scores.items()
                        if k in ("overall", "BPC", "BER", "SPC", "DTP", "completion_status")
                    }
            # Sort by overall score descending so LLM sees the ranking clearly
            tables[tid] = dict(sorted(table.items(), key=lambda x: x[1]["overall"], reverse=True))
    return tables


def write_answer(
    profile: UserProfile,
    result: RecommendationResult,
    method_scores: dict | None = None,
) -> str:
    """
    Ask the LLM to write a brief recommendation grounded in the actual benchmark scores.
    method_scores: the full KB method_scores dict — passed so the LLM sees raw tables,
    not just the pre-ranked list.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Run: export ANTHROPIC_API_KEY=<your-key>  "
            "or use --no-llm mode: sohib-agent chat --no-llm"
        )

    client = anthropic.Anthropic(base_url="https://www.litellm.org", api_key=api_key)
    model = _resolve_model(client)

    score_tables = {}
    if method_scores:
        score_tables = _build_score_tables(result.matched_datasets, method_scores)

    payload = {
        "user_profile": profile.model_dump(),
        "matched_branch": result.matched_branch,
        "matched_datasets": result.matched_datasets,
        "task_score_tables": score_tables,   # full tables — LLM reasons from this
        "warnings": [
            {"method": r["method"], "warnings": r.get("warnings", [])}
            for r in result.recommended_methods
            if r.get("warnings")
        ],
        "confidence_note": result.confidence_note,
    }

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=_PROMPT_PATH.read_text(),
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    return response.content[0].text.strip()
