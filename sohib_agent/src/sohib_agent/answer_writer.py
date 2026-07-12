"""
LLM call: bounded joint method selection + narration, grounded in retrieved benchmark evidence.

matching.py computes a deterministic, evidence-backed CANDIDATE POOL (RecommendationResult.
candidate_methods — up to top_k methods, each with a real composite_score, evidence dict, and
warnings). It does NOT decide the final answer. This module asks the LLM to select and order the
final top-3 from that pool: a "bounded joint decision" where the LLM can weigh things the flat
composite score can't capture (warnings, cross-task-category consistency via method_summaries,
how well a method fits what the user said matters to them) and is free to reorder or pick a
lower-scored candidate over a higher-scored one — but it can never introduce a method or a
number that isn't already in the pool it was given.

The LLM's JSON response is validated against candidate_methods before use (_validate_selection):
any selected method not present in the pool is dropped. If validation leaves fewer than 3 (or
the whole call fails), missing slots are backfilled deterministically via
matching.top_k_by_composite_score, so a malformed or hallucinated response degrades gracefully
to the deterministic ranking instead of shipping a fabricated method.

Raw benchmark numbers (composite_score, overall/BPC/BER/SPC/DTP, similarity, etc.) are sent TO
the LLM as grounding evidence, but are deliberately NOT surfaced in the text shown back to the
USER — the model reasons over them internally and writes a plain-language blurb (what the method
is, why it fits) instead of quoting scores. _format_answer() controls the final shape: one
scenario-summary sentence, up to 3 numbered method blurbs, one closing summary sentence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

from .matching import top_k_by_composite_score
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


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) >= 2 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _build_score_tables(matched_datasets: list[dict], method_scores: dict) -> dict:
    """
    Retrieval step: build a compact score table for each matched benchmark task.
    Returns {task_id: {method: {overall, BPC, BER, SPC, DTP, completion_status}}},
    sorted by overall score descending. Only methods that actually ran are included.
    This is retrieved grounding context, not a selection mechanism.
    """
    tables: dict[str, dict] = {}
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
            tables[tid] = dict(sorted(table.items(), key=lambda x: x[1]["overall"], reverse=True))
    return tables


def _build_method_summaries(candidate_methods: list[dict], method_summaries: dict) -> dict:
    """Retrieval step: pre-computed per-method cross-task-category stats for the candidate
    pool the LLM is choosing from (never the full 42-method catalogue)."""
    return {
        c["method"]: method_summaries[c["method"]]
        for c in candidate_methods
        if c["method"] in method_summaries
    }


def _validate_selection(selected_raw, candidates: list[dict]) -> list[dict]:
    """
    Hallucination guard: keep only LLM-selected entries whose method name is present in the
    real candidate pool. Real composite_score/evidence/warnings always come from the candidate
    dict, never from the LLM — only the free-text "sentence" (the user-facing blurb: what the
    method is + why it fits, no raw numbers) is taken from the model's output.
    """
    if not isinstance(selected_raw, list):
        return []
    by_name = {c["method"]: c for c in candidates}
    validated = []
    seen = set()
    for item in selected_raw:
        if not isinstance(item, dict):
            continue
        name = item.get("method")
        sentence = str(item.get("sentence") or "").strip()
        if name in by_name and name not in seen and sentence:
            entry = dict(by_name[name])
            entry["sentence"] = sentence
            validated.append(entry)
            seen.add(name)
    return validated


class ResolvedSelection(NamedTuple):
    selected: list[dict]
    scenario_summary: str | None
    summary: str | None
    branch_note: str | None
    is_fully_llm_selected: bool


def _resolve_selection(raw_text: str, candidates: list[dict], k: int = 3) -> ResolvedSelection:
    """
    Parse + validate the LLM's response against candidates. Backfills any slots the LLM's
    selection didn't validly fill using the deterministic ranking (no raw score exposed — see
    _fallback_sentence), so the result always has up to k entries drawn only from the real
    candidate pool.
    """
    target = min(k, len(candidates))
    scenario_summary = None
    summary = None
    branch_note = None
    validated: list[dict] = []
    try:
        parsed = json.loads(_strip_fences(raw_text))
        validated = _validate_selection(parsed.get("selected"), candidates)
        scenario_summary = parsed.get("scenario_summary")
        summary = parsed.get("summary")
        branch_note = parsed.get("branch_note")
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    is_fully_llm_selected = len(validated) >= target and target > 0

    if len(validated) < target:
        chosen_names = {v["method"] for v in validated}
        backfill = [c for c in top_k_by_composite_score(candidates, len(candidates)) if c["method"] not in chosen_names]
        for c in backfill:
            if len(validated) >= target:
                break
            validated.append({**c, "sentence": None})

    return ResolvedSelection(validated, scenario_summary, summary, branch_note, is_fully_llm_selected)


def _fallback_sentence(entry: dict) -> str:
    """
    Qualitative-only stand-in for a slot the LLM didn't validly fill. Never surfaces a raw
    score to the user — this is the deterministic ranking's pick, described in plain language.
    """
    if entry.get("composite_score") is None:
        return "one of the SOHIB benchmark's published top methods for this scenario."
    return "selected by the benchmark's deterministic ranking for this scenario."


def _format_answer(resolved: ResolvedSelection) -> str:
    """
    Assembles the user-facing text: one scenario sentence, then each method with a plain-language
    blurb (what it is + why it fits — no raw benchmark numbers), then a closing summary sentence.
    The model's JSON output never reaches the user directly; this function controls the format.
    """
    lines = []
    if resolved.scenario_summary:
        lines.append(str(resolved.scenario_summary))
        lines.append("")
    for i, entry in enumerate(resolved.selected, start=1):
        sentence = entry.get("sentence") or _fallback_sentence(entry)
        lines.append(f"{i}. **{entry['method']}** — {sentence}")
    if resolved.summary:
        lines.append("")
        lines.append(str(resolved.summary))
    if resolved.branch_note:
        lines.append("")
        lines.append(str(resolved.branch_note))
    return "\n".join(lines)


def run_recommendation(
    profile: UserProfile,
    result: RecommendationResult,
    method_scores: dict | None = None,
    method_summaries: dict | None = None,
) -> tuple[RecommendationResult, str]:
    """
    Bounded joint decision: ask the LLM to select and order the final top-3 from
    result.candidate_methods (the deterministic, evidence-backed pool). Falls back to
    matching.top_k_by_composite_score wherever the LLM call fails or its selection doesn't
    validate. Returns (updated_result, display_text) — updated_result has recommended_methods /
    discarded_methods / selection_source filled in from the resolved selection.
    """
    candidates = result.candidate_methods

    if not candidates:
        empty = result.model_copy(update={
            "recommended_methods": [], "discarded_methods": [], "selection_source": "deterministic_fallback",
        })
        return empty, "No candidate methods matched this profile in the benchmark."

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

    task_score_tables = _build_score_tables(result.matched_datasets, method_scores) if method_scores else {}
    candidate_summaries = _build_method_summaries(candidates, method_summaries) if method_summaries else {}

    payload = {
        "user_profile": profile.model_dump(),
        "matched_branch": result.matched_branch,
        "decision_tree_crossref": result.decision_tree_crossref,
        "matched_datasets": result.matched_datasets,
        # The pool the LLM must choose from — it may not name any method outside this list.
        "candidate_methods": candidates,
        # Retrieved grounding context to inform (not replace) the LLM's judgement.
        "task_score_tables": task_score_tables,
        "method_summaries": candidate_summaries,
        "confidence_note": result.confidence_note,
    }

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_PROMPT_PATH.read_text(),
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    raw_text = response.content[0].text.strip()

    resolved = _resolve_selection(raw_text, candidates, k=3)
    selected_names = {s["method"] for s in resolved.selected}
    discarded = [c for c in candidates if c["method"] not in selected_names]

    selection_source = "llm_joint" if resolved.is_fully_llm_selected else "deterministic_fallback"
    confidence_note = result.confidence_note
    if selection_source == "deterministic_fallback":
        confidence_note += (
            " NOTE: the LLM's selection could not be fully validated against the candidate "
            "pool (malformed response or a method it named wasn't in the pool) — missing slots "
            "were filled deterministically by composite_score instead."
        )

    updated = result.model_copy(update={
        "recommended_methods": resolved.selected,
        "discarded_methods": discarded,
        "selection_source": selection_source,
        "confidence_note": confidence_note,
    })
    return updated, _format_answer(resolved)
