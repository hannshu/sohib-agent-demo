"""
SOHIB Recommendation Agent CLI.
Usage: sohib-agent chat
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from . import knowledge_base as kb_module
from .answer_writer import run_recommendation
from .build_knowledge_base import build
from .decision_tree import match_branch
from .h5ad_profile import load_and_extract
from .matching import (
    cell_level_data_available,
    find_similar_datasets,
    rank_methods_cell_level,
    rank_methods_domain_level,
    top_k_by_composite_score,
)
from .models import RecommendationResult, UserProfile
from .profile_extraction import extract_profile_from_text

console = Console()

_KB_PATH = Path(__file__).parent.parent.parent / "data" / "clean" / "knowledge_base.json"
_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

_AGENT_TAG = "[bold magenta]Agent>[/bold magenta] "

# h5ad Tier-3 fields whose inferred value is surfaced for user confirmation before recommending
# (high-weight or gating fields only — a wrong guess here materially affects matching).
_INFERENCE_EXPLANATION = {
    "species": "gene-symbol naming convention",
    "omics_type": "feature-name (var_names) patterns",
}


def _agent_say(message: str) -> None:
    """Print agent-originated text with a visible tag, mirroring the 'You>' input prompt, so
    it's unambiguous which lines are the user's own input vs. the agent's response."""
    console.print(_AGENT_TAG + message)


def _ensure_kb() -> dict:
    if not _KB_PATH.exists():
        console.print("[yellow]Knowledge base not found. Building from raw data…[/yellow]")
        _KB_PATH.parent.mkdir(parents=True, exist_ok=True)
        kb = build(_RAW_DIR)
        import json
        with open(_KB_PATH, "w") as f:
            json.dump(kb, f, indent=2)
        console.print(f"[green]Knowledge base built: {_KB_PATH}[/green]")
    return kb_module.load(_KB_PATH)


def _run_recommendation(profile: UserProfile, kb: dict) -> RecommendationResult:
    """
    Pure-Python step: computes candidate_methods (a deterministic, evidence-backed pool) and a
    deterministic top-3 baseline (recommended_methods). No LLM involved here — this is what
    --no-llm mode returns as-is. In LLM mode, answer_writer.run_recommendation() takes this
    result and lets the LLM make a bounded joint selection from candidate_methods, replacing
    recommended_methods/discarded_methods/selection_source with its (validated) choice.
    """
    dataset_profiles = kb["dataset_profiles"]
    method_scores = kb["method_scores"]
    cell_level_scores = kb.get("cell_level_scores", {})

    # batch_effect_severity is never nagged for (see models._HIGH_VALUE_FIELDS): if the user
    # never described integrating data from different technologies, assume "minimal" (same-tech)
    # for matching purposes rather than asking. Kept as a local override — the profile object
    # itself stays truthful (None means "not stated") for /profile and the LLM narration payload.
    batch_defaulted = profile.batch_effect_severity is None
    if batch_defaulted:
        profile = profile.model_copy(update={"batch_effect_severity": "minimal"})

    # Step 1: decision tree — used as annotation only, never as the source of candidate_methods.
    # For domain-level profiles with enough fields, the fine-grained similarity matcher below
    # is always the primary path. The branch is recorded as decision_tree_crossref.
    branch = match_branch(profile)

    # Step 2: find similar datasets (task_knowledge carries batch-effect info).
    # Fine-grained matching is the default whenever the profile has enough non-null fields
    # to compute a similarity score against at least one benchmark dataset.
    task_knowledge = kb.get("task_knowledge", {})
    matched = find_similar_datasets(profile, dataset_profiles, task_knowledge=task_knowledge, top_k=5)

    # Step 3: build the candidate pool deterministically. This pool — not the LLM — is the
    # source of truth for which methods/scores can ever appear in the final answer.
    if profile.target_resolution == "cell":
        candidates, _ = rank_methods_cell_level(profile, matched, cell_level_scores)
        is_static = all(m["composite_score"] is None for m in candidates)
        selection_source = "static_branch" if is_static else "deterministic_fallback"
        if is_static:
            reason = (
                "no cell-level score data has been supplied for the benchmark at all"
                if not cell_level_data_available(cell_level_scores)
                else "none of the matched benchmark tasks for this profile carry single-cell "
                     "ground truth (only some tasks do)"
            )
            confidence_note = (
                "Cell-level ranking: candidate_methods is the SOHIB benchmark's published branch "
                f"answer (Scanorama, Harmony, BANKSY) — NOT a freshly computed, dataset-matched "
                f"ranking, because {reason}. There is no pool to jointly select from; these three "
                f"are narrated as-is."
            )
        else:
            confidence_note = (
                f"Cell-level ranking: candidate_methods is a deterministic, evidence-backed pool "
                f"of up to {len(candidates)} methods computed from cell_level_scores across "
                f"{len(matched)} matched datasets, for the LLM to jointly select the final top-3 from."
            )
    else:
        candidates, _ = rank_methods_domain_level(profile, matched, method_scores)
        is_static = False
        selection_source = "deterministic_fallback"
        top_sim = matched[0]["similarity"] if matched else 0
        confidence_note = (
            f"Fine-grained similarity matching against {len(matched)} benchmark datasets "
            f"(best similarity: {top_sim:.2f}). candidate_methods is a deterministic, "
            f"evidence-backed pool of up to {len(candidates)} methods; recommended_methods is "
            f"either the LLM's bounded joint selection from that pool, or (in --no-llm mode, or "
            f"if the LLM's selection fails validation) the deterministic top-ranked subset."
        )

    # Decision tree cross-reference: informational annotation, not the answer source.
    dt_crossref = None
    if branch:
        dt_crossref = (
            f"SOHIB decision-tree branch '{branch['branch']}' also matches this profile "
            f"({branch['description']}). This is a cross-reference, not the source of the candidate pool above."
        )

    if batch_defaulted and not is_static:
        # The static cell-level branch answer (see rank_methods_cell_level) doesn't read any
        # profile field, including batch_effect_severity — nothing to disclose in that case.
        confidence_note += (
            " batch_effect_severity was not stated, so 'minimal' (same-technology integration) "
            "was assumed for matching; mention it explicitly (e.g. combining data from different "
            "platforms) if that's not the case."
        )

    baseline = candidates if is_static else top_k_by_composite_score(candidates, 3)
    baseline_names = {c["method"] for c in baseline}

    return RecommendationResult(
        matched_branch=branch["branch"] if branch else None,
        matched_datasets=[
            {"slice_id": d["slice_id"],
             "similarity": round(d["similarity"], 3),
             "task_id": d["profile"].get("task_id"),
             "omics_type": d["profile"].get("omics_type"),
             "st_category": d["profile"].get("st_category"),
             "sequencing_technique": d["profile"].get("sequencing_technique"),
             "species": d["profile"].get("species"),
             "tissue": d["profile"].get("tissue"),
             "num_locations": d["profile"].get("num_locations")}
            for d in matched
        ],
        candidate_methods=candidates,
        recommended_methods=baseline,
        discarded_methods=[c for c in candidates if c["method"] not in baseline_names],
        confidence_note=confidence_note,
        decision_tree_crossref=dt_crossref,
        selection_source=selection_source,
    )


def _print_profile(profile: UserProfile) -> None:
    import json
    data = {k: v for k, v in profile.model_dump().items() if v is not None and v != False}
    console.print(Panel(
        Syntax(json.dumps(data, indent=2), "json", theme="monokai"),
        title="[bold cyan]Current Profile[/bold cyan]",
    ))


def _merge_extracted(profile: UserProfile, extracted: dict) -> UserProfile:
    """Merge LLM-extracted fields into current profile (non-null values only)."""
    current = profile.model_dump()
    for k, v in extracted.items():
        if v is not None and k in current:
            current[k] = v
    return UserProfile(**current)


@click.group()
def main() -> None:
    pass


def _format_missing_field_question(missing_field_desc: str) -> str:
    """
    Turn a 'field (question text)' description (as produced by missing_required_fields())
    into a standalone question. Avoids str.capitalize(), which would lowercase any acronym
    in the description (e.g. "MERFISH") — only the first character is adjusted here.
    """
    question = missing_field_desc.split("(", 1)[1].rstrip(")").strip() if "(" in missing_field_desc else missing_field_desc
    if not question:
        return question
    question = question[0].upper() + question[1:]
    return question if question.endswith("?") else question + "?"


def _cell_level_would_be_static(profile: UserProfile, kb: dict) -> bool:
    """
    True if a cell-level recommendation right now would fall back to the static published
    branch answer (Scanorama/Harmony/BANKSY) regardless of any further detail the user could
    supply. Only some tasks carry cell-level data (single-cell-resolution ones — see
    cell_level_data_available), so this can't be answered from a single global flag: a MERFISH
    brain query might match a task that has real data (worth asking for more detail) while a
    Visium cortex query, even with target_resolution=="cell", might not (nothing to ask for).
    Actually runs the (cheap, pure-Python, no-LLM) matching pipeline to find out.
    """
    if profile.target_resolution != "cell":
        return False
    cell_level_scores = kb.get("cell_level_scores", {})
    if not cell_level_data_available(cell_level_scores):
        return True
    matched = find_similar_datasets(
        profile, kb["dataset_profiles"], task_knowledge=kb.get("task_knowledge", {}), top_k=5
    )
    candidates, _ = rank_methods_cell_level(profile, matched, cell_level_scores)
    return all(c.get("composite_score") is None for c in candidates)


def _relevant_optional_fields(profile: UserProfile, kb: dict) -> list[str]:
    """
    profile.missing_optional_fields(), but suppressed when a cell-level recommendation would
    fall back to the static branch answer regardless — in that state, no further detail changes
    the outcome, so asking for it would falsely imply otherwise.
    """
    if _cell_level_would_be_static(profile, kb):
        return []
    return profile.missing_optional_fields()


def _ask_missing_fields(profile: UserProfile, kb: dict) -> bool:
    """
    Batch all still-missing gating + high-value fields into one grouped question. Used by every
    input path (free text, /upload, or a mix of both across turns) so the number and phrasing of
    follow-up questions never depends on which order the user supplied information in — data then
    text, text then data, or interleaved all land here the same way.
    Returns True if anything was asked (i.e. the caller should wait for a reply
    instead of recommending immediately).
    """
    required = profile.missing_required_fields()
    optional = _relevant_optional_fields(profile, kb)
    if not required and not optional:
        return False

    header = (
        "A few required details are still missing before I can recommend:" if required else
        "I can give a recommendation now, but a few more details would make it more reliable:"
    )
    lines = [header]
    # Only required-field descriptions are authored as "field (question text?)" — the optional
    # (high-value) descriptions in models._HIGH_VALUE_FIELDS are example/hint text, not always in
    # that shape (e.g. st_category has nested parens), so only the former go through the
    # question-ifying formatter; the latter are shown as-is, same as before this was batched.
    for q in required:
        lines.append(f"  • {_format_missing_field_question(q)}")
    for q in optional:
        lines.append(f"  • {q}")
    lines.append(
        "Describe them in your next message, upload a .h5ad, or use /set field=value."
        if required else
        "Use /set field=value, describe them in your next message, or type /recommend to use what's given."
    )
    _agent_say("\n".join(lines))
    return True


@main.command()
@click.option("--no-llm", is_flag=True, help="Skip LLM calls; use manual /set commands only.")
def chat(no_llm: bool) -> None:
    """Start an interactive recommendation session."""
    kb = _ensure_kb()
    profile = UserProfile()

    console.print(Panel(
        "[bold green]SOHIB Recommendation Agent[/bold green]\n"
        "Describe your spatial omics dataset and I'll recommend integration methods.\n\n"
        "Commands: [cyan]/profile[/cyan]  [cyan]/reset[/cyan]  "
        "[cyan]/upload <path.h5ad> [description][/cyan]  [cyan]/recommend[/cyan]  [cyan]/quit[/cyan]",
        title="Welcome",
    ))

    # True once we've asked for optional-but-useful fields — asked at most once per profile so
    # the agent doesn't nag every turn. Reset whenever the profile is reset.
    asked_optional = False

    while True:
        try:
            line = console.input("[bold blue]You>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            break

        if not line:
            continue

        # ── slash commands ────────────────────────────────────────────────────
        if line == "/quit" or line == "/exit":
            _agent_say("[dim]Goodbye.[/dim]")
            break

        if line == "/profile":
            _print_profile(profile)
            continue

        if line == "/reset":
            profile = UserProfile()
            asked_optional = False
            _agent_say("[yellow]Profile reset.[/yellow]")
            continue

        if line == "/recommend":
            missing = profile.missing_required_fields()
            if missing:
                _agent_say(
                    "[red]Cannot recommend yet. Still need:[/red]\n"
                    + "\n".join(f"  • {m}" for m in missing)
                )
                continue
            _do_recommend(profile, kb, no_llm)
            continue

        if line.startswith("/upload "):
            rest = line[8:].strip().split(None, 1)
            path = rest[0]
            description = rest[1] if len(rest) > 1 else None
            profile = _do_upload(path, description, profile, kb, no_llm)
            if profile.is_ready_for_recommendation() and not _relevant_optional_fields(profile, kb):
                asked_optional = True
                _do_recommend(profile, kb, no_llm)
            else:
                asked_optional = _ask_missing_fields(profile, kb)
            continue

        if line.startswith("/set "):
            # Manual field setter: /set field=value
            try:
                _, rest = line.split(" ", 1)
                k, v = rest.split("=", 1)
                current = profile.model_dump()
                current[k.strip()] = v.strip()
                profile = UserProfile(**current)
                _agent_say(f"[green]Set {k.strip()} = {v.strip()}[/green]")
            except Exception as e:
                _agent_say(f"[red]Error: {e}[/red]")
            continue

        # ── free-text message ─────────────────────────────────────────────────
        if no_llm:
            _agent_say(
                "[dim]LLM disabled. Use /set field=value to update your profile, "
                "then /recommend.[/dim]"
            )
            continue

        with console.status("Thinking…"):
            try:
                extracted = extract_profile_from_text(line, profile)
            except Exception as e:
                _agent_say(f"[red]LLM call failed: {e}[/red]")
                continue

        # LLM returned a clarifying question instead of fields
        if "clarify" in extracted:
            _agent_say(extracted["clarify"])
            continue

        try:
            profile = _merge_extracted(profile, extracted)
        except Exception as e:
            # LLM emitted a value outside the profile's allowed schema (e.g. a typo'd
            # literal) — surface it and keep the previous, still-valid profile rather
            # than crashing the session.
            _agent_say(f"[red]Could not apply extracted fields ({e}). Ignoring this turn.[/red]")
            continue

        if not profile.is_ready_for_recommendation():
            # Still missing a gating field — batch it (and any relevant optional fields) into
            # one grouped question, same as the /upload path, rather than asking one at a time.
            asked_optional = _ask_missing_fields(profile, kb)
            continue

        if not asked_optional and _relevant_optional_fields(profile, kb):
            # Gating fields are satisfied, so a recommendation is already possible — but
            # ask once, in one batched message, for the fields that would make it more
            # reliable. The user can answer, ignore and keep chatting, or type /recommend
            # to force an answer with what's already given. Suppressed entirely for
            # cell-level queries with no cell-level score data — nothing left to ask would
            # change that answer (see _relevant_optional_fields).
            asked_optional = True
            _ask_missing_fields(profile, kb)
            continue

        _do_recommend(profile, kb, no_llm)


def _do_recommend(profile: UserProfile, kb: dict, no_llm: bool) -> None:
    with console.status("Computing candidate pool…"):
        result = _run_recommendation(profile, kb)

    if no_llm:
        # No LLM available to make the joint decision — recommended_methods is already the
        # deterministic top-3-by-composite_score baseline computed in _run_recommendation.
        import json
        console.print(Syntax(
            json.dumps(result.model_dump(), indent=2),
            "json", theme="monokai",
        ))
        return

    with console.status("Selecting & writing recommendation…"):
        try:
            result, answer = run_recommendation(
                profile, result,
                method_scores=kb.get("method_scores"),
                method_summaries=kb.get("method_summaries"),
            )
            _agent_say(answer)
            if result.selection_source == "deterministic_fallback" and result.candidate_methods:
                console.print(
                    "[dim](Some picks used the benchmark's deterministic ranking instead of "
                    "a full LLM judgement.)[/dim]"
                )
        except Exception as e:
            _agent_say(f"[red]Recommendation call failed: {e}[/red] Showing the deterministic candidate pool instead.")
            import json
            console.print(Syntax(
                json.dumps(result.model_dump(), indent=2),
                "json", theme="monokai",
            ))


def _do_upload(path: str, description: str | None, profile: UserProfile, kb: dict, no_llm: bool) -> UserProfile:
    """
    Load an .h5ad file, optionally alongside a free-text description in the same command
    (e.g. `/upload sample.h5ad human cortex Visium data, want spatial domains`).

    The description (if any) is extracted and merged first; h5ad-derived values then override
    it field-by-field, per the documented precedence (structural/metadata fields read directly
    from the file are more trustworthy than a paraphrased description). Returns the updated
    profile — callers must reassign it (`profile = _do_upload(...)`), it is not mutated in place.
    """
    p = Path(path)
    if not p.exists():
        _agent_say(f"[red]File not found: {path}[/red]")
        return profile

    with console.status(f"Reading {p.name}…"):
        try:
            extracted = load_and_extract(p)
        except Exception as e:
            _agent_say(f"[red]Failed to read h5ad: {e}[/red]")
            return profile

    # Warn about multi-batch before updating profile
    if extracted.get("_multi_batch_aggregate"):
        _agent_say(f"[yellow]Warning:[/yellow] {extracted['_multi_batch_note']}")

    current = profile.model_dump()
    used_text = False

    if description and no_llm:
        _agent_say("[dim]--no-llm mode: description text ignored, using only h5ad-derived fields.[/dim]")
    elif description:
        with console.status("Reading description…"):
            try:
                text_extracted = extract_profile_from_text(description, profile)
            except Exception as e:
                _agent_say(f"[red]Description extraction failed: {e}[/red]")
                text_extracted = {}
        if "clarify" in text_extracted:
            _agent_say(text_extracted["clarify"])
        else:
            for k, v in text_extracted.items():
                if v is not None and k in current:
                    current[k] = v
                    used_text = True

    # h5ad-derived values override text-derived values.
    # Fields with provenance tags: unwrap the value, but track inferred fields for confirmation.
    updated_confirmed = []
    inferred_fields = []

    # Numeric fields are always confirmed (derived from matrix shape)
    for key in ("num_locations", "num_features", "sparsity"):
        v = extracted.get(key)
        if v is not None:
            current[key] = v
            updated_confirmed.append(f"{key}={v}")

    # Metadata fields may be confirmed or inferred
    for src_key, dst_key in (
        ("tissue", "tissue"), ("species", "species"), ("technology", "technology"),
        ("omics_type", "omics_type"),
    ):
        field = extracted.get(src_key)
        if field is None:
            continue
        if isinstance(field, dict):
            val = field.get("value")
            confidence = field.get("confidence", "confirmed")
        else:
            val = field
            confidence = "confirmed"
        if val is None:
            continue
        current[dst_key] = val
        if confidence == "inferred":
            inferred_fields.append((dst_key, val))
        else:
            updated_confirmed.append(f"{dst_key}={val}")

    current["source"] = "text+h5ad" if used_text else "h5ad"

    try:
        profile = UserProfile(**current)
    except Exception as e:
        _agent_say(f"[red]Could not apply extracted fields ({e}). Keeping previous profile.[/red]")
        return profile

    _agent_say(
        f"[green]Loaded {p.name}.[/green] Updated: {', '.join(updated_confirmed) or 'nothing new'}"
    )

    # Surface inferred fields for user confirmation (high-weight/gating fields only)
    high_weight_inferred = [(k, v) for k, v in inferred_fields if k in _INFERENCE_EXPLANATION]
    for field_name, field_val in high_weight_inferred:
        _agent_say(
            f"[yellow]Inferred {field_name}=[bold]{field_val}[/bold] from "
            f"{_INFERENCE_EXPLANATION[field_name]} — is that correct? "
            f"(Use /set {field_name}=<value> to correct, or continue.)[/yellow]"
        )

    return profile


@main.command(name="build-kb")
def build_kb() -> None:
    """Rebuild knowledge_base.json from raw CSVs."""
    import json
    console.print(f"Building from {_RAW_DIR} …")
    kb = build(_RAW_DIR)
    _KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_KB_PATH, "w") as f:
        json.dump(kb, f, indent=2)
    console.print(f"[green]Done: {_KB_PATH}[/green]")
