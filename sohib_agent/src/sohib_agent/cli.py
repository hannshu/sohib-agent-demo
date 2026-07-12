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
            confidence_note = (
                "Cell-level ranking: candidate_methods is the SOHIB benchmark's published branch "
                "answer (Scanorama, Harmony, BANKSY) — NOT a freshly computed, dataset-matched "
                "ranking. The per-dataset cell-level score file has not yet been supplied, so "
                "there is no pool to jointly select from; these three are narrated as-is."
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


def _ask_missing_fields(profile: UserProfile, header: str) -> bool:
    """
    Batch all still-missing gating + high-value fields into one grouped question.
    Returns True if anything was asked (i.e. the caller should wait for a reply
    instead of recommending immediately).
    """
    questions = profile.missing_required_fields() + profile.missing_optional_fields()
    if not questions:
        return False

    lines = [header]
    for q in questions:
        lines.append(f"  • {q}")
    lines.append("Use /set field=value, describe them in your next message, or type /recommend to use what's given.")
    console.print("\n".join(lines))
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
            console.print("[dim]Goodbye.[/dim]")
            break

        if line == "/profile":
            _print_profile(profile)
            continue

        if line == "/reset":
            profile = UserProfile()
            asked_optional = False
            console.print("[yellow]Profile reset.[/yellow]")
            continue

        if line == "/recommend":
            missing = profile.missing_required_fields()
            if missing:
                console.print(
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
            if profile.is_fully_specified():
                asked_optional = True
                _do_recommend(profile, kb, no_llm)
            else:
                asked_optional = _ask_missing_fields(
                    profile, "To improve the recommendation, please provide:"
                )
            continue

        if line.startswith("/set "):
            # Manual field setter: /set field=value
            try:
                _, rest = line.split(" ", 1)
                k, v = rest.split("=", 1)
                current = profile.model_dump()
                current[k.strip()] = v.strip()
                profile = UserProfile(**current)
                console.print(f"[green]Set {k.strip()} = {v.strip()}[/green]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            continue

        # ── free-text message ─────────────────────────────────────────────────
        if no_llm:
            console.print(
                "[dim]LLM disabled. Use /set field=value to update your profile, "
                "then /recommend.[/dim]"
            )
            continue

        with console.status("Thinking…"):
            try:
                extracted = extract_profile_from_text(line, profile)
            except Exception as e:
                console.print(f"[red]LLM call failed: {e}[/red]")
                continue

        # LLM returned a clarifying question instead of fields
        if "clarify" in extracted:
            console.print(extracted["clarify"])
            continue

        try:
            profile = _merge_extracted(profile, extracted)
        except Exception as e:
            # LLM emitted a value outside the profile's allowed schema (e.g. a typo'd
            # literal) — surface it and keep the previous, still-valid profile rather
            # than crashing the session.
            console.print(f"[red]Could not apply extracted fields ({e}). Ignoring this turn.[/red]")
            continue

        if not profile.is_ready_for_recommendation():
            # Still missing a gating field — ask for the single next-most-important one.
            missing = profile.missing_required_fields()
            console.print(_format_missing_field_question(missing[0]))
            continue

        if not asked_optional and profile.missing_optional_fields():
            # Gating fields are satisfied, so a recommendation is already possible — but
            # ask once, in one batched message, for the fields that would make it more
            # reliable. The user can answer, ignore and keep chatting, or type /recommend
            # to force an answer with what's already given.
            asked_optional = True
            _ask_missing_fields(
                profile,
                "I can give a recommendation now, but a few more details would make it more reliable:",
            )
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
            console.print(answer)
            if result.selection_source == "deterministic_fallback" and result.candidate_methods:
                console.print(
                    "[dim](LLM selection did not fully validate against the candidate pool — "
                    "some or all slots used the deterministic ranking instead.)[/dim]"
                )
        except Exception as e:
            console.print(f"[red]Recommendation call failed: {e}[/red] Showing the deterministic candidate pool instead.")
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
        console.print(f"[red]File not found: {path}[/red]")
        return profile

    with console.status(f"Reading {p.name}…"):
        try:
            extracted = load_and_extract(p)
        except Exception as e:
            console.print(f"[red]Failed to read h5ad: {e}[/red]")
            return profile

    # Warn about multi-batch before updating profile
    if extracted.get("_multi_batch_aggregate"):
        console.print(f"[yellow]Warning:[/yellow] {extracted['_multi_batch_note']}")

    current = profile.model_dump()
    used_text = False

    if description and no_llm:
        console.print("[dim]--no-llm mode: description text ignored, using only h5ad-derived fields.[/dim]")
    elif description:
        with console.status("Reading description…"):
            try:
                text_extracted = extract_profile_from_text(description, profile)
            except Exception as e:
                console.print(f"[red]Description extraction failed: {e}[/red]")
                text_extracted = {}
        if "clarify" in text_extracted:
            console.print(text_extracted["clarify"])
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
    for src_key, dst_key in (("tissue", "tissue"), ("species", "species"), ("technology", "technology")):
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
        console.print(f"[red]Could not apply extracted fields ({e}). Keeping previous profile.[/red]")
        return profile

    console.print(
        f"[green]Loaded {p.name}.[/green] Updated: {', '.join(updated_confirmed) or 'nothing new'}"
    )

    # Surface inferred fields for user confirmation (high-weight fields only: species)
    high_weight_inferred = [(k, v) for k, v in inferred_fields if k == "species"]
    for field_name, field_val in high_weight_inferred:
        console.print(
            f"[yellow]Inferred {field_name}=[bold]{field_val}[/bold] from gene-symbol naming convention "
            f"— is that correct? (Use /set {field_name}=<value> to correct, or continue.)[/yellow]"
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
