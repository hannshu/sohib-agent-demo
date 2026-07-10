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
from .answer_writer import write_answer
from .build_knowledge_base import build
from .decision_tree import match_branch
from .h5ad_profile import load_and_extract
from .matching import find_similar_datasets, rank_methods_cell_level, rank_methods_domain_level
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
    dataset_profiles = kb["dataset_profiles"]
    method_scores = kb["method_scores"]
    cell_level_scores = kb.get("cell_level_scores", {})

    # Step 1: decision tree
    branch = match_branch(profile)

    # Step 2: find similar datasets (task_knowledge carries batch-effect info)
    task_knowledge = kb.get("task_knowledge", {})
    matched = find_similar_datasets(profile, dataset_profiles, task_knowledge=task_knowledge, top_k=5)

    # Step 3: rank methods
    if profile.target_resolution == "cell":
        recommended, discarded = rank_methods_cell_level(profile, matched, cell_level_scores)
        is_static = all(m["composite_score"] is None for m in recommended)
        if is_static:
            confidence_note = (
                "Cell-level ranking: this is the SOHIB benchmark's published decision-tree answer "
                "(Scanorama, Harmony, BANKSY). It is NOT a freshly computed, dataset-matched ranking. "
                "The per-dataset cell-level score file has not yet been supplied. "
                "Once supplied, the ranking will be recomputed against your specific dataset characteristics."
            )
        else:
            confidence_note = (
                f"Cell-level ranking based on {len(matched)} matched datasets. "
                "Scores derived from cell_level_scores only."
            )
    else:
        recommended, discarded = rank_methods_domain_level(profile, matched, method_scores)
        if branch:
            confidence_note = (
                f"This recommendation follows the directly published SOHIB decision-tree branch "
                f"'{branch['branch']}'. The top methods are drawn from benchmark evidence "
                f"on {len(matched)} closely matched datasets."
            )
        else:
            top_sim = matched[0]["similarity"] if matched else 0
            confidence_note = (
                f"No exact decision-tree branch matched (profile may be missing num_locations "
                f"or st_category). Ranking is based on similarity matching against {len(matched)} "
                f"benchmark datasets (best similarity: {top_sim:.2f})."
            )

    return RecommendationResult(
        matched_branch=branch["branch"] if branch else None,
        matched_datasets=[
            {"slice_id": d["slice_id"], "similarity": round(d["similarity"], 3),
             "task_id": d["profile"].get("task_id"),
             "omics_type": d["profile"].get("omics_type"),
             "tissue": d["profile"].get("tissue")}
            for d in matched
        ],
        recommended_methods=recommended,
        discarded_methods=discarded,
        confidence_note=confidence_note,
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
        "[cyan]/upload <path.h5ad>[/cyan]  [cyan]/recommend[/cyan]  [cyan]/quit[/cyan]",
        title="Welcome",
    ))

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
            path = line[8:].strip()
            _do_upload(path, profile, kb)
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

        with console.status("Extracting profile fields…"):
            try:
                extracted = extract_profile_from_text(line, profile)
                profile = _merge_extracted(profile, extracted)
            except Exception as e:
                console.print(f"[red]Profile extraction failed: {e}[/red]")
                continue

        _print_profile(profile)

        missing = profile.missing_required_fields()
        if missing:
            console.print(
                "[yellow]I still need a bit more info to give a recommendation:[/yellow]\n"
                + "\n".join(f"  • {m}" for m in missing)
            )
        else:
            _do_recommend(profile, kb, no_llm)


def _do_recommend(profile: UserProfile, kb: dict, no_llm: bool) -> None:
    with console.status("Computing recommendation…"):
        result = _run_recommendation(profile, kb)

    console.print(Panel(
        f"[bold]Matched branch:[/bold] {result.matched_branch or 'none (similarity matching used)'}\n"
        f"[bold]Matched datasets:[/bold] {len(result.matched_datasets)}\n"
        f"[bold]Top methods:[/bold] "
        + ", ".join(m["method"] for m in result.recommended_methods[:5]),
        title="[bold green]Recommendation[/bold green]",
    ))

    if no_llm:
        import json
        console.print(Syntax(
            json.dumps(result.model_dump(), indent=2),
            "json", theme="monokai",
        ))
        return

    with console.status("Writing answer…"):
        try:
            answer = write_answer(profile, result)
            console.print(Panel(answer, title="[bold cyan]Analysis[/bold cyan]"))
        except Exception as e:
            console.print(f"[red]Answer writing failed: {e}[/red]")
            import json
            console.print(Syntax(
                json.dumps(result.model_dump(), indent=2),
                "json", theme="monokai",
            ))


def _do_upload(path: str, profile: UserProfile, kb: dict) -> None:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return
    with console.status(f"Reading {p.name}…"):
        try:
            extracted = load_and_extract(p)
        except Exception as e:
            console.print(f"[red]Failed to read h5ad: {e}[/red]")
            return

    # h5ad-derived values override text-derived values
    current = profile.model_dump()
    field_map = {
        "num_locations": "num_locations",
        "num_features": "num_features",
        "sparsity": "sparsity",
        "tissue": "tissue",
        "species": "species",
        "technology": "technology",
    }
    updated = []
    for src_key, dst_key in field_map.items():
        v = extracted.get(src_key)
        if v is not None:
            current[dst_key] = v
            updated.append(f"{dst_key}={v}")

    profile.__init__(**current)
    console.print(
        f"[green]Loaded {p.name}.[/green] Updated: {', '.join(updated) or 'nothing new'}"
    )


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
