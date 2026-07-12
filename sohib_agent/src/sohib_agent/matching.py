"""
Dataset similarity scoring and method ranking.
All numeric logic lives here. No LLM calls.

Key design: batch_effect_severity is a TASK-level property, not a dataset-level one.
A single tissue slice has no batch effect. dataset_similarity() therefore accepts an
optional task_knowledge entry and reads batch info from there, not from dataset_profile.
"""
from __future__ import annotations

from .method_metadata import METHOD_METADATA
from .models import UserProfile

# ── Section 7.1: dataset similarity weights ───────────────────────────────────

_WEIGHTS = {
    "omics_type": 4.0,
    "st_category": 2.5,    # only applied when both are transcriptomics
    "technology": 3.0,
    "species": 1.5,
    "tissue": 1.5,
    "batch_effect_severity": 2.0,  # compared against task_knowledge, not dataset_profile
    "num_locations": 2.0,  # scaled by min(a,b)/max(a,b)
    "sparsity": 1.0,        # scaled by 1 - abs(a-b)
}

# ── Section 7.2: priority weights ─────────────────────────────────────────────
# (overall_weight, speed_proxy_weight, memory_proxy_weight)
# Since exact runtime/memory numbers are not available, completion_status acts as
# a binary proxy: methods with completion_status != "success" receive a fixed
# penalty of -0.3. This is an approximation, not equivalent to real resource weighting.
_PRIORITY_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "accuracy": (0.85, 0.05, 0.10),
    "speed":    (0.55, 0.35, 0.10),
    "memory":   (0.55, 0.10, 0.35),
    "balanced": (0.70, 0.15, 0.15),
}

_COMPLETION_PENALTY = -0.3  # applied when completion_status != "success"


def dataset_similarity(
    profile: UserProfile,
    dataset_profile: dict,
    task_info: dict | None = None,
) -> float:
    """
    Weighted similarity between a UserProfile and a knowledge-base dataset profile.

    task_info is the task_knowledge entry for the task this dataset belongs to.
    It is used for batch_effect_severity comparison — that is a task-level fact,
    not a per-slice fact, so it must not be read from dataset_profile.

    Returns a non-negative float; higher is more similar.
    """
    score = 0.0

    # omics_type
    if profile.omics_type is not None and dataset_profile.get("omics_type") == profile.omics_type:
        score += _WEIGHTS["omics_type"]

    # st_category (only meaningful when both are transcriptomics)
    if (
        profile.omics_type == "transcriptomics"
        and dataset_profile.get("omics_type") == "transcriptomics"
        and profile.st_category is not None
        and dataset_profile.get("st_category") == profile.st_category
    ):
        score += _WEIGHTS["st_category"]

    # technology (case-insensitive)
    if (
        profile.technology is not None
        and dataset_profile.get("sequencing_technique") is not None
        and profile.technology.lower() == dataset_profile["sequencing_technique"].lower()
    ):
        score += _WEIGHTS["technology"]

    # species
    if (
        profile.species is not None
        and dataset_profile.get("species") is not None
        and profile.species.lower() == dataset_profile["species"].lower()
    ):
        score += _WEIGHTS["species"]

    # tissue
    if (
        profile.tissue is not None
        and dataset_profile.get("tissue") is not None
        and profile.tissue.lower() == dataset_profile["tissue"].lower()
    ):
        score += _WEIGHTS["tissue"]

    # batch_effect_severity — task-level property, sourced from task_info, never dataset_profile
    if (
        profile.batch_effect_severity is not None
        and task_info is not None
        and task_info.get("batch_effect_severity") == profile.batch_effect_severity
    ):
        score += _WEIGHTS["batch_effect_severity"]

    # num_locations (ratio-based)
    if profile.num_locations is not None and dataset_profile.get("num_locations") is not None:
        a, b = profile.num_locations, dataset_profile["num_locations"]
        if a > 0 and b > 0:
            score += _WEIGHTS["num_locations"] * (min(a, b) / max(a, b))

    # sparsity (absolute difference)
    if profile.sparsity is not None and dataset_profile.get("sparsity") is not None:
        diff = abs(profile.sparsity - dataset_profile["sparsity"])
        score += _WEIGHTS["sparsity"] * max(0.0, 1.0 - diff)

    return score


def find_similar_datasets(
    profile: UserProfile,
    dataset_profiles: dict[str, dict],
    task_knowledge: dict[str, dict] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Return top-k most similar datasets, each with slice_id, similarity, profile, and task_info.
    task_knowledge is used to pass task-level batch info into dataset_similarity().
    """
    task_knowledge = task_knowledge or {}
    scored = []
    for slice_id, dp in dataset_profiles.items():
        # Resolve the task_info for this slice — handle multi-task slices by taking the first
        raw_tid = dp.get("task_id")
        task_ids = raw_tid if isinstance(raw_tid, list) else [raw_tid]
        task_info = next(
            (task_knowledge[tid] for tid in task_ids if tid and tid in task_knowledge),
            None,
        )
        sim = dataset_similarity(profile, dp, task_info=task_info)
        scored.append({
            "slice_id": slice_id,
            "similarity": sim,
            "profile": dp,
            "task_info": task_info,
        })
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def _method_score(method: str, task_scores: dict, priority: str) -> float:
    """
    Compute a single composite score for a method on a given task.
    Returns -inf if the method has no scores at all.
    """
    entry = task_scores.get(method)
    if entry is None:
        return float("-inf")

    overall = entry.get("overall")
    if overall is None:
        return float("-inf")

    w_overall, w_speed, w_memory = _PRIORITY_WEIGHTS[priority]
    base = overall * w_overall

    # Binary completion penalty as proxy for speed/memory (approximation)
    if entry.get("completion_status") != "success":
        base += _COMPLETION_PENALTY * (w_speed + w_memory)

    return base


def is_architecturally_excluded(method: str, profile: UserProfile, meta: dict) -> bool:
    """
    Return True if this method must be hard-excluded from recommended_methods.
    architecturally_inapplicable is a binary design fact: the method cannot process
    that omics type by design and must not appear in results at all.
    """
    inapplicable = meta.get("architecturally_inapplicable", [])
    return bool(profile.omics_type and profile.omics_type in inapplicable)


def _build_warnings(method: str, profile: UserProfile, meta: dict) -> list[str]:
    """
    Return a list of soft warnings for a method given the user's profile.
    Warnings are informational — the method still appears in the ranked list.
    The user decides whether the warning is disqualifying for their situation.
    Note: architecturally_inapplicable methods are hard-excluded before this is called;
    they never reach warning generation.
    """
    warnings = []

    if profile.avoid_deep_learning and meta.get("deep_learning", False):
        warnings.append("uses deep learning (user requested non-DL methods)")

    # FuseMap and DECIPHER produce two distinct embedding types from the same underlying model.
    # Warn when the user's resolution goal and the variant's embedding type are mismatched.
    embedding_type = meta.get("embedding_type")
    base = meta.get("base_method")
    if embedding_type == "cell" and profile.target_resolution == "domain":
        warnings.append(
            f"{base} (cell) outputs per-cell embeddings — the (niche) variant is "
            f"better suited for domain-level integration"
        )
    elif embedding_type == "niche" and profile.target_resolution == "cell":
        warnings.append(
            f"{base} (niche) outputs spatially-smoothed neighbourhood embeddings — "
            f"the (cell) variant is better suited for cell-level analysis"
        )

    return warnings


def rank_methods_domain_level(
    profile: UserProfile,
    matched_datasets: list[dict],
    method_scores: dict[str, dict],
    top_k: int = 10,
) -> tuple[list[dict], list[dict]]:
    """
    Aggregate method scores across matched datasets and return ranked methods.
    No hard filters — all methods are ranked. Potential issues are surfaced as
    per-method warnings so the user can make an informed decision.
    The second return value (previously 'discarded') is now always an empty list
    and kept only for API compatibility.
    """
    all_methods: set[str] = set()
    for ds in matched_datasets:
        task_id = ds["profile"].get("task_id")
        task_ids = task_id if isinstance(task_id, list) else [task_id]
        for tid in task_ids:
            if tid and tid in method_scores:
                all_methods.update(method_scores[tid].keys())

    method_agg: dict[str, list[float]] = {m: [] for m in all_methods}

    for method in sorted(all_methods):
        for ds in matched_datasets:
            task_id = ds["profile"].get("task_id")
            task_ids = task_id if isinstance(task_id, list) else [task_id]
            for tid in task_ids:
                if tid and tid in method_scores:
                    # profile.priority is None when the user never stated one — default to
                    # "balanced" for the actual weighting, but this default is never written
                    # back into the profile, so it never reaches the LLM as if the user asked.
                    s = _method_score(method, method_scores[tid], profile.priority or "balanced")
                    if s > float("-inf"):
                        method_agg[method].append(s * ds["similarity"])

    ranked = []
    for method, scores in method_agg.items():
        if not scores:
            continue
        meta = METHOD_METADATA.get(method, {})

        # Hard exclusion: architecturally_inapplicable methods never appear in results.
        if is_architecturally_excluded(method, profile, meta):
            continue

        avg = sum(scores) / len(scores)
        warnings = _build_warnings(method, profile, meta)

        best_ds = matched_datasets[0]
        task_id = best_ds["profile"].get("task_id")
        task_ids = task_id if isinstance(task_id, list) else [task_id]
        evidence = {}
        for tid in task_ids:
            if tid and tid in method_scores and method in method_scores[tid]:
                evidence = {
                    "task_id": tid,
                    "dataset": best_ds["slice_id"],
                    "similarity": round(best_ds["similarity"], 3),
                    **{k: v for k, v in method_scores[tid][method].items()
                       if k in ("overall", "BPC", "BER", "SPC", "DTP", "completion_status")},
                }
                break
        ranked.append({
            "method": method,
            "composite_score": round(avg, 4),
            "evidence": evidence,
            "warnings": warnings,
        })

    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    return ranked[:top_k], []


def cell_level_data_available(cell_level_scores: dict) -> bool:
    """
    True once the cell-level score file has real per-task data (not just the placeholder
    _comment key). While this is False, rank_methods_cell_level() always returns the fixed
    published branch answer (Scanorama, Harmony, BANKSY) regardless of any profile field —
    so callers deciding whether to ask the user for cell-level-irrelevant details (num_locations,
    batch_effect_severity, etc.) should check this first: asking is pointless (and misleading —
    it implies those answers would change the outcome, when nothing currently does) while it's False.
    """
    return any(
        k != "_comment" and isinstance(v, dict)
        for k, v in cell_level_scores.items()
    )


def _static_cell_level_branch() -> tuple[list[dict], list[dict]]:
    """The SOHIB decision-tree's published cell-level answer — used both when no cell-level
    data exists at all, and as a fallback when this profile's matched tasks happen to be among
    the ~17 tasks that don't carry single-cell ground truth (even if other tasks in the KB do)."""
    return (
        [
            {"method": "Scanorama", "composite_score": None,
             "evidence": {"source": "SOHIB decision-tree branch (static)", "task_id": None}},
            {"method": "Harmony",   "composite_score": None,
             "evidence": {"source": "SOHIB decision-tree branch (static)", "task_id": None}},
            {"method": "BANKSY",    "composite_score": None,
             "evidence": {"source": "SOHIB decision-tree branch (static)", "task_id": None}},
        ],
        [],
    )


def rank_methods_cell_level(
    profile: UserProfile,
    matched_datasets: list[dict],
    cell_level_scores: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Cell-level ranking. Reads ONLY from cell_level_scores, never from method_scores.
    Falls back to the static decision-tree branch (Scanorama, Harmony, BANKSY) both when no
    cell-level data exists at all, AND when this profile's matched tasks aren't among the tasks
    that carry it (only single-cell-resolution tasks do — see cell_level_data_available).
    """
    if not cell_level_data_available(cell_level_scores):
        return _static_cell_level_branch()

    all_methods: set[str] = set()
    for ds in matched_datasets:
        task_id = ds["profile"].get("task_id")
        task_ids = task_id if isinstance(task_id, list) else [task_id]
        for tid in task_ids:
            if tid and tid in cell_level_scores:
                all_methods.update(cell_level_scores[tid].keys())

    method_agg: dict[str, list[float]] = {m: [] for m in all_methods}

    for method in sorted(all_methods):
        for ds in matched_datasets:
            task_id = ds["profile"].get("task_id")
            task_ids = task_id if isinstance(task_id, list) else [task_id]
            for tid in task_ids:
                if tid and tid in cell_level_scores and method in cell_level_scores[tid]:
                    overall = cell_level_scores[tid][method].get("overall_cell_level")
                    if overall is not None:
                        method_agg[method].append(overall * ds["similarity"])

    ranked = []
    for m, s in method_agg.items():
        if not s:
            continue
        meta = METHOD_METADATA.get(m, {})
        if is_architecturally_excluded(m, profile, meta):
            continue

        # Evidence from the best-matched dataset's task, mirroring the domain-level style
        # (task_id/dataset/similarity + the real sub-metrics) rather than leaving it empty.
        evidence = {}
        best_ds = matched_datasets[0] if matched_datasets else None
        if best_ds is not None:
            task_id = best_ds["profile"].get("task_id")
            task_ids = task_id if isinstance(task_id, list) else [task_id]
            for tid in task_ids:
                if tid and tid in cell_level_scores and m in cell_level_scores[tid]:
                    evidence = {
                        "task_id": tid,
                        "dataset": best_ds["slice_id"],
                        "similarity": round(best_ds["similarity"], 3),
                        **{k: v for k, v in cell_level_scores[tid][m].items()
                           if k in ("overall_cell_level", "classification_acc", "classification_f1",
                                    "cLISI", "isolated_labels", "ASW_label", "best_classifier",
                                    "completion_status")},
                    }
                    break

        ranked.append({
            "method": m,
            "composite_score": round(sum(s) / len(s), 4),
            "evidence": evidence,
            "warnings": _build_warnings(m, profile, meta),
        })

    if not ranked:
        # Real cell-level data exists in the KB, but none of THIS profile's matched tasks carry
        # it (e.g. the closest matches are Visium cortex tasks, which have no single-cell ground
        # truth) — fall back to the static branch rather than returning an empty pool.
        return _static_cell_level_branch()

    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    return ranked[:10], []


def top_k_by_composite_score(candidates: list[dict], k: int = 3) -> list[dict]:
    """
    Deterministic fallback/baseline selection: top-k candidates by composite_score.
    Used (a) directly in --no-llm mode, where there is no LLM to make a joint decision, and
    (b) by answer_writer.py to backfill slots the LLM's selection failed to validate.
    If scores are None (the cell-level static branch fallback has no computed scores at all),
    there is nothing to rank by — the candidates are returned as given, un-reordered.
    """
    scored = [c for c in candidates if c.get("composite_score") is not None]
    if not scored:
        return candidates[:k]
    return sorted(scored, key=lambda c: c["composite_score"], reverse=True)[:k]
