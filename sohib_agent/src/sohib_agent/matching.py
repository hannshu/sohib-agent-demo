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


def _build_warnings(method: str, profile: UserProfile, meta: dict) -> list[str]:
    """
    Return a list of plain-language warnings for a method given the user's profile.
    Warnings are informational — the method still appears in the ranked list.
    The user decides whether the warning is disqualifying for their situation.
    """
    warnings = []

    if profile.avoid_deep_learning and meta.get("deep_learning", False):
        warnings.append("uses deep learning (user requested non-DL methods)")

    inapplicable = meta.get("architecturally_inapplicable", [])
    if profile.omics_type and profile.omics_type in inapplicable:
        warnings.append(
            f"architecturally designed for transcriptomics — not evaluated on "
            f"{profile.omics_type} data in the benchmark; results may not transfer"
        )

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
                    s = _method_score(method, method_scores[tid], profile.priority)
                    if s > float("-inf"):
                        method_agg[method].append(s * ds["similarity"])

    ranked = []
    for method, scores in method_agg.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        meta = METHOD_METADATA.get(method, {})
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


def rank_methods_cell_level(
    profile: UserProfile,
    matched_datasets: list[dict],
    cell_level_scores: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Cell-level ranking. Reads ONLY from cell_level_scores, never from method_scores.
    Until the cell-level score file is supplied, returns the static decision-tree
    branch (Scanorama, Harmony, BANKSY) with a confidence note.
    """
    has_data = any(
        k != "_comment" and isinstance(v, dict)
        for k, v in cell_level_scores.items()
    )

    if not has_data:
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
        ranked.append({
            "method": m,
            "composite_score": round(sum(s) / len(s), 4),
            "evidence": {},
            "warnings": _build_warnings(m, profile, meta),
        })
    ranked.sort(key=lambda x: x["composite_score"], reverse=True)
    return ranked[:10], []
