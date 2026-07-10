"""
Build knowledge_base.json from raw CSVs.
Run directly: python -m sohib_agent.build_knowledge_base
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

from .method_metadata import METHOD_METADATA, TASK_CROSS_PLATFORM_INFO

# Methods present in overall_ranks.csv but NOT in the manuscript's 40-method evaluation.
# Verified by cross-referencing the manuscript abstract vs. the CSV row index.
# These three rows are cell-resolution variants counted separately in the CSV but
# grouped with their niche counterparts in the manuscript's 40-method count:
#   DECIPHER (cell), FuseMap (cell) — cell-level variants of niche methods
#   Nicheformer — added after manuscript submission
# They are KEPT in method_scores for completeness but flagged here.
EXTRA_METHODS_VS_MANUSCRIPT = {"DECIPHER (cell)", "FuseMap (cell)", "Nicheformer"}


def _parse_omics_type(raw: str) -> tuple[str, str | None]:
    """Return (omics_type, st_category) from raw 'Omics type' cell."""
    raw = raw.strip()
    if "Spatial transcriptomics" in raw:
        if "(sST)" in raw:
            return "transcriptomics", "sST"
        if "(iST)" in raw:
            return "transcriptomics", "iST"
        return "transcriptomics", None
    if "proteomics" in raw.lower():
        return "proteomics", None
    if "metabolomics" in raw.lower():
        return "metabolomics", None
    if "epigenomics" in raw.lower():
        return "epigenomics", None
    return "unknown", None


def _parse_task_ids(raw: str) -> list[str]:
    """Task ID column may contain comma-separated values like 'Task_25, Task_26'."""
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _parse_overall_ranks(df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """
    Returns method_scores[task_id][method_name] = {overall, BPC, BER, SPC, DTP,
                                                    completion_status, architecturally_inapplicable}
    Missing values are preserved as None, never coerced to 0.
    """
    method_scores: dict[str, dict] = {}
    col_pattern = re.compile(r"^task_(\d+)_(.+)_(overall|BPC|BER|SPC|DTP)$")

    col_map: dict[tuple[str, str], str] = {}
    for col in df.columns[1:]:
        m = col_pattern.match(col)
        if m:
            task_id = f"Task_{m.group(1)}"
            suffix = m.group(3)
            col_map[(task_id, suffix)] = col

    all_tasks = sorted({k[0] for k in col_map})

    for task_id in all_tasks:
        method_scores[task_id] = {}
        for _, row in df.iterrows():
            method = row.iloc[0]
            scores = {}
            for suffix in ("overall", "BPC", "BER", "SPC", "DTP"):
                key = (task_id, suffix)
                if key in col_map:
                    val = row[col_map[key]]
                    scores[suffix] = None if pd.isna(val) else float(val)
                else:
                    scores[suffix] = None

            all_none = all(v is None for v in scores.values())
            scores["completion_status"] = "not_applicable" if all_none else "success"
            scores["architecturally_inapplicable"] = False

            method_scores[task_id][method] = scores

    return method_scores


def _build_task_knowledge(data_df: pd.DataFrame) -> dict[str, dict]:
    """
    Build task-level knowledge from data_w_sparsity.csv.
    Batch effect is a property of a TASK (multi-slice integration challenge),
    not of any individual slice. Each task groups multiple slices together.
    """
    # Group slices by task — a slice may belong to multiple tasks (edge case for cross-platform)
    task_slices: dict[str, list[dict]] = defaultdict(list)
    for _, row in data_df.iterrows():
        task_ids = _parse_task_ids(row["Task ID"])
        omics_type, st_category = _parse_omics_type(str(row["Omics type"]))
        for tid in task_ids:
            task_slices[tid].append({
                "slice_id": str(row["Slice ID"]).strip(),
                "omics_type": omics_type,
                "st_category": st_category,
                "sequencing_technique": str(row["Sequencing technique"]).strip(),
                "species": str(row["Species"]).strip(),
                "tissue": str(row["Tissue"]).strip(),
            })

    task_knowledge: dict[str, dict] = {}
    for task_id, slices in sorted(task_slices.items(), key=lambda x: int(x[0].split("_")[1])):
        techs = list(dict.fromkeys(s["sequencing_technique"] for s in slices))
        omics_types = list(dict.fromkeys(s["omics_type"] for s in slices))
        st_cats = list(dict.fromkeys(s["st_category"] for s in slices if s["st_category"]))
        species = list(dict.fromkeys(s["species"] for s in slices))
        tissues = list(dict.fromkeys(s["tissue"] for s in slices))

        entry: dict = {
            "num_slices": len(slices),
            "slice_ids": [s["slice_id"] for s in slices],
            "omics_type": omics_types[0] if len(omics_types) == 1 else omics_types,
            "st_category": st_cats[0] if len(st_cats) == 1 else (st_cats or None),
            "technologies": techs,
            "species": species[0] if len(species) == 1 else species,
            "tissue": tissues[0] if len(tissues) == 1 else tissues,
            # Batch effect fields — only set for multi-slice tasks; None for single-slice stubs
            "batch_type": None,
            "batch_effect_severity": None,
            "batch_description": None,
        }

        # Overlay hand-encoded semantic labels for Tasks 25-28.
        # Technologies are NOT taken from the hand-encoded table — they are already derived
        # from slice data above so they can never drift from the CSV.
        if task_id in TASK_CROSS_PLATFORM_INFO:
            cp = TASK_CROSS_PLATFORM_INFO[task_id]
            entry["batch_type"] = cp["batch_type"]
            entry["batch_effect_severity"] = cp["batch_effect_severity"]
            entry["batch_description"] = cp["description"]
        elif entry["num_slices"] > 1:
            # Regular multi-slice task: batch effect exists but severity not hand-labelled
            if len(techs) == 1:
                entry["batch_type"] = "same_tech"
                # severity stays None — not labelled in manuscript for standard tasks
            else:
                entry["batch_type"] = "cross_tech_mixed"

        task_knowledge[task_id] = entry

    return task_knowledge


def _build_method_summaries(
    method_scores: dict[str, dict],
    task_knowledge: dict[str, dict],
    method_metadata: dict[str, dict],
) -> dict[str, dict]:
    """
    Pre-compute per-method summary statistics broken down by task category.
    Built once at knowledge-base build time — never recomputed per user session.

    Each method gets:
      - overall stats (mean, min, max, n_tasks) across all tasks it ran
      - per-category stats: sST, iST, non_transcriptomic, cross_platform
      - task_coverage: fraction of tasks the method successfully ran
      - worst_tasks: up to 3 tasks where it scored lowest (useful for caveats)
    """
    # Group task IDs by category
    category_tasks: dict[str, list[str]] = {
        "sST": [],
        "iST": [],
        "non_transcriptomic": [],
        "cross_platform": [],
    }
    # cross_platform = only the 4 explicitly hand-labelled Tasks 25-28 (iST/sST cross-technology).
    # Regular same-technology multi-slice tasks (Tasks 1-24) are sST or iST, not cross_platform.
    cross_platform_task_ids = set(TASK_CROSS_PLATFORM_INFO.keys())
    for tid, tk in task_knowledge.items():
        ot = tk.get("omics_type")
        st = tk.get("st_category")
        if tid in cross_platform_task_ids:
            category_tasks["cross_platform"].append(tid)
        elif ot == "transcriptomics" and st == "sST":
            category_tasks["sST"].append(tid)
        elif ot == "transcriptomics" and st == "iST":
            category_tasks["iST"].append(tid)
        elif ot in ("proteomics", "metabolomics", "epigenomics"):
            category_tasks["non_transcriptomic"].append(tid)

    all_task_ids = list(method_scores.keys())
    all_methods = list(next(iter(method_scores.values())).keys())

    summaries: dict[str, dict] = {}
    for method in all_methods:
        # All tasks
        all_scores = [
            method_scores[tid][method]["overall"]
            for tid in all_task_ids
            if method in method_scores[tid] and method_scores[tid][method].get("overall") is not None
        ]

        def stats(scores: list[float]) -> dict | None:
            if not scores:
                return None
            return {
                "mean":   round(sum(scores) / len(scores), 4),
                "min":    round(min(scores), 4),
                "max":    round(max(scores), 4),
                "n_tasks": len(scores),
            }

        # Per-category
        cat_stats: dict[str, dict | None] = {}
        for cat, tids in category_tasks.items():
            cat_scores = [
                method_scores[tid][method]["overall"]
                for tid in tids
                if method in method_scores[tid] and method_scores[tid][method].get("overall") is not None
            ]
            cat_stats[cat] = stats(cat_scores)

        # Task coverage
        n_applicable = sum(
            1 for tid in all_task_ids
            if method in method_scores[tid]
            and method_scores[tid][method].get("completion_status") != "not_applicable"
        )
        coverage = round(n_applicable / len(all_task_ids), 3) if all_task_ids else 0.0

        # Worst tasks (lowest overall score, useful for tradeoff reporting)
        task_score_pairs = [
            (method_scores[tid][method]["overall"], tid)
            for tid in all_task_ids
            if method in method_scores[tid] and method_scores[tid][method].get("overall") is not None
        ]
        task_score_pairs.sort()
        worst_tasks = [{"task_id": tid, "overall": round(s, 4)} for s, tid in task_score_pairs[:3]]

        meta = method_metadata.get(method, {})
        summaries[method] = {
            "overall": stats(all_scores),
            "by_category": cat_stats,
            "task_coverage": coverage,
            "worst_tasks": worst_tasks,
            "deep_learning": meta.get("deep_learning"),
            "omics_agnostic": meta.get("omics_agnostic"),
            "category": meta.get("category"),
            "architecturally_inapplicable": meta.get("architecturally_inapplicable", []),
        }

    return summaries


def build(raw_dir: Path) -> dict:
    data_path = raw_dir / "data_w_sparsity.csv"
    ranks_path = raw_dir / "overall_ranks.csv"

    data_df = pd.read_csv(data_path)

    # ── dataset_profiles ──────────────────────────────────────────────────────
    # A dataset is a single tissue section (slice). It has no batch effect by itself.
    # batch_effect_severity is NOT stored here — see task_knowledge instead.
    dataset_profiles: dict[str, dict] = {}

    for _, row in data_df.iterrows():
        slice_id = str(row["Slice ID"]).strip()
        task_ids = _parse_task_ids(row["Task ID"])
        omics_type, st_category = _parse_omics_type(str(row["Omics type"]))

        sparsity_raw = row["sparsity"]
        sparsity = None if pd.isna(sparsity_raw) else float(sparsity_raw)

        num_locations_raw = row["# Cells/spots"]
        num_locations = None if pd.isna(num_locations_raw) else int(num_locations_raw)

        num_features_raw = row["# Features"]
        num_features = None if pd.isna(num_features_raw) else int(num_features_raw)

        dataset_profiles[slice_id] = {
            "task_id": task_ids[0] if len(task_ids) == 1 else task_ids,
            "omics_type": omics_type,
            "st_category": st_category,
            "sequencing_technique": str(row["Sequencing technique"]).strip(),
            "species": str(row["Species"]).strip(),
            "tissue": str(row["Tissue"]).strip(),
            "batch_id": str(row["Batch ID"]).strip(),
            "num_locations": num_locations,
            "num_features": num_features,
            "sparsity": sparsity,
            # NOTE: no batch_effect_severity here — that is a task-level concept, not slice-level
        }

    # ── task_knowledge ────────────────────────────────────────────────────────
    task_knowledge = _build_task_knowledge(data_df)

    # ── method_scores ─────────────────────────────────────────────────────────
    ranks_df = pd.read_csv(ranks_path, index_col=0)
    ranks_df = ranks_df.reset_index()
    method_scores = _parse_overall_ranks(ranks_df)

    # ── cell_level_scores (scaffold only — data file not yet supplied) ────────
    cell_level_scores: dict[str, dict] = {
        "_comment": (
            "Populated only for single-cell-resolution tasks once the cell-level score "
            "file is supplied. Do not derive from method_scores — domain-level and "
            "cell-level scores are negatively correlated for several methods."
        )
    }

    decision_tree_ref = {"_ref": "See decision_tree.DECISION_TREE in decision_tree.py"}
    method_metadata = {k: dict(v) for k, v in METHOD_METADATA.items()}
    method_summaries = _build_method_summaries(method_scores, task_knowledge, method_metadata)

    return {
        "dataset_profiles": dataset_profiles,
        "task_knowledge": task_knowledge,
        "method_scores": method_scores,
        "method_summaries": method_summaries,
        "cell_level_scores": cell_level_scores,
        "decision_tree": decision_tree_ref,
        "method_metadata": method_metadata,
        "_reconciliation_note": (
            f"overall_ranks.csv contains {ranks_df.shape[0]} methods; "
            f"manuscript evaluates 40. Extra methods vs. manuscript: "
            f"{sorted(EXTRA_METHODS_VS_MANUSCRIPT)}. "
            "These are kept in method_scores but flagged here."
        ),
    }


def main() -> None:
    raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    out_path = Path(__file__).parent.parent.parent / "data" / "clean" / "knowledge_base.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building knowledge base from {raw_dir} …")
    kb = build(raw_dir)
    with open(out_path, "w") as f:
        json.dump(kb, f, indent=2)
    print(f"Written to {out_path}")
    print(f"  datasets      : {len(kb['dataset_profiles'])}")
    tasks = [k for k in kb["method_scores"] if not k.startswith("_")]
    print(f"  tasks         : {len(tasks)}")
    methods = list(next(iter(kb["method_scores"].values())).keys())
    print(f"  methods       : {len(methods)}")
    cp_tasks = [tid for tid, t in kb["task_knowledge"].items() if t.get("batch_type")]
    print(f"  tasks w/ batch info: {len(cp_tasks)} ({', '.join(sorted(cp_tasks))})")
    print(f"  method summaries : {len(kb['method_summaries'])} methods pre-summarised")


if __name__ == "__main__":
    main()
