"""
Decision tree encoding of SOHIB Figure 5.
Encoded as data so it can be updated without touching matching logic.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import UserProfile

# Each branch specifies conditions that must ALL be satisfied by non-null profile fields.
# Special condition keys:
#   num_locations_lt  — profile.num_locations < value
#   num_locations_gte — profile.num_locations >= value
#   omics_type        — may be a list (any-of match)
DECISION_TREE: list[dict] = [
    {
        "branch": "cell_level_analysis",
        "conditions": {"target_resolution": "cell"},
        "top_methods": ["Scanorama", "Harmony", "BANKSY"],
        "reported_axis": "CLP",
        "description": "Cell-level analysis: published SOHIB top-3 for single-cell-resolution tasks.",
    },
    {
        "branch": "non_transcriptomic",
        "conditions": {
            "target_resolution": "domain",
            "omics_type": ["proteomics", "metabolomics", "epigenomics"],
        },
        "top_methods": ["CAST", "BINARY", "DECIPHER (niche)"],
        "reported_axis": ["BPC", "BER", "SPC", "DTP"],
        "description": "Non-transcriptomic spatial omics: methods evaluated on Tasks 29-33.",
    },
    {
        "branch": "sST_small",
        "conditions": {
            "target_resolution": "domain",
            "omics_type": "transcriptomics",
            "st_category": "sST",
            "num_locations_lt": 5000,
        },
        "top_methods": ["STAGATE", "GraphPCA", "STAIR"],
        "reported_axis": ["BPC", "BER", "SPC", "DTP"],
        "description": "Sequencing-based ST, small datasets (<5000 spots): Tasks 1-4.",
    },
    {
        "branch": "sST_large",
        "conditions": {
            "target_resolution": "domain",
            "omics_type": "transcriptomics",
            "st_category": "sST",
            "num_locations_gte": 5000,
        },
        "top_methods": ["STAIR", "CellCharter", "DECIPHER (niche)"],
        "reported_axis": ["BPC", "BER", "SPC", "DTP"],
        "description": "Sequencing-based ST, large datasets (>=5000 spots): Tasks 5-11.",
    },
    {
        "branch": "iST",
        "conditions": {
            "target_resolution": "domain",
            "omics_type": "transcriptomics",
            "st_category": "iST",
        },
        "top_methods": ["STAIR", "STAGATE", "DECIPHER (niche)"],
        "reported_axis": ["BPC", "BER", "SPC", "DTP"],
        "description": "Imaging-based ST: Tasks 12-24.",
    },
]


def _condition_satisfied(condition_key: str, condition_val, profile: "UserProfile") -> bool:
    """Return True if a single condition key/value is satisfied by the profile."""
    if condition_key == "num_locations_lt":
        if profile.num_locations is None:
            return False
        return profile.num_locations < condition_val

    if condition_key == "num_locations_gte":
        if profile.num_locations is None:
            return False
        return profile.num_locations >= condition_val

    profile_val = getattr(profile, condition_key, None)
    if profile_val is None:
        return False

    if isinstance(condition_val, list):
        return profile_val in condition_val

    return profile_val == condition_val


def match_branch(profile: "UserProfile") -> dict | None:
    """
    Return the first branch whose conditions are all satisfied, or None.
    None means the profile does not cleanly map to any branch (e.g. num_locations
    is missing so sST_small vs sST_large cannot be determined).
    """
    for branch in DECISION_TREE:
        if all(
            _condition_satisfied(k, v, profile)
            for k, v in branch["conditions"].items()
        ):
            return branch
    return None
