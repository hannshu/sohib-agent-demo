"""
Extract UserProfile fields from an AnnData (.h5ad) file.
Tier 1: numeric fields always derivable from the matrix shape.
Tier 2: metadata fields extracted by scanning obs/uns columns with alias lists.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np


# Alias lists for Tier 2 field extraction (case-insensitive)
_TISSUE_ALIASES   = ["tissue", "organ", "tissue_type", "anatomical_region"]
_SPECIES_ALIASES  = ["species", "organism", "organism_ontology_term_id"]
_TECH_ALIASES     = ["technology", "platform", "seq_platform", "sequencing_platform",
                      "assay", "assay_ontology_term_id", "protocol"]
_BATCH_ALIASES    = ["batch", "sample", "sample_id", "slice_id", "section",
                      "donor_id", "patient_id", "library_id"]


def extract_tier1(adata: ad.AnnData) -> dict:
    n_obs, n_vars = adata.shape
    sparsity: float | None = None
    try:
        if hasattr(adata.X, "nnz"):
            total = n_obs * n_vars
            sparsity = 1.0 - (adata.X.nnz / total) if total > 0 else None
        elif isinstance(adata.X, np.ndarray):
            total = n_obs * n_vars
            sparsity = float(np.sum(adata.X == 0) / total) if total > 0 else None
    except Exception:
        sparsity = None

    has_spatial = "spatial" in adata.obsm

    return {
        "num_locations": n_obs,
        "num_features": n_vars,
        "sparsity": round(sparsity, 6) if sparsity is not None else None,
        "has_spatial_coords": has_spatial,
    }


def _scan_obs(adata: ad.AnnData, aliases: list[str]) -> str | None:
    """Look up a field in adata.obs by checking alias list (case-insensitive)."""
    obs_cols_lower = {c.lower(): c for c in adata.obs.columns}
    for alias in aliases:
        col = obs_cols_lower.get(alias.lower())
        if col is not None:
            vals = adata.obs[col].dropna().unique()
            if len(vals) == 1:
                return str(vals[0])
            if len(vals) > 1:
                # Return most common value
                return str(adata.obs[col].value_counts().index[0])
    return None


def _scan_uns(adata: ad.AnnData, aliases: list[str]) -> str | None:
    """Look up a field in adata.uns by checking alias list (case-insensitive)."""
    uns_keys_lower = {k.lower(): k for k in adata.uns}
    for alias in aliases:
        key = uns_keys_lower.get(alias.lower())
        if key is not None:
            val = adata.uns[key]
            if isinstance(val, (str, int, float)):
                return str(val)
    return None


def extract_tier2(adata: ad.AnnData) -> dict:
    """
    Best-effort extraction of metadata fields.
    Returns None per field if not found — never guesses.
    Fields not found here stay None and are asked of the user via the text path.
    """
    tissue   = _scan_obs(adata, _TISSUE_ALIASES)   or _scan_uns(adata, _TISSUE_ALIASES)
    species  = _scan_obs(adata, _SPECIES_ALIASES)  or _scan_uns(adata, _SPECIES_ALIASES)
    technology = _scan_obs(adata, _TECH_ALIASES)   or _scan_uns(adata, _TECH_ALIASES)
    batch    = _scan_obs(adata, _BATCH_ALIASES)     or _scan_uns(adata, _BATCH_ALIASES)

    return {
        "tissue": tissue,
        "species": species,
        "technology": technology,
        "batch_id": batch,
    }


def load_and_extract(path: str | Path) -> dict:
    """Load an h5ad file and return merged Tier1 + Tier2 fields."""
    adata = ad.read_h5ad(path)
    t1 = extract_tier1(adata)
    t2 = extract_tier2(adata)
    return {**t1, **t2}
