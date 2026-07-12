"""
Extract UserProfile fields from an AnnData (.h5ad) file.

Tier 1: numeric fields always derivable from the matrix shape.
Tier 2: metadata fields extracted by scanning obs/uns columns with alias lists.
         Returns "confirmed" provenance.
Tier 3 (heuristic): best-effort inference when Tier 2 finds nothing.
         Returns "inferred" provenance — must be confirmed by user before using at full weight
         in matching. Currently implemented:
           - species: inferred from gene-symbol capitalisation convention
           - st_category: inferred from feature count (sST ≥ threshold, iST < threshold)
           - omics_type: inferred from var_names patterns (peak coordinates → epigenomics,
             numeric m/z-style names → metabolomics, small antibody panel → proteomics,
             gene-symbol-like names → transcriptomics). Left null (caller asks the user) when
             the pattern is inconclusive — omics_type is a gating field, so a wrong guess here
             is worse than asking.

Not implemented (scoped out for this pass):
  - batch_effect_severity: requires a batch-mixing diagnostic (e.g. iLISI/kBET) against a
    batch/sample column. The manuscript provides no general numeric thresholds for this field
    outside its own four specific tasks. Classified as Tier 3: computed diagnostics, future work.
  - tissue, exact technology: no reliable structural signal in a raw matrix — always remain None
    from Tiers 1-3 and are requested from the user.

Multi-batch handling: if an uploaded file contains multiple distinct batch/sample values in .obs,
this module computes aggregate statistics and labels them as cross-batch aggregates. Each numeric
field is computed across all cells/spots in the file. This is documented explicitly in the returned
profile dict via _multi_batch_aggregate = True.
"""
from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np


# Alias lists for Tier 2 field extraction (case-insensitive)
_TISSUE_ALIASES   = ["tissue", "organ", "tissue_type", "anatomical_region"]
_SPECIES_ALIASES  = ["species", "organism", "organism_ontology_term_id"]
_TECH_ALIASES     = ["technology", "platform", "seq_platform", "sequencing_platform",
                      "assay", "assay_ontology_term_id", "protocol"]
_OMICS_ALIASES    = ["omics_type", "omics", "modality", "feature_type", "assay_type", "data_type"]
_BATCH_ALIASES    = ["batch", "sample", "sample_id", "slice_id", "section",
                      "donor_id", "patient_id", "library_id"]

# Recognized omics_type values a Tier 2 metadata scan may find verbatim (case-insensitive).
_OMICS_TYPE_VALUES = {"transcriptomics", "proteomics", "metabolomics", "epigenomics"}

# Tier 3 heuristic for omics_type inference from var_names patterns.
_PEAK_NAME_PATTERN = re.compile(r"^(chr)?[0-9xym]+[:_-]\d+[-_]\d+$", re.IGNORECASE)
# Imaging-based proteomics panels (CODEX/MIBI) profile tens to ~a hundred markers — far below
# even the smallest iST gene panels, which is what makes this cutoff a usable signal.
_PROTEOMICS_PANEL_SIZE_CUTOFF = 100

# Tier 3 heuristic threshold for st_category inference.
# Imaging-based ST (iST) uses targeted panels (roughly hundreds to ~a few thousand genes).
# Sequencing-based ST (sST) captures whole-transcriptome data (tens of thousands of genes).
# This cutoff is a heuristic chosen by inspection of the SOHIB benchmark feature counts,
# NOT derived from a general rule stated in the manuscript.
_IST_FEATURE_COUNT_CUTOFF = 5000  # features < this → likely iST


def _wrap(value, confidence: str) -> dict:
    """Wrap a field value with a provenance tag."""
    return {"value": value, "confidence": confidence}


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


def _infer_species_from_var_names(adata: ad.AnnData) -> str | None:
    """
    Tier 3 heuristic: infer species from gene-symbol capitalisation convention.
    Human gene symbols are conventionally ALL_UPPERCASE (e.g. TP53, ACTB).
    Mouse gene symbols are Titlecase (e.g. Trp53, Actb).
    Returns "human", "mouse", or None (inconclusive).
    """
    var_names = list(adata.var_names[:500])  # sample first 500 genes
    if not var_names:
        return None

    # Filter to names that are purely alphabetic (skip IDs like ENSG000...)
    alpha_names = [n for n in var_names if n.isalpha() and len(n) >= 2]
    if len(alpha_names) < 10:
        return None

    upper_count = sum(1 for n in alpha_names if n == n.upper())
    title_count = sum(1 for n in alpha_names if n[0].isupper() and n[1:] == n[1:].lower())
    total = len(alpha_names)

    upper_frac = upper_count / total
    title_frac = title_count / total

    if upper_frac >= 0.6:
        return "human"
    if title_frac >= 0.6:
        return "mouse"
    return None


def _infer_omics_type_from_var_names(adata: ad.AnnData) -> str | None:
    """
    Tier 3 heuristic: infer omics_type from var_names naming patterns.
    Returns "epigenomics", "metabolomics", "proteomics", "transcriptomics", or None (inconclusive).
    Checked in this order because each pattern is a more specific/confident signal than the next:
      1. genomic peak coordinates (e.g. "chr1:100000-100500") → epigenomics (ATAC/ChIP-style)
      2. numeric feature names (e.g. m/z values like "756.5522") → metabolomics (MALDI/MSI)
      3. small feature count (<= _PROTEOMICS_PANEL_SIZE_CUTOFF) → proteomics (antibody panel)
      4. gene-symbol-like alphabetic names → transcriptomics (the common case)
    """
    var_names = [str(n) for n in adata.var_names[:500]]
    if not var_names:
        return None

    peak_count = sum(1 for n in var_names if _PEAK_NAME_PATTERN.match(n))
    if peak_count / len(var_names) >= 0.6:
        return "epigenomics"

    def _is_numeric(n: str) -> bool:
        try:
            float(n)
            return True
        except ValueError:
            return False

    numeric_count = sum(1 for n in var_names if _is_numeric(n))
    if numeric_count / len(var_names) >= 0.6:
        return "metabolomics"

    if len(adata.var_names) <= _PROTEOMICS_PANEL_SIZE_CUTOFF:
        return "proteomics"

    alpha_names = [n for n in var_names if n.isalpha() and len(n) >= 2]
    if len(alpha_names) / len(var_names) >= 0.6:
        return "transcriptomics"

    return None


def _infer_st_category_from_feature_count(n_vars: int) -> str | None:
    """
    Tier 3 heuristic: estimate st_category from feature count.
    Below _IST_FEATURE_COUNT_CUTOFF: likely imaging-based (iST) targeted panel.
    At or above cutoff: likely sequencing-based (sST) whole-transcriptome.
    Returns "iST", "sST", or None.
    """
    if n_vars < _IST_FEATURE_COUNT_CUTOFF:
        return "iST"
    return "sST"


def _detect_multi_batch(adata: ad.AnnData) -> tuple[bool, str | None]:
    """
    Check whether the file contains multiple distinct batch/sample values.
    Returns (is_multi_batch, batch_column_name_or_None).
    """
    obs_cols_lower = {c.lower(): c for c in adata.obs.columns}
    for alias in _BATCH_ALIASES:
        col = obs_cols_lower.get(alias.lower())
        if col is not None:
            n_unique = adata.obs[col].nunique()
            if n_unique > 1:
                return True, col
    return False, None


def extract_tier2(adata: ad.AnnData) -> dict:
    """
    Best-effort extraction of metadata fields with provenance tagging.

    Returns a dict where each field that can be either confirmed or inferred is wrapped as
    {"value": ..., "confidence": "confirmed"|"inferred"}. Fields that are always numeric
    (num_locations, num_features, sparsity) are returned unwrapped for direct use.

    Fields not found here stay None and are requested from the user via the text path.
    tissue and exact technology are not inferred — no reliable structural signal exists.
    omics_type is inferred from var_names patterns (Tier 3) when no explicit metadata column
    names it; if that's also inconclusive it stays None, since it's a gating field and a wrong
    guess would corrupt matching more than asking the user costs.
    """
    # Tier 2: confirmed fields from explicit metadata
    tissue_val    = _scan_obs(adata, _TISSUE_ALIASES)   or _scan_uns(adata, _TISSUE_ALIASES)
    species_val   = _scan_obs(adata, _SPECIES_ALIASES)  or _scan_uns(adata, _SPECIES_ALIASES)
    tech_val      = _scan_obs(adata, _TECH_ALIASES)     or _scan_uns(adata, _TECH_ALIASES)
    batch_val     = _scan_obs(adata, _BATCH_ALIASES)    or _scan_uns(adata, _BATCH_ALIASES)
    omics_val     = _scan_obs(adata, _OMICS_ALIASES)    or _scan_uns(adata, _OMICS_ALIASES)
    if omics_val is not None and omics_val.lower() not in _OMICS_TYPE_VALUES:
        omics_val = None  # metadata column existed but didn't hold a recognized omics_type value

    # Tier 3: infer species from gene names if metadata not found
    if species_val is None:
        inferred_species = _infer_species_from_var_names(adata)
        species_field = _wrap(inferred_species, "inferred") if inferred_species else None
    else:
        species_field = _wrap(species_val, "confirmed")

    # Tier 3: infer omics_type from var_names patterns if metadata not found
    if omics_val is None:
        inferred_omics = _infer_omics_type_from_var_names(adata)
        omics_field = _wrap(inferred_omics, "inferred") if inferred_omics else None
    else:
        omics_field = _wrap(omics_val.lower(), "confirmed")

    # tissue and technology: remain None — no reliable heuristic
    tissue_field = _wrap(tissue_val, "confirmed") if tissue_val is not None else None
    tech_field   = _wrap(tech_val,   "confirmed") if tech_val   is not None else None

    return {
        "tissue":     tissue_field,
        "species":    species_field,
        "technology": tech_field,
        "omics_type": omics_field,
        "batch_id":   batch_val,
    }


def extract_tier3_st_category(n_vars: int, technology: str | None) -> dict | None:
    """
    Infer st_category when technology is not known.
    Returns a wrapped {"value": ..., "confidence": "inferred"} or None if technology is known
    (caller should derive st_category from technology directly in that case).
    """
    if technology is not None:
        return None  # caller should use technology to determine st_category
    inferred = _infer_st_category_from_feature_count(n_vars)
    if inferred is None:
        return None
    return _wrap(inferred, "inferred")


def load_and_extract(path: str | Path) -> dict:
    """
    Load an h5ad file and return merged Tier1 + Tier2 + Tier3 fields.

    Multi-batch handling: if the file contains multiple distinct batch/sample values,
    computes aggregate statistics across all cells/spots and sets _multi_batch_aggregate=True.
    The caller should surface this to the user.
    """
    adata = ad.read_h5ad(path)
    t1 = extract_tier1(adata)
    t2 = extract_tier2(adata)

    # Detect multi-batch
    is_multi_batch, batch_col = _detect_multi_batch(adata)

    # Infer st_category if technology is unknown
    tech_field = t2.get("technology")
    tech_value = tech_field["value"] if isinstance(tech_field, dict) else tech_field
    st_cat_field = extract_tier3_st_category(t1["num_features"], tech_value)

    result = {**t1, **t2}
    if st_cat_field is not None:
        result["st_category_inferred"] = st_cat_field

    result["_multi_batch_aggregate"] = is_multi_batch
    if is_multi_batch:
        result["_multi_batch_note"] = (
            f"File contains multiple distinct values in column '{batch_col}'. "
            "Numeric statistics (num_locations, sparsity) are computed across all batches combined. "
            "Consider splitting by batch and running per-batch profiles for more accurate matching."
        )

    return result
