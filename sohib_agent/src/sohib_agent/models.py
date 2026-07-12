from pydantic import BaseModel
from typing import Literal, Optional

# Non-gating but high-weight fields (see matching._WEIGHTS): missing these doesn't block
# a recommendation, but filling them in materially improves match quality. Used to decide
# whether to ask one more round of questions before answering, or answer immediately.
_HIGH_VALUE_FIELDS: list[tuple[str, str]] = [
    ("technology", "technology (e.g. Visium, MERFISH, Slide-seq, STARMap, Xenium, CosMx)"),
    ("species", "species (human / mouse / other)"),
    ("st_category", "sST (sequencing-based, whole-transcriptome) or iST (imaging-based, targeted panel)"),
    ("num_locations", "approx. number of cells/spots per section"),
    ("batch_effect_severity", "batch_effect_severity (minimal / moderate / maximal) — leave blank if this is a single sample with no batch to integrate"),
]


class UserProfile(BaseModel):
    # Required before final answer — these two fields change the recommendation outright
    target_resolution: Optional[Literal["domain", "cell"]] = None
    omics_type: Optional[Literal["transcriptomics", "proteomics", "metabolomics", "epigenomics"]] = None

    # Only meaningful when omics_type == "transcriptomics"
    st_category: Optional[Literal["sST", "iST"]] = None

    technology: Optional[str] = None
    species: Optional[str] = None
    tissue: Optional[str] = None
    num_locations: Optional[int] = None
    num_features: Optional[int] = None
    sparsity: Optional[float] = None
    batch_effect_severity: Optional[Literal["minimal", "moderate", "maximal"]] = None

    priority: Literal["accuracy", "speed", "memory", "balanced"] = "balanced"
    avoid_deep_learning: bool = False
    source: Literal["text", "h5ad", "text+h5ad"] = "text"

    def is_ready_for_recommendation(self) -> bool:
        return self.target_resolution is not None and self.omics_type is not None

    def missing_required_fields(self) -> list[str]:
        missing = []
        if self.target_resolution is None:
            missing.append("target_resolution (domain-level spatial domains, or single-cell resolution?)")
        if self.omics_type is None:
            missing.append("omics_type (transcriptomics, proteomics, metabolomics, or epigenomics?)")
        return missing

    def missing_optional_fields(self) -> list[str]:
        """
        High-value, non-gating fields still missing. A recommendation can be produced
        without these, but matching quality improves materially if they're filled in.
        st_category is only asked when relevant (transcriptomics).
        """
        missing = []
        for field, description in _HIGH_VALUE_FIELDS:
            if field == "st_category" and self.omics_type != "transcriptomics":
                continue
            if getattr(self, field) is None:
                missing.append(description)
        return missing

    def is_fully_specified(self) -> bool:
        """True when there is nothing left worth asking for before recommending."""
        return self.is_ready_for_recommendation() and not self.missing_optional_fields()


class RecommendationResult(BaseModel):
    matched_branch: Optional[str]         # DEPRECATED: kept for API compat; use decision_tree_crossref
    matched_datasets: list[dict]
    # Deterministic, evidence-backed pool computed by matching.py (up to top_k methods, each with
    # a real composite_score/evidence/warnings). This is ground truth — recommended_methods must
    # only ever contain entries drawn from this pool, never an invented method or score.
    candidate_methods: list[dict] = []
    # Final top-3 (or fewer). Populated either by the LLM's bounded joint selection from
    # candidate_methods (selection_source="llm_joint"), or deterministically by composite_score
    # (selection_source="deterministic_fallback"/"static_branch") in --no-llm mode or when the
    # LLM's selection fails validation against candidate_methods.
    recommended_methods: list[dict]
    discarded_methods: list[dict]         # candidate_methods not chosen into recommended_methods
    confidence_note: str                  # explicit coverage/confidence statement, never omitted
    # Cross-reference noting when the fine-grained answer happens to align with a SOHIB
    # decision-tree branch — informational only, never the source of recommended_methods.
    decision_tree_crossref: Optional[str] = None
    selection_source: Literal["llm_joint", "deterministic_fallback", "static_branch"] = "deterministic_fallback"
