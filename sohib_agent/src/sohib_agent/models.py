from pydantic import BaseModel
from typing import Literal, Optional


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


class RecommendationResult(BaseModel):
    matched_branch: Optional[str]       # decision-tree branch name, if matched
    matched_datasets: list[dict]
    recommended_methods: list[dict]
    discarded_methods: list[dict]
    confidence_note: str                 # explicit coverage/confidence statement, never omitted
