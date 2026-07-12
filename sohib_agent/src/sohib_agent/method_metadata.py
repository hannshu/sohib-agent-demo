# Hand-encoded from the SOHIB manuscript.
# deep_learning: True if the method uses a neural network backbone.
# omics_agnostic: True if the method can process non-transcriptomic inputs without modification.
#   Note: CellCharter achieves cross-omics applicability via per-omics encoder swapping — a different
#   mechanism than CAST/DECIPHER/BINARY (which use omics-agnostic architectures). It is tagged
#   omics_agnostic=False here to distinguish the mechanism; see omics_cross_strategy for details.
# category: coarse grouping used for display only.
# architecturally_inapplicable: list of omics types this method CANNOT run on by design.
#   This is a HARD EXCLUSION — methods in this list are removed from recommended_methods entirely
#   when the user's omics_type is listed. It is not a preference; it is a design constraint.
# designed_resolution: primary resolution the method was designed and performs best for,
#   based on SOHIB manuscript results (Task 14 CLP vs. BPC/BER rankings, Figure 4c/4d).
#   "domain"  — better at spatial domain-level integration (high BPC/BER/SPC/DTP scores)
#   "cell"    — better at cell-type-level integration (high CLP scores)
#   "both"    — performs well on both axes
#   "niche"   — designed for niche/neighbourhood embeddings (FuseMap/DECIPHER niche variants)
#   None      — insufficient benchmark data to determine (foundation models tested on limited tasks)

METHOD_METADATA: dict[str, dict] = {
    # Graph/spatial methods — designed for domain-level integration (spatial domain structure preservation)
    "GraphPCA":        {"deep_learning": False, "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "STAIR":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "STAGATE":         {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    # CellCharter achieves cross-omics via per-omics encoder swapping, not an omics-agnostic architecture.
    "CellCharter":     {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain", "omics_cross_strategy": "per_omics_encoder"},
    "DeepST":          {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "SPIRAL":          {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "SpaBatch":        {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "SEDR":            {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "STAligner":       {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "STAIG":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "spCLUE":          {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "STAMP":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "SpiceMix":        {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "INSTINCT":        {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "MENDER":          {"deep_learning": False, "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "STACI":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "DeepGFT":         {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "MaskGraphene":    {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "PRECAST":         {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "GraphST":         {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "SpaVAE":          {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "SIMVI":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "Novae":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "stClinic":        {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    # Omics-agnostic methods — designed for cross-omics domain-level integration
    "CAST":            {"deep_learning": True,  "omics_agnostic": True,  "category": "end_to_end_spatial", "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    "BINARY":          {"deep_learning": False, "omics_agnostic": True,  "category": "non_parametric",     "architecturally_inapplicable": [],                                          "designed_resolution": "domain"},
    # FuseMap and DECIPHER each produce two distinct embedding types from the same model.
    # The (niche) variant outputs spatially-smoothed neighbourhood embeddings → suited for domain-level tasks.
    # The (cell) variant outputs per-cell embeddings → suited for cell-level tasks.
    # They are scored separately in the benchmark and should be recommended in different scenarios.
    "FuseMap (niche)": {"deep_learning": True,  "omics_agnostic": True,  "category": "multimodal",         "architecturally_inapplicable": [], "embedding_type": "niche", "base_method": "FuseMap",    "designed_resolution": "niche"},
    "FuseMap (cell)":  {"deep_learning": True,  "omics_agnostic": True,  "category": "multimodal",         "architecturally_inapplicable": [], "embedding_type": "cell",  "base_method": "FuseMap",    "designed_resolution": "cell"},
    "DECIPHER (niche)":{"deep_learning": True,  "omics_agnostic": True,  "category": "multimodal",         "architecturally_inapplicable": [], "embedding_type": "niche", "base_method": "DECIPHER",   "designed_resolution": "niche"},
    "DECIPHER (cell)": {"deep_learning": True,  "omics_agnostic": True,  "category": "multimodal",         "architecturally_inapplicable": [], "embedding_type": "cell",  "base_method": "DECIPHER",   "designed_resolution": "cell"},
    # scRNA-adapted methods — transcriptomics-only; perform comparatively better at cell-level
    # (manuscript Figure 4c/4d: Scanorama, Harmony, BANKSY top the CLP-based cell-level branch)
    "scVI":            {"deep_learning": True,  "omics_agnostic": False, "category": "scRNA_adapted",      "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": "cell"},
    "Harmony":         {"deep_learning": False, "omics_agnostic": False, "category": "scRNA_adapted",      "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": "cell"},
    "Scanorama":       {"deep_learning": False, "omics_agnostic": False, "category": "scRNA_adapted",      "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": "cell"},
    # BANKSY: graph-spatial but performs comparatively better at cell-level per manuscript
    "BANKSY":          {"deep_learning": False, "omics_agnostic": False, "category": "graph_spatial",      "architecturally_inapplicable": [],                                          "designed_resolution": "cell"},
    # Niche methods — spatial niche/neighbourhood analysis; transcriptomics-only
    "scNiche":         {"deep_learning": True,  "omics_agnostic": False, "category": "niche",              "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": "domain"},
    "CellNiche":       {"deep_learning": True,  "omics_agnostic": False, "category": "niche",              "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": "domain"},
    "NicheCompass":    {"deep_learning": True,  "omics_agnostic": False, "category": "niche",              "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": "domain"},
    # Foundation/LLM-based methods — transcriptomics-only; limited benchmark coverage
    "CELLama":         {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",          "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": None},
    "scGPT-spatial":   {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",          "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": None},
    "SToFM":           {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",          "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": None},
    "CellPLM":         {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",          "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": None},
    "Nicheformer":     {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",          "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"], "designed_resolution": None},
}

# Hand-encoded task-level metadata for the four cross-platform tasks (25-28).
# Batch effect is a property of the TASK (multi-slice integration challenge), not of any individual slice.
# batch_type describes the source of batch effect:
#   "same_tech"        — all slices from same sequencing technology (within-platform batch)
#   "cross_tech_iST"   — cross-technology, both imaging-based ST (MERFISH + STARMap)
#   "cross_tech_mixed" — cross-technology, iST + sST (MERFISH + Slide-seq)
# batch_effect_severity reflects the gradient described in the manuscript:
#   minimal (same tech, same principle) → moderate (cross-tech, same class) → maximal (cross-tech, different class)
# Technologies are NOT encoded here — they are derived automatically from data_w_sparsity.csv
# in _build_task_knowledge() to prevent encoding drift.
# Only the semantic labels (batch_type, severity, description) are hand-annotated here.
TASK_CROSS_PLATFORM_INFO: dict[str, dict] = {
    "Task_25": {
        "batch_type": "same_tech",
        "batch_effect_severity": "minimal",
        "description": "MERFISH vs MERFISH — same platform, different batches",
    },
    "Task_26": {
        "batch_type": "cross_tech_iST",
        "batch_effect_severity": "moderate",
        "description": "MERFISH vs STARMap — both iST, different imaging platforms",
    },
    "Task_27": {
        "batch_type": "cross_tech_mixed",
        "batch_effect_severity": "maximal",
        "description": "Slide-seq (sST) vs MERFISH (iST) — different sequencing principles",
    },
    "Task_28": {
        "batch_type": "cross_tech_mixed",
        "batch_effect_severity": "maximal",
        "description": "Slide-seq (sST) vs STARMap (iST) — different sequencing principles",
    },
}
