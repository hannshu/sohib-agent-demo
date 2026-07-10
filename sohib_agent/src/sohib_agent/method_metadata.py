# Hand-encoded from the SOHIB manuscript.
# deep_learning: True if the method uses a neural network backbone.
# omics_agnostic: True if the method can process non-transcriptomic inputs without modification.
# category: coarse grouping used for display only.
# architecturally_inapplicable: list of omics types this method CANNOT run on by design.
#   (transcriptomics methods applied to proteomics/metabolomics/epigenomics will fail or are meaningless)

METHOD_METADATA: dict[str, dict] = {
    "GraphPCA":        {"deep_learning": False, "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "STAIR":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "STAGATE":         {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "CAST":            {"deep_learning": True,  "omics_agnostic": True,  "category": "end_to_end_spatial", "architecturally_inapplicable": []},
    "CellCharter":     {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "DeepST":          {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "SPIRAL":          {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "SpaBatch":        {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "SEDR":            {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "STAligner":       {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "STAIG":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "spCLUE":          {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "STAMP":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "BINARY":          {"deep_learning": False, "omics_agnostic": True,  "category": "non_parametric",    "architecturally_inapplicable": []},
    # FuseMap and DECIPHER each produce two distinct embedding types from the same model.
    # The (niche) variant outputs spatially-smoothed neighbourhood embeddings → suited for domain-level tasks.
    # The (cell) variant outputs per-cell embeddings → suited for cell-level tasks.
    # They are scored separately in the benchmark and should be recommended in different scenarios.
    "FuseMap (niche)": {"deep_learning": True,  "omics_agnostic": True,  "category": "multimodal",        "architecturally_inapplicable": [], "embedding_type": "niche",  "base_method": "FuseMap"},
    "FuseMap (cell)":  {"deep_learning": True,  "omics_agnostic": True,  "category": "multimodal",        "architecturally_inapplicable": [], "embedding_type": "cell",   "base_method": "FuseMap"},
    "DECIPHER (niche)":{"deep_learning": True,  "omics_agnostic": True,  "category": "multimodal",        "architecturally_inapplicable": [], "embedding_type": "niche",  "base_method": "DECIPHER"},
    "DECIPHER (cell)": {"deep_learning": True,  "omics_agnostic": True,  "category": "multimodal",        "architecturally_inapplicable": [], "embedding_type": "cell",   "base_method": "DECIPHER"},
    "stClinic":        {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "scNiche":         {"deep_learning": True,  "omics_agnostic": False, "category": "niche",             "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "SpiceMix":        {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "INSTINCT":        {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "MENDER":          {"deep_learning": False, "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "STACI":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "BANKSY":          {"deep_learning": False, "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "DeepGFT":         {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "scVI":            {"deep_learning": True,  "omics_agnostic": False, "category": "scRNA_adapted",     "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "MaskGraphene":    {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "Harmony":         {"deep_learning": False, "omics_agnostic": False, "category": "scRNA_adapted",     "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "PRECAST":         {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "GraphST":         {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "Scanorama":       {"deep_learning": False, "omics_agnostic": False, "category": "scRNA_adapted",     "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "SpaVAE":          {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "SIMVI":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "CELLama":         {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",         "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "CellNiche":       {"deep_learning": True,  "omics_agnostic": False, "category": "niche",             "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "Novae":           {"deep_learning": True,  "omics_agnostic": False, "category": "graph_spatial",     "architecturally_inapplicable": []},
    "scGPT-spatial":   {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",         "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "SToFM":           {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",         "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "NicheCompass":    {"deep_learning": True,  "omics_agnostic": False, "category": "niche",             "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "CellPLM":         {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",         "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
    "Nicheformer":     {"deep_learning": True,  "omics_agnostic": False, "category": "llm_based",         "architecturally_inapplicable": ["proteomics", "metabolomics", "epigenomics"]},
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
