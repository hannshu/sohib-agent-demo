# SOHIB Recommendation Agent — Build Specification

This document is written to be handed directly to a code agent (for example Claude Code) as a build spec. It is not an academic proposal. Every section is meant to be actionable: concrete file schemas, concrete function signatures, concrete prompts, and an ordered task list with acceptance criteria. Where a decision genuinely depends on data the code agent has not seen yet, that is called out explicitly under "Open items" rather than guessed.

## 0. One-paragraph goal

Build a command-line agent, modeled on the existing open-source `soi_bench_agent` project, that takes either a short natural-language description of a new spatial omics dataset or an uploaded `.h5ad` file, matches it against the SOHIB benchmark (164 tissue sections, 40 integration methods, four metric categories plus cell-level metrics), and returns a ranked list of recommended integration methods with the specific benchmark evidence behind each recommendation. All numeric ranking is done by plain, deterministic code. A language model is used only to (a) extract structured fields from free text and (b) narrate the retrieved evidence in prose. The language model never computes or invents a score.

## 1. Tech stack

- Python 3.11+, packaged with `uv` (mirrors the reference project's `pyproject.toml` + `uv sync` pattern).
- `pandas` for the tabular knowledge base.
- `anndata` and `scanpy` for reading `.h5ad` files (do not hand-roll an AnnData parser).
- `pydantic` (preferred over plain dataclasses this time) for the profile schema, since it gives free JSON-schema export for the LLM extraction prompt and free validation.
- An LLM client (Anthropic or OpenAI, whichever the user already has API access to) used only inside two isolated modules: `profile_extraction.py` and `answer_writer.py`.
- `rich` for CLI output, matching the reference project's terminal UX.
- `pytest` for tests.

## 2. Repository layout

```
sohib_agent/
  data/
    raw/                       # the CSVs as supplied: data_w_sparsity.csv, overall_ranks.csv, task_*.csv
    clean/
      knowledge_base.json      # built artifact, see Section 4
  src/sohib_agent/
    models.py                 # UserProfile, RecommendationResult (Section 5)
    knowledge_base.py          # load/save knowledge_base.json
    build_knowledge_base.py    # raw CSVs -> knowledge_base.json (Section 4)
    decision_tree.py           # encodes Figure 5 as data (Section 6)
    matching.py                # dataset similarity + method ranking (Section 7)
    h5ad_profile.py             # Tier 1 / Tier 2 extraction from .h5ad (Section 8)
    profile_extraction.py      # LLM call: free text -> UserProfile fields (Section 9a)
    answer_writer.py           # LLM call: evidence -> prose (Section 9b)
    cli.py                     # chat loop, /profile /reset /quit commands
  prompts/
    extract_profile.md
    recommendation_answer.md
  tests/
    test_matching.py
    test_h5ad_profile.py
    test_leave_one_out.py       # Section 11
  pyproject.toml
  README.md
```

This mirrors the structure of the reference `soi_bench_agent` project directly, which is a deliberate choice: that project is a working implementation of the same architecture, and departing from its structure for no reason adds risk without benefit.

## 3. Input data contracts

Three files are confirmed available today. State any assumption below explicitly in code comments, since a wrong assumption here breaks everything downstream silently.

### 3.1 `data_w_sparsity.csv` — dataset meta-feature table

Columns confirmed present: `Slice ID, Task ID, Omics type, Sequencing technique, Species, Tissue, Batch ID, # Cells/spots, # Features, sparsity`. One row per tissue section (164 rows total).

- `Slice ID` is the join key into per-slice information.
- `Task ID` groups slices into the 33 SOHIB tasks and is the join key into `overall_ranks.csv` column prefixes (`task_1_...`, `task_2_...`, etc.).
- `Omics type` takes values such as `Spatial transcriptomics (sST)` — note the sST/iST distinction (sequencing-based vs. imaging-based) is encoded inside this field's text and must be parsed out into its own boolean/categorical field, since it is a first-level branch in the decision tree (Section 6).

### 3.2 `overall_ranks.csv` — aggregated method x task performance table

Wide format: one row per method (43 methods observed, though the manuscript evaluates 40 — reconcile the 3-method difference before building the knowledge base; do not silently drop or silently keep rows). Columns are named `task_{N}_{platform_path}_{suffix}` where `suffix` is one of `overall, BPC, BER, SPC, DTP`. Example column: `task_5_sST/Array/kidny_overall`.

- Parse the column name into three parts: task number, a platform/tissue path segment (used to cross-check against `data_w_sparsity.csv`), and the metric suffix.
- Missing values (blank cells, seen in the sample for tasks a method could not run) must be preserved as `null`, not coerced to 0 — a 0 score and a "did not run" are different facts and conflating them will silently corrupt the ranking engine.

### 3.3 `task_{N}.csv` — granular pre-aggregation metric table (one file per task)

Only `task_5.csv` has been supplied so far; the same structure is expected for other task numbers when supplied later. Rows are individual metric names (for example `batch_metrics_iLISI`, `batch_metrics_ASW_batch`, `batch_metrics_PCR_comparison`, `batch_metrics_kBET`), columns are method names. These map directly onto the manuscript's named BER metrics (iLISI, batch ASW, PCR, kBET), and analogous row-name patterns should exist for BPC, SPC, and DTP metrics in other task files (`batch_metrics_*` for BER; expect `bio_metrics_*`, `spatial_metrics_*`, `downstream_metrics_*` or similarly prefixed rows for the other three categories — confirm the actual prefixes against a second task file before hardcoding a parser, since only one task file has been seen so far).

### 3.4 Cell-level score file — required, not yet supplied

None of the three files supplied so far contain per-method cell-level scores. The manuscript reports, for single-cell-resolution tasks, a separate set of cell-level metrics: classification accuracy, macro-averaged F1, isolated labels, cell-type LISI, and cell-type ASW, computed against cell-type labels rather than domain labels (its Figure 4). A file in the same shape as `overall_ranks.csv` but scoped to single-cell-resolution tasks and these five metrics is required before Section 7.4 can be implemented as anything more than a static lookup of the manuscript's own top-3 cell-level methods (Scanorama, Harmony, BANKSY). Do not approximate this file by reusing `overall_ranks.csv`'s BPC scores as a stand-in — the manuscript's central finding is that domain-level and cell-level scores are negatively correlated for several metrics, so substituting one for the other would silently reproduce the exact confound the benchmark was built to expose.

### 3.5 Fields present in the manuscript but not yet in structured file form

These must be extracted from the manuscript text (already done in this conversation) and hand-encoded once, not re-derived from the CSVs, because the CSVs do not contain them:

- Per-method omics applicability (`architecturally_inapplicable: bool` per method x omics-type pair) — draws on the manuscript's explicit list of methods that cannot run on non-transcriptomic data.
- Per-method-per-task completion status (`success | out_of_memory | not_applicable`) — draws on Figure 3d and the iST panel failure-rate discussion.
- Batch-effect-severity label for Tasks 25–28 (`minimal | moderate | maximal`) — draws directly from the manuscript's stated gradient.
- The five-branch decision tree itself (Figure 5) — see Section 6.

## 4. Knowledge base build step

`build_knowledge_base.py` should produce one `knowledge_base.json` with two top-level keys, mirroring the reference project's `dataset_profiles` pattern:

```json
{
  "dataset_profiles": {
    "Dataset_1": {
      "task_id": "Task_1",
      "omics_type": "transcriptomics",
      "st_category": "sST",
      "sequencing_technique": "Visium",
      "species": "Human",
      "tissue": "Cortex",
      "batch_id": "151507",
      "num_locations": 4226,
      "num_features": 33538,
      "sparsity": 0.9580,
      "batch_effect_severity": null
    }
  },
  "method_scores": {
    "Task_1": {
      "GraphPCA": {"overall": 1.0, "BPC": 1.0, "BER": 0.695, "SPC": 0.853, "DTP": 0.938,
                    "completion_status": "success", "architecturally_inapplicable": false}
    }
  },
  "cell_level_scores": {
    "_comment": "populated only for single-cell-resolution tasks, once Section 3.4's file is supplied; do not derive from method_scores",
    "Task_14": {
      "BANKSY": {"acc": null, "f1": null, "isolated_labels": null, "cLISI": null, "cASW": null, "overall_cell_level": null}
    }
  },
  "decision_tree": { "...": "see Section 6" },
  "method_metadata": {
    "GraphPCA": {"deep_learning": true, "omics_agnostic": false, "category": "end_to_end_spatial"}
  }
}
```

Write this as a pure function `build(raw_dir: Path) -> dict`, unit-testable without any LLM or CLI involved.

## 5. Profile schema (`models.py`)

```python
from pydantic import BaseModel
from typing import Literal, Optional

class UserProfile(BaseModel):
    target_resolution: Optional[Literal["domain", "cell"]] = None      # required before final answer
    omics_type: Optional[Literal["transcriptomics", "proteomics",
                                   "metabolomics", "epigenomics"]] = None  # required before final answer
    st_category: Optional[Literal["sST", "iST"]] = None                 # only meaningful if omics_type == transcriptomics
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

class RecommendationResult(BaseModel):
    matched_branch: Optional[str]          # decision-tree branch name, if matched
    matched_datasets: list[dict]
    recommended_methods: list[dict]
    discarded_methods: list[dict]
    confidence_note: str                    # explicit statement of coverage/confidence, never omitted
```

`target_resolution` and `omics_type` are marked required before the agent gives a final answer (not required to start the conversation) because the manuscript's own findings show these two fields change the recommendation outright, not incrementally.

## 6. Decision tree encoding (`decision_tree.py`)

Encode Figure 5 as data, not as nested if-statements, so it can be updated without touching matching logic:

```python
DECISION_TREE = [
    {
        "branch": "cell_level_analysis",
        "conditions": {"target_resolution": "cell"},
        "top_methods": ["Scanorama", "Harmony", "BANKSY"],
        "reported_axis": "CLP",
    },
    {
        "branch": "non_transcriptomic",
        "conditions": {"target_resolution": "domain", "omics_type": ["proteomics", "metabolomics", "epigenomics"]},
        "top_methods": ["CAST", "BINARY", "DECIPHER_niche"],
        "reported_axis": ["BPC", "BER", "SPC", "DTP"],
    },
    {
        "branch": "sST_small",
        "conditions": {"target_resolution": "domain", "omics_type": "transcriptomics", "st_category": "sST", "num_locations_lt": 5000},
        "top_methods": ["STAGATE", "GraphPCA", "STAIR"],
    },
    {
        "branch": "sST_large",
        "conditions": {"target_resolution": "domain", "omics_type": "transcriptomics", "st_category": "sST", "num_locations_gte": 5000},
        "top_methods": ["STAIR", "CellCharter", "DECIPHER_niche"],
    },
    {
        "branch": "iST",
        "conditions": {"target_resolution": "domain", "omics_type": "transcriptomics", "st_category": "iST"},
        "top_methods": ["STAIR", "STAGATE", "DECIPHER_niche"],
    },
]

def match_branch(profile: "UserProfile") -> dict | None: ...
```

`match_branch` returns the first branch whose conditions are all satisfied by non-null profile fields, or `None` if the profile does not fall cleanly into any branch (for example a profile with `num_locations` missing cannot be routed into `sST_small` vs `sST_large`, so it should return `None` and fall through to Section 7's full matching engine rather than guessing).

## 7. Matching and ranking engine (`matching.py`)

### 7.1 Dataset similarity score

Plain weighted sum, following the same structure as the reference project, adapted to SOHIB's fields:

| Property match | Points |
|---|---|
| Exact `omics_type` match | +4.0 |
| Exact `st_category` match (only checked if both are transcriptomics) | +2.5 |
| Exact `technology` match | +3.0 |
| Exact `species` match | +1.5 |
| Exact `tissue` match | +1.5 |
| `batch_effect_severity` exact match | +2.0 |
| `num_locations` closeness, `min(a,b)/max(a,b)` scaled | up to +2.0 |
| `sparsity` closeness, `1 - abs(a-b)` scaled | up to +1.0 |

Implement as `dataset_similarity(profile, dataset_profile) -> float`, unit-tested against at least three hand-checked cases before anything is built on top of it.

### 7.2 Method ranking

For each of the top-k matched datasets, pull that dataset's task's method scores. Combine `overall` with sub-scores according to priority, same weighting pattern as the reference project:

| Priority | overall weight | speed/OOM-risk weight | memory weight |
|---|---|---|---|
| accuracy | 0.85 | 0.05 | 0.10 |
| speed | 0.55 | 0.35 | 0.10 |
| memory | 0.55 | 0.10 | 0.35 |
| balanced | 0.70 | 0.15 | 0.15 |

Since exact runtime/memory numbers are not available (see Section 3.4), substitute a binary `completion_status` penalty in place of a continuous speed/memory bonus until real resource logs exist: any method with `completion_status != "success"` on a closely matched dataset gets a fixed penalty (suggest -0.3) rather than a smooth bonus. This must be documented in the code as an approximation, not presented as equivalent to true runtime/memory weighting.

### 7.3 Hard filters (never merely down-rank, remove outright)

- `avoid_deep_learning` and `method_metadata[method].deep_learning` → remove.
- `architecturally_inapplicable[method][profile.omics_type]` → remove, and do not count this as a "poor score," since it is a different fact (cannot run vs. ran and scored low).

### 7.4 Cell-level ranking mode (separate function, not a parameter of 7.2)

If `profile.target_resolution == "cell"`, method ranking must call a distinct function, `rank_methods_cell_level(profile, matched_datasets)`, which reads exclusively from `knowledge_base["cell_level_scores"]` and never touches `method_scores`. Do not implement this as an `if resolution == "cell": weight BPC differently` branch inside 7.2 — that would still risk domain-level scores leaking into a cell-level answer through a shared code path. The two functions should not call each other and should not share a weighting table, only the upstream `matched_datasets` list from Section 7.1.

Until Section 3.4's cell-level score file is supplied, `rank_methods_cell_level` should not attempt to compute anything from `method_scores`; it should return the decision tree's static `cell_level_analysis` branch methods (Scanorama, Harmony, BANKSY) with a `confidence_note` stating plainly that this is the benchmark's published branch answer, not a freshly computed ranking, because the underlying per-dataset cell-level data is not yet available. Once the file is supplied, replace this static fallback with real matching, following the same pattern as 7.1–7.2.

## 8. h5ad extraction module (`h5ad_profile.py`)

```python
import anndata as ad

def extract_tier1(adata: ad.AnnData) -> dict:
    n_obs, n_vars = adata.shape
    sparsity = 1 - (adata.X.nnz / (n_obs * n_vars)) if hasattr(adata.X, "nnz") else None
    has_spatial = "spatial" in adata.obsm
    return {"num_locations": n_obs, "num_features": n_vars, "sparsity": sparsity, "has_spatial_coords": has_spatial}

def extract_tier2(adata: ad.AnnData) -> dict:
    # best-effort lookup across common naming conventions; return None per field if not found,
    # never guess. Check adata.obs columns (case-insensitive) against a small alias list per field,
    # e.g. tissue: ["tissue", "Tissue", "organ"]; species: ["species", "organism"];
    # batch: ["batch", "sample", "slice_id", "section"].
    ...
```

Test this against at least two real public `.h5ad` files with different metadata conventions (not just one clean example), since the whole point of Tier 2 is handling inconsistency. Any field Tier 2 cannot find stays `None` and is asked of the user through the text path — this must not silently fall back to a guess.

## 9. LLM prompts

### 9a. `prompts/extract_profile.md`

Adapt the reference project's `extract_profile.md` pattern directly, replacing its field list with the `UserProfile` schema from Section 5. Keep its extraction rules style: normalize known synonyms, leave unsupported fields `null`, never guess species/tissue/technology from an unsupported message.

### 9b. `prompts/recommendation_answer.md`

Adapt the reference project's `recommendation_answer.md` section structure (Recommendation / Why These Methods / Tradeoffs / Next Best Questions), with two SOHIB-specific additions:

- If `matched_branch` is not null, the answer must state that the recommendation follows a directly published SOHIB decision-tree branch, not an inferred match.
- If any recommended method's evidence includes a `completion_status` other than `success` on the closest matched dataset, this must be stated as a tradeoff explicitly, in the same sentence as the method name, not buried in a footnote.
- If `profile.target_resolution == "cell"`, the answer must draw only on `cell_level_scores` evidence (or the static decision-tree fallback, per Section 7.4) and must never cite a domain-level BPC, BER, SPC, or DTP number as supporting evidence for a cell-level recommendation, even if it is the only number available for that method.

## 10. CLI flow

Mirror the reference project: `sohib-agent chat` starts a loop; `/profile` prints the current structured profile; `/reset` clears it; `/quit` exits. Add one additional command, `/upload <path.h5ad>`, which runs Section 8's extraction and merges the result into the current profile, with file-derived values overriding text-derived values for any field both supply.

## 11. Validation plan (`tests/test_leave_one_out.py`)

For each of the 33 tasks: remove it from `knowledge_base.json`, construct a synthetic profile from its own `data_w_sparsity.csv` row, run the full matching pipeline against the remaining 32 tasks, and check (a) the top matched dataset shares the same `omics_type` and `st_category`, and (b) the top-ranked recommended method appears in that held-out task's own top-3 methods per `overall_ranks.csv`. Report the hit rate. This must run and pass at some documented threshold before the CLI is presented as reliable — do not skip this step to save time.

Once Section 3.4's cell-level score file exists, repeat the same leave-one-out procedure restricted to single-cell-resolution tasks, calling `rank_methods_cell_level` instead of the domain-level ranking function, and report its hit rate separately rather than averaging it into the domain-level result — a single combined number would hide exactly the kind of domain/cell-level divergence the benchmark exists to measure.

## 12. Ordered task list for the code agent

1. Write `models.py` and get it to import cleanly with no other dependencies.
2. Write `build_knowledge_base.py` against the actual supplied CSVs; write `tests/test_knowledge_base.py` asserting row counts and a handful of spot-checked values match the raw CSVs exactly.
3. Hand-encode `method_metadata` (deep_learning, omics_agnostic, category) and the omics-applicability table from the manuscript text into a small JSON or Python literal — this is manual transcription work, not inference, and should be reviewed by the applicant before being trusted.
4. Write `decision_tree.py` and `matching.py` with unit tests before anything else touches them, including `rank_methods_cell_level` as the static decision-tree fallback described in Section 7.4 — do not stub it as a TODO, since it is the only cell-level answer available until Section 3.4's file is supplied.
5. Write `h5ad_profile.py`; test against at least two differently-structured public `.h5ad` files.
6. Write the two prompt files and the two thin LLM-calling modules; keep them thin — all scoring logic must already be complete and tested before this step.
7. Wire up `cli.py`.
8. Write and run `test_leave_one_out.py`; report the result honestly in the README, including cases where it fails.

## 13. Open items requiring the applicant's input before or during the build

- Reconcile the 43 rows in `overall_ranks.csv` against the 40 methods stated in the manuscript abstract.
- Confirm the raw-metric row-name prefixes for BPC, SPC, and DTP categories using a second `task_{N}.csv` file (only the BER-category prefixes are confirmed from `task_5.csv` so far).
- Confirm whether any runtime/memory logs exist anywhere outside the manuscript's qualitative failure descriptions; if none exist, Section 7.2's binary-penalty approximation should be treated as a permanent design choice, not a placeholder.
- Decide the integration-mode field (cross-slice vs. one-slice/cross-slice multiomics) is out of scope for this version, since SOHIB's own task design does not appear to vary this axis the way the reference project's benchmark does — confirm before building it into the schema.
- Supply the per-method cell-level score file described in Section 3.4. Until it exists, treat cell-level requests as answered only by the static decision-tree branch (Section 7.4), and say so explicitly in every cell-level answer rather than presenting the static branch as a freshly computed, dataset-matched recommendation.