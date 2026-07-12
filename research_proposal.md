# SOHIB Recommendation Agent — Build Specification & Implementation Status

This document serves dual purpose: the original build specification handed to the code agent, and an updated record of what was actually built, the design decisions made, and deviations from the original plan.

---

## 0. One-paragraph goal

Build a command-line agent that takes either a short natural-language description of a new spatial omics dataset or an uploaded `.h5ad` file, matches it against the SOHIB benchmark (164 tissue sections, 42 integration methods, 33 tasks, four metric categories), and returns a ranked list of recommended integration methods with the specific benchmark evidence behind each recommendation. All numeric ranking is done by plain, deterministic code. A language model is used only to (a) extract structured fields from free text and (b) write a brief, score-grounded recommendation in prose. The language model never computes or invents a score.

---

## 1. Tech stack

- Python 3.11+, packaged with `uv` (`pyproject.toml` + `uv sync` pattern).
- `pandas` for tabular KB operations during the build step.
- `anndata` and `scanpy` for reading `.h5ad` files.
- `pydantic` v2 for the profile schema (free JSON-schema export for the LLM extraction prompt, free validation).
- Anthropic API (`anthropic` SDK) for both LLM calls. Model resolved at runtime via `/v1/models` to support both direct keys and gateway-routed keys (e.g. LiteLLM with `us.anthropic.*` prefixes).
- `rich` for CLI output.
- `pytest` for tests.

---

## 2. Repository layout

```
sohib_agent/
  data/
    raw/                          # gitignored — place benchmark CSVs here before build-kb
    clean/
      knowledge_base.json         # pre-built artifact, committed to repo
  src/sohib_agent/
    models.py                     # UserProfile, RecommendationResult
    method_metadata.py            # per-method metadata + Task 25-28 batch labels
    build_knowledge_base.py       # raw CSVs → knowledge_base.json
    knowledge_base.py             # load/save with LRU cache
    decision_tree.py              # Figure 5 encoded as data
    matching.py                   # similarity scoring + method ranking
    h5ad_profile.py               # extract profile fields from .h5ad files
    profile_extraction.py         # LLM call 1: text → UserProfile fields
    answer_writer.py              # LLM call 2: score tables → prose
    cli.py                        # chat loop + build-kb command
  prompts/
    extract_profile.md
    recommendation_answer.md
  pyproject.toml
  README.md
  .gitignore
```

Raw CSVs are not committed. The pre-built `knowledge_base.json` is committed so users can run the agent without needing the original benchmark files.

---

## 3. Input data contracts

### 3.1 `data_w_sparsity.csv` — dataset meta-feature table

164 rows, one per tissue section. Columns: `Slice ID, Task ID, Omics type, Sequencing technique, Species, Tissue, Batch ID, # Cells/spots, # Features, sparsity`.

- `Omics type` encodes the sST/iST distinction inside the text value (e.g. `Spatial transcriptomics (sST)`) — parsed into a separate `st_category` field during build.
- `Task ID` may be multi-valued (e.g. `"Task_25, Task_26, Task_27"`) for slices shared across cross-platform tasks — handled by `_parse_task_ids()`.

### 3.2 `overall_ranks.csv` — aggregated method × task performance table

42 rows (43 in original file — reconciliation: `DECIPHER (cell)`, `FuseMap (cell)`, and `Nicheformer` are the 3 extra rows vs. the manuscript's 40-method count; all kept in the KB and flagged). Missing values preserved as `null`, never coerced to 0.

### 3.3 `task_{N}.csv` — granular pre-aggregation metric table

33 files (one per task; the raw file names are `Task_1.csv` … `Task_33.csv`, row-per-metric / column-per-method, same layout as `overall_ranks.csv`). Row names follow prefixes: `batch_metrics_*` (BER), `bio_metrics_*` (BPC), `spatial_metrics_*` (SPC), `downstream_metrics_*` (DTP) — used only informationally; `overall_ranks.csv`'s aggregated scores drive all domain-level ranking. A fifth prefix, `sc_metrics_*`, is parsed separately — see 3.4.

### 3.4 Cell-level score file — `sc_metrics_*` rows in `task_{N}.csv`

Only single-cell-resolution tasks carry `sc_metrics_*` rows at all: **16 of the 33 tasks** — `Task_10`–`Task_24` and `Task_29`. Tasks without single-cell ground truth (small-scale sST 1–4, cross-platform batch tasks 25–28, most non-transcriptomic tasks 30–33) simply have no such rows; their absence in `cell_level_scores` is a real fact about the benchmark, not a gap.

Per-slice classifier rows (e.g. `sc_metrics_0_KNN_acc`, `sc_metrics_MsBrainAgingSpatialDonor_1_RF_f1` — the slice-id segment is not standardized across tasks) are **not** parsed; only six fixed, task-independent aggregate rows are (`_parse_cell_level_scores()` in `build_knowledge_base.py`):

| Row | KB field |
|---|---|
| `sc_metrics_best_classifier_mean_acc` | `classification_acc` |
| `sc_metrics_best_classifier_mean_f1` | `classification_f1` |
| `sc_metrics_cLISI` | `cLISI` |
| `sc_metrics_isolated_labels` | `isolated_labels` |
| `sc_metrics_ASW_label` | `ASW_label` |
| `sc_metrics_best_classifier` | `best_classifier` (categorical: which of KNN/LR/RF scored best) |

`overall_cell_level` — the number `rank_methods_cell_level()` actually ranks by — is `mean(classification_acc, classification_f1)`, the two headline classification metrics for cell-level preservation (CLP): whichever classifier scored best per method-task pair, averaged across accuracy and macro F1. The other four fields are kept as auxiliary evidence, mirroring how BPC/BER/SPC/DTP sit alongside `overall` on the domain-level side. Never approximate `overall_cell_level` from `method_scores` — the benchmark's central finding is that domain-level and cell-level scores are negatively correlated for several methods; `cell_level_scores` is strictly separate.

Six methods are recorded under a different column string in `task_{N}.csv` than in `overall_ranks.csv` (verified by diffing every column name across all 33 files against the 42 canonical names) — normalized by `_normalize_method_name()`: `DECIPHER_niche`→`DECIPHER (niche)`, `DECIPHER_cell`→`DECIPHER (cell)`, `FuseMap_niche`→`FuseMap (niche)`, `FuseMap_cell`→`FuseMap (cell)`, `scGPT_sp`→`scGPT-spatial`, `MENDER_wo_pcs`→`MENDER`.

**Fallback when a query's matched task has no cell-level data.** Even with the file supplied, a cell-level query can still match a task outside the 16 (e.g. Visium cortex tasks 1–4) — `rank_methods_cell_level()` falls back to the static published branch answer (Scanorama, Harmony, BANKSY) in that case too, not just when the file is entirely absent. This can't be decided by a single global "is data available" flag; `cli._cell_level_would_be_static()` actually runs the matching pipeline for the specific profile to find out, since asking the user for more detail is only pointless when *this* query's outcome genuinely can't change.

### 3.5 Fields hand-encoded from the manuscript

- **Per-method metadata**: `deep_learning`, `omics_agnostic`, `architecturally_inapplicable`, `embedding_type` (for FuseMap/DECIPHER variants), `base_method` — in `method_metadata.py`.
- **Cross-platform task labels** (Tasks 25–28): `batch_type`, `batch_effect_severity`, `description` — in `TASK_CROSS_PLATFORM_INFO`. Technologies are auto-derived from slice data to prevent encoding drift.

---

## 4. Knowledge base design

`knowledge_base.json` has six top-level keys:

```
dataset_profiles     164 tissue sections — omics, technology, tissue, spots, sparsity
                     NOTE: no batch_effect_severity here — that is a task-level concept

task_knowledge       33 tasks — num_slices, technologies, omics_type, st_category,
                     batch_type, batch_effect_severity, batch_description
                     Batch effect is a property of the TASK (multi-slice integration
                     challenge), not of any individual slice.

method_scores        Per-task scores for 42 methods: overall, BPC, BER, SPC, DTP,
                     completion_status. Missing = null, never 0.

method_summaries     Pre-computed per-method statistics by category (sST / iST /
                     non_transcriptomic / cross_platform). Built once at build-kb time,
                     not recomputed per session.

cell_level_scores    Populated for the 16 single-cell-resolution tasks (Task_10-24, Task_29)
                     from task_{N}.csv's sc_metrics_* rows — see 3.4. Falls back to a
                     scaffold-only placeholder if those files aren't in raw_dir. Strictly
                     separated from method_scores — never derived from domain-level scores.

method_metadata      deep_learning, omics_agnostic, architecturally_inapplicable,
                     embedding_type, base_method, category per method.
```

Built by `build_knowledge_base.py` as a pure function `build(raw_dir) -> dict`, unit-testable without any LLM or CLI.

### 4.1 Task batch severity gradient (Tasks 25–28)

| Task | Technologies | batch_type | batch_effect_severity |
|---|---|---|---|
| Task_25 | MERFISH + MERFISH | same_tech | minimal |
| Task_26 | MERFISH + STARMap | cross_tech_iST | moderate |
| Task_27 | Slide-seq + MERFISH | cross_tech_mixed | maximal |
| Task_28 | Slide-seq + STARMap | cross_tech_mixed | maximal |

Task_27 was initially mis-encoded as `cross_tech_iST / moderate` — corrected after verifying actual slice data showed Slide-seq (sST) + MERFISH (iST).

---

## 5. Profile schema (`models.py`)

```python
class UserProfile(BaseModel):
    target_resolution: Optional[Literal["domain", "cell"]] = None   # required before answer
    omics_type: Optional[Literal["transcriptomics", "proteomics",
                                  "metabolomics", "epigenomics"]] = None  # required before answer
    st_category: Optional[Literal["sST", "iST"]] = None
    technology: Optional[str] = None
    species: Optional[str] = None
    tissue: Optional[str] = None
    num_locations: Optional[int] = None      # per slide, not total across slides
    num_features: Optional[int] = None
    sparsity: Optional[float] = None
    batch_effect_severity: Optional[Literal["minimal", "moderate", "maximal"]] = None
    priority: Literal["accuracy", "speed", "memory", "balanced"] = "balanced"
    avoid_deep_learning: bool = False
    source: Literal["text", "h5ad", "text+h5ad"] = "text"
```

`target_resolution` and `omics_type` are the two gating fields — the agent will not give a final recommendation until both are present.

---

## 6. Decision tree (`decision_tree.py`)

Encoded as a data list (not nested if-statements) so it can be updated without touching matching logic. Five branches from SOHIB Figure 5:

| Branch | Key conditions | Published top methods |
|---|---|---|
| `cell_level_analysis` | target_resolution = cell | Scanorama, Harmony, BANKSY |
| `non_transcriptomic` | domain + proteomics/metabolomics/epigenomics | CAST, BINARY, DECIPHER (niche) |
| `sST_small` | domain + sST + spots < 5000 | STAGATE, GraphPCA, STAIR |
| `sST_large` | domain + sST + spots ≥ 5000 | STAIR, CellCharter, DECIPHER (niche) |
| `iST` | domain + iST | STAIR, STAGATE, DECIPHER (niche) |

`match_branch()` returns the first fully-satisfied branch, or `None` if any condition field is null (e.g. missing `num_locations` cannot route sST_small vs sST_large — falls through to similarity matching rather than guessing).

---

## 7. Matching and ranking engine (`matching.py`)

### 7.1 Dataset similarity score

Weighted sum over profile fields vs. benchmark dataset profiles:

| Field | Weight | Notes |
|---|---|---|
| omics_type exact match | 4.0 | |
| st_category exact match | 2.5 | only if both are transcriptomics |
| technology exact match | 3.0 | case-insensitive |
| species exact match | 1.5 | case-insensitive |
| tissue exact match | 1.5 | case-insensitive |
| batch_effect_severity match | 2.0 | read from task_knowledge, NOT dataset_profile |
| num_locations closeness | up to 2.0 | min(a,b)/max(a,b) |
| sparsity closeness | up to 1.0 | 1 - abs(a-b) |

`batch_effect_severity` is explicitly sourced from `task_knowledge` (not `dataset_profile`) because a single slice has no batch effect — it is a task-level property.

### 7.2 Method ranking

Composite score = overall × w₁ + completion_penalty × (w₂ + w₃), weighted by dataset similarity.

| Priority | overall | speed proxy | memory proxy |
|---|---|---|---|
| accuracy | 0.85 | 0.05 | 0.10 |
| speed | 0.55 | 0.35 | 0.10 |
| memory | 0.55 | 0.10 | 0.35 |
| balanced | 0.70 | 0.15 | 0.15 |

Completion penalty: −0.3 applied when `completion_status != "success"` as a binary proxy for speed/memory, since exact runtime/memory logs are not available.

### 7.3 Warnings (replacing hard filters)

Methods are never removed. Issues are surfaced as per-method `warnings` so the user decides:
- `avoid_deep_learning = true` → warning on every DL method
- `architecturally_inapplicable` → warning on methods not designed for the user's omics type
- Embedding type mismatch → warning when a FuseMap/DECIPHER variant's embedding type does not match `target_resolution`

### 7.4 FuseMap and DECIPHER — two embedding variants

Both methods produce two distinct output types from the same model. Treated as separate methods in the KB with an `embedding_type` field:

| Variant | Embedding type | Suited for |
|---|---|---|
| FuseMap (niche), DECIPHER (niche) | Spatially-smoothed neighbourhood | Domain-level integration |
| FuseMap (cell), DECIPHER (cell) | Per-cell representation | Cell-level analysis |

On domain-level tasks, the niche variants consistently outperform the cell variants (e.g. DECIPHER niche 0.735 vs. cell 0.504 on Task_1). Both appear in results; the mismatched variant carries a warning.

### 7.5 Cell-level ranking (`rank_methods_cell_level`)

Completely separate function from domain-level ranking. Reads only from `cell_level_scores`, never from `method_scores` — it does NOT fall back to domain-level scores as a proxy (an earlier draft of this spec said it would; the implementation never did, precisely to avoid the domain/cell conflation the benchmark's central finding warns against). When there's nothing real to rank by — no cell-level data supplied at all, or this profile's matched task isn't among the 16 that carry it (3.4) — it falls back to the static published branch answer (Scanorama, Harmony, BANKSY) instead, clearly flagged via `RecommendationResult.selection_source == "static_branch"`. The two functions share no weighting table and do not call each other.

---

## 8. h5ad extraction (`h5ad_profile.py`)

Two tiers:
- **Tier 1** (always available): `num_locations`, `num_features`, `sparsity`, `has_spatial_coords` — derived from matrix shape.
- **Tier 2** (best-effort): `tissue`, `species`, `technology`, `batch_id` — scanned from `.obs` columns and `.uns` using alias lists. Returns `None` if not found, never guesses.

File-derived values override text-derived values when merging into the profile.

---

## 9. LLM architecture — two isolated calls

```
User text
    │
    ▼
[Call 1: profile_extraction.py]
  Input:  user message + current profile state + UserProfile JSON schema
  Output: partial JSON of changed fields only, OR {"clarify": "question"}
  Model does: entity recognition constrained to a fixed schema
  Model cannot: touch any scores (none present in this call)
    │
    ▼
UserProfile (stateful across turns)
    │
    ├─ Decision tree check (pure Python)
    ├─ Similarity matching (pure Python)
    └─ Method ranking (pure Python) → candidate_methods (deterministic, evidence-backed pool)
            │
            ▼
    RecommendationResult (candidate_methods populated; recommended_methods = deterministic
    top-3-by-composite_score baseline — this is what --no-llm mode ships as-is)
            │
            ▼
[Call 2: answer_writer.py — run_recommendation(), a BOUNDED JOINT DECISION]
  Input:  user profile + matched_datasets (with actual benchmark fields) +
          candidate_methods (the pool — up to top_k methods, each with real composite_score,
          evidence, warnings) + task_score_tables (full scores for matched tasks) +
          method_summaries (candidates' cross-task-category stats)
  Output: {"selected": [{"method", "sentence"}, × up to 3], "branch_note"}
  Model does: SELECTS and ORDERS the final top-3 from candidate_methods — may reorder or
              prefer a lower-scored candidate (e.g. one with no disqualifying warning, or one
              that's consistently strong per method_summaries rather than a one-task spike) —
              and writes one grounded sentence per pick
  Model cannot: name a method outside candidate_methods, or cite a number not present in the
                data it was given
            │
            ▼
    _resolve_selection() validates the model's output against candidate_methods (hallucination
    guard): any method name not in the pool is dropped; any slot the LLM didn't validly fill is
    backfilled deterministically via top_k_by_composite_score. RecommendationResult.
    recommended_methods / discarded_methods / selection_source are updated with the resolved,
    validated result — selection_source records whether the final answer was "llm_joint" (the
    model's choice fully validated) or "deterministic_fallback" (backfill occurred, or the LLM
    call failed entirely).
```

`candidate_methods` — not the LLM's training knowledge, and not an unbounded free-form catalogue — is the ceiling on what can ever appear in the final answer: it is computed deterministically (weighted similarity + composite score, no embeddings; the corpus is small and structured enough for exact field/task-ID matching) before the LLM is ever called. `task_score_tables` and `method_summaries` are additional retrieved grounding context that inform the LLM's judgement without expanding what it's allowed to select from.

### 9.1 Model resolution

Both modules resolve the model ID at runtime by calling `/v1/models` and matching against a preference list. This handles both direct Anthropic keys (`claude-sonnet-4-6`) and gateway-routed keys (`us.anthropic.claude-sonnet-4-6`). No hardcoded model string that breaks silently.

### 9.2 Answer format (Call 2)

The LLM's raw output is strict JSON (`{"selected": [...], "branch_note": ...}`) — not the display text directly. `answer_writer._format_answer()` assembles the final display from the validated selection, guaranteeing the format regardless of what the model writes:
```
1. **Method** — one grounded sentence (LLM's choice + wording).
2. **Method** — one grounded sentence.
3. **Method** — one grounded sentence.
```
No intro line, no scenario description, no benchmark explanation, no closing sentence beyond an optional one-line `branch_note`. If a warning applies (embedding mismatch, failed run), it is woven into that method's own sentence by the model.

Any slot the model's selection didn't validly fill (see 9.3) gets a synthesized fallback sentence (`composite score {score}`) instead of the model's prose, so the display never has a blank or broken line.

### 9.3 Hallucination prevention — two layers

**Selection layer (new: bounded joint decision).** The LLM may only select methods present in `candidate_methods` — a pool it did not choose the contents of. `answer_writer._validate_selection()` checks every model-selected method name against the pool; anything not present (a typo, a method from training memory, a different benchmark's method) is silently dropped. `_resolve_selection()` then backfills any resulting gap deterministically (`matching.top_k_by_composite_score`), so a partially- or fully-hallucinated response degrades to the deterministic ranking rather than shipping a fabricated method. `RecommendationResult.selection_source` records which happened (`"llm_joint"` vs `"deterministic_fallback"`) so this is visible in the output, not silently absorbed.

**Content layer (existing).** A known failure mode was the LLM substituting the user's technology name (e.g. "Xenium") for the benchmark's actual technology in matched datasets. Fixed by:
- Including `sequencing_technique`, `species`, `tissue`, and `task_id` from the actual benchmark in `matched_datasets`
- Prompt explicitly instructs: use `sequencing_technique` from `matched_datasets`; never use the user's technology to describe benchmark data unless it exactly matches

### 9.4 Clarifying questions (Call 1)

When the user's message is too vague to extract any fields, the LLM returns `{"clarify": "..."}` with a single natural-language question for the most important missing field. The CLI displays this question directly — no field-name lists, no bullet points. The two gating fields (`target_resolution`, `omics_type`) are always prioritised in clarifying questions.

### 9.5 Two-stage clarification: gating fields, then one optional-fields round

The agent does not ask everything up front, and it does not stop at the bare minimum either:

1. **Gating fields missing** (`target_resolution`, `omics_type`): asked one at a time, every turn, via Call 1's `clarify` mechanism — a recommendation is impossible without these, so this loop repeats until both are present.
2. **Gating fields present, high-value fields missing** (`technology`, `species`, `st_category`, `num_locations`, `batch_effect_severity` — see `UserProfile.missing_optional_fields()`): the agent *could* recommend now, but asks once, in a single batched message, for whatever of these is still missing, since they materially change which benchmark datasets match. If the user answers, ignores the prompt and keeps chatting, or types `/recommend`, the agent proceeds with whatever is present — it never asks a second time in the same session.
3. **Everything the profile schema tracks is already present** (`UserProfile.is_fully_specified()` — e.g. the user described their dataset fully in one message, or uploaded an `.h5ad` with rich `.obs` metadata): the agent skips straight to recommending, no questions asked.

This applies identically to the `/upload` path: an uploaded `.h5ad` file's `.obs`/`.uns` metadata may already answer most high-value fields, in which case the agent recommends immediately after loading it; otherwise it asks once for what's still missing.

---

## 10. CLI flow (`cli.py`)

`sohib-agent chat` starts a stateful loop. The `UserProfile` object accumulates fields across turns.

| Command | Effect |
|---|---|
| `/profile` | Show current structured profile |
| `/upload <path.h5ad> [description]` | Load h5ad; an optional trailing description is extracted via Call 1 and merged first, then h5ad-derived fields override it field-by-field (structural/metadata fields read directly from the file outrank a paraphrased description) |
| `/recommend` | Force recommendation with current profile |
| `/reset` | Clear profile |
| `/set field=value` | Manual field override |
| `/quit` | Exit |

`sohib-agent chat --no-llm` — bypasses both LLM calls; use `/set` + `/recommend`. Under `--no-llm`, an `/upload` description is accepted but ignored (no LLM call is made to extract it) — only the h5ad's own Tier 1/2/3 fields are used.
`sohib-agent build-kb` — rebuilds `knowledge_base.json` from raw CSVs.

---

## 11. Validation (`tests/test_leave_one_out.py`)

Leave-one-out across all 33 tasks: remove task, build synthetic profile from its own data row, run matching against remaining 32, check (a) top matched dataset shares `omics_type` and `st_category`, (b) top-ranked method appears in held-out task's own top-3.

**Scope note (post joint-decision architecture, §9):** this validates `candidate_methods` — the deterministic pool — and the `top_k_by_composite_score` baseline that ships in `--no-llm` mode. It does not, and cannot without live LLM calls, validate the LLM's joint-selected `recommended_methods` in normal (LLM) mode, since that selection can legitimately reorder or substitute within the pool based on qualitative evidence (warnings, `method_summaries` consistency) the leave-one-out check doesn't score. Treat these numbers as a lower bound on end-to-end accuracy — the candidate pool the LLM chooses from — not as measuring the final shipped answer.

| Metric | Result | Threshold |
|---|---|---|
| Omics/category match rate | 93.3% (28/30) | ≥ 80% |
| Top-method hit rate | 56.7% (17/30) | ≥ 40% |

Both thresholds pass. 13 misses are near-misses within the same top-method cluster (STAIR / STAGATE / GraphPCA all appear in the sST top group). No miss crosses a major category boundary.

Cell-level leave-one-out is not yet run — the cell-level score file is now supplied (Section 3.4), so this is implementable, just not yet built. When it is, results must be reported separately from domain-level, not averaged, to avoid hiding the domain/cell divergence the benchmark exists to measure.

---

## 12. Open items

| Item | Status |
|---|---|
| Cell-level score file (Section 3.4) | Supplied — parsed from `task_{N}.csv`'s `sc_metrics_*` rows for the 16 single-cell-resolution tasks. `rank_methods_cell_level` now returns real computed rankings for queries matching those tasks; other cell-level queries still fall back to the static published branch answer. Cell-level leave-one-out validation (Section 11) not yet built. |
| Runtime / memory logs | Not available. Binary completion penalty (−0.3) used as proxy. If logs are supplied later, replace the binary penalty with continuous weighting. |
| 42 vs. 40 method reconciliation | Documented. `DECIPHER (cell)`, `FuseMap (cell)`, `Nicheformer` are the 3 extra rows. All kept in KB, flagged in `_reconciliation_note`. |
| Cross-slice multiomics integration mode | Out of scope for this version. |

---

## 13. Design decisions and rationale

**No hard filters.** Methods are never removed from results. Potential issues (deep learning, architectural mismatch, wrong embedding variant) are surfaced as per-method warnings. The researcher decides whether a warning is disqualifying — the agent should not make that call.

**No method knowledge graph.** Score correlations between methods are moderate (0.49–0.74), not strong enough to reliably assert substitutability. The pre-computed `method_summaries` (per-category mean/min/max across sST/iST/non-transcriptomic tasks) capture the structurally useful relationships as flat lookups. A graph would add maintenance cost without adding reliability.

**Pre-computed method summaries.** Built once at `build-kb` time. The LLM does not re-scan 164 datasets per query. Summaries include per-category statistics and worst-performing tasks, available for display without any runtime computation.

**Batch effect is task-level, not slice-level.** A single tissue slice has no batch effect. Batch effect is a property of the multi-slice integration challenge. `batch_effect_severity` lives in `task_knowledge`, not `dataset_profiles`. `dataset_similarity()` accepts a `task_info` argument to access this field — it cannot be read from `dataset_profile`.

**LLM sees raw score tables, not pre-ranked list.** The answer writer receives the full method × metric table for each matched task, sorted by overall score. The LLM reasons from actual data. This is different from RAG — retrieval and ranking are algorithmic; the LLM only narrates and selects from numbers it was given.
