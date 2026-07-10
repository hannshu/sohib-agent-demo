# SOHIB Recommendation Agent

A command-line agent that recommends spatial omics integration methods for your dataset, backed by the [SOHIB benchmark](https://www.biorxiv.org/content/10.1101/2024.04.13.589366) — 164 tissue sections, 42 integration methods, 33 tasks, four metric categories.

You describe your dataset in plain text or upload a `.h5ad` file. The agent matches it against the benchmark and returns a ranked list of methods with the specific benchmark evidence behind each recommendation.

---

## Why this gives better answers than asking an LLM directly

When you ask a general-purpose LLM "what integration method should I use for my Visium data?", it answers from training memory — a lossy representation of papers it read. It may recall that STAGATE and Harmony exist, but it cannot tell you that on your closest benchmark match (Task_1, Visium cortex):

- GraphPCA scores **1.0 overall** (BPC=1.0, BER=0.695, SPC=0.853, DTP=0.938)
- Harmony scores **0.641** — ranked ~20th
- DECIPHER **(cell)** scores 0.504 while DECIPHER **(niche)** scores 0.735 on the same task

This agent solves that by keeping the LLM out of the scoring entirely. The LLM is called exactly twice — once to extract structured fields from your text, once to narrate the pre-computed result. All ranking is deterministic Python over the benchmark's actual CSV data.

```
Your text
   │
   ▼
[LLM: extract profile fields]       ← entity recognition over a fixed schema
   │                                  cannot invent scores; no scores exist here
   ▼
UserProfile {omics_type, st_category,
             technology, tissue, ...}
   │
   ├─ Decision tree (SOHIB Figure 5) ──► direct branch match → top-3 from paper
   │
   ├─ Similarity matching ──────────► score all 164 datasets, pick top-k
   │   omics(4) + st_category(2.5)
   │   + technology(3) + species(1.5)
   │   + tissue(1.5) + batch_severity(2)
   │   + num_locations(2) + sparsity(1)
   │
   ├─ Method ranking ───────────────► aggregate scores from matched tasks
   │   composite = overall × w₁       priority: accuracy / speed / memory / balanced
   │   + completion_penalty × (w₂+w₃) warnings: DL preference, embedding mismatch
   │
   └─ RecommendationResult {matched_branch, matched_datasets,
                             recommended_methods, warnings, confidence_note}
          │
          ▼
      [LLM: narrate evidence]        ← narrates numbers it was given; forbidden
                                       from citing anything not in the JSON
```

---

## Installation

Requires Python 3.11+. Uses [`uv`](https://github.com/astral-sh/uv) for packaging.

```bash
git clone <this-repo>
cd sohib_agent

# with uv (recommended)
uv sync
uv run sohib-agent --help

# or with pip
pip install -e ".[dev]"
sohib-agent --help
```

---

## First-time setup: build the knowledge base

The raw benchmark CSVs (`data_w_sparsity.csv`, `overall_ranks.csv`, `task_*.csv`) are not committed to the repo. Place them in `data/raw/` then run:

```bash
sohib-agent build-kb
```

This produces `data/clean/knowledge_base.json` — a single JSON file containing:

| Key | Contents |
|---|---|
| `dataset_profiles` | 164 tissue sections with omics type, technology, tissue, spot count, sparsity |
| `task_knowledge` | 33 tasks with slice counts, technologies, batch type and severity |
| `method_scores` | Per-task scores for 42 methods: overall, BPC, BER, SPC, DTP, completion status |
| `method_summaries` | Pre-computed per-method statistics by category (sST / iST / non-transcriptomic / cross-platform) |
| `method_metadata` | Deep-learning flag, omics-agnostic flag, embedding type for FuseMap/DECIPHER variants |
| `cell_level_scores` | Scaffold only — populated once the cell-level score file is supplied |

The raw CSVs are not needed after this step and are excluded from version control via `.gitignore`.

---

## Usage

### Interactive chat (recommended)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY

sohib-agent chat
```

Describe your dataset in plain text. The agent extracts your profile, asks for anything missing, then gives a ranked recommendation.

```
You> I have 3 Visium slides from human dorsolateral prefrontal cortex, about 4000 spots each

▶ Decision tree match: sST_small
  Published top methods: STAGATE, GraphPCA, STAIR

  Rank   Method     Overall   BPC    BER    SPC    DTP
     1   GraphPCA   1.000     1.000  0.695  0.853  0.938
     2   STAGATE    0.939     0.798  0.655  0.846  0.983
     3   STAIR      0.947     0.791  0.580  1.000  0.921
```

### No-LLM mode (no API key required)

```bash
sohib-agent chat --no-llm
```

Set profile fields manually and call `/recommend`:

```
You> /set omics_type=transcriptomics
You> /set st_category=sST
You> /set technology=Visium
You> /set num_locations=4000
You> /recommend
```

### Upload a .h5ad file

Within the chat session:

```
You> /upload /path/to/mydata.h5ad
```

The agent reads `num_locations`, `num_features`, and `sparsity` directly from the file. It also scans `.obs` and `.uns` for tissue, species, and technology fields using common naming conventions. Values from the file override anything you typed.

### Slash commands

| Command | Effect |
|---|---|
| `/profile` | Show the current structured profile |
| `/upload <path.h5ad>` | Load a .h5ad file and merge into profile |
| `/recommend` | Run the recommendation pipeline now |
| `/reset` | Clear profile and start over |
| `/quit` | Exit |

---

## The five decision-tree branches

The agent first checks whether your profile falls into one of the five branches from the SOHIB paper (Figure 5). If it does, the top methods come directly from the published benchmark — no inference.

| Branch | Condition | Published top methods |
|---|---|---|
| `sST_small` | Sequencing-based ST, domain-level, < 5000 spots | STAGATE, GraphPCA, STAIR |
| `sST_large` | Sequencing-based ST, domain-level, ≥ 5000 spots | STAIR, CellCharter, DECIPHER (niche) |
| `iST` | Imaging-based ST, domain-level | STAIR, STAGATE, DECIPHER (niche) |
| `non_transcriptomic` | Proteomics / metabolomics / epigenomics, domain-level | CAST, BINARY, DECIPHER (niche) |
| `cell_level_analysis` | Any omics, cell-level resolution | Scanorama, Harmony, BANKSY |

If no branch matches (e.g. `num_locations` is unknown), the agent falls through to similarity matching against all 164 benchmark datasets.

---

## FuseMap and DECIPHER: two embedding modes

FuseMap and DECIPHER each produce **two distinct embedding types** from the same underlying model:

| Variant | Embedding | Best for |
|---|---|---|
| `FuseMap (niche)` | Spatially-smoothed neighbourhood embedding | Domain-level integration |
| `FuseMap (cell)` | Per-cell embedding | Cell-level analysis |
| `DECIPHER (niche)` | Neighbourhood embedding | Domain-level integration |
| `DECIPHER (cell)` | Per-cell embedding | Cell-level analysis |

Both variants appear in results for any query. When the variant's embedding type does not match your `target_resolution`, a warning is shown — you decide whether that matters for your use case.

---

## Batch effect information

Batch effect is a **task-level** property — a single tissue slice has no batch effect by itself. It is a property of the multi-slice integration challenge. The `task_knowledge` section of the knowledge base stores this correctly.

The four cross-platform tasks form a severity gradient:

| Task | Technologies | Type | Severity |
|---|---|---|---|
| Task_25 | MERFISH + MERFISH | same_tech | minimal |
| Task_26 | MERFISH + STARMap | cross_tech_iST | moderate |
| Task_27 | Slide-seq + MERFISH | cross_tech_mixed | maximal |
| Task_28 | Slide-seq + STARMap | cross_tech_mixed | maximal |

If you specify `batch_effect_severity` in your profile, the similarity matching will favour tasks at the same severity level.

---

## Scoring details

### Dataset similarity weights

| Field | Points |
|---|---|
| `omics_type` exact match | 4.0 |
| `st_category` exact match (transcriptomics only) | 2.5 |
| `technology` exact match | 3.0 |
| `species` exact match | 1.5 |
| `tissue` exact match | 1.5 |
| `batch_effect_severity` exact match (task-level) | 2.0 |
| `num_locations` closeness — `min(a,b)/max(a,b)` | up to 2.0 |
| `sparsity` closeness — `1 - abs(a-b)` | up to 1.0 |

### Priority weighting

| Priority | overall | speed proxy | memory proxy |
|---|---|---|---|
| `accuracy` | 0.85 | 0.05 | 0.10 |
| `speed` | 0.55 | 0.35 | 0.10 |
| `memory` | 0.55 | 0.10 | 0.35 |
| `balanced` (default) | 0.70 | 0.15 | 0.15 |

Since exact runtime/memory logs are not in the benchmark, a binary completion penalty (-0.3) is applied as a proxy for methods that failed on a matched task. This is an approximation documented in the code.

---

## Metric categories

| Code | Full name | What it measures |
|---|---|---|
| BPC | Biological preservation (cluster) | How well biological clusters are preserved after integration |
| BER | Batch effect removal | How well batch effects are corrected |
| SPC | Spatial coherence | Whether spatially adjacent spots have coherent representations |
| DTP | Downstream task performance | Performance on spatial domain identification tasks |

---

## Running the demo

```bash
python demo.py
```

Runs four representative cases through the full pipeline — no API key required. Shows the decision tree, similarity matching, ranked results, warnings, pre-computed method summaries, and batch severity information.

---

## Running tests

```bash
pytest tests/ -v
```

The test suite includes:

- `test_knowledge_base.py` — row counts, spot-checked values, correct task-level batch encoding
- `test_matching.py` — hand-verified similarity scores, warning generation
- `test_leave_one_out.py` — leave-one-out validation across all 33 tasks

### Leave-one-out validation results

| Metric | Result | Threshold |
|---|---|---|
| Omics/category match rate | 93.3% (28/30) | ≥ 80% |
| Top-method hit rate | 56.7% (17/30) | ≥ 40% |

The 13 misses are near-misses within the same top-method cluster (e.g. STAIR vs STAGATE vs GraphPCA all appear in the sST top group). No miss crosses a major category boundary.

---

## Known limitations

| Gap | Status |
|---|---|
| Cell-level ranking | Returns the static SOHIB branch (Scanorama, Harmony, BANKSY) until the per-method cell-level score file is supplied. Supply the CSV and the path activates automatically. |
| Runtime / memory advice | No timing logs in the benchmark. Binary pass/fail proxy used instead. |
| Methods outside SOHIB | Only the 42 benchmarked methods are ranked. New methods require a benchmark run. |
| Failure reasons | Why a method failed on a task is not in the CSVs — only that it failed. |

---

## Repository layout

```
sohib_agent/
  data/
    raw/                        # place benchmark CSVs here (gitignored)
    clean/
      knowledge_base.json       # built by sohib-agent build-kb
  src/sohib_agent/
    models.py                   # UserProfile, RecommendationResult (Pydantic)
    method_metadata.py          # per-method metadata + Task 25-28 batch labels
    build_knowledge_base.py     # raw CSVs → knowledge_base.json
    knowledge_base.py           # load / save
    decision_tree.py            # Figure 5 encoded as data
    matching.py                 # similarity scoring + method ranking
    h5ad_profile.py             # extract profile fields from .h5ad files
    profile_extraction.py       # LLM call: text → UserProfile fields
    answer_writer.py            # LLM call: evidence → prose
    cli.py                      # chat loop + build-kb command
  prompts/
    extract_profile.md          # system prompt for profile extraction
    recommendation_answer.md    # system prompt for answer narration
  tests/
    test_knowledge_base.py
    test_matching.py
    test_leave_one_out.py
  demo.py                       # standalone demo, no API key needed
  pyproject.toml
  README.md
```
