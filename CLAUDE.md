# CLAUDE.md — perturbseq-pipeline

Project rules and conventions for Claude Code working in this repository.

## 1. What this project is

A simple, reproducible Perturb-seq analysis pipeline that turns CRISPR
single-cell screening data into a self-contained QC + analysis report.

It is derived from two prototype notebooks (kept in `notebooks/prototype/` for
reference):

- `analyze_TF_perturbseq_from_rawcounts.ipynb` — starts from 10x
  `filtered_feature_bc_matrix` folders (multi-lane).
- `analyze_TF_perturbseq_small_scale_from_h5ad.ipynb` — starts from an existing
  (Seurat-converted) `.h5ad`.

The pipeline must support **both entry points**.

## 2. Hard requirements (from the project owner)

These are the acceptance criteria. Do not silently drop any of them.

### Inputs — two supported modes

1. **Count matrix mode** — one or more 10x MTX directories
   (`barcodes.tsv.gz`, `features.tsv.gz`, `matrix.mtx.gz`) containing both
   `Gene Expression` and guide (`Custom` / `CRISPR Guide Capture`) features.
2. **h5ad mode** — an existing `.h5ad`. Guide information may arrive as
   (a) guide features inside `var`, (b) a companion guide `.h5ad`, or
   (c) a pre-computed per-cell guide/genotype column in `obs`
   (e.g. `genotype = "AFF4-P1P2.2"`). All three must be handled.

### Sample metadata

- Datasets with **multiple lanes require a user-supplied sample metadata file**
  (CSV/TSV). One row per lane/sample; the key column joins to the lane ID.
- All metadata columns are merged into `adata.obs` and carried into the h5ad.
- Single-lane runs may omit it; the pipeline synthesizes a minimal record.

### Report — must cover three sections

1. **QC** — standard scRNA-seq QC *and* Perturb-seq-specific guide QC.
2. **Clustering analysis** results.
3. **Perturbation strength** — test whether the targeted gene's own expression
   is *lower* in perturbed cells than in control cells.

The report contains diagnostic figures plus representative figures for the
**strongest perturbation effects**.

### Deliverables of every run

1. A processed `.h5ad`.
2. The report (self-contained HTML).
3. **All** diagnostic figures on disk. When there are many targeted genes, the
   per-gene figures that did *not* make it into the report still go to a
   separate figures folder — nothing is silently dropped.
4. A **`.tar.gz` archive** of the run directory holding everything *except* the
   `.h5ad` matrices, so a whole run can be shared as one file. Controlled by
   `output.archive` / `archive_name` / `archive_exclude`; built last so the
   finished report is inside it, and it never packs itself.

### Demo

- A small-scale demo using the shared count matrix (one lane), plus a
  **runnable notebook** showing how to run that demo through the pipeline.
- Demo data is fetched from the owner's public Drive folder
  (`https://drive.google.com/drive/folders/1tU89UlsmZ6qTKPj348decXm523McLpvo`)
  via `gdown`. This requires the folder to be shared as "Anyone with the link";
  the fetch script must fail with a clear, actionable message if it is not.

## 3. File-size and storage rules

**Never commit large data to git.** GitHub is for code, configs, docs, and small
text tables only.

- Anything > ~50 MB, and *any* `.h5ad`, `.mtx`, `.mtx.gz`, `.rds`, or raw 10x
  directory, goes to Google Drive at:

  ```
  /content/drive/MyDrive/Colab Notebooks/scGPT/pancreatic/data/TF-PerturbSeq/small_scale/pipeline
  ```

  Refer to this path as `DRIVE_PIPELINE_DIR` in code and docs; make it a config
  value, never a hard-coded literal scattered through modules.
- Keep `.gitignore` covering: `*.h5ad`, `*.mtx*`, `*.rds`, `results/`,
  `*.h5`, `cache/`, `__pycache__/`, `.ipynb_checkpoints/`.
- Prototype notebooks must be committed **without output cells** (they carry
  multi-MB embedded PNGs otherwise).
- Small result tables (CSV/TSV) and the HTML report may be committed only when
  explicitly requested; by default they live under `results/` and are ignored.

## 4. Environment

- Target platform is **Google Colab** (Python 3.12, ~8 cores, ~50 GB RAM) with
  Drive mounted at `/content/drive`. It must also run as a plain CLI on a
  workstation or HPC login node.
- Core stack: `scanpy`, `anndata`, `numpy`, `pandas`, `scipy`, `matplotlib`,
  `seaborn`, `leidenalg`/`igraph`, `jinja2`, `pyyaml`, `statsmodels`.
- Pin nothing tighter than a lower bound unless a bug forces it; Colab
  pre-installs a lot and version fights are the main source of demo breakage.
- Do not assume a GPU. Do not add a heavy dependency (e.g. `scvi-tools`,
  `scGPT`) to the core path — optional extras only.

## 5. Code conventions

- Python package lives in `src/perturbseq_pipeline/`, installable, with a
  console entry point `perturbseq-pipeline`.
- **Config-driven**: one YAML file drives a run. Every threshold that appears in
  the prototype notebooks becomes a named config key with a documented default —
  no magic numbers buried in functions.
- One module per pipeline stage (`io`, `qc`, `guides`, `cluster`,
  `perturbation`, `plots`, `report`). Stages take and return `AnnData`; they do
  not reach into global state.
- Every figure is written to disk by a single helper that also registers the
  figure (path, caption, section) so the report builder can find it. Never call
  `plt.show()` in library code; use a non-interactive backend.
- Prefer vectorized numpy/pandas over the per-cell Python loops used in the
  prototypes (the guide-assignment loop over ~34k cells is the main offender).
- Log progress with `logging`, not bare `print`.
- Type-hint public functions. Docstrings state units and expected layers
  (raw counts vs log-normalized) — mixing those up is the classic bug here.

## 6. Analysis conventions

- `adata.layers["counts"]` always holds raw integer counts; `adata.X` holds
  log1p-normalized values after preprocessing. Perturbation tests run on
  log-normalized data.
- Guide-to-target-gene parsing is configurable. The demo library uses
  `TARGET_SUFFIX_N` (`AFF4_P1P2_3`) and the h5ad variant uses `TARGET-SUFFIX.N`
  (`AFF4-P1P2.2`) — both resolve to target `AFF4`.
- Non-targeting guides (`non`, `non_targeting`, `NTC`, `scramble`, …) are
  detected by a configurable pattern. The perturbation test is reported against
  **both** control definitions side by side:
  1. `ntc` — cells assigned to non-targeting guides (preferred, cleaner);
  2. `other` — all cells assigned to a *different* target gene (the prototype
     notebooks' behavior; larger n but the controls carry real effects).
  When a dataset has no non-targeting guides, the `ntc` columns are reported as
  not available and the report says so explicitly.
- Cells are labeled `unassigned` (no guide counts) or `ambiguous` (top guide not
  dominant over the second) — these are reported, never quietly discarded.
- Perturbation strength is directional: report effect size *and* direction, and
  correct p-values across genes (Benjamini–Hochberg). A gene is only called
  effectively perturbed if expression is **lower** in perturbed cells.

## 7. Git / GitHub workflow

- Repo: `weili-lab/perturbseq-pipeline` (default branch `main`).
- Develop on feature branches (`feat/…`, `fix/…`). Do not commit directly to
  `main`.
- **Do not commit or push unless the project owner asks.** Same for opening PRs,
  creating releases, or changing repo settings.
- Commit messages: imperative subject, one logical change per commit.

## 8. Working style in this repo

- Before changing analysis logic, check it against the prototype notebooks —
  they are the scientific reference for what the owner expects to see.
- Verify with the demo dataset before declaring a stage done; report real
  runtimes and real cell/gene counts, not estimates.
- If a run is skipped or a check fails, say so plainly rather than reporting
  partial success.
