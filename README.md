# perturbseq-pipeline

A simple, reproducible Perturb-seq pipeline: from counts to a QC + clustering +
perturbation-strength report, in one command.

```bash
perturbseq-pipeline run --config config/demo.local.yaml
```

Every run produces three deliverables:

1. a processed **`.h5ad`**,
2. a self-contained **HTML report**, and
3. **all diagnostic figures** on disk — including the per-target figures that
   were too numerous to embed in the report.

---

## What it does

**1 · Quality control**
Standard single-cell QC (genes/UMIs per cell, mitochondrial, ribosomal and
hemoglobin fractions, per-lane breakdowns) *plus* Perturb-seq-specific guide QC:
guide UMI depth, guides detected per cell (MOI), top-vs-second guide dominance,
assignment outcome per lane, and guide/target library representation.

**2 · Clustering**
Library-size normalization, log1p, HVG selection, PCA, optional Harmony batch
correction, UMAP and Leiden clustering — with the embedding coloured by cluster,
lane, QC metrics, assignment class and target gene, so technical artefacts are
visible rather than implicit.

**3 · Perturbation strength**
For every target gene also measured in the expression matrix, the gene's *own*
expression is compared between perturbed and control cells. Effective CRISPR
perturbation lowers it, so the call is directional: a target is **effective**
only at BH-FDR < 0.05 **and** log2FC < 0.

Results are reported against **two control definitions** side by side:

| Control | Definition | Note |
|---|---|---|
| `ntc` | cells carrying non-targeting guides | preferred; same handling, no on-target effect |
| `other` | cells assigned to a *different* target gene | larger n, but controls are themselves perturbed |

**4 · Cluster enrichment**
The follow-on question: did losing the gene push cells into a particular
transcriptional state? Every target is tested against every cluster with
Fisher's exact test (BH-FDR across all pairs), reporting odds ratio, direction,
and **how many of the target's guides independently agree** — a real phenotype
appears across several guides, a single-guide artefact does not. Set
`enrichment.stratify_by: lane_id` on a multi-lane run for a Cochran–Mantel–
Haenszel test that controls for lane differences in cluster composition.

On the demo lane this recovers SMARCC1 (core SWI/SNF) taking over one cluster,
and EZH2 with SUZ12 — both core PRC2 subunits — independently landing in the
same one.

**5 · Per-cell perturbation response** *(optional)*
Sections 3 and 4 treat all cells carrying a guide as one group, but a perturbed
population is rarely uniform. This stage scores **each cell** using
[PS_python](https://github.com/weili-lab/PS_python), the lab's scMAGeCK-style
perturbation score, and combines it with the target's own expression to separate
**confirmed knockdowns** from **escapers** — cells that carry the guide and show
the signature yet still express the gene.

```bash
pip install -e ".[ps]"    # brings in pertps from PS_python
```

It also builds PS_python's **supervised LDA embedding**: where the UMAP in
section 2 is unsupervised and knows nothing about which guide a cell carries,
this one is trained on the perturbation labels, so its axes are chosen to
separate perturbations. Scores are shown in that space, one figure per target.
Disable with `ps_score.compute_lda_umap: false` if the extra few minutes and few
GB are not worth it.

Without the extra the stage is skipped and the report says so; set
`ps_score.require: true` to make it a hard failure.

**6 · lochNESS neighbourhood enrichment**
Ported from [pertTF](https://github.com/davidliwei/pertTF). For every cell and
every perturbation, the share of that cell's 300 nearest neighbours carrying the
perturbation, divided by its overall share, minus one — so 0 is background and
positive means locally over-represented. Continuous and cluster-free, so unlike
section 4 it also sees structure inside a cluster or across two, and it maps
*where* a perturbation accumulates. One figure per perturbation.

On the demo lane it independently reproduces the section-4 result (SALL4 and
SMARCC1 strongest; EZH2, SUZ12, NANOG and CTNNB1 all peaking in the same
cluster) without using clusters at all.

---

## Install

```bash
git clone https://github.com/weili-lab/perturbseq-pipeline.git
cd perturbseq-pipeline
pip install -e .

# optional extras
pip install -e ".[harmony]"   # batch correction across lanes
pip install -e ".[demo]"      # gdown, for fetching the demo data
```

Requires Python >= 3.9. Runs on Colab (Drive mounted), a workstation, or an HPC
login node. No GPU needed.

---

## Quick start: the demo

One lane of a human ESC transcription-factor screen — 416 guides, 61 targets,
30 non-targeting controls.

```bash
python demo/fetch_demo_data.py --dest demo_data --write-config config/demo.local.yaml
perturbseq-pipeline run --config config/demo.local.yaml
```

Already have the data (e.g. on a mounted Drive)? Skip the download:

```bash
python demo/fetch_demo_data.py --source "/path/to/raw_counts" --dest demo_data
```

There is also a runnable notebook that walks through the whole thing:
**[`notebooks/demo_run_pipeline.ipynb`](notebooks/demo_run_pipeline.ipynb)**.

---

## Inputs

Start a config from the documented defaults:

```bash
perturbseq-pipeline init-config my_run.yaml
```

### Option 1 — 10x count matrices

```yaml
input:
  mode: mtx
  mtx_dirs:
    S1lane1: /path/to/filtered_feature_bc_matrix_S1lane1
    S1lane2: /path/to/filtered_feature_bc_matrix_S1lane2
metadata:
  file: my_samples.csv
cluster:
  batch_key: lane_id     # Harmony correction across lanes
```

Each directory needs `barcodes.tsv.gz`, `features.tsv.gz` and `matrix.mtx.gz`,
with both `Gene Expression` and guide (`Custom` / `CRISPR Guide Capture`)
features.

### Option 2 — an existing `.h5ad`

Guide information can arrive three ways; the pipeline detects them in order:

```yaml
input:
  mode: h5ad
  h5ad: /path/to/my_data.h5ad

  # (a) guide features already in var['feature_types'] — nothing else needed
  # (b) a companion guide count matrix:
  guide_h5ad: /path/to/my_guides.h5ad
  # (c) a pre-computed per-cell label, e.g. a Seurat 'genotype' column:
  guide_obs_column: genotype
  # (d) a barcode -> guide table (the PS_python demo layout):
  guide_table: BARCODE_10x_Merged.txt

  # Seurat exports often keep counts in X and log values in a layer:
  normalized_layer: logcounts
  # ...or use "X" when the object holds only normalized values and no counts:
  # normalized_layer: "X"
```

A barcode table lists one row per detected guide per cell, so a cell with two
guides appears twice. The pipeline resolves those with the **same dominance rule
as the count-matrix path** (top guide must clear `guides.min_umi` and beat the
runner-up by `guides.dominance_ratio`) rather than keeping whichever row came
last, which would pick a guide at random for every multiplet.

### Sample metadata

**Required for any run spanning more than one lane.** One row per lane, joined
on `lane_id`; every column is merged into `adata.obs` and travels with the
output `.h5ad`.

```csv
lane_id,sample,lane,cell_line,condition,replicate
S1lane1,S1,1,ESC,TF_screen,1
S1lane2,S1,2,ESC,TF_screen,1
```

---

## Outputs

```
results/<run>/
├── report.html                      # deliverable 2 — self-contained
├── processed.h5ad                   # deliverable 1
├── processed_guides.h5ad            # guide count matrix
├── figures/                         # deliverable 3
│   ├── qc/                          # cell QC, before and after filtering
│   ├── guides/                      # Perturb-seq guide QC
│   ├── clustering/                  # PCA, UMAPs, cluster composition
│   └── perturbation/
│       ├── perturbation_volcano.png
│       ├── perturbation_waterfall.png
│       └── per_gene/                # EVERY target, not just those in the report
├── tables/                          # CSVs for all report tables
├── logs/  run.log + resolved_config.yaml
└── <run>_results.tar.gz             # shareable bundle, matrices excluded
```

### The guide barcode table

Every run also exports the guide count matrix as a long `barcode -> guide`
table (`<run>_guide_barcodes.txt`) — the format
[PS_python](https://github.com/weili-lab/PS_python) consumes. It was verified to
reproduce that project's `BARCODE_10x_Merged.txt` exactly: filtering the matrix
at >= 3 UMIs matches the original on per-cell totals and guides-per-cell for
100% of shared cells.

```yaml
output:
  write_guide_table: true
  guide_table_min_umi: 3
```

Two deliberate differences: the `gene` column uses the pipeline's target parser
(so `CD81.2` collapses to `CD81` instead of becoming its own target), and an
`assignment` column carries the pipeline's dominance-rule call so consumers get
the same per-cell answer. Rows are ordered with the dominant guide last, so even
a naive "last row wins" reader lands on the right guide. See
[`docs/ps_python_proposal.md`](docs/ps_python_proposal.md).

### The results archive

Every run bundles its outputs into one `.tar.gz` — the report, all figures,
tables and logs, with the `.h5ad` matrices left out so the archive stays small
enough to email or attach to a GitHub release. It unpacks into a single
directory named after the run.

```yaml
output:
  archive: true                 # set false to skip
  archive_name: null            # defaults to <run.name>_results.tar.gz
  archive_exclude: ["*.h5ad", "*.h5", "*.loom", "*.tar.gz"]
```

Patterns are matched against paths relative to the run directory (and against
bare filenames), so `"figures/perturbation/per_gene/*"` would drop just the
per-target figures.

Large outputs can be redirected off local disk (useful on Colab):

```yaml
output:
  large_file_dir: "/content/drive/MyDrive/.../pipeline"
  large_file_threshold_mb: 50
```

### The processed `.h5ad`

| Slot | Contents |
|---|---|
| `X`, `layers['lognorm']` | log1p of library-size-normalized counts |
| `layers['counts']` | raw integer counts |
| `obs['target_gene']` | assigned target, or `ambiguous` / `unassigned` / `non-targeting` |
| `obs['perturbation_class']` | `targeting` / `non-targeting` / `ambiguous` / `unassigned` |
| `obs['guide_id']`, `top_guide_count`, `second_guide_count` | guide-call diagnostics |
| `obs['total_guide_counts']`, `n_guides_detected` | guide depth and MOI |
| `obs['leiden']`, `obsm['X_umap']` | clustering and embedding |
| `obs[...]` | all sample metadata columns |

---

## Guide calling

A cell is assigned to its top guide when that guide has at least
`guides.min_umi` UMIs **and** exceeds the runner-up by `guides.dominance_ratio`:

```yaml
guides:
  min_umi: 3
  dominance_ratio: 2.0
  target_split_delims: ["_", "-", "."]   # AFF4_P1P2_1 -> AFF4
  target_regex: null                     # override when targets contain a delimiter
```

Cells failing the dominance rule are `ambiguous`; cells with no guide counts are
`unassigned`. Both stay in the object and are reported — they are never silently
dropped.

Non-targeting guides are detected by pattern (`non`, `non_targeting`, `NTC`,
`scramble`, …) and used as the preferred control group.

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The suite generates a synthetic dataset with a **known ground truth** — some
targets are genuinely knocked down, others are not — and asserts that the
pipeline recovers exactly those, through both entry points and all three h5ad
layouts.

---

## Repository layout

```
src/perturbseq_pipeline/   io · qc · guides · cluster · perturbation · plots · report · cli
config/                    default.yaml (all documented defaults) · demo.yaml
demo/                      fetch_demo_data.py · sample_metadata.csv
notebooks/                 demo_run_pipeline.ipynb · prototype/ (original analyses)
tests/                     synthetic data generator + end-to-end tests
```

`CLAUDE.md` records the project conventions, including where large files belong.
