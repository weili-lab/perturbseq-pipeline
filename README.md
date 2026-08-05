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

Every directory needs `barcodes.tsv.gz`, `features.tsv.gz` and `matrix.mtx.gz`.
Which of the two layouts below you have depends on how the run was quantified.

#### 1.1 Guides and gene expression in one matrix

The CellRanger layout: one directory per lane, holding both `Gene Expression`
and guide (`Custom` / `CRISPR Guide Capture`) features, told apart by the third
column of `features.tsv.gz`. This is the ESC TF Perturb-seq screen.

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

The pipeline splits the two feature classes itself. If your file names the guide
class something else, set `input.guide_feature_types`.

#### 1.2 Gene expression and guides quantified separately

The STARsolo layout: each lane has two independent MTX directories, and
`features.tsv.gz` carries no usable class column — guides are often labelled
`Gene Expression` too, so there is nothing to split on. This is the THP-1 /
M0 / M1 screen:

```
count_matrices/{THP1,M0,M1}/{ch_1,ch_2}/
  GEX/filtered/    <- genes, called cells only
  sgRNA/raw/       <- guides, the entire barcode whitelist
```

Add `guide_mtx_dirs` alongside `mtx_dirs`, **using the same lane keys**:

```yaml
input:
  mode: mtx
  mtx_dirs:                                     # gene expression
    THP1_ch1: /path/THP1/ch_1/GEX/filtered
    M0_ch1:   /path/M0/ch_1/GEX/filtered
    M1_ch1:   /path/M1/ch_1/GEX/filtered
  guide_mtx_dirs:                               # guide counts, same keys
    THP1_ch1: /path/THP1/ch_1/sgRNA/raw
    M0_ch1:   /path/M0/ch_1/sgRNA/raw
    M1_ch1:   /path/M1/ch_1/sgRNA/raw
  var_names: gene_symbols

guides:
  # Strip only a trailing _<number>, so multi-token targets survive:
  #   ADGRV1_1 -> ADGRV1, gene_desert_1 -> gene_desert, non-targeting_20 -> non-targeting
  # The default first-delimiter split would give a target called "gene".
  target_regex: '^(.+)_\d+$'
  ntc_patterns: ["^non[-_.]?targeting$"]

metadata:
  file: my_samples.csv
cluster:
  batch_key: lane_id
```

Three things worth knowing about this layout:

**The keys must match exactly.** A lane in one mapping and not the other is
rejected at config load, rather than after a long read.

**The guide matrix is usually much larger than the cell set.** STARsolo emits
guide counts over the whole barcode whitelist — 737,280 barcodes against 36,364
called cells in the THP-1 run. The pipeline subsets it to the barcodes in the
expression matrix, filling any missing ones with zeros rather than dropping
those cells, and logs the match rate per lane (it was 100% for all three THP-1
channels). No overlap at all is an error, since that almost always means the two
matrices use different barcode formats.

**Point GEX at the called cells and guides at whatever exists.** In the THP-1
tree that is `GEX/filtered` and `sgRNA/raw` — `sgRNA` has no `filtered` output.

Check your guide naming before a long run:

```bash
python -c "
from perturbseq_pipeline.config import Config
from perturbseq_pipeline.guides import parse_target_genes, is_non_targeting
cfg = Config.from_yaml('config/my_run.yaml')
names = ['ADGRV1_1', 'gene_desert_3', 'non-targeting_20']
t = parse_target_genes(names, cfg.guides)
print(dict(zip(names, t)), dict(zip(t, is_non_targeting(t, cfg.guides))))"
```

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
├── processed.h5ad                   # deliverable 1 — includes the guide matrix
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
| `obsm['guide_counts']` | the raw guide count matrix, sparse (cells x guides) |
| `uns['guide_names']`, `uns['guide_target_genes']` | guide IDs and their parsed targets |
| `obs['target_gene']` | assigned target, or `ambiguous` / `unassigned` / `non-targeting` |
| `obs['perturbation_class']` | `targeting` / `non-targeting` / `ambiguous` / `unassigned` |
| `obs['guide_id']`, `top_guide_count`, `second_guide_count` | guide-call diagnostics |
| `obs['total_guide_counts']`, `n_guides_detected` | guide depth and MOI |
| `obs['leiden']`, `obsm['X_umap']` | clustering and embedding |
| `obs[...]` | all sample metadata columns |

### One file, both matrices

The guide counts live **inside** the processed `.h5ad`, in
`obsm['guide_counts']`, so a run is a single file rather than a pair that can
drift apart. They sit in `obsm` rather than being concatenated onto `var`
because guide counts are not gene expression: putting them in `var` would feed
them to normalization, HVG selection and scaling along with the genes.

They stay sparse — on the THP-1 run that is 7.2M non-zeros in a 103,151 x 694
matrix, about 58 MB in memory against 286 MB dense. `obsm` also keeps the rows
tied to the cells through any subsetting, which two separate files do not.

```python
import scanpy as sc
adata = sc.read_h5ad("results/my_run/processed.h5ad")
guides = adata.obsm["guide_counts"]          # sparse, cells x guides
names  = adata.uns["guide_names"]            # column labels
```

A merged object can be fed straight back to the pipeline — the reader detects
`obsm['guide_counts']` and rebuilds the guide matrix, no companion file needed.

```yaml
output:
  merge_guides_into_h5ad: true   # guides inside the processed .h5ad
  guide_obsm_key: guide_counts
  write_guide_h5ad: false        # also write the old separate file
```

---

## Guide calling

For each cell the pipeline takes the highest and second-highest guide counts and
applies one rule:

```python
assigned = (
    top >= guides.min_umi
    and top > guides.dominance_ratio * second
    and (guides.max_second_umi < 0 or second <= guides.max_second_umi)
)
```

| Condition | Label in `obs['target_gene']` |
|---|---|
| top count is 0 | `unassigned` |
| top ≥ `min_umi`, top > `dominance_ratio` × second, and second within `max_second_umi` | the guide's target gene |
| anything else | **`ambiguous`** |

There is no separate "ambiguous threshold" — `ambiguous` is the fallback when a
cell *has* guide counts but fails the rule, either because its best guide is too
weak or because the runner-up is too close to it.

```yaml
guides:
  min_umi: 3            # the top guide must reach this many UMIs
  dominance_ratio: 2.0  # ...and exceed the runner-up by this factor
  max_second_umi: -1    # hard cap on the runner-up; -1 disables the gate
  detection_threshold: 3         # MOI statistics only — NOT used for assignment
  target_split_delims: ["_", "-", "."]   # AFF4_P1P2_1 -> AFF4
  target_regex: null             # override when targets contain a delimiter
  ambiguous_label: ambiguous     # the strings written into obs
  unassigned_label: unassigned
  ntc_label: non-targeting
```

### Tuning the ambiguous rate

**`dominance_ratio` is the knob that matters.** At 2.0 a cell with counts 40 and
25 is ambiguous (40 < 50); at 1.5 it would be assigned. That one value drives
most of the ambiguous rate — 24% of cells on the ESC screen. The prototype
notebooks used 1.2 in one and 2.0 in the other, which is why it is an explicit
key rather than a number buried in the code.

`min_umi` matters less on deeply sequenced guide libraries, where the top guide
is usually far above 3, but raising it is the right move when guide capture is
shallow and low-count calls are unreliable.

Watch out for **`detection_threshold`**, which looks similar but is only used for
the guides-per-cell and MOI statistics in the QC section. Changing it does not
move a single assignment.

### Removing droplet multiplets

A cell carrying genuine counts of a *second* guide is usually two cells in one
droplet. The dominance ratio alone does not catch those, because a ratio scales
with sequencing depth: at `dominance_ratio: 2.0` a cell with 1,000 and 100 UMIs
passes, even though 100 UMIs of a second guide is real signal rather than
ambient. `max_second_umi` puts an absolute cap on the runner-up, which does not
scale.

It is **off by default** (`-1`). To calibrate it, the THP-1 M0_ch1 channel was
compared against that study's own published cell set, which keeps only cells
strictly expressing one sgRNA (14,156 of 48,554 cells):

| Setting | Cells assigned | Recall | Precision | F1 |
|---|---|---|---|---|
| `-1` (off) | 30,208 | 0.970 | 0.455 | 0.619 |
| `max_second_umi: 5` | 12,345 | 0.832 | 0.954 | 0.889 |
| **`max_second_umi: 8`** | **14,580** | **0.920** | **0.893** | **0.906** |
| `max_second_umi: 10` | 15,563 | 0.942 | 0.857 | 0.897 |

Raising `dominance_ratio` instead does **not** work as well — its best setting
(20) reaches only F1 0.817, because it cannot distinguish a deep singlet from a
deep doublet. The two knobs address different things and are worth setting
independently.

Note this is a property of the *library and chemistry*, not a universal
constant: the runner-up count on Seurat-retained cells had a median of 3 and a
99th percentile of 16, whereas rejected cells sat at a median of 33. Re-derive
it for a new dataset rather than copying the number.

### What happens to ambiguous cells

They are **kept** in the object and counted in the report, but excluded from
perturbation testing and from *both* control groups. So loosening
`dominance_ratio` does not merely relabel cells — it moves them into the tested
populations. On the four-lane run that category holds 27,566 cells (25.4%),
so the setting has real leverage over every downstream result.

The same rule is applied when guides arrive as a barcode table
(`input.guide_table`), reading the same two keys, so a run from a count matrix
and a run from a barcode table produce identical per-cell calls.

Non-targeting guides are detected by pattern (`non`, `non_targeting`, `NTC`,
`scramble`, …) via `guides.ntc_patterns` and used as the preferred control group.

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
