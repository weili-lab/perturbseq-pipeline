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

Inputs can be 10x MTX directories or an existing `.h5ad`; guides can be called
one per cell or as scaffold pairs; every analysis stage after guide assignment
is switched on or off in the YAML config. The same code runs a one-lane demo
and a multi-million-cell screen on an HPC node.

---

## What it does

**1 · Quality control**
Standard single-cell QC (genes/UMIs per cell, mitochondrial, ribosomal and
hemoglobin fractions, per-lane breakdowns) *plus* Perturb-seq-specific guide QC:
guide UMI depth, guides detected per cell (MOI), top-vs-second guide dominance,
assignment outcome per lane, and guide/target library representation. Every
loaded cell is also written to an all-cells `.h5ad` before any filter, so the
before/after cell counts are on disk, not only in the report.

**2 · Guide assignment**
Three modes, selected with `guides.assignment_mode`. `single_guide` (the default)
assigns each cell to its dominant guide. `dual_guide_pair` resolves the
strongest guide of each scaffold class and interprets the pair through an
optional construct reference. `high_moi` calls every guide above a per-cell
threshold and stores a cells x targets membership matrix in
`obsm['perturbation_membership']` (see [docs/high_moi_assignment.md](docs/high_moi_assignment.md)).
All three write the same `obs['target_gene']` and `obs['perturbation_class']`
columns, which every later stage reads; in `high_moi` mode that label is the
cell's primary (highest-UMI) target. See [Guide calling](#guide-calling).

**3 · Clustering**
Library-size normalization, log1p, HVG selection, PCA, optional Harmony batch
correction, UMAP and Leiden clustering — with the embedding coloured by cluster,
lane, QC metrics, assignment class and target gene, so technical artefacts are
visible rather than implicit.

**4 · Perturbation strength**
For every target gene also measured in the expression matrix, the gene's *own*
expression is compared between perturbed and control cells. Effective CRISPR
perturbation lowers it, so the call is directional: a target is **effective**
only at BH-FDR < 0.05 **and** log2FC < 0.

Results are reported against **two control definitions** side by side:

| Control | Definition | Note |
|---|---|---|
| `ntc` | cells carrying non-targeting guides | preferred; same handling, no on-target effect |
| `other` | cells assigned to a *different* target gene | larger n, but controls are themselves perturbed |

**5 · Cluster enrichment** *(optional)*
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

**6 · Per-cell perturbation response (PS score)** *(optional)*
Sections 4 and 5 treat all cells carrying a guide as one group, but a perturbed
population is rarely uniform. This stage scores **each cell** using
[PS_python](https://github.com/weili-lab/PS_python), the lab's scMAGeCK-style
perturbation score, and combines it with the target's own expression to separate
**confirmed knockdowns** from **escapers** — cells that carry the guide and show
the signature yet still express the gene.

```bash
pip install -e ".[ps]"    # brings in pertps from PS_python
```

It also builds PS_python's **supervised LDA embedding**: where the UMAP in
section 3 is unsupervised and knows nothing about which guide a cell carries,
this one is trained on the perturbation labels, so its axes are chosen to
separate perturbations. Scores are shown in that space, one figure per target.
Disable with `ps_score.compute_lda_umap: false` if the extra few minutes and few
GB are not worth it.

Without the extra the stage is skipped and the report says so; set
`ps_score.require: true` to make it a hard failure.

**7 · lochNESS neighbourhood enrichment** *(optional)*
Ported from [pertTF](https://github.com/davidliwei/pertTF). For every cell and
every perturbation, the share of that cell's 300 nearest neighbours carrying the
perturbation, divided by its overall share, minus one — so 0 is background and
positive means locally over-represented. Continuous and cluster-free, so unlike
section 5 it also sees structure inside a cluster or across two, and it maps
*where* a perturbation accumulates. One figure per perturbation.

On the demo lane it independently reproduces the section-5 result (SALL4 and
SMARCC1 strongest; EZH2, SUZ12, NANOG and CTNNB1 all peaking in the same
cluster) without using clusters at all.

**8 · Co-functional modules & gene programs** *(optional)*
The "regulome" map, after
[Chen et al. 2023](https://www.nature.com/articles/s41586-023-06733-x). A
perturbation×gene matrix of log2FC-vs-control is clustered on both axes:
perturbations into **co-functional modules** (Spearman-correlation clustering)
and downstream genes into **co-regulated programs** (Pearson-correlation
clustering). It then relates the two — a signed module×program strength matrix
and alluvial — and draws the TF-hub and module–module networks. Every cell is
also scored for each program, and programs are annotated by over-representation
against full MSigDB collections (Hallmark, Reactome, GO BP; KEGG for human),
downloaded at run time through gseapy for `input.species` (`human` or `mouse`)
and the release in `modules.program_enrichment.msigdb_version`. Modules (`M1..`) and programs (`P1..`)
are numbered clusters, not biological labels; their member TFs and top genes
are in the tables so you can annotate them. The number of modules/programs is
configurable (`modules.n_modules` / `modules.n_programs`, or an automatic
dendrogram cut).

```bash
pip install -e ".[networks]"   # optional: the network-graph layouts (networkx)
```

The stage skips itself when a run has too few perturbations to be meaningful (a
single small lane), so it is most useful on a full multi-lane screen.

**9 · Perturbation distance** *(optional)*
How far did each perturbation move the whole transcriptome, not just its own
target? The **energy distance** between perturbed and control cells in PCA
space (with **MMD** as a secondary metric) measures the global displacement of
the cell distribution; a permutation test and BH-FDR say whether the shift is
larger than chance. Cells are subsampled deterministically
(`distance.max_cells_per_target`, `distance.max_control_cells`) so the cost is
bounded on large screens.

**10 · Perturbation distance space** *(optional)*
The same distance between every pair of perturbations gives a
perturbation×perturbation matrix, its PCoA coordinates, the nearest phenotypic
neighbours of each target, and **phenotype modules** from hierarchical
clustering. Where section 8 groups perturbations by shared downstream genes,
this groups them by similarity of the cell states they produce. The cost grows
with the square of the target count, so it is off in the large-screen example.

**Reference / inspiration:** The perturbation-distance analyses in sections 9
and 10 were conceptually inspired by and developed with reference to the
perturbation-analysis approaches available in
[scverse Pertpy](https://github.com/scverse/pertpy). The implementation in
this pipeline is integrated into the unified Perturb-seq workflow described
here.

**11 · Master perturbation table** *(optional)*
`tables/perturbation_meta.csv` joins the target-level results of sections 4
and 6 to 10 into one table, with the atlas, PS-versus-distance and module
concordance figures.

Stages 5 to 11 are enabled per section (`enrichment.enabled`,
`modules.enabled`, `ps_score.enabled`, `lochness.enabled`,
`distance.enabled`, `distance_space.enabled`, `meta_analysis.enabled`). The
defaults keep a run light: cluster enrichment and the master table are on,
the other five are off until the config enables them (the demo config turns
on PS score, lochNESS and modules). A disabled stage writes nothing and
appears as `disabled` in the report's module status table.

---

## Install

```bash
git clone https://github.com/weili-lab/perturbseq-pipeline.git
cd perturbseq-pipeline
pip install -e .

# optional extras
pip install -e ".[harmony]"   # batch correction across lanes
pip install -e ".[ps]"        # per-cell PS scores (pertps from PS_python)
pip install -e ".[networks]"  # network-graph layouts for the modules stage
pip install -e ".[gpu]"       # CuPy / cuML / rapids-singlecell acceleration
pip install -e ".[demo]"      # gdown, for fetching the demo data
pip install -e ".[dev]"       # pytest
```

Requires Python >= 3.9. Runs on Colab (Drive mounted), a workstation, or an HPC
node. No GPU needed; when one is present and the `gpu` extra is installed,
`compute.backend: auto` uses it for the dense linear algebra.

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

Start a config from the documented defaults, or from one of the examples:

```bash
perturbseq-pipeline init-config my_run.yaml
```

| Example | Shows |
|---|---|
| `config/demo.yaml` | one 10x lane, single-guide, the demo's stages on |
| `config/examples/mtx.yaml` | 10x MTX lanes with the default (core) stages only |
| `config/examples/h5ad.yaml` | `.h5ad` input with the optional stages chosen explicitly |
| `config/examples/all_modules.yaml` | every optional stage switched on |
| `config/examples/paired_guide.yaml` | separate GEX and guide matrices, paired-guide assignment, per-lane runs |
| `config/examples/large_hpc.yaml` | scaling, compute and storage settings for a multi-million-cell screen |
| `config/examples/basic_qc_samples.yaml` | several 10x wells, guide FASTQ counting, QC-only checkpoint |

`input.mode` is `auto` by default and follows whichever of `mtx_dirs`, `h5ad`
or `samples` is set; `mtx` and `h5ad` force one layout.

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
matrices use different barcode formats; a bare-versus-`-1` suffix mismatch is
reconciled automatically first.

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

When guide ids do not encode the target at all (10x Flex CRISPRi libraries name
guides by transcription start site), point `guides.target_feature_column` at
the guide-feature column that does, e.g. `target_gene_name`; values listed in
`guides.ignored_target_values` become `unassigned`.

#### 1.3 Per-lane and combined runs

```bash
perturbseq-pipeline run -c my_run.yaml --lane S1lane1            # -> <outdir>/samples/S1lane1
perturbseq-pipeline run -c my_run.yaml --combined-subdir combined # -> <outdir>/combined
```

With `input.cell_id_format: prefix` cells are named `<lane>_<barcode>` in both,
so a per-lane object is an exact row subset of the combined object.

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

### Option 3 — several 10x wells, QC checkpoint only

For a fresh sequencing run the first question is often just "what did we get".
A `samples:` block lists the wells, `run.stop_after: qc` runs a QC-only stage
and stops: per-well expression QC with MAD or fixed thresholds, Scrublet
doublet scores and guide-derived multiplet flags (flagged, never removed),
guide UMIs counted straight from the feature FASTQs or taken from a matrix, and
one all-cells `.h5ad` plus one expression-QC-pass `.h5ad` with a QC report.
`config/examples/basic_qc_samples.yaml` is a complete example.

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
├── report.md                        # the same content, figures by relative path
├── processed.h5ad                   # deliverable 1 — includes the guide matrix
├── processed_all_cells.h5ad         # every loaded cell, before QC filtering
├── figures/                         # deliverable 3
│   ├── qc/                          # cell QC, before and after filtering
│   ├── guides/                      # Perturb-seq guide QC (+ pair status in pair mode)
│   ├── clustering/                  # PCA, UMAPs, cluster composition
│   ├── perturbation/
│   │   ├── perturbation_volcano.png
│   │   ├── perturbation_waterfall.png
│   │   └── per_gene/                # EVERY target, not just those in the report
│   ├── enrichment/  modules/  ps_score/  lochness/
│   └── distance/  distance_space/
├── tables/                          # CSVs for all report tables
│   ├── qc_steps.csv  guide_assignment.csv  clusters.csv  perturbation.csv
│   ├── perturbation_distance.csv  phenotype_modules.csv  perturbation_meta.csv
│   ├── module_status.csv            # one row per stage: completed / skipped / disabled
│   └── compute_profile.csv          # seconds and memory per stage
├── logs/  run.log + resolved_config.yaml + run_manifest.json
└── <run>_results.tar.gz             # shareable bundle, matrices excluded
```

### The run manifest

`logs/run_manifest.json` records what ran, from which code, on which inputs:

| Field | Contents |
|---|---|
| `git` | branch, commit, whether the tree had uncommitted changes |
| `execution` | the exact command line, interpreter and working directory |
| `config_path`, `resolved_config` | the config given and the fully resolved copy |
| `random_seed` | `run.seed` |
| `input` | every input path with existence and size, the lanes actually loaded, metadata and pair-reference files |
| `guides.assignment_mode` | `single_guide`, `dual_guide_pair` or `high_moi` |
| `enabled_modules` | the `enabled` flag of every optional stage |
| `module_status` | per stage: `completed`, `skipped` (with the reason), or `disabled`, and seconds spent |
| `execution_mode`, `compute` | STANDARD or LARGE; backend, workers, CPUs seen, GPU present |
| `environment` | host, conda env, SLURM job, package versions |

The same facts appear in the report's *Run provenance* section, and
`logs/module_status.json` is updated after every stage so a failed run still
shows how far it got.

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
| `uns['guide_scaffolds']`, `uns['guide_pair_ids']` | scaffold class and construct id per guide, when known |
| `obs['target_gene']` | assigned target, or `ambiguous` / `unassigned` / `non-targeting` |
| `obs['perturbation_class']` | `targeting` / `non-targeting` / `ambiguous` / `unassigned` |
| `obs['guide_id']`, `top_guide_count`, `second_guide_count` | guide-call diagnostics |
| `obs['guide_assignment_mode']`, `pair_assignment_status`, `guide_A_*`, `guide_C_*` | pair-mode call and per-slot detail |
| `obsm['perturbation_membership']`, `obsm['guide_membership']`, `uns['membership_targets']` | high-MOI mode: cells x targets and cells x guides membership (sparse, 0/1) and the target column names |
| `obs['n_guides_assigned']`, `n_targets_assigned`, `is_ntc_only` | high-MOI mode: membership counts per cell |
| `obs['total_guide_counts']`, `n_guides_detected` | guide depth and MOI |
| `obs['leiden']`, `obsm['X_umap']` | clustering and embedding |
| `obs['ps_score']`, `lochness_self`, `obsm['X_lda_umap']` | per-cell scores from the optional stages |
| `obs[...]` | all sample metadata columns |

Column and key names that would be illegal or fragile in HDF5 (a target such as
`LIPA (rs1412444)` becomes `lochness_LIPA__rs1412444_`) are sanitised on write;
the original names stay in the tables and the mapping is stored in
`uns['column_name_mapping']` and next to the file as
`<stem>_column_name_mapping.csv`.

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
  assignment_mode: single_guide  # or dual_guide_pair, see below
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

### Paired-guide mode

Dual-guide libraries put one guide on a scaffold-A vector position and one on
a scaffold-C position; the single-guide rule labels every such cell
`ambiguous` by design. `dual_guide_pair` applies the same gate **within each
scaffold class** and then interprets the pair:

```yaml
guides:
  assignment_mode: dual_guide_pair
  pair_assignment_primary: true
  pair_reference: guide_reference.csv   # optional: guide_id, target, scaffold, construct id
  scaffold_classes: [A, C]
  require_complete_pair: true
  ntc_partner_policy: ambiguous         # targeting + NTC without a designed construct
```

| `pair_assignment_status` | `perturbation_class` |
|---|---|
| `pair_targeting` (same designed target in both slots) | `targeting` |
| `pair_non_targeting` (two non-targeting guides) | `non-targeting` — the primary control |
| `pair_targeting_plus_ntc` (designed target + NTC construct) | `targeting`, or `ambiguous` when `designed_targeting_plus_ntc_primary: false` |
| `dual_target_ambiguous`, `incomplete_pair`, `ambiguous_scaffold_*`, `unknown_guide`, `below_min_umi`, `unresolved_pair` | `ambiguous` |
| `no_guide` | `unassigned` |

Downstream stages see only `target_gene` and `perturbation_class`, so they run
unchanged. Pair-mode runs add per-lane pair QC, per-target ECDFs and a
single-guide diagnostic to the report. Details and every status are in
[`docs/paired_guide_assignment.md`](docs/paired_guide_assignment.md).

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

One place ambiguous cells still act is **clustering**, which by default runs on
all QC-passing cells. Multiplets pass gene-expression QC and fragment the
embedding into many small clusters (on M0_ch1, 123 instead of 37).

`cluster.assigned_only: true` splits the run into two objects rather than
discarding anything:

| | Cells | Written as | Used for |
|---|---|---|---|
| all cells | every QC-passing cell | `<name>_all_cells.h5ad` | the `all_cells_*` UMAPs, where ambiguous/unassigned cells are visible |
| analysed | guide-assigned singlets | `<name>.h5ad` | clustering, perturbation, enrichment, PS scores, lochNESS, distance |

Each is embedded independently and is internally complete, so **cluster labels
do not carry across the two files**. Set `output.write_unfiltered_h5ad: false`
to keep the figures but skip the extra (large) matrix.

The same rule is applied when guides arrive as a barcode table
(`input.guide_table`), reading the same two keys, so a run from a count matrix
and a run from a barcode table produce identical per-cell calls.

### Knockdown mask

`knockdown_filter` marks targeting cells whose own target is not knocked down.
It **removes nothing**: every analysis still runs on all cells, so section 3's
perturbation strength stays an independent estimate. The mask and those
estimates are written side by side for the final filtering:

| Where | Contents |
|---|---|
| `obs['kd_ratio']` | target expression / mean in non-targeting cells of the same context |
| `obs['kd_status']` | `knockdown`, `escaper`, `failed_group`, `low_control_expression`, `not_measured`, `non_testable`, `unfiltered_context`, `control`, `untouched` |
| `obs['kd_keep']` | False for the marked statuses (`escaper`, `failed_group`, `low_control_expression`, `not_measured`) |
| `obs['pert_log2fc_ntc']`, `pert_ks_fdr_ntc`, … | section-3 estimates for the cell's target |
| `tables/knockdown_filter.csv` | one row per target × context, with both views |

Per group: (1) pass if the mean `kd_ratio` < `max_mean_ratio`, i.e. mean
target-cell over mean control expression (a median would be 0 for any target
detected in under half of the cells, knockdown or not); (2) in a passing group,
mark cells with `kd_ratio` >= `max_cell_ratio` as escapers; (3) groups under
`min_cells` are `non_testable` and left unmarked. `mode` sets the group:
`pooled` (per target, cell-weighted mean across contexts), `per_context` (per
target × context), or `any_context` (step 1 per context with
`max_mean_ratio_any`; a target that passes anywhere is filtered only in its
passing contexts and keeps every cell elsewhere). The control baseline is per
context in every mode whenever `context_key` is set.

The default is `mode: pooled` with `context_key: null`: one baseline over all
control cells and one group per target, so `enabled: true` runs as is. With
several cell lines or conditions, set `context_key` and, if knockdown may
differ between them, `mode: per_context`:

```yaml
knockdown_filter:
  enabled: true
  mode: per_context
  context_key: cell_line
```

```python
final = adata[adata.obs["kd_keep"] & (adata.obs["perturbation_class"] != "ambiguous")]
```

`method: count_model` replaces steps 1-2 with a negative-binomial mixture on
the target's raw counts (`layers['counts']`; the run stops if the first 10,000
cells of that layer are not non-negative integers). Each target cell is either
unperturbed (an escaper), with the mean and overdispersion of its context's
controls scaled by its library size, or knocked down to a fraction `rho` of
that mean. Maximum likelihood fits `rho` and the escaper fraction per group. The group passes
when `rho < max_rho`, so escapers no longer dilute the test, and the escaper
fraction is below `max_escaper_fraction` (0.5: most cells must be knocked down;
without it, a group with no knockdown can be fitted as a near-zero majority plus
escapers, which passed 1.7% of null groups built from Nadig control cells), and a cell is an
escaper when `obs['kd_escaper_prob']` reaches `min_escaper_prob`. The table
gains `rho`, `escaper_fraction`, `control_mean_counts` and `control_dispersion`.
The dispersion is shared across genes as in DESeq2: each gene's estimate from
its context's controls is shrunk toward a trend over all genes, weighted by how
noisy it is (measured by splitting the controls in half). This helps when a
context has few controls (about 20-35% lower error at 30-100 cells in
simulation) and changes little with thousands.
The ratio cut calls every cell with a single count an escaper when the target is
weakly expressed. The count model does not: a single count is weak evidence,
so the posterior stays near the escaper fraction and such cells are left as
knockdowns. The posterior is calibrated but conservative: when escaper and
knocked-down counts overlap, many escapers stay below the 0.9 cut.

Non-targeting guides are detected by pattern (`non`, `non_targeting`, `NTC`,
`scramble`, …) via `guides.ntc_patterns` and used as the preferred control group.

---

## Scaling

The same branch runs the demo lane and a screen of millions of cells. Nothing
about the statistics changes with size; the memory strategy does.

| Section | Keys | What they control |
|---|---|---|
| `scaling` | `mode: auto \| standard \| large`, `large_n_cells`, `large_n_perturbations`, `marker_max_cells`, `effect_gene_chunk`, `guide_chunk_size` | STANDARD keeps the historical in-memory paths; LARGE keeps matrices sparse (PCA scaling without densification), estimates HVGs on a bounded sample, accumulates the effect matrix in gene chunks, scores lochNESS in target chunks and samples cells for scatter plots |
| `compute` | `backend: auto \| cpu \| gpu`, `n_jobs`, `blas_threads_per_worker`, `<stage>_n_jobs` | target-level tests, permutations and pairwise distances run in worker pools; the worker count follows `n_jobs` and the SLURM allocation; the GPU is used only for dense linear algebra when present |
| `storage` | `mode: auto \| in_memory \| backed`, `backed_threshold_cells` | backed `.h5ad` reading above a cell count; embeddings stay in memory |

`tables/compute_profile.csv` records seconds and memory per stage, and the run
manifest records the mode that was actually used. Job templates and resource
guidance for a scheduler are in [`docs/slurm.md`](docs/slurm.md); real datasets
belong in a job, not on a login node.

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The suite generates a synthetic dataset with a **known ground truth** — some
targets are genuinely knocked down, others are not — and asserts that the
pipeline recovers exactly those, through both entry points and all three h5ad
layouts. Further files cover paired-guide assignment on a synthetic dual-guide
library, the QC-only stage on synthetic 10x wells and guide FASTQs, the
distance and module stages, STANDARD-versus-LARGE numerical consistency, the
compute backend, and the run manifest, module switches and HDF5-safe naming of
this branch.

---

## Repository layout

```
src/perturbseq_pipeline/   io · qc · guides · dual_guides · cluster · perturbation · enrichment ·
                           modules · ps_score · lochness · distance · meta · basic_qc ·
                           compute · data_access · run_manifest · plots · report · cli
config/                    default.yaml (all documented defaults) · demo.yaml · examples/
demo/                      fetch_demo_data.py · sample_metadata.csv
docs/                      methods.md · paired_guide_assignment.md · slurm.md · PIPELINE_DEVELOPMENT.md
notebooks/                 demo_run_pipeline.ipynb · prototype/ (original analyses)
tests/                     synthetic data generators + end-to-end and unit tests
```
