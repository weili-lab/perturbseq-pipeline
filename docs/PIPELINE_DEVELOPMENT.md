# Perturb-seq Pipeline: Extended Analyses and Scalable Execution

## Overview

This document describes the analytical stages, modular architecture, and centralized scalable execution framework of the `perturbseq-pipeline` package.

The pipeline provides an end-to-end workflow for CRISPR pooled single-cell screens (Perturb-seq), taking raw count matrices or processed AnnData objects and producing:
1. Standard single-cell and guide-specific quality control (QC)
2. Guide assignment and multiplet classification
3. Normalization, highly variable gene (HVG) selection, embedding, and Leiden clustering
4. Directional perturbation-strength testing against dual control groups (`ntc` and `other`)
5. Cluster-level perturbation enrichment testing (Fisher's exact, Cochran–Mantel–Haenszel, guide concordance, omnibus permutation)
6. Co-functional perturbation modules and co-regulated gene programs (regulome discovery)
7. Per-cell perturbation response scores (via the `pertps` / `PS_python` framework)
8. Continuous neighbourhood perturbation enrichment via lochNESS
9. Centralized scalable execution supporting datasets ranging from standard screens (~300k cells, e.g. Replogle) to multi-million-cell pan-genome libraries (>2.6M cells, e.g. KOLF)
10. Complete deliverables: processed `.h5ad`, structured tabular CSV outputs, publication-ready vector/raster figures, and a self-contained interactive HTML report.

---

## Pipeline Workflow

The complete analysis pipeline executes 14 sequential stages managed centrally by `perturbseq_pipeline.cli.run_pipeline`:

```text
 [1. Input Loading]
         │
         ▼
 [2. QC Filtering] ─────────────► [Cell & Gene QC Statistics]
         │
         ▼
 [3. Guide Assignment] ─────────► [Dominance / Multiplet Gate]
         │
         ▼
 [4. Normalization & Cluster] ──► [HVG -> PCA -> UMAP -> Leiden]
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
 [5. Perturbation Strength] ────► [Directional Knockdown vs NTC/Other]
         │                                          │
         ├────────────────────┐                     ▼
         │                    │              [6. Cluster Enrichment]
         ▼                    ▼              [Fisher / CMH / Concordance]
 [7. Modules & Programs]   [8. PS Scoring]
 [Log2FC Matrix -> M/P]    [Cellular Penetrance]
         │                    │
         ├────────────────────┘
         │
         ▼
 [9. lochNESS] ─────────────────► [k-NN Neighbourhood Topology (k=300)]
         │
         ▼
 [10. Distance & DistanceTest] ─► [Energy Distance & Permutation FDR vs NTC]
         │
         ▼
 [11. DistanceSpace] ───────────► [Pairwise Distance Matrix, PCoA, Phenotype Modules]
         │
         ▼
 [12. Master Meta & Atlas] ─────► [tables/perturbation_meta.csv & Cross-Layer Plots]
         │
         ▼
 [13. Output Writing] ──────────► [Processed H5AD, Guide Table, CSVs, Figures, Tarball]
         │
         ▼
 [14. HTML Report] ─────────────► [Self-contained Interactive Report & Manifest]
```

### Execution Stages in `cli.py`

1. **Stage 1/14: Input Loading & Validation**: Resolves 10x MTX directories, companion guide matrices, or `.h5ad` inputs; merges sample metadata; applies `Config.validate()`; determines scaling execution mode (`standard` vs `large`).
2. **Stage 2/14: Quality Control**: Annotates mitochondrial, ribosomal, and hemoglobin genes; filters low-quality cells and unexpressed genes; records step-by-step filtering tables.
3. **Stage 3/14: Guide Assignment**: Determines dominant and runner-up guide counts per cell; applies dominance ratio and optional second-guide UMI gates; parses target gene identities.
4. **Stage 4/14: Normalization, Embedding & Clustering**: Performs library-size normalization and log1p transformation; identifies HVGs; computes PCA, k-NN graph, UMAP projection, and Leiden clustering. If `cluster.assigned_only: true`, optionally re-embeds singlets while preserving the all-cell embedding.
5. **Stage 5/14: Perturbation Strength (Efficacy)**: Measures target gene knockdown in perturbed cells versus Non-Targeting Controls (`ntc`) and Other-Targeting Controls (`other`); calculates log2 fold changes and Benjamini–Hochberg FDR values.
6. **Stage 6/14: Cluster Enrichment**: Evaluates whether perturbation targets alter cluster occupancy using Fisher's exact test, stratified CMH tests across lanes, guide concordance verification, and omnibus permutation testing.
7. **Stage 7/14: Co-functional Modules & Gene Programs (Regulome)**: Constructs a perturbation × gene effect matrix; clusters genes into co-regulated programs (Pearson correlation) and perturbations into co-functional modules (Spearman correlation); identifies TF hubs and connectivity networks.
8. **Stage 8/14: Per-cell Perturbation Scores (PS Penetrance)**: Executes per-cell signature scoring via `pertps`; stratifies cells into responders and non-responders/escapers; optionally constructs supervised LDA projections.
9. **Stage 9/14: lochNESS Neighbourhood Topology**: Quantifies continuous manifold over-representation across $k=300$ nearest neighbours without relying on discrete cluster boundaries.
10. **Stage 10/14: Perturbation Distance & Permutation DistanceTest**: Computes multivariate distribution distances (Energy Distance / MMD) against unperturbed control with finite-permutation empirical p-values and BH-FDR.
11. **Stage 11/14: Perturbation Distance Space & Phenotype Modules**: Evaluates all-vs-all pairwise Energy Distance matrix, classical MDS / PCoA coordinates, nearest phenotypic neighbors, and Phenotype Modules.
12. **Stage 12/14: Master Perturbation Metadata & Integrated Visualizations**: Merges all target-level metrics into `tables/perturbation_meta.csv` and renders the Perturbation Atlas, PS $\times$ Distance map, and Module concordance figures.
13. **Stage 13/14: Output Writing & Lean H5AD**: Saves the integrated `.h5ad` containing cell-level observations and latent embeddings; writes structured tables (`tables/`), diagnostic figures (`figures/`), compute profile (`tables/compute_profile.csv`), and optional results archive (`.tar.gz`).
14. **Stage 14/14: HTML Report Generation & Manifest**: Renders a standalone, self-contained HTML report with interactive data tables, figure previews, and deliverables manifest.

---

## Input Compatibility

The pipeline supports diverse sequencing chemistries, alignment pipelines, and single-cell data formats through `perturbseq_pipeline.io`:

| Input Layout | Configuration Key | Description |
|---|---|---|
| **Combined 10x MTX** | `input.mode: mtx`<br>`input.mtx_dirs: {lane: path}` | Standard 10x Genomics CellRanger output where GEX and CRISPR guide capture features are stored in a single matrix and differentiated by `features.tsv.gz`. |
| **Separated GEX & Guide MTX** | `input.mode: mtx`<br>`input.mtx_dirs: {lane: path}`<br>`input.guide_mtx_dirs: {lane: path}` | STARsolo layout where gene expression and guide barcodes are quantified independently. The pipeline matches cell barcodes and fills missing guide entries with zero counts. |
| **Integrated H5AD** | `input.mode: h5ad`<br>`input.h5ad: path.h5ad` | AnnData object with guide features already in `var` (`feature_types`), in an existing layer/slot, or with precomputed guide calls. |
| **Companion Guide H5AD** | `input.mode: h5ad`<br>`input.h5ad: path.h5ad`<br>`input.guide_h5ad: guides.h5ad` | Separate expression and guide AnnData files aligned by cell barcode (`obs_names`). |
| **Precomputed Guide Labels** | `input.mode: h5ad`<br>`input.guide_obs_column: col_name` | Datasets (such as KOLF or Replogle) with precomputed guide assignments in `obs` (e.g. `gene_target` or `gene`). The pipeline parses targets directly without reconstructing guide count matrices. |
| **Barcode Guide Table** | `input.mode: h5ad`<br>`input.guide_table: table.txt` | Long-format `barcode -> guide -> count` lookup table (e.g., PS_python input format). |

---

## Guide Assignment and Classification

### Decision Gate
For count-based guide assignment, the pipeline computes top and second-highest guide UMIs per cell and evaluates:

$$\text{Assigned} = (\text{top} \ge \text{min\_umi}) \land (\text{top} > \text{dominance\_ratio} \times \text{second}) \land (\text{max\_second\_umi} < 0 \lor \text{second} \le \text{max\_second\_umi})$$

Cells are assigned to one of four mutually exclusive perturbation classes:
* **`targeting`**: Cell passed the assignment gate and carries a guide targeting a known gene.
* **`non-targeting`**: Cell passed the assignment gate and carries a guide matching non-targeting patterns (`guides.ntc_patterns`).
* **`ambiguous`**: Cell has guide counts but failed the dominance ratio or maximum runner-up UMI threshold.
* **`unassigned`**: Cell has zero detected guide counts.

### Scalable Implementation Details
* **Sparse CSR & Numba Search**: For large guide libraries, `_csr_top_two_numba` and `_csr_top_two_python` search non-zero entries in sparse CSR format without dense matrix allocations.
* **Categorical Unique Parsing**: When precomputed labels are provided in `obs` (e.g., KOLF), target regular expressions are evaluated once per unique category rather than millions of times across all rows.

---

## Guide-to-Target Mapping for Metadata-Rich CRISPR Libraries

### Problem & Motivation

Single-cell pooled CRISPR screens historically utilized guide identifiers that directly encoded the target gene symbol (e.g., `AFF4_P1P2_1` -> `AFF4` or `CD81.2` -> `CD81`). However, newer chemistries such as **10x Genomics Chromium Flex CRISPRi** decouple guide feature identifiers from target gene names.

In the 1.23-million cell K562 10x Flex CRISPRi dataset (`K562_1M_CRISPR_filtered.h5ad`):
* **Dataset dimensions**: 1,233,421 cells × 25,349 features (18,446 Gene Expression, 6,903 CRISPR Guide Capture).
* **Guide identifiers**: Encode TSS / transcript / genomic IDs, for example:
  ```text
  guide ID: TSS100020_17082653_23-ENST00000606659
  ```
* **True biological target**: Stored explicitly in feature metadata:
  ```text
  var["target_gene_name"] == "CNOT7"
  ```
* **Failure of historical guide-name parsing**: Empirical comparison across all 6,726 targeting guides in the library revealed **0.00% exact agreement** between delimiter-based guide ID parsing and true `target_gene_name`. Naive parsing produced bogus target labels such as `TSS100020`, `TSS100176`, and `TSS10038` instead of actual genes (`CNOT7`, `CDCA2`, `DDX51`, `DCLRE1C`).
* **Non-biological metadata annotations**: The library also annotates non-targeting controls as `target_gene_name = "Non-Targeting"` and non-targeting/unassigned controls as `target_gene_name = "Ignore"`.

All 6,903 guide features have populated `target_gene_name` and `target_gene_id` fields in `var`.

---

### Authoritative Metadata Resolution Architecture

To address metadata-rich CRISPR libraries while strictly preserving historical behavior, the pipeline introduces an optional, authoritative metadata mapping mechanism in `GuideConfig` and `resolve_guide_targets()`:

```text
       [Guide Count Matrix]
                │
                ▼ (top vs runner-up UMI)
        [Dominant Guide ID]
                │
                ├── target_feature_column is null ────────► Legacy Guide-ID Parsing (target_split_delims / target_regex)
                │
                └── target_feature_column is set
                            │
                            ▼
                    Read guides.var[col]
                            │
                            ├── In ignored_target_values? ──► unassigned_label ("unassigned") -> CLASS_UNASSIGNED
                            ├── Empty / NaN / Null? ────────► unassigned_label ("unassigned") -> CLASS_UNASSIGNED
                            ├── Matches ntc_patterns? ──────► ntc_label ("ntc") -> CLASS_NTC
                            └── Biological gene symbol ─────► Biological target -> CLASS_TARGETING
```

#### Key Components:

1. **`guides.target_feature_column`** (Default: `null`):
   * When set (e.g. `target_gene_name`), the specified column in `guides.var` is treated as the authoritative source of target gene names.
   * If the configured column is missing from `guides.var`, the pipeline immediately raises a clear `ValueError` rather than silently inferring incorrect targets from guide names.
   * When `null` (default), legacy delimiter and regex-based guide ID parsing remains 100% active.

2. **`guides.ignored_target_values`** (Default: `["Ignore"]`):
   * Values matching any entry in this list (case-insensitively, e.g. `"Ignore"`, `"ignore"`) are mapped directly to `guides.unassigned_label` (`"unassigned"`).
   * These guides are classified as `CLASS_UNASSIGNED` and are excluded from downstream biological target tests, co-functional modules, and perturbation hit calling.

3. **Non-Targeting Control Handling**:
   * Metadata values such as `"Non-Targeting"` or `"non-targeting"` are detected case-insensitively via the configurable `guides.ntc_patterns` (e.g. `r"^non[-_. ]?targeting"`).
   * They collapse to `guides.ntc_label` (`"ntc"` or `"non-targeting"`) with `CLASS_NTC`, establishing the unperturbed control baseline.

4. **Export Safeguards (`write_guide_table`)**:
   * `write_guide_table()` checks `guides.var["target_gene"]` first before falling back to `parse_target_genes()`.
   * This guarantees that exported long-format guide tables reproduce the authoritative metadata target names instead of re-parsing TSS IDs.

5. **Downstream Decoupling**:
   * Downstream modules (`qc`, `cluster`, `perturbation`, `enrichment`, `modules`, `ps_score`, `lochness`, `report`) interface strictly with `expr.obs["guide_id"]`, `expr.obs["target_gene"]`, and `expr.obs["perturbation_class"]`.
   * Guide concordance in enrichment uses `guide_id` for individual guide identity and `target_gene` for biological gene identity.

---

### Backward-Compatibility Guarantees

| Screen / Input Type | Configuration | Target Resolution Path | Backward Compatibility Status |
|---|---|---|---|
| **Replogle Screens** | `input.guide_obs_column: gene`<br>`guides.target_feature_column: null` | `_assign_from_labels()` (precomputed `obs['gene']`) | **100% Unchanged**. No guide count matrix or var column involved. |
| **KOLF Pan Genome** | `input.guide_obs_column: gene_target`<br>`guides.target_feature_column: null` | `_assign_from_labels()` (precomputed `obs['gene_target']`) | **100% Unchanged**. Categorical parsing of unique labels preserved. |
| **Conventional Guide Matrices** | `input.mode: h5ad` / `mtx`<br>`guides.target_feature_column: null` | `_assign_from_matrix()` + `parse_target_genes()` | **100% Unchanged**. Existing delimiter splitting and regex overrides active. |
| **10x Flex CRISPRi** | `input.mode: h5ad`<br>`guides.target_feature_column: target_gene_name` | `_assign_from_matrix()` + `resolve_guide_targets()` | **New Feature**. Authoritative `guides.var['target_gene_name']` mapping. |

---

### Interaction with LARGE Execution

* **One-Pass Guide Resolution**: Guide target resolution operates across unique guide features ($N_{\text{guides}} = 6,903$) rather than per single cell ($N_{\text{cells}} = 1,233,421$), adding zero measurable memory overhead.
* **Sparse Top-Two Guide Search**: Operates directly on CSR sparse guide matrices via `_csr_top_two_numba` / `_csr_top_two_python`.
* **Logging & Observability**: Clear log messages report target mapping mode:
  ```text
  Guide target mapping: using var['target_gene_name'] for 6903 guide features (177 ignored, 0 missing)
  ```
  or in legacy mode:
  ```text
  Guide target mapping: parsing target names from guide IDs
  ```

---

### Example configuration for a 10x Flex CRISPRi screen with guide metadata

```yaml
run:
  name: K562_1M_CRISPRi
  outdir: results/k562_1m_crispri

input:
  mode: h5ad
  h5ad: ../data/K562_1M_CRISPR_filtered.h5ad
  feature_type_column: feature_types
  gex_feature_type: "Gene Expression"
  guide_feature_types:
    - "CRISPR Guide Capture"
  counts_layer: counts

scaling:
  mode: auto
  large_n_cells: 1000000

guides:
  min_umi: 3
  dominance_ratio: 2.0
  max_second_umi: -1
  detection_threshold: 3
  target_feature_column: target_gene_name
  ignored_target_values:
    - Ignore
  target_regex: null
  target_split_delims: []
  ntc_label: ntc
  ntc_patterns:
    - "^non[-_. ]?targeting"
    - "^non$"
    - "^ntc"
    - scramble
    - "^safe[-_. ]?harbor"
  unassigned_label: unassigned
  ambiguous_label: ambiguous

ps_score:
  compute_lda_umap: false

modules:
  draw_networks: false

output:
  archive: false
  write_guide_table: false
```

---

### Validation & Testing

1. **Unit & Regression Test Suite (`tests/test_guide_metadata.py`)**:
   * **Test A & B**: Verified metadata target overrides guide parsing (`TSS100020_...` -> `CNOT7`, `TSS100176_...` -> `CDCA2`).
   * **Test C**: Verified `Non-Targeting` metadata maps to configured `ntc_label` and `CLASS_NTC`.
   * **Test D**: Verified `Ignore` metadata maps to `unassigned_label` and `CLASS_UNASSIGNED` (never a biological target).
   * **Test E**: Verified missing `target_feature_column` raises clear `ValueError`.
   * **Test F**: Verified legacy guide-matrix parsing remains identical when `target_feature_column: null`.
   * **Test G & H**: Verified Replogle and KOLF label paths are completely unaffected.
   * **Test I**: Verified YAML round-trip serialization and backward compatibility with old dictionaries.
   * **Test J**: Verified regression assertions (`"TSS100020" not in targets`, `"Ignore" not in targets`, `"CNOT7" in targets`).
2. **Real-Data Validation on `K562_1M_CRISPR_filtered.h5ad`**:
   * Evaluated on full 6,903-guide library and a 5,000-cell slice.
   * Confirmed 849 resolved biological targets including `CNOT7`, `CDCA2`, `DDX51`, and `DCLRE1C`.
   * Confirmed 0 TSS-prefixed labels and 0 `Ignore` entries in biological targets.
   * Confirmed 98 non-targeting cells correctly assigned `CLASS_NTC`.

---

## Perturbation-Strength Analysis

Perturbation-strength testing (`perturbseq_pipeline.perturbation`) verifies whether a guide knock-down successfully depletes the target gene's own mRNA transcript.

* **Target Expression**: Evaluated on log-normalized counts (`layers['lognorm']` or `X`).
* **Dual Control Groups**:
  * **`ntc` (Non-Targeting Controls)**: Cells carrying non-targeting guides (unperturbed baseline).
  * **`other` (Other-Targeting Controls)**: Cells carrying guides targeting other genes.
* **Statistical Tests**: Mann–Whitney U test and Kolmogorov–Smirnov test.
* **Directional Hit Calling**: A perturbation is called an effective hit (`is_hit_<control>`) if:
  1. $\text{BH-FDR} < \text{fdr\_alpha}$ (default: $0.05$)
  2. $\log_2\text{FC} < \text{max\_log2fc\_for\_hit}$ (default: $0.0$)
* **Threshold Guards**: Minimum cell count (`min_cells_per_target`, default: 10) and minimum percentage of control cells expressing the target (`min_pct_expressing_control`, default: 1.0%).

---

## Cluster-Level Perturbation Enrichment

Cluster enrichment (`perturbseq_pipeline.enrichment`) tests whether gene perturbations drive cells toward or away from specific transcriptional states.

* **Pairwise Association**: $2 \times 2$ contingency tables for each target-cluster pair evaluated with Fisher's exact test and Haldane–Anscombe corrected odds ratios.
* **Stratified CMH Test**: When `enrichment.stratify_by` (e.g., `lane_id` or `batch`) is configured, a Cochran–Mantel–Haenszel test controls for batch composition differences.
* **Guide Concordance**: Computes the percentage of independent guides for a target gene that agree in enrichment direction, filtering out single-guide artefacts.
* **Omnibus Permutation Test**: Global Monte Carlo permutation test evaluating overall perturbation-cluster dependency across the full screen.

---

## Co-functional Modules and Co-regulated Programs

The modules stage (`perturbseq_pipeline.modules`) discovers regulatory structure inspired by regulome mapping approaches (e.g., Chen et al., *Nature* 2023):

```text
                Perturbation × Gene Effect Matrix (Log2FC vs Control)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
Gene–Gene Correlation (Pearson)               Perturbation–Perturbation Correlation (Spearman)
            │                                               │
            ▼                                               ▼
  Co-regulated Programs (P1, P2, ...)           Co-functional Modules (M1, M2, ...)
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    ▼
                      Module × Program Strength Matrix
                      Transcription Factor (TF) Hubs
                      TF–TF Directed Regulatory Edges
                      Module–Module Connectivity Network
                      Per-Cell Program Activity Scores
```

### Methodological Details
* **Candidate Gene Selection**: Highly variable genes (HVGs), cluster marker genes, and perturbation marker genes.
* **Hierarchical Clustering**: Average linkage hierarchical clustering applied to Pearson distance for genes and Spearman distance for perturbations.
* **Cluster Nomenclature**: Program labels (`P1`, `P2`, ...) and module labels (`M1`, `M2`, ...) are numbered cluster identifiers. Member genes and TFs are output in tables for downstream biological annotation.
* **Underpowered Protection**: Automatically skips execution if the screen contains fewer than `min_perturbations` or `min_genes`.

---

## Per-Cell Perturbation Scores (PS Score)

The PS score stage (`perturbseq_pipeline.ps_score`) interfaces with the `pertps` / `PS_python` framework to quantify perturbation response at single-cell resolution:

* **Biomarker Construction**: Selects top up- and down-regulated biomarker genes for each target against control baselines.
* **Per-Cell Scoring**: Calculates continuous perturbation scores ($0$ to $1$) separating confirmed knockdowns from unperturbed cells or escapers.
* **Supervised LDA Projection**: Builds an LDA embedding trained on perturbation labels to maximize separation between perturbation states.
* **Computational Safeguards**:
  * Supervised LDA visualization can be disabled via `ps_score.compute_lda_umap: false` while retaining all PS scoring and statistics.
  * In large mode, LDA projection uses a bounded, stratified cell subset capped by `ps_score.lda_large_max_cells` (default: $150,000$).

---

## lochNESS: Neighbourhood Perturbation Enrichment

Ported from `pertTF`, lochNESS (`perturbseq_pipeline.lochness`) measures local neighbourhood enrichment on the phenotypic manifold:

$$\text{lochNESS}_{i, g} = \frac{\text{Local Fraction of Target } g \text{ in } k\text{-NN of Cell } i}{\text{Global Fraction of Target } g} - 1$$

* **Neighbourhood Size**: Evaluated across $k=300$ nearest neighbours in PCA space.
* **Interpretation**: A score of $0$ reflects background expectation; positive values indicate localized accumulation on the manifold.
* **Large-Screen Optimization**: In large mode, `lochness_self` (the score of each cell for its own assigned perturbation) is computed directly in bounded chunks, avoiding the memory cost of materializing full multi-gigabyte $N_{\text{cells}} \times N_{\text{targets}}$ matrices.

---

## Large-Dataset Execution Architecture

### Overview

The pipeline provides a centralized scaling architecture designed to handle single-cell screens spanning multiple orders of magnitude:

* **STANDARD mode**: Tailored for conventional single-cell Perturb-seq datasets such as Replogle Weissman 2022 K562 Essential (~310,000 cells, ~1,800 perturbations). Preserves historical Scanpy behavior, zero-centered dense scaling on HVGs, and standard in-memory structures.
* **LARGE mode**: Engineered for multi-million-cell or very high-perturbation datasets such as KOLF Pan Genome (~2.66 million cells, ~11,700 targets, 37.5k genes). Switches to sparse-preserving variance scaling, bounded reference sampling for marker/HVG discovery, chunked matrix operations, and lean in-memory table previews.
* **AUTO mode**: Automatically selects between `STANDARD` and `LARGE` based on dataset dimensions evaluated against centralized thresholds in `ScalingConfig`.

### Critical Invariants

1. **STANDARD preserves original pipeline behavior**: Historical defaults and analytical routines remain unchanged for typical screens.
2. **LARGE changes execution strategy, not biology**: LARGE mode alters *how* calculations are performed (chunking, sparsity preservation, bounded estimation), never the underlying biological definitions or statistical definitions.
3. **Explicit overrides**: Users can explicitly set `scaling.mode: standard` or `scaling.mode: large` to override automatic heuristics.
4. **AUTO is purely a mode-selection mechanism**: It inspects total cell count ($N_{\text{cells}}$) and perturbation count ($N_{\text{targets}}$) and delegates execution to either STANDARD or LARGE.

### Centralized Configuration Keys

All scaling controls reside under `scaling` in `Config` (with specialized large-data switches in stage sections):

| Configuration Key | Type | Default | Description |
|---|---|---|---|
| `scaling.mode` | `str` | `"auto"` | Execution mode: `"auto"`, `"standard"`, or `"large"`. |
| `scaling.large_n_cells` | `int` | `1000000` | Minimum cell count to trigger `LARGE` mode automatically. |
| `scaling.large_n_perturbations` | `int` | `5000` | Minimum perturbation count to trigger `LARGE` mode automatically. |
| `scaling.marker_max_cells` | `int` | `200000` | Maximum cells sampled for reproducible HVG and marker discovery in `LARGE` mode. |
| `scaling.effect_gene_chunk` | `int` | `256` | Gene chunk size for calculating perturbation × gene effect matrices. |
| `scaling.guide_chunk_size` | `int` | `20000` | Cell chunk size for dense guide assignment processing. |
| `scaling.guide_max_dense_elements` | `int` | `20000000` | Dense matrix ceiling ($N_{\text{cells}} \times N_{\text{guides}}$) before forcing sparse CSR search. |
| `scaling.collect_between_stages` | `bool` | `true` | Explicitly trigger `gc.collect()` between major pipeline stages in `LARGE` mode. |
| `scaling.log_memory` | `bool` | `true` | Log process RSS memory after each stage and during heavy operations. |
| `scaling.report_preview_rows` | `int` | `500` | Table row limit stored in memory for interactive HTML report preview tables. |
| `ps_score.lda_large_max_cells` | `int` | `150000` | Cell ceiling for supervised LDA embedding when LDA is enabled in `LARGE` mode. |
| `ps_score.lda_large_stratified` | `bool` | `true` | Use class-stratified subsampling when building the large LDA subset. |
| `lochness.target_chunk_size` | `int` | `256` | Number of perturbation targets evaluated per block during lochNESS calculation. |
| `lochness.max_targets_in_obs` | `int` | `50` | Maximum individual target score columns added directly to `obs` in `LARGE` mode. |

### Configuration Example

```yaml
scaling:
  mode: auto
  large_n_cells: 1000000
  large_n_perturbations: 5000
  marker_max_cells: 200000
  effect_gene_chunk: 256
  collect_between_stages: true
  log_memory: true
```

---

## Stage-by-Stage Large-Data Behavior

| Stage | STANDARD Mode Behavior | LARGE Mode Behavior |
|---|---|---|
| **QC** | Full sequential per-cell metric calculation and gene filtering passes. | Reuses existing QC metrics when present and valid in input `.h5ad` (e.g. pre-filtered KOLF); combines filtering masks in a single pass; avoids redundant AnnData cloning. |
| **Guide Assignment** | Vectorized chunked dense top-two search. | Precomputed labels in `obs` are parsed via unique categorical mapping; count matrices use bounded cell chunks; sparse CSR top-two search prevents huge dense matrix allocations. |
| **Normalization** | Normalizes and makes full duplicate copy of matrix into `layers['lognorm']`. | Raw counts remain strictly immutable in `layers['counts']`; normalized matrix is assigned directly to `layers['lognorm']` and referenced without redundant full-object duplication. |
| **HVG Selection** | Computes dispersion over all $N$ cells across all genes. | Estimates HVGs on a reproducible random sample of up to `scaling.marker_max_cells` (default: 200k cells). **All cells** are retained for PCA, nearest-neighbour graph, and clustering. |
| **PCA & Scaling** | Applies `sc.pp.scale(..., zero_center=True, max_value=10.0)` followed by standard PCA. | Applies sparse-safe variance scaling with `zero_center=False` and `max_value=None`, followed by `sc.tl.pca(..., zero_center=True, svd_solver="arpack")`. Centering is performed implicitly by ARPACK, keeping memory sparse throughout. |
| **Harmony** | Batch correction on dense cells × PCs representation. | Operates strictly on cells × PCs ($N \times 50$) rather than cells × genes; results are cast to `float32` immediately upon completion to halve persistent embedding memory. |
| **Perturbation Strength** | In-memory evaluation across all target genes. | Bounded gene-wise extraction directly from sparse matrices; never materializes full cells × targets dense matrices. |
| **Cluster Enrichment** | Vectorized contingency tables. | Fast contingency-table accumulation from categorical `obs` columns; avoids target × cell dense tables; preserves exact Fisher, CMH, odds ratio, and BH-FDR statistics. |
| **Modules & Programs** | Full-matrix slice accumulation for perturbation × gene effect matrix. | Effect matrix is accumulated in bounded gene chunks (`scaling.effect_gene_chunk: 256`); marker discovery uses bounded representative sampling; network layout plotting can be independently disabled (`draw_networks: false`). |
| **PS Scoring** | Scores all cells and computes full supervised LDA/UMAP. | Core PS statistics remain fully enabled; supervised LDA/UMAP can be disabled (`compute_lda_umap: false`) or bounded to `lda_large_max_cells` without altering PS hit calling. |
| **lochNESS** | Calculates full $N_{\text{cells}} \times N_{\text{targets}}$ score matrix. | Bypasses dense all-target matrices; calculates `lochness_self` and summary statistics in target chunks (`target_chunk_size: 256`); retains sparse $k$-NN graph. |
| **Plotting** | Renders full cell scatter plots. | Large-mode scatter plots sample cell backgrounds for rendering efficiency; statistical outputs and tables are never altered by plotting sampling. |
| **H5AD Output** | Standard full object serialization. | Direct sparse matrix writing; eliminates unnecessary intermediate object duplication during file writing. |

---

## Error and Resolution: KOLF Stage-4 OOM

### Incident and Symptoms

During initial production execution on the **KOLF Pan Genome** screen:
* **Dataset dimensions**: 2,659,209 cells × 37,567 genes across ~11,688 perturbation/control targets.
* **Execution progress**: Stages 1 through 3 completed successfully. The job reached **Stage 4/11: Normalization, embedding, and clustering**.
* **Log output**:
  ```text
  Creating PCA working matrix:
  2659209 cells x 3000 HVGs

  Scaling HVG working matrix with max_value=10.0
  ```
* **Scanpy warning emitted**:
  ```text
  UserWarning: zero-centering a sparse array/matrix densifies it.
  ```
* **Failure outcome**: Slurm terminated the job with an out-of-memory error:
  ```text
  Killed
  Detected 1 oom_kill event
  ```

### Root Cause Analysis

In standard Scanpy workflows, `sc.pp.scale(adata, zero_center=True)` subtracts each gene's mean from every element in the matrix. Because the mean of expressed single-cell genes is non-zero, every structural zero in the sparse matrix becomes a non-zero floating-point number.

For KOLF:
* $2,659,209 \text{ cells} \times 3,000 \text{ HVGs} = 7,977,627,000 \text{ elements}$ (~7.98 billion floats).
* In `float32`: $\approx 29.72 \text{ GiB}$.
* In `float64`: $\approx 59.44 \text{ GiB}$.

During scaling and subsequent truncated SVD / PCA, temporary copies, center vectors, and working arrays are allocated. Multiple simultaneous dense arrays rapidly consumed hundreds of gigabytes of RAM, triggering an OOM kill. Simply increasing node RAM was not an architecturally sound solution for a dataset that is fundamentally sparse.

### Architectural Resolution

The LARGE sparse clustering path in `cluster.py` was re-engineered:

1. **Variance Scaling without Densification**:
   ```python
   sc.pp.scale(
       pca_expr,
       zero_center=False,
       max_value=None,
   )
   ```
   Setting `zero_center=False` scales each gene column by its standard deviation without shifting zero entries. The matrix remains sparse.

2. **Implicit Mean Centering in PCA**:
   ```python
   sc.tl.pca(
       pca_expr,
       n_comps=n_pcs,
       zero_center=True,
       svd_solver="arpack",
       random_state=cfg.run.seed,
   )
   ```
   Scanpy's sparse ARPACK SVD solver performs mean centering implicitly during linear operator multiplications ($\mathbf{v} \mapsto (\mathbf{X} - \boldsymbol{\mu})\mathbf{v} = \mathbf{X}\mathbf{v} - \boldsymbol{\mu}(\mathbf{1}^T \mathbf{v})$) without ever allocating a dense centered matrix.

3. **Omission of `max_value` Clipping**:
   `scale_max_value` clipping is intentionally omitted in this sparse path because clipping uncentered values ($x / \sigma$) before mean subtraction is mathematically non-equivalent to clipping centered $z$-scores ($(x - \mu) / \sigma$) and would distort PCA coordinates. STANDARD mode retains historical clipping.

4. **Early Failure for `regress_out`**:
   `cluster.regress_out` in LARGE mode now raises a clear `ValueError` early rather than allowing Scanpy to silently densify multi-million-cell matrices during linear regression.

### How to Recognize the Fixed Path in Logs

A healthy execution on KOLF will display the following sequence in Stage 4 logs:

```text
Pipeline execution mode: LARGE
Large dataset detected: 2659209 cells x 37567 genes
Estimating 3000 HVGs using a reproducible subset of 200000 cells
Creating PCA working matrix: 2659209 cells x 3000 HVGs (dense equivalent 29.7 GiB float32 / 59.4 GiB float64)
PCA working matrix storage: sparse
LARGE sparse scaling: zero_center=False to preserve sparsity; gene means will be centered by PCA. cluster.scale_max_value=10.0 is intentionally not applied in this path because clipping before centering is not equivalent to clipping centered z-scores.
Running PCA: n_comps=50, solver=arpack, zero_center=True
PCA: 50 components on 3000 HVGs
```

> [!IMPORTANT]
> **Warning Guard**: In LARGE mode, you should **never** see `UserWarning: zero-centering a sparse array/matrix densifies it` during Stage 4 scaling. If this warning appears, stop the run immediately and verify that the active environment is importing the updated `perturbseq_pipeline` package.

---

## Validation and Regression Testing

The scaling architecture and fixes underwent multi-tier verification:

### 1. Test Suite
* **Result**: **120 tests passed / collected** (118 passed, 2 skipped due to optional external dependencies, 0 failed).
* **Test Duration**: ~6 minutes 42 seconds.
* **Regression Coverage**: `tests/test_scaling_consistency.py` explicitly tests sparse PCA scaling, early `regress_out` failure, guide assignment consistency, enrichment table matching, and lochNESS numerical identity.

### 2. Standard Mode Smoke Test
* **Dataset**: Synthetic Perturb-seq dataset (600 cells × 120 genes, 10 targets + NTC, 3 batches).
* **Mode**: `scaling.mode: standard`.
* **Outcome**: **PASS** — Complete 11-stage execution.

### 3. Forced Large Mode Smoke Test
* **Dataset**: Same 600 cells × 120 genes synthetic dataset.
* **Mode**: `scaling.mode: large`.
* **Outcome**: **PASS** — Verified that large-mode code paths execute correctly on small datasets without regressions.

### 4. 100k-Cell Large Stress Test
* **Dataset**: 100,000 cells × 2,000 genes across 500 perturbation targets.
* **Matrix Storage**: Realistic sparse Poisson counts ($<5\%$ density).
* **Outcome**: **PASS** — Completed all stages in **~15.9 minutes** with a peak process RSS of **~1.63 GB**.

### 5. Numerical Consistency Verification
Outputs between STANDARD and LARGE modes were compared programmatically:
* **Guide Assignment**: Identical cell-to-guide mappings, top UMIs, runner-up UMIs, and class labels.
* **Perturbation Strength**: Log2 fold changes, percent knockdown, Mann–Whitney U p-values, KS p-values, and BH-FDR matched to machine precision.
* **Cluster Enrichment**: 2×2 contingency counts, odds ratios, Fisher exact p-values, CMH p-values, and FDR matched.
* **Modules**: Perturbation × gene effect matrices matched within numerical tolerance ($< 10^{-6}$).
* **PS Score**: Biomarker gene sets, cell-level response scores, and responder classifications matched.
* **lochNESS**: `lochness_self` scores matched within floating-point tolerance.

---

## Known Scale-Dependent Considerations

While the pipeline is engineered for multi-million-cell screens, memory consumption is governed by irreducible data structures:

1. **Persistent Large In-Memory Structures**:
   * Full sparse count matrix (`layers['counts']`): typically 5–15 GB for 2.66M cells depending on non-zero sparsity.
   * Full sparse normalized matrix (`layers['lognorm']`): shared or referenced to minimize duplication.
   * PCA embedding matrix ($N_{\text{cells}} \times 50$ in `float32`): $\approx 0.53 \text{ GiB}$.
   * Harmony batch correction buffer: requires dense $N_{\text{cells}} \times 50$ `float64` input ($\approx 1.06 \text{ GiB}$).
   * Nearest-neighbour graph ($N_{\text{cells}} \times N_{\text{cells}}$ sparse adjacency with $k=15$ to $k=300$): multi-gigabyte sparse CSR representation.
2. **Memory Estimates for KOLF (2.66M cells × 37.5k genes)**:
   * Dense equivalent of 3,000 HVGs: 29.7 GiB (`float32`) / 59.4 GiB (`float64`) — *now bypassed via sparse scaling*.
   * lochNESS $k=300$ graph: several gigabytes in memory.
   * **Estimated Full-Run Peak RSS**: **$\approx 135\text{--}155 \text{ GB}$**.
   *(Note: Peak memory depends on matrix sparsity, graph connectivity, library versions, and node-level memory allocator behavior).*

---

## Recommended Production Settings for KOLF

For multi-million-cell screens like KOLF Pan Genome, use the following production configuration:

### Configuration (example)
```yaml
scaling:
  mode: auto                      # Automatically selects LARGE
  large_n_cells: 1000000
  large_n_perturbations: 5000
  marker_max_cells: 200000
  effect_gene_chunk: 256
  collect_between_stages: true
  log_memory: true

ps_score:
  enabled: true
  compute_lda_umap: false         # Disable supervised LDA visualization

modules:
  enabled: true
  draw_networks: false            # Disable expensive network layout generation

lochness:
  enabled: true
  target_chunk_size: 256

output:
  archive: false                  # Skip building multi-gigabyte .tar.gz during validation
```

### Slurm Resource Guidelines
* **CPUs**: 16–32 cores
* **Memory (RAM)**: 180–200 GB
* **Wall Time**: ~12 hours
* **GPU**: None required (all large-data routines run on CPU).

---

## Conda Environment and Diagnostics

The production pipeline should be executed within the dedicated `perturbseq-pipeline` conda environment:

```bash
conda activate perturbseq-pipeline
```

### Runtime Environment Diagnostics

To verify that the CLI and imported package resolve to the intended development repository:

```bash
which python
which perturbseq-pipeline

python - <<'PY'
import perturbseq_pipeline
print("Package location:", perturbseq_pipeline.__file__)
PY
```

Expected location:
`.../perturbseq-pipeline/src/perturbseq_pipeline/__init__.py`

---

## Pipeline Outputs

```text
results/<run_name>/
├── processed.h5ad                   # Final AnnData (expression counts, lognorm, obsm['guide_counts'])
├── report.html                      # Self-contained HTML report with embedded figures
├── tables/                          # Tabular CSV outputs
│   ├── qc_summary.csv
│   ├── clusters.csv
│   ├── perturbation.csv             # Formatted hit calls
│   ├── perturbation_full.csv        # Complete statistics for all targets
│   ├── enrichment.csv               # Fisher/CMH cluster enrichment
│   ├── cofunctional_modules.csv     # Module assignments
│   ├── gene_programs.csv            # Program assignments
│   ├── effect_matrix.csv            # Perturbation x gene log2FC matrix
│   ├── ps_score.csv                 # PS response summaries
│   ├── lochness.csv                 # lochNESS summaries
│   └── figure_manifest.csv          # Catalog of all generated plots
├── figures/                         # Diagnostic figures (PNG / PDF)
│   ├── qc/
│   ├── guides/
│   ├── clustering/
│   ├── perturbation/
│   │   └── per_gene/                # Individual volcano and expression plots
│   ├── enrichment/
│   ├── modules/
│   ├── ps_score/
│   └── lochness/
│       └── per_target/              # Individual neighbourhood maps
├── logs/
│   ├── run.log
│   └── resolved_config.yaml         # Fully resolved configuration for reproducibility
└── <run_name>_results.tar.gz        # Portable results archive (matrices excluded)
```

---

## Scientific Safeguards and Design Principles

1. **Explicit Ambiguity Representation**: Ambiguous and multiplet cells are explicitly labelled and tracked rather than silently discarded or forced into single-guide classes.
2. **Centralized Scaling Decisions**: Operational mode decisions reside in `Config`, preventing conflicting heuristics across modules.
3. **Preservation of Scientific Invariants**: Large-scale optimizations modify memory allocation, chunking, and computational representations while preserving statistical methods, test statistics, and significance thresholds.
4. **Reproducibility**: The exact parameters, thresholds, and runtime settings used for every run are dumped to `logs/resolved_config.yaml`.

---

## Development History & Changelog

### 2026-09-23: Unified branch (feature/unified-perturbseq-pipeline)
* **One implementation for every input and assignment mode**: MTX and H5AD input, single-guide, paired-guide (`guides.assignment_mode: dual_guide_pair`) and high-MOI membership (`guides.assignment_mode: high_moi`) assignment, the QC-only `samples` stage, and the optional stages (enrichment, modules, PS score, lochNESS, distance, distance space, meta table) selected through `<section>.enabled`. All optional stages read `obs['target_gene']` / `obs['perturbation_class']` from the guide-assignment stage.
* **Run manifest** (`run_manifest.py`): `logs/run_manifest.json` records pipeline version, git commit and dirty state, the execution command, the input manifest, assignment mode, enabled modules, random seed, execution and compute mode, and per-stage completion status (`completed` / `skipped` / `disabled`); `logs/module_status.json` is updated after every stage and `tables/module_status.csv` plus a *Run provenance* section are added to the HTML and Markdown reports.
* **Repository policy**: code-only branch based on `origin/main`; dataset configs, job scripts with site paths, results and reports are not tracked. Portable examples live in `config/examples/`; scheduler guidance in `docs/slurm.md`; pair-mode reference in `docs/paired_guide_assignment.md`.
* **Dependencies**: `pertpy` removed from the hard dependencies (never imported).
* **Defaults**: `modules`, `ps_score`, `lochness`, `distance` and `distance_space` are off in `config/default.yaml`; `enrichment` and `meta_analysis` stay on. `config/demo.yaml` enables PS score, lochNESS and modules explicitly so the demo output matches the original; `config/examples/all_modules.yaml` turns everything on.

### 2026-09-13: Basic QC stage (multi-well 10x h5, guide FASTQ counting, flag-only doublets)
* **New first-class stage** `run.stop_after: qc` (`src/perturbseq_pipeline/basic_qc.py`), driven from `cli.run_pipeline` before Stage 1. Loads per-GEM-well 10x `.h5`/MTX inputs described under a new top-level `samples:` block, computes per-sample expression QC, runs Scrublet per sample, quantifies guides, flags guide-derived multiplets, concatenates, writes QC-level objects and stops. **Nothing is normalised and no doublet or guide multiplet is removed**; `gex_qc_pass` depends only on expression flags.
* **Config** (`config.py`): `RunConfig.stop_after`, `Config.samples` + `SampleConfig`, `QCConfig.thresholds` (`QCThresholdConfig`: MAD/fixed, floors/ceilings, per-condition mt caps, per-sample overrides), `QCConfig.doublets` (`DoubletConfig`, deliberately without a `remove` switch), `GuideConfig.source/design/fastq/multiplet` (`GuideDesignConfig`, `GuideFastqConfig`, `GuideMultipletConfig`); `^no[-_. ]?target` added to the default `ntc_patterns`; `resolved_mode()` returns `"samples"`.
* **IO** (`io.py`): `read_10x_h5`, `read_10x_mtx_sample`, `read_10x_guide_features`, `strip_barcode_suffix`; integer CSR counts, `obs["cell_barcode"]`, globally unique `<sample>_<barcode>` obs names.
* **QC** (`qc.py`): `compute_basic_qc_metrics`, `resolve_sample_thresholds`, `flag_expression_qc`, `expression_qc_step_table`; gene classes annotated without rewriting `var_names`.
* **New modules**: `doublets.py` (Scrublet wrapper, annotation only), `guide_design.py` (workbook parser, all designed guides retained, configurable control recognition), `guide_counting.py` (streaming scaffold-aware UMI counter, parallel per FASTQ, MTX + stats output), `guide_qc.py` (guide attachment, per-scaffold summaries, `guide_multiplet_flag`, Scrublet-vs-guide cross-tab), `qc_plots.py`, `provenance.py`; `report.build_qc_report` + `templates/qc_report.html`.
* **Tests**: `tests/test_basic_qc.py` (22 tests) with `tests/make_synthetic_basic_qc.py` (two-well synthetic h5 + guide FASTQ + design workbook). Explicitly asserts `n_cells_before == n_cells_after` for doublet and guide-multiplet detection and that flagged cells persist in both combined objects.
* **Assessment aids**: `guides.multiplet.detection_min_fraction_of_top` (optional depth-aware detection rule, off by default), `tables/guide_detection_sensitivity.tsv` + figure over a threshold/fraction grid, and Scrublet `threshold_suspect` diagnostics (automatic threshold beyond the observed scores or calling < 0.5% of cells) with per-sample score quantiles in `doublet_summary.tsv`. Nothing is forced; flags stay flags.
* **Dependencies**: `h5py`, `openpyxl`, `scikit-image` added to `pyproject.toml` (all already present in the `perturbseq-pipeline` env).

### 2026-08-23: 320K 10x Flex CRISPRi configuration
* **320K Flex Configuration (`config/320CRISPRiflex.yaml`)**: Configured the 312,195-cell K562 10x Flex CRISPRi dataset (`../data/K562_320K_CRISPR_filtered.h5ad`), following the exact biological and input logic of the 1M Flex dataset rather than KOLF.
* **Metadata-Aware Guide Resolution**: Uses authoritative `guides.var["target_gene_name"]` (`target_feature_column: target_gene_name`) with non-targeting pattern matching (`"Non-Targeting"` -> `ntc`), `ignored_target_values: ["Ignore"]` (`"Ignore"` -> `unassigned`), and missing values -> `unassigned`. TSS transcript identifiers (e.g. `TSS100020_...`) are never treated as biological gene targets.
* **Automatic Scaling Behavior**: Retains `scaling.mode: auto` with thresholds `large_n_cells: 1000000` and `large_n_perturbations: 5000`. Because 320K Flex contains 312,195 cells and 849 candidate perturbations, AUTO scaling resolves to `STANDARD` execution mode (compared to 1M Flex with 1.23M cells resolving automatically to `LARGE` mode).
* **Dataset Independence & Compatibility**:
  - **320K Flex & 1M Flex**: Combined Gene Expression + CRISPR Guide Capture matrices split via `var["feature_types"]`, guide targets resolved via guide metadata `target_gene_name`.
  - **Replogle**: Legacy guide-ID parsing or `obs['gene']` remains unaffected.
  - **KOLF**: Independent precomputed-label path (`obs["gene_target"]`) and batch-stratified CMH analysis remain unaffected.
  - No existing dataset architecture was changed.
* **Real-Data Validation Results**:
  - Validated via `Config.from_yaml("config/320CRISPRiflex.yaml")` -> `CONFIG VALID`.
  - Verified 312,195 cells, 18,446 GEX features, 6,903 CRISPR Guide Capture features.
  - Resolved execution mode: `STANDARD`.
  - Tested on a real 5,000-cell subset: successfully separated GEX and guide matrices, correctly mapped 629 biological targets (including CNOT7), mapped 97 NTC cells to `ntc`, mapped 1,493 Ignore-dominant cells to `unassigned`, and confirmed 0 TSS identifiers in `target_gene`.
  - Passed full test suite (132/132 tests passing).

### 2026-08-23: Authoritative Metadata Guide Target Mapping for 10x Flex CRISPRi
* **Authoritative Guide-to-Target Metadata**: Added `target_feature_column` to `GuideConfig` in `config.py` and implemented `resolve_guide_targets()` in `guides.py` to support metadata-rich libraries (e.g. 10x Flex CRISPRi) where guide IDs encode TSS/transcript details rather than target genes.
* **Ignored & Non-Targeting Annotations**: Added `ignored_target_values` (default `["Ignore"]`) to safely map non-biological annotations to `unassigned` class; added case-insensitive non-targeting pattern support for `"Non-Targeting"`.
* **Export Decoupling**: Updated `write_guide_table()` in `io.py` to prioritize `guides.var["target_gene"]` over guide-ID reparsing.
* **Backward-Compatibility Guarantees**: Preserved 100% fidelity for Replogle `obs['gene']`, KOLF `obs['gene_target']`, and legacy guide-ID parsing when `target_feature_column: null`.
* **Configuration**: Created and validated `config/1MCRISPRiflex.yaml` for 1.23M-cell K562 10x Flex CRISPRi screen; updated `config/default.yaml`.
* **Testing & Real-Data Validation**: Added `tests/test_guide_metadata.py` (12 tests covering tests A–J); performed lightweight validation against real `K562_1M_CRISPR_filtered.h5ad` verifying 849 real gene targets, 0 TSS labels, and correct NTC mapping.

### 2026-08-23: Large-Data Execution Hardening & KOLF OOM Resolution
* **Centralized Scaling Architecture**: Added `ScalingConfig` in `config.py` supporting `STANDARD`, `LARGE`, and `AUTO` modes with unified thresholds across all pipeline stages.
* **KOLF Sparse PCA OOM Fix**: Resolved Stage 4 memory exhaustion on 2.66M-cell KOLF dataset by replacing densifying zero-centered scaling with sparse-safe variance scaling (`zero_center=False`) and implicit mean centering via ARPACK SVD.
* **Early Guard for Regress-Out**: Added early validation error for `cluster.regress_out` in large mode to prevent accidental dense array allocation.
* **Scalable Guide Assignment**: Implemented sparse CSR search (`_csr_top_two_numba` / `_csr_top_two_python`) and unique categorical string parsing for precomputed `obs` columns.
* **Chunked Modules & Regulome Discovery**: Implemented sufficient-statistics accumulation in gene blocks (`scaling.effect_gene_chunk: 256`) and bounded marker gene sampling.
* **Scalable Cluster Enrichment**: Implemented contingency-count-based table generation from categorical vectors without target × cell dense allocations.
* **Scalable lochNESS**: Implemented chunked target evaluation and direct `lochness_self` calculation to prevent $N_{\text{cells}} \times N_{\text{targets}}$ dense allocations.
* **Bounded PS Scoring**: Added `ps_score.compute_lda_umap` switch and stratified cell ceiling (`ps_score.lda_large_max_cells`) for large screens.
* **Bounded Visualization**: Added representative background sampling for large-mode scatter plots without affecting statistical summaries.
* **Validation & Regression Testing**: Implemented `tests/test_scaling_consistency.py`, verified 120-item test suite, validated STANDARD/LARGE smoke tests, and verified 100k-cell stress test (15.9 min, 1.63 GB RSS peak).
