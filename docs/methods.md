# Methods

What the pipeline computes, why, and where it deliberately differs from the
prototype notebooks in `notebooks/prototype/`.

---

## 1. Input handling

### 10x MTX mode

Each lane is read with `sc.read_10x_mtx(..., gex_only=False)` so guide features
come along with gene expression. Lanes are concatenated with
`sc.concat(label='lane_id', keys=..., index_unique='-')`.

`sc.concat` keeps only `var` columns that are identical across objects, which
silently drops `feature_types` when lanes disagree. The prototype notebook
worked around this by reassigning `adata_full.var = adata_full1.var` after the
fact. Here the feature sets are **checked for identity first** — a lane
quantified against a different reference or guide library raises an error rather
than producing a matrix whose `var` annotation belongs to a different lane.

Features are then split into gene expression and guides. Guides are re-indexed
by their unique feature ID (`gene_ids`, e.g. `AFF4_P1P2_1`), because the symbol
column repeats across the six guides targeting one gene.

### h5ad mode

Three layouts are detected, in order:

1. **guide features in `var`** — split exactly as in MTX mode;
2. **companion guide `.h5ad`** — joined on the barcode intersection, with a
   warning if the two files do not fully overlap;
3. **pre-computed per-cell label** in `obs` (e.g. a Seurat `genotype` column) —
   parsed directly; guide-count diagnostics are then unavailable, which the
   report states.

Seurat exports commonly keep raw counts in `X` and log-normalized values in
`layers['logcounts']`. Point `input.counts_layer` / `input.normalized_layer` at
them and the pipeline will not re-normalize already-normalized data.

### Sample metadata

Required for any run spanning more than one lane. Joined on `lane_id`; every
column is merged into `obs` and written into the output `.h5ad`. Missing rows
for a present lane are an error, not a silent `NaN`. The requirement is waived
when the input already carries sample annotation (re-analyzing an `.h5ad` this
pipeline wrote).

---

## 2. Quality control

QC metrics are computed **before** the strict filters run, so the diagnostic
figures show the distribution the thresholds acted on rather than the already
truncated one.

| Stage | Default | Purpose |
|---|---|---|
| pre-filter | `min_genes_per_cell: 200`, `min_cells_per_gene: 3` | drop empty droplets and never-detected genes |
| metrics | `pct_counts_mt`, `pct_counts_ribo`, `pct_counts_hb` | via `sc.pp.calculate_qc_metrics` |
| strict filter | `min_genes_final: 1000`, `max_pct_mt: 20` | matches the prototype notebook |

Gene classes are matched case-insensitively, and a warning is emitted when the
mitochondrial prefix matches nothing (the usual cause is a mouse dataset with
human `MT-` settings, which would otherwise yield `pct_counts_mt = 0` and a
filter that silently does nothing).

Every filtering step records cells and genes before/after; that table appears in
the report.

### Perturb-seq-specific QC

* guide UMI depth per cell,
* guides detected per cell above `detection_threshold` → MOI estimate,
* top-vs-second guide counts, with the dominance threshold drawn on the plot,
* assignment outcome (targeting / non-targeting / ambiguous / unassigned),
  overall and **per lane**,
* guide and target representation, showing under-represented or absent guides.

The report also carries automatic warnings: low assignment rate, no
non-targeting controls detected, high ambiguous fraction, high multiplet rate.

---

## 3. Guide calling

For each cell, the highest and second-highest guide counts are found, then:

```
top == 0                                        -> unassigned
top >= min_umi  and  top > ratio x second       -> assigned to that guide
otherwise                                       -> ambiguous
```

Defaults: `min_umi = 3`, `dominance_ratio = 2.0`.

**Differences from the prototypes.** The notebooks looped over every cell in
Python (`for i in range(adata_guide.X.shape[0])`, printing progress every 10,000
cells); this is a chunked vectorized top-2 search that densifies at most 20,000
rows at a time. On the demo lane it assigns 27,541 cells in about one second.
The notebooks also used two different dominance ratios (1.2 in one, 2.0 in the
other) and no minimum UMI threshold; both are now single documented config keys.

Ambiguous and unassigned cells are **kept** in the object and counted in the
report. They are excluded from perturbation testing but never deleted, so the
assignment rate stays auditable.

### Target-gene parsing

Guide identifiers are split on `_`, `-` and `.`, taking the first field:

| Guide ID | Target |
|---|---|
| `AFF4_P1P2_1` (10x features file) | `AFF4` |
| `AFF4-P1P2.2` (Seurat `genotype`) | `AFF4` |
| `non_targeting_7` | `non` → non-targeting |

Set `guides.target_regex` when target names themselves contain a delimiter
(`NKX2-5`, for example).

Non-targeting guides are recognized by pattern (`non`, `non_targeting`, `NTC`,
`scramble`, `safe_harbor`) and collapsed to one `non-targeting` label.

---

## 4. Clustering

Library-size normalization → `log1p` → HVG selection (3,000 by default) →
scaling → PCA (50 PCs) → optional Harmony on `cluster.batch_key` → neighbor
graph → UMAP → Leiden (`flavor="igraph"`).

The layer contract afterwards:

| Slot | Contents |
|---|---|
| `layers['counts']` | raw integer counts |
| `layers['lognorm']` | log1p of normalized counts |
| `X` | the same log-normalized values |

Scaling is applied for PCA only; `X` is restored to log-normalized values
afterwards. **Perturbation tests read `layers['lognorm']` explicitly**, never
`X`, so a scaling step cannot silently change what is being tested — a real risk
in the notebook workflow, where `sc.pp.scale` and the KS test both operated on
whatever `X` happened to hold.

---

## 5. Perturbation strength

For each target gene *g* that is also measured in the expression matrix, the
expression of *g* itself is compared between:

* **perturbed cells** — assigned to a guide against *g*, and
* **control cells** — under two definitions, reported side by side.

### Control definitions

| Key | Cells | Trade-off |
|---|---|---|
| `ntc` | assigned to non-targeting guides | same transduction and selection, no on-target effect — the cleaner comparison |
| `other` | assigned to a *different* target gene | much larger n, but every control cell is itself perturbed, which dilutes apparent effects |

The prototype notebooks used `ambiguous` cells, or "all cells not labelled *g* or
`unassigned`", as controls. Since the library carries 30 dedicated
non-targeting guides, `ntc` is the default primary control here; `other`
reproduces the notebook comparison. Ambiguous and unassigned cells are never
used as controls. The report states which control drove the ranking, and a
scatter plot compares effect sizes under both — targets that disagree between
the two are exactly the ones worth a second look.

### Statistics

For each target/control pair:

| Quantity | Definition |
|---|---|
| `log2fc` | `log2(mean(expm1(perturbed)) / mean(expm1(control)))` — fold change on de-logged means |
| `pct_knockdown` | the same restated as a percentage drop |
| `ks_stat`, `ks_pval` | two-sample Kolmogorov–Smirnov (as in the notebooks) |
| `mwu_pval_less` | one-sided Mann–Whitney U testing *perturbed < control* |
| `ks_fdr`, `mwu_fdr` | Benjamini–Hochberg across all tested targets |

A target is called **effective** when `ks_fdr < 0.05` **and** `log2fc < 0`.

The direction requirement matters: a KS test is two-sided and will happily
report a significant difference for a target whose expression went *up*. The
notebooks reported the p-value alone, leaving direction to visual inspection of
the ECDF. Multiple-testing correction is likewise new — testing ~60 targets at
an uncorrected α = 0.05 expects around three false positives.

Targets that cannot be tested — not measured in the expression matrix, or fewer
than `min_cells_per_target` cells — are listed with the reason rather than
dropped.

---

## 6. Cluster enrichment

Section 5 asks whether a guide knocked its target down. This asks the follow-on
question: **did losing that gene push cells into a particular transcriptional
state?** For most screens this is the phenotype of interest.

### The test

For every (target, cluster) pair, a 2x2 table

```
              in cluster    elsewhere
    target        a             b
    reference     c             d
```

is tested with **Fisher's exact test** (two-sided), with BH-FDR across all pairs
within a control arm.

An exact test is used rather than chi-square residuals for a concrete reason: on
the demo lane **26% of the expected counts are below 5**, because the rare
clusters hold very few reference cells. Chi-square is therefore reported only as
an omnibus screen, alongside a **permutation p-value** that does not rely on the
approximation.

### Why `other` is the default reference here

The perturbation-strength test defaults to non-targeting controls. This one does
not, and the demo data shows why:

| Cluster | NTC cells | Other-target cells |
|---|---|---|
| 7 | 3 | 127 |
| 9 | 4 | 142 |

Clusters 7 and 9 are exactly where the strongest effects live. Against a 3-cell
denominator an odds ratio is nearly meaningless, and any moderate effect would
be invisible. Both arms are still computed and reported; `other` drives ranking
and hit calling.

### Guarding the numbers

* **Haldane-Anscombe correction** (+0.5 to each cell) keeps odds ratios finite
  when a count is zero.
* Pairs whose reference contributes fewer than `min_reference_cells` cells in a
  cluster are **flagged low-power** rather than silently trusted.
* Clusters smaller than `min_cells_per_cluster` are not tested at all.

### Guide-level concordance

Each target carries several guides, so a real phenotype should appear across
more than one of them; a single-guide artefact will not. For every significant
pair the pipeline reports how many of the target's guides independently show the
same effect.

Agreement is judged **against the observed direction** — a guide supports an
enrichment when its cells sit in the cluster more often than the reference, and
supports a depletion when they sit there less often. Testing only the enrichment
direction would report every genuine depletion as "0 guides agreeing", which
reads as the exact opposite of the truth.

### Multi-lane runs

Setting `enrichment.stratify_by` (e.g. to `lane_id`) replaces the pooled Fisher
test with **Cochran-Mantel-Haenszel** across strata, so a cluster that merely
differs in size between lanes cannot masquerade as a perturbation effect. Off by
default; the demo is a single lane.

### What it found on the demo lane

Omnibus chi-square = 9,218 on 590 df, permutation p < 0.001 — perturbation
identity and cluster are strongly associated. Twelve pairs reach FDR < 0.05,
and the pattern is biologically coherent:

* **SMARCC1** (core BAF/SWI-SNF subunit) takes over cluster 7 — 45% of its cells
  versus 0.2% of the reference — and is correspondingly depleted from three
  other clusters. All five testable guides agree.
* **EZH2 and SUZ12**, both core PRC2 subunits, are independently enriched in the
  same cluster 9, together with SALL4, NANOG and CTNNB1.

Two members of one complex landing on the same phenotype, from independent
guides, is the analysis validating itself — no prior knowledge of the complexes
enters the computation.

## 7. Figures and the report

Every figure passes through a registry that records its path, title, caption and
section, so a figure cannot be produced without being reachable from the report.

Per-target figures (ECDF + violin + UMAP highlight) are generated for **every**
tested target. The `perturbation.top_n_report` strongest are embedded inline;
the rest are written to `figures/perturbation/per_gene/` and linked from the
report by name. Nothing is silently truncated.

The report embeds figures as base64 data URIs, making it a single portable HTML
file that can be shared without an accompanying folder.
