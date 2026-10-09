# High-MOI guide assignment

`guides.assignment_mode: high_moi` is for screens in which each cell carries
**several** guides by design (multiplicity of infection well above 1). The
default `single_guide` rule, which needs one guide to dominate the runner-up,
calls almost every cell *ambiguous* in such data; the high-MOI mode instead
records which guides each cell carries.

## Rule

For every cell, a guide is **called** (the cell is a *member* of that guide) when

* `method: threshold` (default): `umi >= min_umi` **and**
  `umi >= min_frac_of_top * top_umi` of the cell (the second condition removes
  the ambient background that deep guide libraries push over a small absolute
  threshold);
* `method: knee`: among guides with `>= min_umi` UMIs, the guides above the
  largest drop in `log1p(umi)` between consecutive ranked guides.

A cell is a member of every **target** its called guides map to (guide to
target mapping as in the other modes: `target_feature_column`, `target_regex`
or `target_split_delims`). Non-targeting guides collapse into one
`guides.ntc_label` column. Cells with more than `max_guides_per_cell` called
guides are classed `ambiguous` (doublet-like) and keep no membership.

```yaml
guides:
  assignment_mode: high_moi
  high_moi:
    method: threshold          # or knee
    min_umi: 10
    min_frac_of_top: 0.02
    max_guides_per_cell: 30
    min_guides_per_cell: 1
    membership_obsm_key: perturbation_membership
    guide_membership_obsm_key: guide_membership
    rank_profile_max_rank: 20
  multiplet:
    expected_guides_per_cell: null   # several guides per cell are expected; no multiplet flag
perturbation:
  primary_control: other     # NTC-only cells are rare in a high-MOI design
```

Choose `min_umi` from the **rank-ordered guide UMI profile** figure and table
(`figures/guides/high_moi_rank_umi_profile`, `tables/high_moi_rank_umi_profile.csv`):
real integrations sit on a plateau of tens to hundreds of UMIs, ambient guides
at a few UMIs, and the threshold should fall in the gap.

## What is written

| Where | Content |
|---|---|
| `obsm['perturbation_membership']` | cells x targets, sparse CSR int8; 1 = the cell carries at least one guide of the target; NTC column last |
| `obsm['guide_membership']` | cells x guides, sparse CSR int8; 1 = the guide is called in the cell |
| `uns['membership_targets']`, `uns['membership_guides']` | column names of the two matrices |
| `obs['n_guides_assigned']`, `obs['n_targets_assigned']` | row sums (targets exclude the NTC column) |
| `obs['n_guides_called']` | guides passing the call before the `max_guides_per_cell` gate (over-cap cells keep their real count here; their membership rows are cleared) |
| `obs['is_ntc_only']` | at least one NTC guide and no targeting guide |
| `obs['perturbation_class']` | `targeting` / `non-targeting` (NTC-only) / `ambiguous` (above the cap) / `unassigned` (no called guide) |
| `obs['target_gene']`, `obs['guide_id']` | **primary label**: the cell's highest-UMI targeting guide (NTC guide for NTC-only cells) |
| `obs['top_guide_count']`, `second_guide_count`, `total_guide_counts`, `n_guides_detected` | the same diagnostics the single-guide path writes |
| `tables/guide_assignment.csv` | per target: `n_cells` (membership), `n_cells_primary` (primary label), `testable` (enough primary-label cells: what the downstream stages test in this version), `testable_membership` (enough member cells) |
| `tables/guide_representation.csv` | per guide: cells carrying it (membership) and as primary label |
| `tables/high_moi_calling.csv`, `high_moi_rank_umi_profile.csv`, `high_moi_cells_per_target.csv` | calling summary (overall and per lane), knee profile, cells per target |
| `<run>_guide_barcodes.txt` | the long guide table gains an `is_member` column |

## What the downstream stages do

**Membership-aware stages** (perturbation strength, cluster enrichment,
co-functional modules, perturbation distance and distance space) read the
membership matrix:

* perturbed(*t*) = cells carrying at least one guide of *t*;
* `other` control = targeting cells that do **not** carry *t* (a cell carrying
  several targets is a control for every target it does not carry);
* `ntc` control = NTC-only cells (usually too few in a high-MOI design; the
  arm is skipped below `perturbation.min_control_cells` and the report says so);
* in STANDARD and LARGE execution the contingency counts are the same; LARGE
  builds them once with sparse products (`membership.T @ onehot`), with
  reference totals counted over *cells*, never summed over target rows.

**Negative-control pseudo-targets** (`guides.high_moi.ntc_pseudo_targets`,
default on): every non-targeting guide is tested exactly like a target
(perturbed = cells carrying that NTC guide, `other` = targeting cells not
carrying it) in the enrichment and distance stages. They have their own BH
family and never enter the real-target tables; `tables/enrichment_pseudo_targets.csv`
holds the tests and `tables/enrichment_pseudo_summary.csv` the empirical
false-positive rate next to the real targets' hit rate. Perturbation strength
has no pseudo-targets (an NTC guide has no own gene to test).

**Per-cell scores on membership.** PS scores (target-wise PS_python runs,
perturbed = cells carrying the target, reference = NTC-only cells) and lochNESS
(indicator = cells carrying the target, overall fraction = members / cells,
computed for chunks of targets with one sparse product each) are computed for
every (cell, carried target) pair **whose target is eligible for the stage** and
stored sparsely in `obsm['ps_score_membership']` / `obsm['lochness_membership']`
(column names in `uns['ps_score_membership_targets']` /
`uns['lochness_membership_targets']`). Eligibility follows the stage thresholds:
lochNESS needs `lochness.min_cells_per_target` member cells; PS needs
`ps_score.min_cells_per_target`, a measured and expressed target gene and a
successful PS_python fit. A membership column without a score column therefore
means the target was skipped by that stage (see the stage's skipped table), not
that the cell does not carry it. The matrices exist only when the stage is
enabled.
`obs['ps_score']`, `obs['ps_quadrant']` and `obs['lochness_self']` keep one
value per cell: the score for the cell's primary (highest-UMI) target. The
PS_python LDA/UMAP embedding is not computed in this mode (it needs one label
per cell).

**Knockdown filter on membership** (when enabled). Each target's member cells go
through the unchanged per-(target, context) machinery; the per-cell results are
stored for every (cell, carried target) pair in `obsm['kd_ratio_membership']` and
`obsm['kd_status_membership']` (status codes in `uns['kd_status_codes']`,
columns in `uns['kd_membership_targets']`), and `obs['kd_status']` /
`kd_ratio` / `kd_keep` keep the primary target's values.
`tables/guide_assignment.csv` keeps both `testable` (primary-label rule) and
`testable_membership`.

**Membership regression** (`regression.enabled`, off by default, high-MOI only;
`regression.py`). All targets are fitted jointly, one linear model per gene over
the assigned cells (targeting + NTC-only):
`lognorm ~ membership (all targets) + n_guides + log(total_counts) + lane`,
with a ridge penalty (`regression.ridge_alpha`) on the membership coefficients
only. Each effect is therefore adjusted for the targets co-carried in the same
cells, which the pseudobulk contrasts above are not. Each design (the observed
one and every permutation) is factorised once (targets x targets) and processed
in turn, so only one factor is held in memory; `Xᵀ Y` is accumulated over
`scaling.effect_gene_chunk` gene chunks from the sparse layer for each design.
`regression.n_permutations` permutations shuffle the membership rows (with
`n_guides`) across cells within `regression.batch_key`; they (1) calibrate each
target's t statistic by genomic control (divided by `sqrt(lambda)`,
`lambda = max(1, permuted median t² / expected median)`) before its t-test
p-value, BH across all (target, gene) pairs (`regression.fdr_scope: global`,
the default) or across genes within each target (`target`, the modules
convention), and (2) give the empirical FDR of the call set: the same calls made on the permuted data, mean count over the
observed count (`tables/regression_design.csv`: `empirical_fdr`; it treats every
target as null in the permutations, so it is conservative; it also reuses the
permutations that set lambda). `empirical_target_fdr` is the same for targets
with at least one DE gene. With many null targets, per-target BH lets roughly
`fdr_alpha` x (number of null targets) targets carry a spurious call: on the ESC
full-scale screen (2,084 targets, 889 genes) `fdr_scope: target` called 443
targets with an empirical target FDR of 0.32, `global` 413 targets with 0.016 —
hence the default. A purely empirical
p-value is not used because it cannot go below `1 / (1 + permutations x genes)`,
which would cap the q-value of a target with a single real hit at about
`1 / permutations`. Outputs: `tables/regression_effect_matrix.csv` (targets x
genes, `log2fc` = lognorm coefficient / ln 2, a log2 ratio of geometric means of
normalised counts + 1), `regression_fdr.csv`, `regression_de.csv` (significant
pairs), `regression_summary.csv` (per target: DE counts, own-gene effect,
`gc_lambda`). `regression.genes: modules` (default) fits the modules stage's
gene selection, so `modules.effect_source: regression` can build the modules
and programs from the adjusted matrix and its FDR instead of the pseudobulk
log2FC. `n_guides` is close to the membership row sum, so it is identified only
through cells whose guide count differs from their target count and through
the ridge penalty: a response shared by every target is attributed to guide
burden, not to the targets.

Caveats: with ~9 co-carried guides per cell a strong perturbation leaks into
the `other` control of the targets it co-occurs with (diluted roughly by
1 / number of targets); the membership regression above adjusts for it, the
pseudobulk stages do not. Distance-space similarities are inflated by shared
cells and should be read with that in mind.

Consistency: on cells that carry exactly one target, the membership paths
reproduce the `single_guide` results to floating-point precision
(`tests/test_high_moi_membership_stats.py`), and the `single_guide` /
`dual_guide_pair` modes are unchanged (`tests/test_low_moi_golden.py`).
