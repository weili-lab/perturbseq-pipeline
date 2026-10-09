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

**Primary-label stages** (per-cell PS scores, lochNESS, the knockdown filter)
still evaluate each cell under its primary (highest-UMI) target; the report
warnings say so. `tables/guide_assignment.csv` therefore keeps both
`testable` (primary-label rule, what those stages use) and `testable_membership`.

Caveats: with ~9 co-carried guides per cell a strong perturbation leaks into
the `other` control of the targets it co-occurs with (diluted roughly by
1 / number of targets); a regression estimator that adjusts for co-carried
guides is the planned next step. Distance-space similarities are inflated by
shared cells and should be read with that in mind.

Consistency: on cells that carry exactly one target, the membership paths
reproduce the `single_guide` results to floating-point precision
(`tests/test_high_moi_membership_stats.py`), and the `single_guide` /
`dual_guide_pair` modes are unchanged (`tests/test_low_moi_golden.py`).
