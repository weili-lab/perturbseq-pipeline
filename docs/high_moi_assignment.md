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
| `obs['is_ntc_only']` | at least one NTC guide and no targeting guide |
| `obs['perturbation_class']` | `targeting` / `non-targeting` (NTC-only) / `ambiguous` (above the cap) / `unassigned` (no called guide) |
| `obs['target_gene']`, `obs['guide_id']` | **primary label**: the cell's highest-UMI targeting guide (NTC guide for NTC-only cells) |
| `obs['top_guide_count']`, `second_guide_count`, `total_guide_counts`, `n_guides_detected` | the same diagnostics the single-guide path writes |
| `tables/guide_assignment.csv` | per target: `n_cells` (membership), `n_cells_primary` (primary label), `testable` (enough primary-label cells: what the downstream stages test in this version), `testable_membership` (enough member cells) |
| `tables/guide_representation.csv` | per guide: cells carrying it (membership) and as primary label |
| `tables/high_moi_calling.csv`, `high_moi_rank_umi_profile.csv`, `high_moi_cells_per_target.csv` | calling summary (overall and per lane), knee profile, cells per target |
| `<run>_guide_barcodes.txt` | the long guide table gains an `is_member` column |

## What the downstream stages do in this version

Every later stage (clustering, perturbation strength, enrichment, modules, PS
score, lochNESS, distance) reads `obs['target_gene']` / `obs['perturbation_class']`
and therefore tests each cell under its **primary target only**; the `other`
control is cells with a different primary target, which may still carry the
tested target as a secondary membership. The report says so in its warnings. Membership-aware statistics (perturbed = cells
carrying the target, control = assigned cells not carrying it, non-targeting
guides as negative-control pseudo-targets) are the next step and will read the
`obsm` matrices; the `single_guide` and `dual_guide_pair` modes are not affected
by this mode (guarded by `tests/test_low_moi_golden.py`).
