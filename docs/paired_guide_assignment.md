# Paired-guide assignment

`guides.assignment_mode` selects how a cell receives its perturbation label.

| Mode | Rule | Label written to `obs['target_gene']` |
|---|---|---|
| `single_guide` (default) | top guide reaches `min_umi`, beats the runner-up by `dominance_ratio`, runner-up within `max_second_umi` | the top guide's target |
| `dual_guide_pair` (alias `pair`) | the strongest guide of **each scaffold class** passes the same gate against the runner-up of its own class; the (A, C) pair is then interpreted | the pair's target |

Both modes write the same downstream contract: `obs['target_gene']`,
`obs['perturbation_class']` (`targeting` / `non-targeting` / `ambiguous` /
`unassigned`) and `obs['guide_id']` (`"<A guide>|<C guide>"` in pair mode).
Every later stage reads these columns and nothing else, so clustering,
perturbation strength, enrichment, PS score, lochNESS, distance, distance
space and modules behave identically under both modes.

## Configuration

```yaml
guides:
  assignment_mode: dual_guide_pair
  pair_assignment_primary: true        # must stay true in pair mode
  pair_reference: guide_reference.csv  # optional; one row per designed guide
  scaffold_column: auto                # scaffold | scaffold_class | scaffold_id
  pair_id_column: auto                 # pair_id | construct_id | vector_id
  pair_id_delimiter: ";"               # a guide may list several construct ids
  scaffold_classes: [A, C]
  require_complete_pair: true
  ntc_partner_policy: ambiguous        # or provisional_target
  designed_targeting_plus_ntc_primary: true
  single_guide_diagnostic: false       # also evaluate the single-guide rule, for comparison only
  min_umi: 3
  dominance_ratio: 2.0
  dominance_pseudocount: 1.0           # (top + pc) / (second + pc) >= dominance_ratio
```

The scaffold class of each guide comes from the pair reference, or from a
`scaffold` column of the guide features, and is stored in
`uns['guide_scaffolds']` of the processed `.h5ad`. When the reference carries
construct ids, a pair is valid only if both guides share a designed construct;
without construct ids the provisional same-target rule applies.

## Status categories

`obs['pair_assignment_status']` records why each cell got its label.

| Status | Meaning | `perturbation_class` |
|---|---|---|
| `pair_targeting` | both slots resolved, same designed target | `targeting` |
| `pair_non_targeting` | both slots resolved, both non-targeting | `non-targeting` (primary control) |
| `pair_targeting_plus_ntc` | designed targeting + NTC construct | `targeting` when `designed_targeting_plus_ntc_primary`, else `ambiguous` |
| `pair_targeting_plus_ntc_provisional` | targeting + NTC without a construct id, `ntc_partner_policy: provisional_target` | `targeting` (flagged provisional) |
| `dual_target_ambiguous` | two different targets | `ambiguous` |
| `incomplete_pair` | one slot resolved, the other empty | `ambiguous` (with `require_complete_pair: false`: assigned from the resolved slot, `construct_type = single_slot`) |
| `ambiguous_scaffold_A` / `_C` / `_A_and_C` | several strong guides in a class | `ambiguous` |
| `unknown_guide` | strong guides only among features without a scaffold class | `ambiguous` |
| `below_min_umi` | counts present, none reaching `min_umi` | `ambiguous` |
| `no_guide` | no guide UMIs | `unassigned` |
| `unresolved_pair` | combination not in the construct reference (reason in `pair_resolution_detail`) | `ambiguous` |

Per-slot detail lives in `guide_A_id`, `guide_A_count`, `guide_A_second_count`,
`guide_A_slot_status`, `guide_A_dominance_ratio` (and the `_C` counterparts).

## Outputs added in pair mode

| File | Contents |
|---|---|
| `tables/pair_assignment_summary.csv`, `pair_assignment_per_lane.csv` | cells per status, overall and per lane |
| `tables/cell_counts_before_after.csv` | expression-QC accounting per lane |
| `tables/pair_guide_qc_per_lane.csv` | per-scaffold UMI depth and strong-guide counts |
| `tables/pair_perturbation_by_target.csv`, `pair_perturbation_by_pair.csv` | target- and construct-level depletion tests with `fdr_ks`, `log2fc`, `neg_log10_fdr` |
| `tables/single_guide_diagnostic_vs_pair.csv` | cross-tab of the single-guide rule against the pair status (diagnostic only) |
| `figures/guides/pair_*`, `figures/perturbation/ecdf/` | status counts, per-target ECDFs |

Pair mode needs a guide count matrix. Inputs that carry only a pre-computed
per-cell label (`input.guide_obs_column`) cannot be pair-resolved and are
rejected at guide assignment.
