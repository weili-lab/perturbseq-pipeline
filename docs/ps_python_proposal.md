# Proposed changes to PS_python

Findings from integrating [PS_python](https://github.com/weili-lab/PS_python)
into `perturbseq-pipeline`, and the changes they suggest — on both sides.
Everything below was verified against the shared S1lane1 guide count matrix and
the demo's `BARCODE_10x_Merged.txt`.

---

## 1. `BARCODE_10x_Merged.txt` is derivable from the guide count matrix

**It is, exactly.** The file is the guide count matrix in long form, keeping only
entries of **at least 3 UMIs**. Checked against `filtered_feature_bc_matrix_S1lane1`:

| Threshold | Per-cell total UMIs match | Guides-per-cell match | Cells with any guide |
|---|---|---|---|
| >= 1 | 0.07% | 0.07% | 33,881 |
| >= 2 | 24.56% | 24.56% | 32,113 |
| **>= 3** | **100.00%** | **100.00%** | **29,166** (file: 29,166) |
| >= 4 | 67.63% | 67.63% | 27,680 |

So the file needs no separate provenance — it can be regenerated from the matrix
at any time. `perturbseq-pipeline` now emits it directly
(`output.write_guide_table`), which removes it as a hand-maintained side file.

Two incidental observations: `read_count` equals `umi_count` in **100%** of rows,
so one of the columns carries no information; and `barcode` and `sgrna` are
identical throughout.

---

## 2. Bug: the `gene` column keeps a `.N` suffix for some guides

The `gene` column looks like `sgrna.split("_")[0]`. That works for guides whose
names contain an underscore, but not for the ones that do not, because those
carry R's `make.unique` suffix instead:

| sgrna | current `gene` | should be |
|---|---|---|
| `ARID1B_P1` | `ARID1B` | `ARID1B` ✅ |
| `TFCP2L1_P1P2.1` | `TFCP2L1` | `TFCP2L1` ✅ |
| `CD81.2` | **`CD81.2`** | `CD81` ❌ |
| `CD151.1` | **`CD151.1`** | `CD151` ❌ |
| `TFRC.1` | **`TFRC.1`** | `TFRC` ❌ |
| `CD55.1` | **`CD55.1`** | `CD55` ❌ |
| `NGFRAP1.1` | **`NGFRAP1.1`** | `NGFRAP1` ❌ |

The effect is that one target is split into several. `CD81` becomes `CD81.1`
(15,116 cells) and `CD81.2` (21,417 cells) — two "genes" that are really one, so
each is analysed with half its cells and against a control set that contains the
other half. The file reports **64 distinct genes** where the library targets ~57.

**Suggested fix** — split on any of `_`, `-`, `.`:

```python
import re
gene = re.split(r"[_\-.]", sgrna)[0]
```

with the existing special case mapping `non_targeting*` to `Non-Targeting`.
`perturbseq-pipeline` uses exactly this rule
(`guides.target_split_delims: ["_", "-", "."]`), which is why it reports `CD81`
as a single target.

---

## 3. Multiplets should follow the dominance rule, not row order

`PertPS.py` collapses the table with:

```python
barcode_map = bc_frame.set_index('cell')['gene'].to_dict()
adata.obs['gene'] = adata.obs_names.map(barcode_map).fillna('Other')
```

`set_index(...).to_dict()` keeps **whichever row for a barcode came last**. The
file averages 2.30 guides per cell and reaches 38 for some cells, so this
effectively assigns a guide at random for every multiplet, and the result depends
on row order rather than on the data.

**Suggested fix** — the same rule `perturbseq-pipeline` uses everywhere, so the
two codebases agree cell for cell:

```python
def assign_guides(bc, min_umi=3, dominance_ratio=2.0):
    """Assign each cell to its dominant guide, else 'Ambiguous'."""
    bc = bc.sort_values(["cell", "umi_count"], ascending=[True, False])
    top = bc.groupby("cell").head(1).set_index("cell")
    second = bc.groupby("cell").nth(1).reindex(top.index)
    second_umi = second["umi_count"].fillna(0)

    assigned = (top["umi_count"] >= min_umi) & (
        top["umi_count"] > dominance_ratio * second_umi
    )
    out = top["gene"].where(assigned, "Ambiguous")
    out[top["umi_count"] < min_umi] = "Unassigned"
    return out
```

A cell is assigned only when its top guide clears `min_umi` **and** beats the
runner-up by `dominance_ratio`; otherwise it is `Ambiguous`, and with no guide
above threshold it is `Unassigned`. Both categories should be excluded from the
target and control groups rather than silently folded into one of them.

On the demo lane this is not a small correction: 27,541 QC-passing cells split
into 70.9% assigned, 24.0% ambiguous and 0.3% unassigned. Taking the last row
would hand roughly a quarter of the cells an arbitrary label.

Until this lands, files written by `perturbseq-pipeline` sort each cell's rows so
the **highest-count guide comes last**, which makes the existing `to_dict()` call
pick the dominant guide rather than an arbitrary one. The written file also
carries an `assignment` column with the pipeline's own call, including the
`ambiguous` and `unassigned` labels.

---

## 4. The quadrant expression cut is degenerate for most genes

`PertPS.py` places the horizontal cut at the control population's **median**
expression:

```python
h_thresh = df_ctrl['Expression'].median()
```

Single-cell counts are zero-inflated, so for most genes that median is exactly
0 — on the demo lane, **35 of 46 scored targets**. The cut then collapses
"low expression" into "expression is exactly zero", and the plotted dashed line
sits on the axis.

Measured across all 46 targets (percentages are medians over targets):

| Cut | Cut value | Perturbed called KD | Controls called KD | Net | Degenerate (cut = 0) |
|---|---|---|---|---|---|
| median (current) | 0.00 | 27.55% | 2.88% | 23.73% | **35 / 46** |
| **mean** | 0.27 | 27.33% | 2.88% | **23.60%** | **0 / 46** |
| 75th percentile | 0.60 | 29.48% | 3.22% | 24.74% | 17 / 46 |

**Recommendation: use the control mean.** It changes the net signal by about
0.1 percentage points, so no conclusion moves, but it is never degenerate and
the quadrant plot becomes readable. The upper quantile gives a marginally larger
net signal yet is still degenerate for 17 targets and has no principled
justification for the choice of 0.75.

The wider point is visible in the same table: the **vertical (score) cut does
almost all the work** — only ~3% of control cells land in the knockdown quadrant
under any of the three cuts. Arguments about the horizontal cut matter far less
than they appear to.

`perturbseq-pipeline` now defaults to the mean, with
`ps_score.expression_cut: mean | median | quantile` if you want to compare.

---

## 5. Report the control baseline next to every knockdown percentage

Whatever cut is chosen, the same classification applied to **control** cells says
what the number means. "SALL4: 66% knocked down" is only interpretable next to
"controls: 3% knocked down".

`perturbseq-pipeline` now reports `pct_controls_called_kd` and `net_pct_kd` per
target for exactly this reason. Worth adding to the PS_python scatter plots as a
subtitle.

---

## 6. Smaller items

* `pertps/analyzer.py` uses a bare `except:` around `rank_genes_groups`, which
  will also swallow `KeyboardInterrupt`. `except Exception:` and logging the
  target that failed would make debugging much easier.
* `calculate_ps_score` normalizes each target's scores by their own maximum
  (`eff_scores /= np.max(eff_scores)`), so scores are **not comparable between
  targets** — a 0.8 for one gene does not mean the same as a 0.8 for another.
  Worth stating in the README, since the fixed 0.5 threshold is applied across
  all targets.
* `setup.py` does not list `umap-learn`, but `analyzer.py` imports `umap` at
  module scope, so `import pertps` fails on a clean environment. Adding
  `umap-learn` to `install_requires` would fix it.
* A gene that controls do not express cannot be shown to be knocked down: with a
  control median of 0 every cell is "low", so any high score reads as a success.
  Before guarding for this, the olfactory-receptor controls topped the knockdown
  ranking on the demo lane (OR2D3 at 51%, OR6A2 at 39%). `perturbseq-pipeline`
  now skips targets expressed in under 1% of control cells; PS_python would
  benefit from the same check.

---

## Changes already made on the `perturbseq-pipeline` side

* Emits the barcode table from the guide count matrix (`output.write_guide_table`),
  with correct target parsing and an `assignment` column.
* Reads the barcode-table layout as a fourth input mode (`input.guide_table`),
  applying the dominance rule to multiplets.
* Defaults the quadrant cut to the control mean, configurable.
* Reports the control baseline and net effect per target.
* Skips targets not detectably expressed in control cells.
