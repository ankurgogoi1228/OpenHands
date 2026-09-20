# Simulation 1 — paper-style figures

Everything needed to regenerate the article and appendix figures for the median-based
committee-selection simulation, using the figure vocabulary from the paper's spec
(panel grid, dual-metric overlay, aggregate summary) and nothing else.

## Files

| File | Purpose |
|---|---|
| `sim1_median_committee.py` | Simulation. Scores are integers in {0..5}, rank map (5,4,3,2,0), round-half-up Euclidean, 6 preference models, k ∈ {1,2,3,4}, 2,000 trials/cell. Data only — no figures. |
| `make_figures.py` | Renders the three paper figure types (PNG 300 dpi + PDF vector) and the three LaTeX tables. |
| `diag_identity.py` | Diagnostic for the change-rate vs success-rate identity (see "Finding" below). |

## Run order

```bash
python sim1_median_committee.py          # ~2 min on 2 cores, writes CSV + tables
python make_figures.py                   # main figures, k in {2,3,4}   (paper setting)
python make_figures.py --k 1,2,3,4 --subdir appendix   # appendix variant
```

Outputs land in `output_median_project/Simulation 1/{graphs,tables}`; nothing outside this folder.

## Figure set actually generated

| File | Type | Content |
|---|---|---|
| `sim1_cutoff_change_rate.{png,pdf}` | 2×3 panel grid | Cutoff attack, committee change rate vs k |
| `sim1_overlap_cutoff.{png,pdf}` | 2×3 panel grid | Cutoff attack, average overlap vs k (optional 4th figure) |
| `sim1_bottom_change_and_success.{png,pdf}` | 2×3 dual-metric overlay | Bottom attack, CR (solid/filled) and SR (dashed/open), same colour |
| `sim1_aggregate_comparison.{png,pdf}` | 1×2 aggregate | Mean over the six models, one panel per attack |
| `appendix/sim1_cutoff_success_rate.*`, `appendix/sim1_bottom_change_rate.*` | 2×3 panel grid | Single-metric appendix versions |
| `tables/sim1_table_{cutoff_change_rate,bottom_change_rate,bottom_success_rate}.tex` | booktabs + `S` columns | Exact numbers |

## Conventions implemented

* Panel order fixed and identical in every figure — row-major: Mallows, Plackett–Luce, 2D Euclidean / IAC, Pólya Urn, Impartial Culture. **All six models appear in every figure**, including the aggregate (as the 6-model mean).
* Okabe–Ito colours, one colour per rule, reused everywhere: evaluative `#D55E00` (circles), k-median `#0072B2` (squares), median-Shapley `#009E73` (triangles).
* Quadratic-spline smoothing through the actual points, markers overlaid so nothing is hidden. With three `k` values the smoothing is an exact quadratic interpolant; with four it is a C¹ piecewise-quadratic spline through the midpoints.
* Shared y-range within a figure, integer x ticks, dotted grid (α = 0.3), top/right spines removed, single framed legend bottom-centre, PNG 300 dpi + PDF vector.

## Recommendation

1. **Main text:** cutoff change rate (grid), bottom dual-metric overlay (grid), aggregate summary. These three carry the whole empirical message.
2. **Plot k ∈ {2,3,4}.** That matches the paper's own k range. The simulated k = 1 extends the range to the single-winner case and, in the bottom attack, it is the region where evaluative voting is *not* the most vulnerable rule (IAC 29.7 vs 42.9 median; Plackett–Luce 9.05 vs 32.35; Pólya Urn 25.05 vs 47.30). Including k = 1 is defensible and arguably interesting — but it changes the visual message from "medians dominate" to "medians dominate except intermediated cases at k = 1", so keep it in the appendix and label it as the k = 1 extension.
3. **Keep the aggregate figure change-rate only.** Do not put the overlap panel next to a rate panel: different units, and a shared axis would be meaningless. The overlap grid is already the dedicated place for overlap.
4. **Do not hide the identity failure** — it is a result, not an embarrassment. See below.

## Finding you should not lose: CR ≠ SR for k-median, bottom attack

For the cutoff attack, both k-median and evaluative voting satisfy CR ≡ SR exactly (max |CR − SR| = 0.0000), and median-Shapley diverges by at most 11.15 pp. That reproduces the paper's Remark and the overlay shows it cleanly.

Under the **bottom** attack, however:

| Rule | max CR − SR (bottom) |
|---|---|
| Evaluative voting | 0.00 pp |
| k-Median rule | **29.45 pp** |
| k-Median-Shapley | 39.65 pp |

The k-median rule's identity breaks down as soon as the committee is large enough that the target can enter *without* displacing the incumbent. For Pólya Urn, k = 2: CR = 65.15 %, SR = 36.30 %. In 100 % of the "changed but target failed" trials, the ordering *among the non-target candidates* had reshuffled — the manipulators' zeros demote the incumbent below a third candidate, so the committee changes and the target still loses.

This is not an artefact of the lower median: rerunning the same configurations with standard `np.median` gives 29.20 pp instead of 29.45 pp (Pólya Urn) and 16.80 instead of 17.30 (IAC). It is structural.

It persists at the paper's manipulation scale: with 99 truthful voters + 2 manipulators the gap is still 5.3–6.95 pp for Plackett–Luce, Pólya Urn and IAC at k = 2. So the correct claim is not "CR ≡ SR for median rules" but:

> CR ≡ SR holds under the cutoff attack and for evaluative voting generally; under the bottom attack it holds only for evaluative voting, while both median-based rules admit committee changes that leave the target outside the committee.

Suggested sentence for the caption of the overlay figure: *"Solid curves show committee change rate and dashed curves target success rate; the two coincide for evaluative voting in both attacks, while the median-based rules separate under the bottom attack."*

## Configuration notes

* Default run: **80 truthful + 20 manipulators = 100 voters**, lower median on both the 80-voter baseline and the 100-voter attacked profile, 2,000 trials/cell (Monte Carlo 95 % margin ≤ 2.2 pp).
* Paper-mirror run: set `N_HONEST = 99`, `N_MANIP = 2`, `NUM_ITERATIONS = 50000` at the top of `sim1_median_committee.py`. **Caveat:** 99 is odd but 99 + 2 = 101 is odd too — good — however the honest profile then uses an odd-length median while the attacked profile also does, so the two remain comparable. The 80/100 default is worth stating explicitly in the caption, since the baseline and the attacked profile have different sizes there.
* Targets are rule-specific (each rule's own honest ranking), and at k = 4 the cutoff target is the bottom candidate, so the k = 4 rows of cutoff and bottom coincide exactly.
