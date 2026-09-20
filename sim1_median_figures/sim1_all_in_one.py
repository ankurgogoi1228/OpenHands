#!/usr/bin/env python3
"""
========================================================================================
SIMULATION 1 (all-in-one): Median-Based Committee Selection
m = 5 candidates, k <= 4, total electorate = N_HONEST + N_MANIP
========================================================================================

This single file does everything:
    1. generates the simulation data (multiprocessing, one task per culture x k),
    2. writes the raw CSV and the summary tables,
    3. renders the paper-style figures (2x3 panel grids, dual-metric overlay,
       1x2 aggregate summary) as PNG (300 dpi) and PDF (vector),
    4. writes the LaTeX tables (booktabs + siunitx S columns).

Default configuration:
    N_HONEST = 80 truthful voters, N_MANIP = 20 manipulators  ->  total n = 100

Paper-mirror configuration (edit the three lines marked CONFIG below):
    N_HONEST = 99, N_MANIP = 2, NUM_ITERATIONS = 50000

Score discipline (STRICT):
  - Voter scores: integers in {0,1,2,3,4,5}.
  - Rank -> score map: (5, 4, 3, 2, 0).
  - 2D Euclidean: floor(x + 0.5), clipped to [0,5].
  - Manipulator ballots: 5 for target, 0 elsewhere (integers).
  - INTEGER_MEDIAN = True  ->  for even n, use the LOWER median (integer valued).
    This applies both to the honest profile (80 voters) and to the attacked
    profile (100 voters).

Command line
------------
    python sim1_median_committee.py                       # full run, k in {2,3,4}
    python sim1_median_committee.py --k 1,2,3,4           # main figures over every k
    python sim1_median_committee.py --trials 500          # quick test run
    python sim1_median_committee.py --skip-simulation     # re-plot an existing CSV
    python sim1_median_committee.py --skip-figures        # data only
========================================================================================
"""

import argparse
import itertools
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc="Processing", unit="task"):
        total = total or len(iterable)
        for i, item in enumerate(iterable):
            yield item
            pct = ((i + 1) / total) * 100
            print(f"\r{desc}: [{i+1}/{total}] ({pct:.1f}%)", end="", flush=True)
        print()

np.random.seed(42)


# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================
# ---------------- CONFIG -----------------------------------------------------------------
N_HONEST = 80              # truthful voters      (paper-mirror: 99)
N_MANIP = 20               # manipulators         (paper-mirror: 2)
NUM_ITERATIONS = 2000      # trials per cell      (paper-mirror: 50000)
# -----------------------------------------------------------------------------------------

M = 5                      # number of candidates
SCALE_MAX = 5              # integer scale {0,1,2,3,4,5}
TOTAL_N = N_HONEST + N_MANIP
COMMITTEE_SIZES = [1, 2, 3, 4]     # k <= m - 1

# Strict integer rank -> score map (contains grade 3; worst-ranked gets 0)
RANK_TO_SCORE = np.array([5, 4, 3, 2, 0], dtype=int)

# For even n use the LOWER of the two middle order statistics (always an integer).
INTEGER_MEDIAN = True

TARGETS = ['Cutoff', 'Bottom']
RULES = ['Evaluative Voting', 'k-Median Rule', 'k-Median-Shapley']


def smart_median(arr, axis=0):
    """Median that returns an integer for even-length inputs when INTEGER_MEDIAN=True."""
    if not INTEGER_MEDIAN:
        return np.median(arr, axis=axis)
    arr = np.asarray(arr)
    n = arr.shape[axis]
    if n % 2 == 1:
        return np.median(arr, axis=axis)
    sorted_arr = np.sort(arr, axis=axis)
    lower_idx = n // 2 - 1
    return np.take(sorted_arr, lower_idx, axis=axis).astype(float)


# --- Precompute Shapley combinatorial structures ---------------------------------------
subsets_by_len = [[] for _ in range(M + 1)]
for r in range(M + 1):
    for combo in itertools.combinations(range(M), r):
        subsets_by_len[r].append(combo)

all_subsets = [c for sub in subsets_by_len for c in sub]
subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
num_subsets = len(all_subsets)

subset_mask = np.zeros((num_subsets, M), dtype=float)
for i, T in enumerate(all_subsets):
    for c in T:
        subset_mask[i, c] = 1.0

fact = math.factorial
candidate_steps = []
for j in range(M):
    steps_j = []
    other = [c for c in range(M) if c != j]
    for r in range(M):
        w = (fact(r) * fact(M - r - 1)) / fact(M)
        for T in itertools.combinations(other, r):
            idx_T = subset_to_idx[T]
            idx_T_j = subset_to_idx[tuple(sorted(T + (j,)))]
            steps_j.append((w, idx_T, idx_T_j))
    candidate_steps.append(steps_j)


# ======================================================================================
# 2. VOTING RULES
# ======================================================================================
def utilitarian_rule(scores, k):
    total = np.sum(scores, axis=0)
    ranking = np.argsort(-total, kind='mergesort')
    return ranking, set(ranking[:k]), total


def median_rule(scores, k):
    med = smart_median(scores, axis=0)
    ranking = np.argsort(-med, kind='mergesort')
    return ranking, set(ranking[:k]), med


def shapley_median_rule(scores, k):
    voter_subset_sums = scores @ subset_mask.T
    delta = smart_median(voter_subset_sums, axis=0)
    delta[0] = 0.0

    phi = np.zeros(M)
    for j in range(M):
        for w, idx_T, idx_T_j in candidate_steps[j]:
            phi[j] += w * (delta[idx_T_j] - delta[idx_T])

    ranking = np.argsort(-phi, kind='mergesort')
    return ranking, set(ranking[:k]), phi


# ======================================================================================
# 3. PREFERENCE GENERATIVE MODELS (STRICT INTEGER SCORES)
# ======================================================================================
def generate_mallows(n, m=M, phi=0.7, scale_max=SCALE_MAX):
    """1. Mallows model (phi = 0.7): integer map (5,4,3,2,0) + integer noise in
    {-1,0,+1} with probabilities (0.1, 0.8, 0.1), clipped to [0,5]."""
    profile = np.zeros((n, m), dtype=int)
    for i in range(n):
        ranking = []
        for j in range(m):
            probs = np.array([phi ** (j - pos) for pos in range(j + 1)])
            probs /= probs.sum()
            pos = np.random.choice(j + 1, p=probs)
            ranking.insert(pos, j)
        for rank_pos, cand in enumerate(ranking):
            base = int(RANK_TO_SCORE[rank_pos])
            noise = int(np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]))
            profile[i, cand] = int(np.clip(base + noise, 0, scale_max))
    return profile


def generate_plackett_luce(n, m=M, scale_max=SCALE_MAX):
    """2. Plackett-Luce: probabilistic ranking from latent qualities gamma,
    integer map (5,4,3,2,0), no additive noise."""
    gamma = np.random.gamma(shape=2.5, scale=1.0, size=m)
    profile = np.zeros((n, m), dtype=int)
    for i in range(n):
        remaining = list(range(m))
        ranking = []
        for _ in range(m):
            probs = np.array([gamma[c] for c in remaining])
            probs /= probs.sum()
            chosen = np.random.choice(remaining, p=probs)
            ranking.append(chosen)
            remaining.remove(chosen)
        for rank_pos, cand in enumerate(ranking):
            profile[i, cand] = int(RANK_TO_SCORE[rank_pos])
    return profile


def generate_euclidean(n, m=M, dim=2, scale_max=SCALE_MAX):
    """3. 2D Euclidean: score = clip(floor(5*(1 - d/sqrt(dim)) + 0.5), 0, 5)."""
    candidates = np.random.uniform(0, 1, size=(m, dim))
    voters = np.random.uniform(0, 1, size=(n, dim))
    profile = np.zeros((n, m), dtype=int)
    d_max = np.sqrt(dim)
    for i in range(n):
        dists = np.linalg.norm(candidates - voters[i], axis=1)
        raw = scale_max * (1.0 - dists / d_max)
        scores = np.floor(raw + 0.5).astype(int)
        profile[i, :] = np.clip(scores, 0, scale_max)
    return profile


def generate_iac(n, m=M, scale_max=SCALE_MAX):
    """4. IAC (score-based): per candidate, a Dirichlet distribution over the six
    discrete scores, sampled independently for each voter."""
    profile = np.zeros((n, m), dtype=int)
    for j in range(m):
        probs = np.random.dirichlet(np.ones(scale_max + 1))
        profile[:, j] = np.random.choice(scale_max + 1, size=n, p=probs)
    return profile


def generate_urn(n, m=M, alpha=0.2, scale_max=SCALE_MAX):
    """5. Polya-Eggenberger urn (alpha = 0.2): 8 integer ballots in the urn,
    uniform draw per voter, copied back with probability alpha."""
    urn = [np.random.randint(0, scale_max + 1, size=m) for _ in range(8)]
    profile = np.zeros((n, m), dtype=int)
    for i in range(n):
        idx = np.random.randint(len(urn))
        ballot = urn[idx]
        profile[i, :] = ballot
        if np.random.rand() < alpha:
            urn.append(ballot)
    return profile


def generate_ic(n, m=M, scale_max=SCALE_MAX):
    """6. IC (score-based interpretation): every score drawn uniformly from {0,...,5}."""
    return np.random.randint(0, scale_max + 1, size=(n, m))


BEST_MODELS = {
    'Mallows (phi=0.7)': generate_mallows,
    'Plackett-Luce': generate_plackett_luce,
    '2D Euclidean': generate_euclidean,
    'Impartial Anonymous Culture (IAC)': generate_iac,
    'Polya Urn (alpha=0.2)': generate_urn,
    'Impartial Culture (IC)': generate_ic,
}


# ======================================================================================
# 4. MULTIPROCESSING WORKER
# ======================================================================================
def execute_task(args):
    cult_name, k, seed = args
    np.random.seed(seed)
    gen_fn = BEST_MODELS[cult_name]

    stats = {t: {r: {'changed': 0, 'overlap': 0.0, 'success': 0} for r in RULES} for t in TARGETS}

    for _ in range(NUM_ITERATIONS):
        scores_h = gen_fn(N_HONEST, M)

        r_uti, c_uti, _ = utilitarian_rule(scores_h, k)
        r_med, c_med, _ = median_rule(scores_h, k)
        r_shp, c_shp, _ = shapley_median_rule(scores_h, k)

        rule_pack = [
            ('Evaluative Voting', utilitarian_rule,    r_uti, c_uti),
            ('k-Median Rule',     median_rule,         r_med, c_med),
            ('k-Median-Shapley',  shapley_median_rule, r_shp, c_shp),
        ]

        for t_type in TARGETS:
            for r_name, r_fn, r_h, c_h in rule_pack:
                target_cand = r_h[k] if t_type == 'Cutoff' else r_h[M - 1]

                manip = np.zeros((N_MANIP, M), dtype=int)
                manip[:, target_cand] = SCALE_MAX
                scores_m = np.vstack([scores_h, manip])

                r_post, c_post, _ = r_fn(scores_m, k)

                if c_h != c_post:
                    stats[t_type][r_name]['changed'] += 1
                stats[t_type][r_name]['overlap'] += len(c_h & c_post) / k
                if target_cand in c_post:
                    stats[t_type][r_name]['success'] += 1

    results = []
    for t_type in TARGETS:
        for r_name in RULES:
            results.append({
                'Culture': cult_name,
                'k': k,
                'n_total': TOTAL_N,
                'n_honest': N_HONEST,
                'n_manip': N_MANIP,
                'trials': NUM_ITERATIONS,
                'Target': t_type,
                'Rule': r_name,
                'Change_Rate':  (stats[t_type][r_name]['changed'] / NUM_ITERATIONS) * 100.0,
                'Avg_Overlap':  stats[t_type][r_name]['overlap'] / NUM_ITERATIONS,
                'Success_Rate': (stats[t_type][r_name]['success'] / NUM_ITERATIONS) * 100.0,
            })
    return results


# ======================================================================================
# 5. SANITY CHECKS
# ======================================================================================
def sanity_check_profiles():
    print("\n[*] Sanity check: integer scores only, admissible range, grade coverage...")
    all_ok = True
    for name, fn in BEST_MODELS.items():
        profile = fn(1000, M)
        if profile.dtype.kind not in 'iu':
            print(f"    [FAIL] {name}: non-integer dtype = {profile.dtype}")
            all_ok = False
            continue
        if profile.min() < 0 or profile.max() > SCALE_MAX:
            print(f"    [FAIL] {name}: scores outside [0,{SCALE_MAX}]")
            all_ok = False
            continue
        grades = sorted(np.unique(profile).tolist())
        print(f"    {name:38s} dtype={profile.dtype}  "
              f"min={profile.min()}  max={profile.max()}  grades={grades}")
    if all_ok:
        print("    [OK] All models produce integer scores in {0,...,5}.")
    print()
    return all_ok


def sanity_check_median():
    print("[*] Sanity check: aggregated median under even n ...")
    probe = np.array([[1, 4, 2, 5, 3],
                      [2, 3, 4, 1, 0]])
    if INTEGER_MEDIAN:
        med = smart_median(probe, axis=0)
        assert np.array_equal(med, [1, 3, 2, 1, 0]), med
        print(f"    [OK] even n = 2, lower-median per column = {med.tolist()} (all integers)")
    else:
        med = smart_median(probe, axis=0)
        print(f"    [note] INTEGER_MEDIAN=False -> np.median per column = {med.tolist()}")
    print()


# ======================================================================================
# 6. FIGURE STYLE AND HELPERS (paper conventions)
# ======================================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
})

RULE_COLORS = {
    "Evaluative Voting": "#D55E00",   # Okabe-Ito vermillion
    "k-Median Rule":     "#0072B2",   # Okabe-Ito blue
    "k-Median-Shapley":  "#009E73",   # Okabe-Ito green
}
RULE_MARKERS = {"Evaluative Voting": "o", "k-Median Rule": "s", "k-Median-Shapley": "^"}
RULE_LABELS = {
    "Evaluative Voting": "Evaluative voting",
    "k-Median Rule":     "k-Median rule",
    "k-Median-Shapley":  "k-Median-Shapley",
}

# panel order: 2 rows x 3 cols, row-major — identical in every figure
PANEL_ORDER = [
    "Mallows (phi=0.7)",
    "Plackett-Luce",
    "2D Euclidean",
    "Impartial Anonymous Culture (IAC)",
    "Polya Urn (alpha=0.2)",
    "Impartial Culture (IC)",
]
PANEL_TITLES = {
    "Mallows (phi=0.7)": "Mallows ($\\varphi = 0.7$)",
    "Plackett-Luce": "Plackett–Luce",
    "2D Euclidean": "2D Euclidean",
    "Impartial Anonymous Culture (IAC)": "IAC",
    "Polya Urn (alpha=0.2)": "Pólya Urn ($\\alpha = 0.2$)",
    "Impartial Culture (IC)": "Impartial Culture",
}
LATEX_CULTURE = {
    "Mallows (phi=0.7)": r"Mallows ($\varphi=0.7$)",
    "Plackett-Luce": "Plackett--Luce",
    "2D Euclidean": "2D Euclidean",
    "Impartial Anonymous Culture (IAC)": "IAC",
    "Polya Urn (alpha=0.2)": r"P\'olya Urn ($\alpha=0.2$)",
    "Impartial Culture (IC)": "Impartial Culture",
}

MAIN_LW, OVERLAY_LW = 2.6, 1.9
MS, MS_OPEN = 9.5, 9.5


def _quad_bezier(A, C, B, t):
    t = t[:, None]
    return (1 - t) ** 2 * A + 2 * (1 - t) * t * C + t ** 2 * B


def smooth_curve(x, y, npts_per_seg=60):
    """Smooth quadratic interpolation that passes through every data point.

    With exactly three x-values a single quadratic (the least-squares fit is exact)
    interpolates all points. Otherwise a piecewise quadratic Bezier chain through the
    segment midpoints is used (the standard C1 quadratic spline).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    P = np.column_stack([x, y])

    if len(x) == 3:                                   # exact single quadratic
        coeffs = np.polyfit(x, y, 2)
        xs = np.linspace(x[0], x[-1], 3 * npts_per_seg)
        return xs, np.polyval(coeffs, xs)

    if len(x) == 2:                                   # straight line
        return x, y

    t = np.linspace(0, 1, npts_per_seg)
    mids = [(P[i - 1] + P[i]) / 2.0 for i in range(1, len(P))]

    pts = [_quad_bezier(P[0], P[0], mids[0], t)]
    for i in range(1, len(P) - 1):
        pts.append(_quad_bezier(mids[i - 1], P[i], mids[i], t))
    pts.append(_quad_bezier(mids[-1], P[-1], P[-1], t))

    curve = np.vstack(pts)
    return curve[:, 0], curve[:, 1]


def panel_grid(df, metric, target, k_values, ylabel, ylim, out_png, out_pdf,
               suptitle, dual_metric=False, metric2="Success_Rate"):
    """2 x 3 panel grid, one panel per preference model (row-major fixed order)."""
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), sharey=True)
    axes = axes.flatten()

    for idx, cult in enumerate(PANEL_ORDER):
        ax = axes[idx]
        sub = df[(df["Culture"] == cult) & (df["Target"] == target)]

        for rule in RULES:
            d = sub[sub["Rule"] == rule].sort_values("k")
            x = d["k"].values

            y = d[metric].values
            xs, ys = smooth_curve(x, y)
            ax.plot(xs, ys, color=RULE_COLORS[rule], linewidth=MAIN_LW,
                    linestyle="-", zorder=3)
            ax.plot(x, y, linestyle="none", marker=RULE_MARKERS[rule],
                    markersize=MS, markerfacecolor=RULE_COLORS[rule],
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=RULE_LABELS[rule] if not dual_metric else None, zorder=4)

            if dual_metric:                                  # same colour, second metric
                y2 = d[metric2].values
                xs2, ys2 = smooth_curve(x, y2)
                ax.plot(xs2, ys2, color=RULE_COLORS[rule], linewidth=OVERLAY_LW,
                        linestyle="--", zorder=3)
                ax.plot(x, y2, linestyle="none", marker=RULE_MARKERS[rule],
                        markersize=MS_OPEN, markerfacecolor="white",
                        markeredgecolor=RULE_COLORS[rule], markeredgewidth=1.6, zorder=4)

        ax.set_title(PANEL_TITLES[cult], fontsize=11, fontweight="bold")
        ax.set_xlabel("Committee size $k$")
        if idx % 3 == 0:
            ax.set_ylabel(ylabel)
        ax.set_xticks(k_values)
        ax.set_ylim(*ylim)
        ax.set_xlim(min(k_values) - 0.25, max(k_values) + 0.25)

    if dual_metric:
        handles = [
            plt.Line2D([], [], color=RULE_COLORS[r], marker=RULE_MARKERS[r], linestyle="-",
                       linewidth=MAIN_LW, markersize=MS, markerfacecolor=RULE_COLORS[r],
                       markeredgecolor="white", label=RULE_LABELS[r]) for r in RULES]
        handles += [
            plt.Line2D([], [], color="0.25", marker="o", linestyle="--",
                       linewidth=OVERLAY_LW, markersize=MS_OPEN, markerfacecolor="white",
                       markeredgecolor="0.25", label="change rate (solid, filled)"),
            plt.Line2D([], [], color="0.25", marker="o", linestyle="--",
                       linewidth=OVERLAY_LW, markersize=MS_OPEN, markerfacecolor="white",
                       markeredgecolor="0.25", label="target success rate (dashed, open)"),
        ]
        ncol = 5
    else:
        handles = [
            plt.Line2D([], [], color=RULE_COLORS[r], marker=RULE_MARKERS[r], linestyle="-",
                       linewidth=MAIN_LW, markersize=MS, markerfacecolor=RULE_COLORS[r],
                       markeredgecolor="white", label=RULE_LABELS[r]) for r in RULES]
        ncol = 3

    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=True,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}  /  {os.path.basename(out_pdf)}")


def aggregate_figure(df, k_values, ylabel, ylim, out_png, out_pdf, suptitle):
    """1 x 2 aggregate summary: mean over the six preference models, one panel per attack."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)

    for ax, target in zip(axes, TARGETS):
        sub = df[df["Target"] == target]
        for rule in RULES:
            d = (sub[sub["Rule"] == rule].groupby("k")[["Change_Rate", "Success_Rate"]]
                 .mean().reindex(k_values))
            xs, ys = smooth_curve(d.index.values, d["Change_Rate"].values)
            ax.plot(xs, ys, color=RULE_COLORS[rule], linewidth=MAIN_LW, label=RULE_LABELS[rule])
            ax.plot(d.index.values, d["Change_Rate"].values, linestyle="none",
                    marker=RULE_MARKERS[rule], markersize=MS,
                    markerfacecolor=RULE_COLORS[rule], markeredgecolor="white",
                    markeredgewidth=1.2, zorder=4)
        ax.set_title(f"{target} attack", fontsize=12, fontweight="bold")
        ax.set_xlabel("Committee size $k$")
        ax.set_xticks(k_values)
        ax.set_ylim(*ylim)
        ax.set_xlim(min(k_values) - 0.25, max(k_values) + 0.25)
    axes[0].set_ylabel(ylabel)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.085, 1, 0.955))
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}  /  {os.path.basename(out_pdf)}")


def latex_table(df, metric, target, k_values, caption, label, out_tex):
    """booktabs table with siunitx S columns — the 'exact numbers' companion."""
    head = " & ".join([f"{{$k={k}$}}" for k in k_values])
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \begin{tabular}{@{}ll" + "S[table-format=3.2]" * len(k_values) + r"@{}}",
        r"    \toprule",
        r"    & & \multicolumn{" + str(len(k_values)) + r"}{c}{Committee size $k$} \\",
        r"    \cmidrule(lr){3-" + str(2 + len(k_values)) + r"}",
        r"    Culture & Rule & " + head + r" \\",
        r"    \midrule",
    ]
    sub = df[df["Target"] == target]
    short_rule = {"Evaluative Voting": "Evaluative",
                  "k-Median Rule": "$k$-Median",
                  "k-Median-Shapley": "$k$-Median-Shapley"}
    for cult in PANEL_ORDER:
        for i, rule in enumerate(RULES):
            d = (sub[(sub["Culture"] == cult) & (sub["Rule"] == rule)]
                 .set_index("k").reindex(k_values))
            cells = " & ".join(f"{v:.2f}" for v in d[metric].values)
            if i == 0:
                lines.append(f"    \\multirow{{{len(RULES)}}}{{*}}{{{LATEX_CULTURE[cult]}}} "
                             f"& {short_rule[rule]} & {cells} \\\\")
            else:
                lines.append(f"    & {short_rule[rule]} & {cells} \\\\")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"\end{table}", ""]
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines))
    print(f"    [+] {os.path.basename(out_tex)}")


# ======================================================================================
# 7. PIPELINE STEPS
# ======================================================================================
def setup_directories(target_windows_path=r"D:\PYTHON\Project - Median"):
    if platform.system() == "Windows":
        base_dir = os.path.join(target_windows_path, "Simulation 1")
    else:
        base_dir = os.path.abspath(os.path.join("./output_median_project", "Simulation 1"))

    graphs_dir = os.path.join(base_dir, "graphs")
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 80)
    print(f"[*] Base Output Directory:   {base_dir}")
    print(f"[*] Graphs Output Directory: {graphs_dir}")
    print(f"[*] Tables Output Directory: {tables_dir}")
    print("=" * 80)
    return base_dir, graphs_dir, tables_dir


def simulate(base_dir):
    """Run every culture x k cell and return the raw results DataFrame."""
    tasks = []
    seed = 2026
    for cult_name in BEST_MODELS:
        for k in COMMITTEE_SIZES:
            tasks.append((cult_name, k, seed))
            seed += 1

    num_workers = cpu_count()
    print(f"[*] {len(tasks)} batches across {num_workers} CPU cores...\n")
    t0 = time.time()

    raw_results = []
    with Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap(execute_task, tasks),
                        total=len(tasks),
                        desc="Simulating Elections", unit="batch"):
            raw_results.append(res)

    flat = [x for sub in raw_results for x in sub]
    df = pd.DataFrame(flat)
    df.to_csv(os.path.join(base_dir, "sim1_raw_results.csv"), index=False)
    print(f"\n[+] Raw results saved ({time.time()-t0:.2f}s).")
    return df


def export_tables(df, tables_dir):
    print("[*] Exporting summary tables...")
    for t_type in TARGETS:
        sub = df[df['Target'] == t_type]
        for metric, nd in [('Change_Rate', 2), ('Success_Rate', 2), ('Avg_Overlap', 3)]:
            sub.pivot_table(index=['Culture', 'Rule'], columns='k', values=metric).round(nd).to_csv(
                os.path.join(tables_dir, f"sim1_{metric.lower()}_{t_type.lower()}.csv"))

    agg = (df.groupby(['Target', 'Rule'])[['Change_Rate', 'Success_Rate', 'Avg_Overlap']]
             .mean().round(3))
    agg.to_csv(os.path.join(tables_dir, "sim1_aggregate_summary.csv"))
    print(agg)


def render_figures(df, base_dir, graphs_dir, tables_dir, main_k):
    """Main figures for `main_k` (paper setting), appendix figures for every simulated k."""
    n_honest = int(df["n_honest"].iloc[0])
    n_manip = int(df["n_manip"].iloc[0])
    trials = int(df["trials"].iloc[0])
    context = (f"{n_honest} truthful + {n_manip} manipulators, "
               f"{trials:,} trials per cell")

    main_df = df[df["k"].isin(main_k)]
    appendix_k = sorted(df["k"].unique().tolist())

    print(f"\n[*] Rendering main figures for k in {{{', '.join(map(str, main_k))}}}  ({context})")
    print(f"[*] Output: {graphs_dir}\n")

    panel_grid(main_df, "Change_Rate", "Cutoff", main_k,
               "Committee change rate (%)", (0, 104),
               os.path.join(graphs_dir, "sim1_cutoff_change_rate.png"),
               os.path.join(graphs_dir, "sim1_cutoff_change_rate.pdf"),
               f"Cutoff attack: committee change rate vs $k$  ({context})")

    panel_grid(main_df, "Avg_Overlap", "Cutoff", main_k,
               "Average overlap with initial committee", (0, 1.02),
               os.path.join(graphs_dir, "sim1_overlap_cutoff.png"),
               os.path.join(graphs_dir, "sim1_overlap_cutoff.pdf"),
               f"Cutoff attack: average overlap vs $k$  ({context})")

    panel_grid(main_df, "Change_Rate", "Bottom", main_k,
               "Rate (%)", (0, 104),
               os.path.join(graphs_dir, "sim1_bottom_change_and_success.png"),
               os.path.join(graphs_dir, "sim1_bottom_change_and_success.pdf"),
               f"Bottom attack: committee change rate and target success rate vs $k$  ({context})",
               dual_metric=True)

    aggregate_figure(main_df, main_k, "Committee change rate (%)", (0, 104),
                     os.path.join(graphs_dir, "sim1_aggregate_comparison.png"),
                     os.path.join(graphs_dir, "sim1_aggregate_comparison.pdf"),
                     f"Mean over the six preference models ({context})")

    appendix_dir = os.path.join(graphs_dir, "appendix")
    os.makedirs(appendix_dir, exist_ok=True)
    print(f"\n[*] Rendering appendix figures for k in {{{', '.join(map(str, appendix_k))}}}")
    panel_grid(df, "Success_Rate", "Cutoff", appendix_k,
               "Target success rate (%)", (0, 104),
               os.path.join(appendix_dir, "sim1_cutoff_success_rate.png"),
               os.path.join(appendix_dir, "sim1_cutoff_success_rate.pdf"),
               f"Cutoff attack: target success rate vs $k$  ({context})")
    panel_grid(df, "Change_Rate", "Bottom", appendix_k,
               "Committee change rate (%)", (0, 104),
               os.path.join(appendix_dir, "sim1_bottom_change_rate.png"),
               os.path.join(appendix_dir, "sim1_bottom_change_rate.pdf"),
               f"Bottom attack: committee change rate only vs $k$  ({context})")

    print("\n[*] Exporting LaTeX tables...")
    latex_table(df, "Change_Rate", "Cutoff", appendix_k,
                f"Committee change rate (\\%) under the cutoff attack ({context}).",
                "tab:sim1-cutoff-change",
                os.path.join(tables_dir, "sim1_table_cutoff_change_rate.tex"))
    latex_table(df, "Change_Rate", "Bottom", appendix_k,
                f"Committee change rate (\\%) under the bottom attack ({context}).",
                "tab:sim1-bottom-change",
                os.path.join(tables_dir, "sim1_table_bottom_change_rate.tex"))
    latex_table(df, "Success_Rate", "Bottom", appendix_k,
                f"Target success rate (\\%) under the bottom attack ({context}).",
                "tab:sim1-bottom-success",
                os.path.join(tables_dir, "sim1_table_bottom_success_rate.tex"))


# ======================================================================================
# 8. MAIN
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="Simulation 1: median-based committee selection "
                                             "(simulation + tables + figures in one file).")
    ap.add_argument("--k", default="2,3,4",
                    help="committee sizes for the MAIN figures (default 2,3,4; app 1,2,3,4)")
    ap.add_argument("--trials", type=int, default=None, help="override NUM_ITERATIONS")
    ap.add_argument("--honest", type=int, default=None, help="override N_HONEST")
    ap.add_argument("--manip", type=int, default=None, help="override N_MANIP")
    ap.add_argument("--skip-simulation", action="store_true",
                    help="reuse the existing sim1_raw_results.csv")
    ap.add_argument("--skip-figures", action="store_true", help="data and tables only")
    args = ap.parse_args()

    global N_HONEST, N_MANIP, NUM_ITERATIONS, TOTAL_N
    if args.honest is not None:
        N_HONEST = args.honest
    if args.manip is not None:
        N_MANIP = args.manip
    if args.trials is not None:
        NUM_ITERATIONS = args.trials
    TOTAL_N = N_HONEST + N_MANIP
    main_k = [int(v) for v in args.k.split(",")]

    print("=" * 80)
    print(f"SIMULATION 1 (all-in-one): m={M}, k<=4, total n={TOTAL_N} "
          f"({N_HONEST} truthful + {N_MANIP} manipulators)")
    print(f"Trials per cell = {NUM_ITERATIONS}  |  INTEGER_MEDIAN = {INTEGER_MEDIAN}")
    print(f"Rank -> Score map = {RANK_TO_SCORE.tolist()}")
    print("=" * 80)

    sanity_check_profiles()
    sanity_check_median()

    base_dir, graphs_dir, tables_dir = setup_directories(r"D:\PYTHON\Project - Median")

    csv_path = os.path.join(base_dir, "sim1_raw_results.csv")
    if args.skip_simulation:
        if not os.path.exists(csv_path):
            raise SystemExit(f"--skip-simulation given but {csv_path} does not exist")
        df = pd.read_csv(csv_path)
        print(f"[*] Reusing existing results: {csv_path}  ({len(df)} rows)")
    else:
        df = simulate(base_dir)

    export_tables(df, tables_dir)

    if not args.skip_figures:
        render_figures(df, base_dir, graphs_dir, tables_dir, main_k)
        print("\n[*] Done. PNG (300 dpi) and PDF (vector) versions were written.")
    else:
        print("\n[*] Done (figures skipped).")

    print("=" * 80)
    print("SIMULATION 1 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == '__main__':
    main()
