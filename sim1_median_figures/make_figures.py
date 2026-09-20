#!/usr/bin/env python3
"""
========================================================================================
PAPER-STYLE FIGURE RENDERER  (Simulation 1 — median-based committee selection)
========================================================================================

Produces exactly the three figure types described in the paper's figure spec:

  Type 1  Panel-grid line chart (2 x 3)      -> change rate vs k, cutoff attack
                                             -> average overlap vs k, cutoff attack
                                             -> target success rate vs k, cutoff attack  (appendix)
  Type 2  Dual-metric overlay (2 x 3)        -> change rate (solid/filled) and
                                                success rate (dashed/open), same colour
  Type 3  Aggregate summary line chart (1x2) -> mean over the six models, one panel
                                                per attack

Conventions (from the spec):
  * Fixed panel order, 2 rows x 3 cols, row-major:
        Mallows | Plackett-Luce | 2D Euclidean
        IAC     | Polya Urn     | IC
  * Okabe-Ito colourblind-safe palette, one colour per rule, reused everywhere
  * Quadratic-spline smoothing through the actual simulated points, markers overlaid
  * Shared y-range within a figure, dotted grid (alpha 0.3), top/right spines removed,
    legend bottom-centre in one framed row, integer ticks only
  * Every figure saved as PNG (300 dpi) and PDF (vector)

Usage
-----
    python make_figures.py                 # main figures, k in {2,3,4}  (paper setting)
    python make_figures.py --k 1,2,3,4     # appendix variant using every simulated k
    python make_figures.py --k 1,2,3,4 --subdir appendix
========================================================================================
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Global style
# --------------------------------------------------------------------------------------
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

RULE_ORDER = ["Evaluative Voting", "k-Median Rule", "k-Median-Shapley"]
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


# --------------------------------------------------------------------------------------
# Quadratic-spline smoothing through the actual simulated points
# --------------------------------------------------------------------------------------
def _quad_bezier(A, C, B, t):
    t = t[:, None]
    return (1 - t) ** 2 * A + 2 * (1 - t) * t * C + t ** 2 * B


def smooth_curve(x, y, npts_per_seg=60):
    """Return a smooth quadratic interpolation that passes through every data point.

    With exactly three x-values a single quadratic (least-squares fit is exact)
    interpolates all points. Otherwise a piecewise quadratic Bezier chain through
    the segment midpoints is used, which is the standard C1 quadratic spline.
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

    pts = [_quad_bezier(P[0], P[0], mids[0], t)]                       # P0 -> m1
    for i in range(1, len(P) - 1):                                     # m_i -> m_{i+1}
        pts.append(_quad_bezier(mids[i - 1], P[i], mids[i], t))
    pts.append(_quad_bezier(mids[-1], P[-1], P[-1], t))                # m_last -> P_last

    curve = np.vstack(pts)
    return curve[:, 0], curve[:, 1]


# --------------------------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------------------------
def panel_grid(df, metric, target, k_values, ylabel, ylim, out_png, out_pdf,
               suptitle, dual_metric=False, metric2="Success_Rate", legend_anchor=None):
    """2 x 3 panel grid, one panel per preference model (row-major fixed order)."""
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), sharey=True)
    axes = axes.flatten()

    for idx, cult in enumerate(PANEL_ORDER):
        ax = axes[idx]
        sub = df[(df["Culture"] == cult) & (df["Target"] == target)]

        for rule in RULE_ORDER:
            d = sub[sub["Rule"] == rule].sort_values("k")
            x = d["k"].values

            y = d[metric].values                                   # primary metric
            xs, ys = smooth_curve(x, y)
            ax.plot(xs, ys, color=RULE_COLORS[rule], linewidth=MAIN_LW,
                    linestyle="-", zorder=3)
            ax.plot(x, y, linestyle="none", marker=RULE_MARKERS[rule],
                    markersize=MS, markerfacecolor=RULE_COLORS[rule],
                    markeredgecolor="white", markeredgewidth=1.2,
                    label=RULE_LABELS[rule] if not dual_metric else None, zorder=4)

            if dual_metric:                                        # same colour, second metric
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

    # one legend for the whole grid
    if dual_metric:
        handles = [
            plt.Line2D([], [], color=RULE_COLORS[r], marker=RULE_MARKERS[r], linestyle="-",
                       linewidth=MAIN_LW, markersize=MS, markerfacecolor=RULE_COLORS[r],
                       markeredgecolor="white", label=RULE_LABELS[r]) for r in RULE_ORDER]
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
                       markeredgecolor="white", label=RULE_LABELS[r]) for r in RULE_ORDER]
        ncol = 3

    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=True,
               bbox_to_anchor=legend_anchor or (0.5, 0.005))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}  /  {os.path.basename(out_pdf)}")


def aggregate_figure(df, k_values, ylabel, ylim, out_png, out_pdf, suptitle):
    """1 x 2 aggregate summary: mean over the six preference models, one panel per attack."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharey=True)

    for ax, target in zip(axes, ["Cutoff", "Bottom"]):
        sub = df[df["Target"] == target]
        for rule in RULE_ORDER:
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


# --------------------------------------------------------------------------------------
# LaTeX tables (booktabs + siunitx S columns) — the "exact numbers" companion
# --------------------------------------------------------------------------------------
def latex_table(df, metric, target, k_values, caption, label, out_tex):
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
        for i, rule in enumerate(RULE_ORDER):
            d = (sub[(sub["Culture"] == cult) & (sub["Rule"] == rule)]
                 .set_index("k").reindex(k_values))
            cells = " & ".join(f"{v:.2f}" for v in d[metric].values)
            if i == 0:
                lines.append(f"    \\multirow{{{len(RULE_ORDER)}}}{{*}}{{{LATEX_CULTURE[cult]}}} "
                             f"& {short_rule[rule]} & {cells} \\\\")
            else:
                lines.append(f"    & {short_rule[rule]} & {cells} \\\\")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"\end{table}", ""]
    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines))
    print(f"    [+] {os.path.basename(out_tex)}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("output_median_project", "Simulation 1",
                                                   "sim1_raw_results.csv"))
    ap.add_argument("--k", default="2,3,4", help="comma-separated committee sizes to plot")
    ap.add_argument("--subdir", default="", help="sub-directory under graphs/ for the output")
    args = ap.parse_args()

    k_values = [int(v) for v in args.k.split(",")]
    if not os.path.exists(args.data):
        raise SystemExit(f"data file not found: {args.data}\n"
                         f"run sim1_median_committee.py first")
    df = pd.read_csv(args.data)
    df = df[df["k"].isin(k_values)]

    base = os.path.dirname(os.path.abspath(args.data))
    graphs = os.path.join(base, "graphs", args.subdir) if args.subdir else os.path.join(base, "graphs")
    tables = os.path.join(base, "tables")
    os.makedirs(graphs, exist_ok=True)
    os.makedirs(tables, exist_ok=True)

    n_honest = int(df["n_honest"].iloc[0])
    n_manip = int(df["n_manip"].iloc[0])
    trials = int(df["trials"].iloc[0])
    context = (f"{n_honest} truthful + {n_manip} manipulators, "
               f"{trials:,} trials per cell")
    ks = ", ".join(str(k) for k in k_values)

    print(f"[*] Rendering figures for k in {{{ks}}}  ({context})")
    print(f"[*] Output: {graphs}\n")

    # ---- Type 1: panel-grid line charts ------------------------------------------------
    panel_grid(df, "Change_Rate", "Cutoff", k_values,
               "Committee change rate (%)", (0, 104),
               os.path.join(graphs, "sim1_cutoff_change_rate.png"),
               os.path.join(graphs, "sim1_cutoff_change_rate.pdf"),
               f"Cutoff attack: committee change rate vs $k$  ({context})")

    panel_grid(df, "Avg_Overlap", "Cutoff", k_values,
               "Average overlap with initial committee", (0, 1.02),
               os.path.join(graphs, "sim1_overlap_cutoff.png"),
               os.path.join(graphs, "sim1_overlap_cutoff.pdf"),
               f"Cutoff attack: average overlap vs $k$  ({context})")

    # ---- Type 2: dual-metric overlay (bottom attack) -----------------------------------
    panel_grid(df, "Change_Rate", "Bottom", k_values,
               "Rate (%)", (0, 104),
               os.path.join(graphs, "sim1_bottom_change_and_success.png"),
               os.path.join(graphs, "sim1_bottom_change_and_success.pdf"),
               f"Bottom attack: committee change rate and target success rate vs $k$  ({context})",
               dual_metric=True)

    # ---- Type 3: aggregate summary -----------------------------------------------------
    aggregate_figure(df, k_values, "Committee change rate (%)", (0, 104),
                     os.path.join(graphs, "sim1_aggregate_comparison.png"),
                     os.path.join(graphs, "sim1_aggregate_comparison.pdf"),
                     f"Mean over the six preference models ({context})")

    # ---- appendix extras ---------------------------------------------------------------
    appendix = os.path.join(graphs, "appendix")
    os.makedirs(appendix, exist_ok=True)
    panel_grid(df, "Success_Rate", "Cutoff", k_values,
               "Target success rate (%)", (0, 104),
               os.path.join(appendix, "sim1_cutoff_success_rate.png"),
               os.path.join(appendix, "sim1_cutoff_success_rate.pdf"),
               f"Cutoff attack: target success rate vs $k$  ({context})")
    panel_grid(df, "Change_Rate", "Bottom", k_values,
               "Committee change rate (%)", (0, 104),
               os.path.join(appendix, "sim1_bottom_change_rate.png"),
               os.path.join(appendix, "sim1_bottom_change_rate.pdf"),
               f"Bottom attack: committee change rate only vs $k$  ({context})")

    # ---- LaTeX tables ------------------------------------------------------------------
    latex_table(df, "Change_Rate", "Cutoff", k_values,
                f"Committee change rate (\\%) under the cutoff attack ({context}).",
                "tab:sim1-cutoff-change",
                os.path.join(tables, "sim1_table_cutoff_change_rate.tex"))
    latex_table(df, "Change_Rate", "Bottom", k_values,
                f"Committee change rate (\\%) under the bottom attack ({context}).",
                "tab:sim1-bottom-change",
                os.path.join(tables, "sim1_table_bottom_change_rate.tex"))
    latex_table(df, "Success_Rate", "Bottom", k_values,
                f"Target success rate (\\%) under the bottom attack ({context}).",
                "tab:sim1-bottom-success",
                os.path.join(tables, "sim1_table_bottom_success_rate.tex"))

    print("\n[*] Done. Both PNG (300 dpi) and PDF (vector) versions were written.")


if __name__ == "__main__":
    main()
