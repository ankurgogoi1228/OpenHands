#!/usr/bin/env python3
"""
========================================================================================
SIMULATION 1 (manipulation-intensity sweep): Median-Based Committee Selection
========================================================================================
DESIGN
    total electorate        n   = 100 voters  (FIXED for every run)
    manipulators            x   = 1% ... 49% of the electorate  -> 1 ... 49 voters
    honest voters               = 100 - x   (99 down to 51)
    candidates              m   = 5
    committee sizes         k   = 2, 3, 4      (k = 1 is a single winner, k = 5 takes all)
    attacks                     Cutoff (the candidate just outside the committee)
                                Bottom (the worst-ranked candidate)
    rules                       3  ->  Evaluative, k-Median, k-Median-Shapley
    trials                      NUM_ITERATIONS Monte Carlo repetitions per cell
    The manipulated electorate is always the full 100 voters: the x manipulators vote the
    extreme integer ballot (5 for their target, 0 for everyone else) and the remaining
    100 - x voters vote honestly. Only the composition changes as x grows, never the size.
COMMON RANDOM NUMBERS
    For every trial a pool of 99 honest voters is drawn once, and intensity x uses the
    first 100 - x voters of that pool. Every intensity therefore faces the same underlying
    electorate, so curves are paired across intensities and their differences are not
    diluted by independent resampling. Each marginal profile is still a valid sample from
    the culture: prefixes of an i.i.d. sample (and of the sequential Polya urn) are
    themselves valid samples.
SCORE DISCIPLINE (STRICT)
    - All voter scores are integers in {0,1,2,3,4,5}.
    - Rank -> score map (5, 4, 3, 2, 0).
    - 2D Euclidean: floor(x + 0.5), clipped to [0,5].
    - Manipulator ballots: 5 for the target, 0 elsewhere (integers).
    - INTEGER_MEDIAN = True: for even n the LOWER of the two middle order statistics is
      used, which is an integer. Applies to every median, including the Shapley values.
    - GRADE COVERAGE: the 5-slot map alone can never produce grade 1, so both rank-based
      models (Mallows, Plackett-Luce) add independent integer noise in {-1,0,+1} with
      probabilities (0.1, 0.8, 0.1), clipped to [0,5]. 2D Euclidean rescales by the largest
      realised distance so that grade 0 is attainable. The sanity check enforces that all
      six models use all six grades.
OUTPUT  ->  D:\PYTHON\Project - Median\Simulation 1-49%\   (or ./output_median_project/Simulation 1-49%)
    sim1_raw_results.csv                     full long-format results
    graphs/                                  ALL figures, one folder, PNG (300 dpi) + PDF
    tables/                                  pivot CSVs, tipping points, LaTeX tables
Command line
------------
    python sim1_intensity_sweep.py                     # full run, 1000 trials per cell
    python sim1_intensity_sweep.py --trials 100        # quick test run
    python sim1_intensity_sweep.py --skip-figures      # data and tables only
    python sim1_intensity_sweep.py --skip-simulation   # re-plot an existing CSV
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
from matplotlib.colors import LinearSegmentedColormap
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
N_VOTERS = 100             # FIXED total electorate
MANIP_PCT_MIN = 1          # 1%  of the electorate -> 1 manipulator
MANIP_PCT_MAX = 49         # 49% of the electorate -> 49 manipulators
NUM_ITERATIONS = 1000      # Monte Carlo trials per culture (each trial covers all x, k)
# -----------------------------------------------------------------------------------------
# Output folder created under the base path:  D:\PYTHON\Project - Median\Simulation 1-49%
OUTPUT_DIRNAME = "Simulation 1-49%"
M = 5                      # number of candidates
SCALE_MAX = 5              # integer score scale {0,1,2,3,4,5}
COMMITTEE_SIZES = [2, 3, 4]        # k = 1 is a single winner, k = m takes everything
# x is a percentage of a 100-voter electorate, so x% == x voters
MANIP_COUNTS = list(range(MANIP_PCT_MIN, MANIP_PCT_MAX + 1))
MAX_HONEST = N_VOTERS - MANIP_PCT_MIN          # largest honest profile needed (99)
RANK_TO_SCORE = np.array([5, 4, 3, 2, 0], dtype=int)
INTEGER_MEDIAN = True
PL_RANK_TO_SCORE = RANK_TO_SCORE               # one shared map for both rank-based models
RANK_MODEL_NOISE = (0.1, 0.8, 0.1)             # integer noise, needed to reach grade 1
EUCLIDEAN_SCALE = "realized-max"               # "sqrt2" for the paper's exact formula
REQUIRED_GRADES = set(range(SCALE_MAX + 1))
TARGETS = ['Cutoff', 'Bottom']
RULES = ['Evaluative Voting', 'k-Median Rule', 'k-Median-Shapley']


def smart_median(arr, axis=0):
    """Median returning an integer for even-length input when INTEGER_MEDIAN=True."""
    if not INTEGER_MEDIAN:
        return np.median(arr, axis=axis)
    arr = np.asarray(arr)
    n = arr.shape[axis]
    if n % 2 == 1:
        return np.median(arr, axis=axis)
    sorted_arr = np.sort(arr, axis=axis)
    return np.take(sorted_arr, n // 2 - 1, axis=axis).astype(float)


# --- Shapley structures (all 2^m subsets), vectorised ----------------------------------
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
# candidate_steps[j] = list of (weight, idx(S), idx(S u {j})) -- kept for the reference
# implementation and for the equivalence check; the fast path uses the array form below.
candidate_steps = []
weight_arrays, idx_arrays, idxj_arrays = [], [], []
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
    weight_arrays.append(np.array([s[0] for s in steps_j]))
    idx_arrays.append(np.array([s[1] for s in steps_j]))
    idxj_arrays.append(np.array([s[2] for s in steps_j]))
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
    """Vectorised Shapley value of the median-based coalition game.
    phi_j = sum_{S subseteq C\\{j}} w(|S|) [ v(S u {j}) - v(S) ],  v(S) = median of the
    coalition's summed scores. Identical to the original Python loop; the equivalence is
    asserted by sanity_check_shapley() at start-up.
    """
    subset_sums = scores @ subset_mask.T
    delta = smart_median(subset_sums, axis=0)
    delta[0] = 0.0
    phi = np.zeros(M)
    for j in range(M):
        phi[j] = weight_arrays[j] @ (delta[idxj_arrays[j]] - delta[idx_arrays[j]])
    ranking = np.argsort(-phi, kind='mergesort')
    return ranking, set(ranking[:k]), phi


def shapley_median_rule_reference(scores, k):
    """Original step-by-step implementation, kept only for the equivalence check."""
    subset_sums = scores @ subset_mask.T
    delta = smart_median(subset_sums, axis=0)
    delta[0] = 0.0
    phi = np.zeros(M)
    for j in range(M):
        for w, idx_T, idx_T_j in candidate_steps[j]:
            phi[j] += w * (delta[idx_T_j] - delta[idx_T])
    ranking = np.argsort(-phi, kind='mergesort')
    return ranking, set(ranking[:k]), phi


RULE_FUNCTIONS = [utilitarian_rule, median_rule, shapley_median_rule]
# ======================================================================================
# 3. PREFERENCE GENERATIVE MODELS (STRICT INTEGER SCORES)
# ======================================================================================


def _add_integer_noise(profile, probs=None, scale_max=SCALE_MAX):
    """Integer noise in {-1,0,+1}, clipped: makes grade 1 reachable for rank models."""
    if probs is None:
        probs = RANK_MODEL_NOISE
    if probs is None:
        return profile.astype(np.int64)
    noise = np.random.choice([-1, 0, 1], size=profile.shape, p=list(probs))
    return np.clip(profile + noise, 0, scale_max).astype(np.int64)


def generate_mallows(n, m=M, phi=0.7, scale_max=SCALE_MAX):
    profile = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        ranking = []
        for j in range(m):
            probs = np.array([phi ** (j - pos) for pos in range(j + 1)])
            probs /= probs.sum()
            pos = np.random.choice(j + 1, p=probs)
            ranking.insert(pos, j)
        for rank_pos, cand in enumerate(ranking):
            profile[i, cand] = int(RANK_TO_SCORE[rank_pos])
    return _add_integer_noise(profile, scale_max=scale_max)


def generate_plackett_luce(n, m=M, scale_max=SCALE_MAX):
    gamma = np.random.gamma(shape=2.5, scale=1.0, size=m)
    profile = np.zeros((n, m), dtype=np.int64)
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
            profile[i, cand] = int(PL_RANK_TO_SCORE[rank_pos])
    return _add_integer_noise(profile, scale_max=scale_max)


def generate_euclidean(n, m=M, dim=2, scale_max=SCALE_MAX):
    candidates = np.random.uniform(0, 1, size=(m, dim))
    voters = np.random.uniform(0, 1, size=(n, dim))
    d = np.linalg.norm(voters[:, None, :] - candidates[None, :, :], axis=2)
    d_ref = np.sqrt(dim) if EUCLIDEAN_SCALE == "sqrt2" else d.max()
    raw = scale_max * (1.0 - d / d_ref)
    return np.clip(np.floor(raw + 0.5), 0, scale_max).astype(np.int64)


def generate_iac(n, m=M, scale_max=SCALE_MAX):
    profile = np.zeros((n, m), dtype=np.int64)
    for j in range(m):
        probs = np.random.dirichlet(np.ones(scale_max + 1))
        profile[:, j] = np.random.choice(scale_max + 1, size=n, p=probs).astype(np.int64)
    return profile


def generate_urn(n, m=M, alpha=0.2, scale_max=SCALE_MAX):
    urn = [np.random.randint(0, scale_max + 1, size=m, dtype=np.int64) for _ in range(8)]
    profile = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        ballot = urn[np.random.randint(len(urn))]
        profile[i, :] = ballot
        if np.random.rand() < alpha:
            urn.append(ballot)
    return profile


def generate_ic(n, m=M, scale_max=SCALE_MAX):
    return np.random.randint(0, scale_max + 1, size=(n, m), dtype=np.int64)


BEST_MODELS = {
    'Mallows (phi=0.7)': generate_mallows,
    'Plackett-Luce': generate_plackett_luce,
    '2D Euclidean': generate_euclidean,
    'Impartial Anonymous Culture (IAC)': generate_iac,
    'Polya Urn (alpha=0.2)': generate_urn,
    'Impartial Culture (IC)': generate_ic,
}
# ======================================================================================
# 4. MONTE CARLO WORKER  (one task per culture; every task sweeps x and k internally)
# ======================================================================================


def run_culture(args):
    cult_name, trials, seed = args
    np.random.seed(seed)
    gen_fn = BEST_MODELS[cult_name]
    n_x, n_k, n_t, n_r = (len(MANIP_COUNTS), len(COMMITTEE_SIZES),
                          len(TARGETS), len(RULES))
    changed = np.zeros((n_x, n_k, n_t, n_r))
    success = np.zeros((n_x, n_k, n_t, n_r))
    overlap = np.zeros((n_x, n_k, n_t, n_r))
    attacked = np.zeros((N_VOTERS, M), dtype=np.int64)
    t0 = time.time()
    for _ in range(trials):
        pool = gen_fn(MAX_HONEST, M)                    # 99 honest voters, reused below
        for ix, n_manip in enumerate(MANIP_COUNTS):
            n_honest = N_VOTERS - n_manip
            honest = pool[:n_honest]
            attacked[:n_honest] = honest
            attacked[n_honest:] = 0                     # manipulator block, zeros elsewhere
            for ik, k in enumerate(COMMITTEE_SIZES):
                honest_state = [fn(honest, k)[:2] for fn in RULE_FUNCTIONS]
                for it, t_type in enumerate(TARGETS):
                    for ir, rule_fn in enumerate(RULE_FUNCTIONS):
                        r_h, c_h = honest_state[ir]
                        target = r_h[k] if t_type == 'Cutoff' else r_h[M - 1]
                        attacked[n_honest:, target] = SCALE_MAX
                        _, c_post, _ = rule_fn(attacked, k)
                        attacked[n_honest:, target] = 0
                        if c_h != c_post:
                            changed[ix, ik, it, ir] += 1
                        overlap[ix, ik, it, ir] += len(c_h & c_post) / k
                        if target in c_post:
                            success[ix, ik, it, ir] += 1
    rows = []
    for ix, n_manip in enumerate(MANIP_COUNTS):
        for ik, k in enumerate(COMMITTEE_SIZES):
            for it, t_type in enumerate(TARGETS):
                for ir, r_name in enumerate(RULES):
                    rows.append({
                        'Culture': cult_name, 'k': k, 'n_total': N_VOTERS,
                        'n_honest': N_VOTERS - n_manip, 'n_manip': n_manip,
                        'manip_pct': n_manip, 'trials': trials, 'Target': t_type,
                        'Rule': r_name,
                        'Change_Rate': changed[ix, ik, it, ir] / trials * 100.0,
                        'Success_Rate': success[ix, ik, it, ir] / trials * 100.0,
                        'Avg_Overlap': overlap[ix, ik, it, ir] / trials,
                    })
    return rows, f"{cult_name:36s} {time.time()-t0:6.1f}s  ({trials} trials)"
# ======================================================================================
# 5. SANITY CHECKS
# ======================================================================================


def sanity_check_profiles(sample=20000):
    print(f"\n[*] Sanity check: integer scores, range [0,{SCALE_MAX}], grade coverage "
          f"({sample} voters per model)...")
    all_ok = True
    for name, fn in BEST_MODELS.items():
        profile = fn(sample, M)
        if profile.dtype.kind not in 'iu':
            print(f"    [FAIL] {name}: non-integer dtype = {profile.dtype}")
            all_ok = False
            continue
        grades = sorted(np.unique(profile).tolist())
        missing = sorted(REQUIRED_GRADES - set(grades))
        out_of_range = profile.min() < 0 or profile.max() > SCALE_MAX
        all_ok &= (not missing and not out_of_range)
        print(f"    [{'OK ' if (not missing and not out_of_range) else 'FAIL'}] "
              f"{name:38s} dtype={profile.dtype}  min={profile.min()}  "
              f"max={profile.max()}  grades={grades}")
        if missing:
            print(f"           unreachable grades: {missing}")
    print("    [OK] All six models produce integers in {0,...,5} and use every grade."
          if all_ok else "    [FAIL] grade coverage incomplete.")
    return all_ok


def sanity_check_median():
    probe = np.array([[1, 4, 2, 5, 3], [2, 3, 4, 1, 0]])
    med = smart_median(probe, axis=0)
    assert np.array_equal(med, [1, 3, 2, 1, 0]), med
    print(f"[*] Sanity check: lower median (even n) = {med.tolist()}  [OK]")


def sanity_check_shapley():
    """The vectorised Shapley must reproduce the original loop up to round-off.
    Rankings can legitimately differ only when two candidates tie exactly at the k-th
    boundary (the argsort order is then arbitrary), so ranking equality is asserted only
    when the boundary margin is above the numerical tolerance.
    """
    np.random.seed(1234)
    worst_phi, ties, checked = 0.0, 0, 0
    for _ in range(40):
        n = np.random.randint(5, N_VOTERS)
        scores = np.random.randint(0, SCALE_MAX + 1, size=(n, M)).astype(np.int64)
        for k in COMMITTEE_SIZES:
            _, set_fast, phi_fast = shapley_median_rule(scores, k)
            _, set_ref, phi_ref = shapley_median_rule_reference(scores, k)
            worst_phi = max(worst_phi, float(np.abs(phi_fast - phi_ref).max()))
            assert np.allclose(phi_fast, phi_ref, atol=1e-9), "vectorisation mismatch"
            if k < M:
                ordered = np.sort(phi_fast)[::-1]
                margin = ordered[k - 1] - ordered[k]
                if margin > 1e-9:
                    assert set_fast == set_ref, "top-k differs without a tie"
                    checked += 1
                else:
                    ties += 1
    # efficiency: sum_j phi_j == v(C) - v(empty) == v(C), where v(S) is the median over
    # voters of the coalition's summed score, so v(C) is the median of the per-voter totals
    scores = np.random.randint(0, SCALE_MAX + 1, size=(50, M)).astype(np.int64)
    _, _, phi = shapley_median_rule(scores, 2)
    grand = float(smart_median(scores.sum(axis=1)))
    print(f"[*] Sanity check: vectorised Shapley == reference loop "
          f"(max |diff| = {worst_phi:.2e}); {checked} non-tied top-k sets matched, "
          f"{ties} exact ties; efficiency |sum(phi) - v(C)| = {abs(phi.sum()-grand):.2e}  [OK]")


def sanity_check_design():
    print(f"[*] Design: n = {N_VOTERS} voters fixed; manipulators {MANIP_COUNTS[0]}% ... "
          f"{MANIP_COUNTS[-1]}% ({MANIP_COUNTS[0]} ... {MANIP_COUNTS[-1]} voters); "
          f"honest {N_VOTERS-MANIP_COUNTS[-1]} ... {N_VOTERS-MANIP_COUNTS[0]}")
    print(f"[*] m = {M} candidates, k = {COMMITTEE_SIZES}, attacks = {TARGETS}")
# ======================================================================================
# 6. AUDIT -- verifies that every combination ran and that every result is valid
# ======================================================================================


def audit_results(df, base_dir):
    """Check the whole design and every output cell; write an audit report next to the data.
    Returns True when every check passes.
    """
    print("\n" + "=" * 80)
    print("[*] AUDIT: checking the design, the coverage and every result cell")
    report, ok = [], True
    def check(label, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        line = f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  ->  {detail}" if detail else "")
        print(line)
        report.append(line)
    # ---- design / coverage ----------------------------------------------------------
    n_expected = (len(BEST_MODELS) * len(MANIP_COUNTS) * len(COMMITTEE_SIZES)
                  * len(TARGETS) * len(RULES))
    check("result rows complete", len(df) == n_expected, f"{len(df)} of {n_expected}")
    check("all six preference cultures", set(df.Culture) == set(BEST_MODELS),
          f"{df.Culture.nunique()} cultures")
    check("all committee sizes k = 2, 3, 4", sorted(df.k.unique()) == COMMITTEE_SIZES,
          f"k = {sorted(int(v) for v in df.k.unique())}")
    check("every integer intensity 1% ... 49%",
          sorted(df.manip_pct.unique()) == MANIP_COUNTS, f"{df.manip_pct.nunique()} levels")
    check("both attacks present", set(df.Target) == set(TARGETS), f"{sorted(set(df.Target))}")
    check("all three rules present", set(df.Rule) == set(RULES), f"{len(set(df.Rule))} rules")
    check("no duplicate cells",
          df.duplicated(subset=["Culture", "k", "manip_pct", "Target", "Rule"]).sum() == 0)
    check("electorate is always 100 voters",
          bool((df.n_honest + df.n_manip == N_VOTERS).all()),
          f"min {int(df.n_honest.min())} honest, max {int(df.n_manip.max())} manipulators")
    check("honest count inside 51 ... 99", bool(df.n_honest.between(51, 99).all()))
    # ---- value validity -------------------------------------------------------------
    check("no missing values", bool(~df.isna().any().any()))
    rates = df[["Change_Rate", "Success_Rate"]]
    check("all rates inside [0, 100]",
          bool(((rates >= 0) & (rates <= 100)).all().all()),
          f"max rate {rates.max().max():.1f}%")
    check("overlap inside [0, 1]", bool(df.Avg_Overlap.between(0, 1).all()),
          f"range {df.Avg_Overlap.min():.3f} ... {df.Avg_Overlap.max():.3f}")
    # ---- behavioural identities -----------------------------------------------------
    gap = df.Change_Rate - df.Success_Rate
    check("evaluative voting: change rate == success rate (both attacks)",
          float(gap[df.Rule == 'Evaluative Voting'].abs().max()) < 1e-9,
          f"max |diff| = {float(gap[df.Rule == 'Evaluative Voting'].abs().max()):.2e} pp")
    cut = gap[(df.Rule == 'k-Median Rule') & (df.Target == 'Cutoff')]
    check("cutoff attack: change rate == success rate for the k-Median rule",
          float(cut.abs().max()) < 1e-9, f"max |diff| = {float(cut.abs().max()):.2e} pp")
    cut_shap = float(gap[(df.Rule == 'k-Median-Shapley') & (df.Target == 'Cutoff')].max())
    bot_med = float(gap[(df.Rule == 'k-Median Rule') & (df.Target == 'Bottom')].max())
    bot_shap = float(gap[(df.Rule == 'k-Median-Shapley') & (df.Target == 'Bottom')].max())
    for msg in [f"  [INFO] cutoff attack, Shapley: change minus success up to "
                f"{cut_shap:.1f} pp (expected: Shapley can reshuffle the committee without "
                f"admitting the target)",
                f"  [INFO] bottom attack, k-Median: change minus success up to {bot_med:.1f} pp",
                f"  [INFO] bottom attack, Shapley: change minus success up to {bot_shap:.1f} pp"]:
        print(msg)
        report.append(msg)
    # ---- robustness ordering --------------------------------------------------------
    for target in TARGETS:
        for metric in ["Success_Rate", "Change_Rate"]:
            w = (df[df.Target == target]
                 .pivot_table(index=["Culture", "k", "manip_pct"], columns="Rule",
                              values=metric))
            share = float(((w["k-Median-Shapley"] <= w["k-Median Rule"])
                           & (w["k-Median Rule"] <= w["Evaluative Voting"])).mean()) * 100
            check(f"{target} / {metric}: Shapley <= Median <= Evaluative",
                  share >= 75.0, f"{share:.1f}% of the {len(w)} cells")
    # ---- monotone response to more manipulation -------------------------------------
    for target in TARGETS:
        steps = (df[df.Target == target].groupby(["Culture", "k", "Rule"])["Success_Rate"]
                 .apply(lambda s: s.sort_index().diff().dropna()))
        rising = float((steps >= 0).mean()) * 100
        check(f"{target} / success rate non-decreasing as manipulation grows",
              rising >= 90.0, f"{rising:.1f}% of consecutive steps")
    # ---- the attack must actually be able to work -----------------------------------
    tip = df[(df.Target == 'Bottom') & (df.Rule == 'Evaluative Voting')]
    worst = float(tip.groupby(["Culture", "k"])["Success_Rate"].max().min())
    check("every culture x k reacts to the attack (success exceeds 50% somewhere)",
          worst > 50.0, f"worst case maximum success = {worst:.1f}%")
    header = (f"AUDIT REPORT - Simulation 1 (manipulation intensity 1%-49%)\n"
              f"n = {N_VOTERS} voters fixed, m = {M} candidates, k = {COMMITTEE_SIZES}, "
              f"trials per cell = {int(df.trials.iloc[0])}\n"
              f"rows = {len(df)} (cultures x k x intensities x attacks x rules)\n"
              + "-" * 78 + "\n")
    with open(os.path.join(base_dir, "sim1_audit_report.txt"), "w") as fh:
        fh.write(header + "\n".join(report) + "\n" + "-" * 78 + "\n"
                 + ("OVERALL: ALL CHECKS PASSED\n" if ok else "OVERALL: SOME CHECKS FAILED\n"))
    print("  [OK] audit report written to sim1_audit_report.txt")
    print(f"  OVERALL: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    print("=" * 80)
    return ok


# ======================================================================================
# 7. FIGURE STYLE
# ======================================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5, "legend.fontsize": 9, "axes.grid": True,
    "grid.linestyle": ":", "grid.alpha": 0.3, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 300,
})
RULE_COLORS = {"Evaluative Voting": "#D55E00", "k-Median Rule": "#0072B2",
               "k-Median-Shapley": "#009E73"}
RULE_MARKERS = {"Evaluative Voting": "o", "k-Median Rule": "s", "k-Median-Shapley": "^"}
RULE_LABELS = {"Evaluative Voting": "Evaluative voting", "k-Median Rule": "k-Median rule",
               "k-Median-Shapley": "k-Median-Shapley"}
K_LINESTYLES = {2: (0, (1, 1.6)), 3: (0, (5, 2)), 4: "-"}
K_MARKERS = {2: "o", 3: "s", 4: "^"}
K_LABELS = {2: r"$k=2$", 3: r"$k=3$", 4: r"$k=4$"}
K_COLORS = {2: "#4C72B0", 3: "#DD8452", 4: "#55A868"}
MARK_AT = [1, 13, 25, 37, 49]
PANEL_ORDER = list(BEST_MODELS.keys())
PANEL_TITLES = {
    "Mallows (phi=0.7)": "Mallows ($\\varphi = 0.7$)",
    "Plackett-Luce": "Plackett–Luce",
    "2D Euclidean": "2D Euclidean",
    "Impartial Anonymous Culture (IAC)": "IAC",
    "Polya Urn (alpha=0.2)": "Pólya Urn ($\\alpha = 0.2$)",
    "Impartial Culture (IC)": "Impartial Culture",
}
LATEX_CULTURE = {
    "Mallows (phi=0.7)": r"Mallows ($\varphi=0.7$)", "Plackett-Luce": "Plackett--Luce",
    "2D Euclidean": "2D Euclidean", "Impartial Anonymous Culture (IAC)": "IAC",
    "Polya Urn (alpha=0.2)": r"P\'olya Urn ($\alpha=0.2$)", "Impartial Culture (IC)": "Impartial Culture",
}
XLABEL = "Manipulators (% of the 100-voter electorate)"
ANCHORS = [10, 20, 30, 40, 49]           # intensities used in the LaTeX tables


def _design_note(df):
    return (f"n = {int(df['n_total'].iloc[0])} voters, m = 5 candidates, "
            f"k = 2,3,4, {int(df['trials'].iloc[0]):,} trials per cell")


def _two_block_legend(fig, k_values, loc=(0.5, 0.005), ncol=None):
    handles = [plt.Line2D([], [], color=RULE_COLORS[r], lw=2.6,
                          marker=RULE_MARKERS[r], markersize=8, label=RULE_LABELS[r])
               for r in RULES]
    handles += [plt.Line2D([], [], color="0.25", lw=2.0, linestyle=K_LINESTYLES[k],
                           marker=K_MARKERS[k], markersize=8, label=K_LABELS[k])
                for k in k_values]
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles),
               frameon=True, bbox_to_anchor=loc)
# ======================================================================================
# 8. FIGURES
# ======================================================================================


def fig_intensity_grid(df, target, metric, out_png, out_pdf, ylabel, suptitle):
    """2x3 culture grid: one panel per culture, colour = rule, line style/marker = k."""
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6), sharey=True)
    for ax, cult in zip(axes.flatten(), PANEL_ORDER):
        sub = df[(df["Culture"] == cult) & (df["Target"] == target)]
        for k in COMMITTEE_SIZES:
            for rule in RULES:
                d = (sub[(sub["Rule"] == rule) & (sub["k"] == k)]
                     .sort_values("manip_pct"))
                ax.plot(d["manip_pct"], d[metric], color=RULE_COLORS[rule],
                        linestyle=K_LINESTYLES[k], linewidth=2.0, zorder=3)
                dm = d[d["manip_pct"].isin(MARK_AT)]
                ax.plot(dm["manip_pct"], dm[metric], linestyle="none",
                        marker=K_MARKERS[k], markersize=7, color=RULE_COLORS[rule],
                        markeredgecolor="white", markeredgewidth=1.0, zorder=4)
        ax.set_title(PANEL_TITLES[cult])
        ax.set_xlabel(XLABEL)
        ax.set_xticks([1, 10, 20, 30, 40, 49])
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 104)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    _two_block_legend(fig, COMMITTEE_SIZES)
    fig.suptitle(f"{suptitle}\n{_design_note(df)}", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}")


def fig_dual_metric(df, out_png, out_pdf, suptitle):
    """Bottom attack, both metrics: solid = change rate, dashed = success rate.
    Mean over k = 2,3,4 with the shaded band spanning the k values, so the two metrics
    and the effect of committee size are visible in the same panel.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6), sharey=True)
    for ax, cult in zip(axes.flatten(), PANEL_ORDER):
        sub = df[(df["Culture"] == cult) & (df["Target"] == "Bottom")]
        for rule in RULES:
            d = sub[sub["Rule"] == rule].groupby("manip_pct")[["Change_Rate", "Success_Rate"]]
            mean = d.mean().reindex(MANIP_COUNTS)
            lo, hi = d.min().reindex(MANIP_COUNTS), d.max().reindex(MANIP_COUNTS)
            c = RULE_COLORS[rule]
            ax.fill_between(mean.index, lo["Change_Rate"], hi["Change_Rate"], color=c, alpha=0.15, lw=0)
            ax.fill_between(mean.index, lo["Success_Rate"], hi["Success_Rate"], color=c, alpha=0.08, lw=0)
            ax.plot(mean.index, mean["Change_Rate"], color=c, lw=2.6,
                    marker=RULE_MARKERS[rule], markevery=[0, 12, 24, 36, 48],
                    markersize=6, label=RULE_LABELS[rule])
            ax.plot(mean.index, mean["Success_Rate"], color=c, lw=1.9, linestyle="--",
                    marker=RULE_MARKERS[rule], markevery=[0, 12, 24, 36, 48], markersize=6,
                    markerfacecolor="white", markeredgecolor=c)
        ax.set_title(PANEL_TITLES[cult])
        ax.set_xlabel(XLABEL)
        ax.set_xticks([1, 10, 20, 30, 40, 49])
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 104)
    for ax in axes[:, 0]:
        ax.set_ylabel("Rate (%)")
    handles = [plt.Line2D([], [], color=RULE_COLORS[r], lw=2.6, marker=RULE_MARKERS[r],
                          markersize=8, label=RULE_LABELS[r]) for r in RULES]
    handles += [plt.Line2D([], [], color="0.25", lw=2.4, label="change rate (solid)"),
                plt.Line2D([], [], color="0.25", lw=1.9, ls="--", marker="o",
                           markerfacecolor="white", markersize=7,
                           label="target success rate (dashed)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{suptitle}\n{_design_note(df)}", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}")


def fig_heatmap(df, out_png, out_pdf, suptitle):
    """Attack-success map: rows = culture x k (18), columns = manipulation bins, one panel
    per rule. Colour = bottom-target success rate (%)."""
    edges = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    labels = [f"{edges[i]}–{edges[i+1]}%" for i in range(len(edges) - 1)]
    sub_all = df[df["Target"] == "Bottom"]
    sub_all = sub_all.assign(bin=pd.cut(sub_all["manip_pct"], bins=edges,
                                        labels=labels, include_lowest=True))
    cmap = LinearSegmentedColormap.from_list("wr", ["#FFFFFF", "#FDD49E", "#FC8D59",
                                                    "#D7301F", "#7F0000"])
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), sharey=True)
    row_labels = [f"{c.replace(' (phi=0.7)','').replace(' (alpha=0.2)','')}  ·  k={k}"
                  for c in PANEL_ORDER for k in COMMITTEE_SIZES]
    for ax, rule in zip(axes, RULES):
        d = (sub_all[sub_all["Rule"] == rule]
             .groupby(["Culture", "k", "bin"], observed=True)["Success_Rate"].mean()
             .unstack("bin").reindex(columns=labels))
        d = d.reindex([(c, k) for c in PANEL_ORDER for k in COMMITTEE_SIZES])
        im = ax.imshow(d.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_title(RULE_LABELS[rule])
        ax.set_yticks(range(len(row_labels)))
        if rule == RULES[0]:
            ax.set_yticklabels([lab.split("  ·  ")[0] if i % 3 == 0 else ""
                                for i, lab in enumerate(row_labels)], fontsize=8.5)
            for i, lab in enumerate(row_labels):
                if i % 3 == 1:
                    ax.text(-0.02, i, lab.split("·")[-1].strip(), transform=ax.get_yaxis_transform(),
                            ha="right", va="center", fontsize=7.5, color="#333333")
        for i in range(1, len(row_labels)):
            ax.axhline(i - 0.5, color="white", lw=1.6)
        for x in range(1, len(labels)):
            ax.axvline(x - 0.5, color="white", lw=0.8)
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.01)
    cb.set_label("Bottom-target success rate (%)")
    fig.suptitle(f"{suptitle}\nrows = culture (3 committee sizes each), "
                 f"columns = manipulation level, {_design_note(df)}", fontsize=12.5)
    fig.savefig(out_png, bbox_inches="tight"); fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}")


def fig_tipping_points(df, out_png, out_pdf, suptitle, threshold=50.0):
    """How much manipulation is needed before the bottom target succeeds in 50% of trials?
    Further right = more robust. A marker on the right edge means the threshold was never
    reached inside the simulated 1-49% range."""
    sub = df[df["Target"] == "Bottom"]
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    yticks, ylabels = [], []
    y = 0
    for cult in PANEL_ORDER:
        for k in COMMITTEE_SIZES:
            yticks.append(y); ylabels.append(f"{PANEL_TITLES[cult]}  ·  k={k}")
            for rule in RULES:
                d = sub[(sub["Culture"] == cult) & (sub["k"] == k) & (sub["Rule"] == rule)]
                d = d.sort_values("manip_pct")
                hit = d[d["Success_Rate"] >= threshold]["manip_pct"]
                x = float(hit.iloc[0]) if len(hit) else float(MANIP_PCT_MAX) + 4
                ax.plot([x], [y + (RULES.index(rule) - 1) * 0.24],
                        marker=RULE_MARKERS[rule], markersize=9, color=RULE_COLORS[rule],
                        markeredgecolor="white", markeredgewidth=1.0, zorder=4)
                if x > MANIP_PCT_MAX:
                    ax.plot([x], [y + (RULES.index(rule) - 1) * 0.24],
                            marker=RULE_MARKERS[rule], markersize=9,
                            markerfacecolor="white", markeredgecolor=RULE_COLORS[rule],
                            markeredgewidth=1.8, zorder=5)
            y += 1
    ax.axvspan(MANIP_PCT_MAX, MANIP_PCT_MAX + 8, color="0.92", zorder=0)
    ax.text(MANIP_PCT_MAX + 4, len(PANEL_ORDER) * len(COMMITTEE_SIZES) - 0.5,
            "never reached\ninside 1–49%", ha="center", va="bottom", fontsize=9,
            color="0.35")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(XLABEL)
    ax.set_xlim(0, MANIP_PCT_MAX + 8)
    ax.set_xticks([1, 10, 20, 30, 40, 49])
    ax.set_ylabel("Culture  ·  committee size")
    handles = [plt.Line2D([], [], color=RULE_COLORS[r], marker=RULE_MARKERS[r],
                          linestyle="none", markersize=9, label=RULE_LABELS[r])
               for r in RULES]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    ax.set_title(f"{suptitle}\nmanipulation level (% of the electorate) at which the bottom "
                 f"target reaches a {threshold:.0f}% success rate  ·  {_design_note(df)}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}")


def fig_crossing_vs_k(df, out_png, out_pdf, suptitle, threshold=50.0):
    """The simplest summary of the whole simulation.
    One panel per culture, x = committee size, y = manipulation level at which the bottom
    target reaches a 50% success rate. A higher curve means the electorate must be
    manipulated harder before the attack works, so a higher curve is a more robust rule.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), sharey=True)
    sub = df[df["Target"] == "Bottom"]
    for ax, cult in zip(axes.flatten(), PANEL_ORDER):
        for rule in RULES:
            xs, ys = [], []
            for k in COMMITTEE_SIZES:
                d = (sub[(sub["Culture"] == cult) & (sub["k"] == k) & (sub["Rule"] == rule)]
                     .sort_values("manip_pct"))
                hit = d[d["Success_Rate"] >= threshold]["manip_pct"]
                xs.append(k)
                ys.append(float(hit.iloc[0]) if len(hit) else np.nan)
            if all(np.isnan(ys)):
                continue
            ys_plot = [y if not np.isnan(y) else MANIP_PCT_MAX for y in ys]
            ax.plot(xs, ys_plot, color=RULE_COLORS[rule], lw=2.8,
                    marker=RULE_MARKERS[rule], markersize=10,
                    markeredgecolor="white", markeredgewidth=1.2, label=RULE_LABELS[rule])
            for x, y, raw in zip(xs, ys_plot, ys):
                if np.isnan(raw):
                    ax.plot([x], [y], marker=RULE_MARKERS[rule], markersize=10,
                            markerfacecolor="white", markeredgecolor=RULE_COLORS[rule],
                            markeredgewidth=2.0, linestyle="none")
        ax.set_title(PANEL_TITLES[cult])
        ax.set_xlabel("Committee size $k$")
        ax.set_xticks(COMMITTEE_SIZES)
        ax.set_xlim(1.7, 4.3)
        ax.set_ylim(0, 52)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Manipulators needed for a\n{threshold:.0f}% success rate (%)")
    handles = [plt.Line2D([], [], color=RULE_COLORS[r], lw=2.8, marker=RULE_MARKERS[r],
                          markersize=10, markeredgecolor="white", label=RULE_LABELS[r])
               for r in RULES]
    handles.append(plt.Line2D([], [], color="0.4", marker="o", linestyle="none",
                              markerfacecolor="white", markeredgecolor="0.4",
                              markeredgewidth=2.0, markersize=10,
                              label=f"never reached within 1–{MANIP_PCT_MAX}%"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{suptitle}\nhigher = harder to manipulate, so higher is better  ·  "
                 f"{_design_note(df)}", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}")


def fig_aggregate(df, out_png, out_pdf, suptitle):
    """Mean over the six cultures: rows = attack, columns = metric, k as line style."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=True)
    for r, target in enumerate(TARGETS):
        for c, metric in enumerate(["Change_Rate", "Success_Rate"]):
            ax = axes[r, c]
            sub = df[df["Target"] == target]
            for k in COMMITTEE_SIZES:
                for rule in RULES:
                    d = (sub[(sub["Rule"] == rule) & (sub["k"] == k)]
                         .groupby("manip_pct")[metric].mean().reindex(MANIP_COUNTS))
                    ax.plot(d.index, d.values, color=RULE_COLORS[rule], lw=2.2,
                            linestyle=K_LINESTYLES[k])
            ax.set_title(f"{target} attack · "
                         f"{'committee change rate' if metric == 'Change_Rate' else 'target success rate'}")
            ax.set_xlabel(XLABEL)
            ax.set_xticks([1, 10, 20, 30, 40, 49])
            ax.set_xlim(0, 50); ax.set_ylim(0, 104)
        axes[r, 0].set_ylabel("Rate (%)")
    _two_block_legend(fig, COMMITTEE_SIZES, loc=(0.5, 0.008))
    fig.suptitle(f"{suptitle}\nmean over the six preference models  ·  {_design_note(df)}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}")


def fig_divergence(df, out_png, out_pdf, suptitle):
    """Bottom attack: change rate minus target success rate (percentage points).
    Zero means 'every committee change admits the target'; a positive value means the
    committee changed while the target still failed. Exact zero for evaluative voting."""
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6), sharey=True)
    for ax, cult in zip(axes.flatten(), PANEL_ORDER):
        sub = df[(df["Culture"] == cult) & (df["Target"] == "Bottom")].copy()
        sub["gap"] = sub["Change_Rate"] - sub["Success_Rate"]
        for k in COMMITTEE_SIZES:
            for rule in RULES:
                d = sub[(sub["Rule"] == rule) & (sub["k"] == k)].sort_values("manip_pct")
                ax.plot(d["manip_pct"], d["gap"], color=RULE_COLORS[rule], lw=2.0,
                        linestyle=K_LINESTYLES[k])
        ax.axhline(0, color="0.3", lw=1.2)
        ax.set_title(PANEL_TITLES[cult])
        ax.set_xlabel(XLABEL)
        ax.set_xticks([1, 10, 20, 30, 40, 49])
        ax.set_xlim(0, 50); ax.set_ylim(-3, 62)
    for ax in axes[:, 0]:
        ax.set_ylabel("Change rate − success rate (pp)")
    _two_block_legend(fig, COMMITTEE_SIZES)
    fig.suptitle(f"{suptitle}\nzero = the target enters whenever the committee changes; "
                 f"positive = committee changed but the target failed  ·  {_design_note(df)}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}")


def fig_advantage(df, out_png, out_pdf, suptitle):
    """Robustness gain of the median rules over evaluative voting, bottom attack:
    success(evaluative) − success(median rule), in percentage points. Bigger = better."""
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6), sharey=True)
    for ax, cult in zip(axes.flatten(), PANEL_ORDER):
        sub = df[(df["Culture"] == cult) & (df["Target"] == "Bottom")]
        for k in COMMITTEE_SIZES:
            piv = (sub[sub["k"] == k]
                   .pivot_table(index="manip_pct", columns="Rule", values="Success_Rate")
                   .reindex(MANIP_COUNTS))
            ax.plot(piv.index, piv["Evaluative Voting"] - piv["k-Median Rule"],
                    color=K_COLORS[k], lw=2.2, label=f"vs k-Median, {K_LABELS[k]}")
            ax.plot(piv.index, piv["Evaluative Voting"] - piv["k-Median-Shapley"],
                    color=K_COLORS[k], lw=1.7, linestyle="--",
                    label=f"vs Shapley, {K_LABELS[k]}")
        ax.axhline(0, color="0.3", lw=1.2)
        ax.set_title(PANEL_TITLES[cult])
        ax.set_xlabel(XLABEL)
        ax.set_xticks([1, 10, 20, 30, 40, 49])
        ax.set_xlim(0, 50); ax.set_ylim(-8, 62)
    for ax in axes[:, 0]:
        ax.set_ylabel("Success reduction vs evaluative (pp)")
    h1 = [plt.Line2D([], [], color=K_COLORS[k], lw=2.4, label=f"committee size {K_LABELS[k]}")
          for k in COMMITTEE_SIZES]
    h2 = [plt.Line2D([], [], color="0.25", lw=2.2, label="vs k-Median rule"),
          plt.Line2D([], [], color="0.25", lw=1.7, ls="--", label="vs k-Median-Shapley")]
    fig.legend(handles=h1 + h2, loc="lower center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{suptitle}\nhigher = median rules keep the bottom target out more often "
                 f"than evaluative voting  ·  {_design_note(df)}", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(out_png); fig.savefig(out_pdf); plt.close(fig)
    print(f"    [+] {os.path.basename(out_png)}")


def make_all_figures(df, graphs_dir):
    print(f"\n[*] Rendering figures -> {graphs_dir}")
    design = "Manipulation-intensity sweep of committee selection"
    fig_intensity_grid(df, "Cutoff", "Change_Rate",
                       os.path.join(graphs_dir, "fig01_cutoff_change_rate.png"),
                       os.path.join(graphs_dir, "fig01_cutoff_change_rate.pdf"),
                       "Committee change rate (%)",
                       "Cutoff attack: committee change rate")
    fig_intensity_grid(df, "Cutoff", "Success_Rate",
                       os.path.join(graphs_dir, "fig02_cutoff_success_rate.png"),
                       os.path.join(graphs_dir, "fig02_cutoff_success_rate.pdf"),
                       "Target success rate (%)",
                       "Cutoff attack: target success rate")
    fig_intensity_grid(df, "Bottom", "Change_Rate",
                       os.path.join(graphs_dir, "fig03_bottom_change_rate.png"),
                       os.path.join(graphs_dir, "fig03_bottom_change_rate.pdf"),
                       "Committee change rate (%)",
                       "Bottom attack: committee change rate")
    fig_intensity_grid(df, "Bottom", "Success_Rate",
                       os.path.join(graphs_dir, "fig04_bottom_success_rate.png"),
                       os.path.join(graphs_dir, "fig04_bottom_success_rate.pdf"),
                       "Target success rate (%)",
                       "Bottom attack: target success rate")
    fig_intensity_grid(df, "Cutoff", "Avg_Overlap",
                       os.path.join(graphs_dir, "fig05_cutoff_overlap.png"),
                       os.path.join(graphs_dir, "fig05_cutoff_overlap.pdf"),
                       "Average overlap with the honest committee",
                       "Cutoff attack: average overlap")
    fig_dual_metric(df, os.path.join(graphs_dir, "fig06_bottom_dual_metric.png"),
                    os.path.join(graphs_dir, "fig06_bottom_dual_metric.pdf"),
                    "Bottom attack: change rate vs target success rate")
    fig_heatmap(df, os.path.join(graphs_dir, "fig07_success_heatmap.png"),
                os.path.join(graphs_dir, "fig07_success_heatmap.pdf"),
                "Bottom-target success map")
    fig_tipping_points(df, os.path.join(graphs_dir, "fig08_tipping_points.png"),
                       os.path.join(graphs_dir, "fig08_tipping_points.pdf"),
                       "How much manipulation does it take to elect the bottom candidate?")
    fig_aggregate(df, os.path.join(graphs_dir, "fig09_aggregate_summary.png"),
                  os.path.join(graphs_dir, "fig09_aggregate_summary.pdf"),
                  "Aggregate summary over the six preference models")
    fig_divergence(df, os.path.join(graphs_dir, "fig10_change_vs_success_gap.png"),
                   os.path.join(graphs_dir, "fig10_change_vs_success_gap.pdf"),
                   "Bottom attack: committee change vs manipulation success")
    fig_advantage(df, os.path.join(graphs_dir, "fig11_median_advantage.png"),
                  os.path.join(graphs_dir, "fig11_median_advantage.pdf"),
                  "Bottom attack: robustness gain of the median rules")
    fig_crossing_vs_k(df, os.path.join(graphs_dir, "fig12_crossing_vs_committee_size.png"),
                      os.path.join(graphs_dir, "fig12_crossing_vs_committee_size.pdf"),
                      "How much manipulation is needed to elect the bottom candidate?")
# ======================================================================================
# 9. TABLES
# ======================================================================================


def export_tables(df, tables_dir):
    print(f"\n[*] Exporting tables -> {tables_dir}")
    for target in TARGETS:
        for metric, nd in [("Success_Rate", 2), ("Change_Rate", 2), ("Avg_Overlap", 3)]:
            sub = df[df["Target"] == target]
            piv = (sub.pivot_table(index=["Culture", "k", "Rule"], columns="manip_pct",
                                   values=metric).round(nd))
            name = f"sim1_{metric.lower()}_{target.lower()}_by_intensity.csv"
            piv.to_csv(os.path.join(tables_dir, name))
            print(f"    [+] {name}")
    # mean over cultures (compact overview)
    for target in TARGETS:
        sub = df[df["Target"] == target]
        for metric in ["Success_Rate", "Change_Rate"]:
            piv = (sub.pivot_table(index=["k", "Rule"], columns="manip_pct", values=metric)
                   .round(2))
            name = f"sim1_mean_over_cultures_{metric.lower()}_{target.lower()}.csv"
            piv.to_csv(os.path.join(tables_dir, name))
            print(f"    [+] {name}")
    # tipping points (first intensity with bottom-target success >= 50%)
    rows = []
    for cult in PANEL_ORDER:
        for k in COMMITTEE_SIZES:
            for rule in RULES:
                d = df[(df["Culture"] == cult) & (df["k"] == k) & (df["Rule"] == rule)
                       & (df["Target"] == "Bottom")].sort_values("manip_pct")
                hit = d[d["Success_Rate"] >= 50.0]["manip_pct"]
                rows.append({"Culture": cult, "k": k, "Rule": rule,
                             "tipping_pct_50": (int(hit.iloc[0]) if len(hit)
                                                else None),
                             "success_at_49pct": round(float(d["Success_Rate"].iloc[-1]), 2)})
    tip = pd.DataFrame(rows)
    tip.to_csv(os.path.join(tables_dir, "sim1_tipping_points_bottom.csv"), index=False)
    print("    [+] sim1_tipping_points_bottom.csv")
    # LaTeX: one table per (metric, target, k), rows = culture x rule, cols = anchors
    for target in TARGETS:
        for metric in ["Success_Rate", "Change_Rate"]:
            for k in COMMITTEE_SIZES:
                _latex_table(df, metric, target, k, tables_dir)
    _latex_table_tipping(tip, tables_dir)


def _latex_table(df, metric, target, k, tables_dir):
    sub = df[(df["Target"] == target) & (df["k"] == k)]
    head = " & ".join(f"{{${a}\\%$}}" for a in ANCHORS)
    lines = [r"\begin{table}[htbp]", r"  \centering",
             f"  \\caption{{{metric.replace('_', ' ')} (\\%) under the {target.lower()} "
             f"attack, committee size $k={k}$. Manipulators are a share of the fixed "
             f"{int(df['n_total'].iloc[0])}-voter electorate.}}",
             f"  \\label{{tab:sim1-{target.lower()}-{metric.lower()}-k{k}}}",
             r"  \begin{tabular}{@{}ll" + "S[table-format=3.2]" * len(ANCHORS) + r"@{}}",
             r"    \toprule", r"    & & \multicolumn{" + str(len(ANCHORS))
             + r"}{c}{Manipulators (\% of electorate)} \\",
             r"    \cmidrule(lr){3-" + str(2 + len(ANCHORS)) + r"}",
             r"    Culture & Rule & " + head + r" \\", r"    \midrule"]
    short = {"Evaluative Voting": "Evaluative", "k-Median Rule": r"$k$-Median",
             "k-Median-Shapley": r"$k$-Median-Shapley"}
    for cult in PANEL_ORDER:
        for i, rule in enumerate(RULES):
            d = (sub[(sub["Culture"] == cult) & (sub["Rule"] == rule)]
                 .set_index("manip_pct").reindex(ANCHORS))
            cells = " & ".join(f"{v:.2f}" for v in d[metric].values)
            if i == 0:
                lines.append(f"    \\multirow{{3}}{{*}}{{{LATEX_CULTURE[cult]}}} & "
                             f"{short[rule]} & {cells} \\\\")
            else:
                lines.append(f"    & {short[rule]} & {cells} \\\\")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"\end{table}", ""]
    name = f"sim1_table_{target.lower()}_{metric.lower()}_k{k}.tex"
    with open(os.path.join(tables_dir, name), "w") as fh:
        fh.write("\n".join(lines))
    print(f"    [+] {name}")


def _latex_table_tipping(tip, tables_dir):
    lines = [r"\begin{table}[htbp]", r"  \centering",
             r"  \caption{Manipulation level (\% of the electorate) at which the bottom "
             r"target first reaches a 50\% success rate. A dash means the threshold was "
             r"not reached within the simulated 1--49\% range.}",
             r"  \label{tab:sim1-tipping}",
             r"  \begin{tabular}{@{}ll" + "S[table-format=2.0]" * len(COMMITTEE_SIZES)
             + r"@{}}", r"    \toprule",
             r"    & & \multicolumn{" + str(len(COMMITTEE_SIZES))
             + r"}{c}{Committee size $k$} \\",
             r"    \cmidrule(lr){3-" + str(2 + len(COMMITTEE_SIZES)) + r"}",
             r"    Culture & Rule & " + " & ".join(f"{{${k}$}}" for k in COMMITTEE_SIZES)
             + r" \\", r"    \midrule"]
    short = {"Evaluative Voting": "Evaluative", "k-Median Rule": r"$k$-Median",
             "k-Median-Shapley": r"$k$-Median-Shapley"}
    for cult in PANEL_ORDER:
        for i, rule in enumerate(RULES):
            cells = []
            for k in COMMITTEE_SIZES:
                v = tip[(tip["Culture"] == cult) & (tip["k"] == k)
                        & (tip["Rule"] == rule)]["tipping_pct_50"].iloc[0]
                cells.append("{" + ("--" if pd.isna(v) else f"{int(v)}") + "}")
            if i == 0:
                lines.append(f"    \\multirow{{3}}{{*}}{{{LATEX_CULTURE[cult]}}} & "
                             f"{short[rule]} & " + " & ".join(cells) + r" \\")
            else:
                lines.append(f"    & {short[rule]} & " + " & ".join(cells) + r" \\")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [r"  \end{tabular}", r"\end{table}", ""]
    with open(os.path.join(tables_dir, "sim1_table_tipping_points.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print("    [+] sim1_table_tipping_points.tex")
# ======================================================================================
# 10. PIPELINE
# ======================================================================================


def setup_directories(target_windows_path=r"D:\PYTHON\Project - Median"):
    base_dir = (os.path.join(target_windows_path, OUTPUT_DIRNAME)
                if platform.system() == "Windows"
                else os.path.abspath(os.path.join("./output_median_project", OUTPUT_DIRNAME)))
    graphs_dir, tables_dir = os.path.join(base_dir, "graphs"), os.path.join(base_dir, "tables")
    for d in (base_dir, graphs_dir, tables_dir):
        os.makedirs(d, exist_ok=True)
    print("=" * 80)
    print(f"[*] Output base : {base_dir}")
    print(f"[*] Figures     : {graphs_dir}   (all figures, one folder)")
    print(f"[*] Tables      : {tables_dir}")
    print("=" * 80)
    return base_dir, graphs_dir, tables_dir


def simulate(base_dir):
    tasks, seed = [], 2026
    for cult in BEST_MODELS:
        tasks.append((cult, NUM_ITERATIONS, seed))
        seed += 1
    workers = min(cpu_count(), len(tasks))
    print(f"[*] {len(tasks)} culture tasks on {workers} cores; each task sweeps "
          f"{len(MANIP_COUNTS)} manipulation levels x {len(COMMITTEE_SIZES)} committee "
          f"sizes x {len(TARGETS)} attacks x {len(RULES)} rules\n")
    t0 = time.time()
    rows = []
    with Pool(processes=workers) as pool:
        for res, msg in tqdm(pool.imap(run_culture, tasks), total=len(tasks),
                             desc="Cultures", unit="culture"):
            rows.extend(res)
            print(f"    {msg}")
    df = pd.DataFrame(rows)
    csv = os.path.join(base_dir, "sim1_raw_results.csv")
    df.to_csv(csv, index=False)
    print(f"[+] {len(df)} rows written to {csv}  ({time.time()-t0:.1f}s)")
    return df


def main():
    ap = argparse.ArgumentParser(description="Simulation 1: manipulation-intensity sweep "
                                             "(n=100, m=5, k=2,3,4).")
    global NUM_ITERATIONS
    ap.add_argument("--trials", type=int, default=NUM_ITERATIONS,
                    help=f"Monte Carlo trials per culture (default {NUM_ITERATIONS})")
    ap.add_argument("--skip-figures", action="store_true", help="data and tables only")
    ap.add_argument("--skip-simulation", action="store_true",
                    help="reuse the existing sim1_raw_results.csv")
    args = ap.parse_args()
    NUM_ITERATIONS = args.trials
    print("=" * 80)
    print("SIMULATION 1 (intensity sweep): median-based committee selection")
    sanity_check_design()
    sanity_check_profiles()
    sanity_check_median()
    sanity_check_shapley()
    base_dir, graphs_dir, tables_dir = setup_directories(r"D:\PYTHON\Project - Median")
    csv = os.path.join(base_dir, "sim1_raw_results.csv")
    if args.skip_simulation:
        if not os.path.exists(csv):
            raise SystemExit(f"--skip-simulation given but {csv} does not exist")
        df = pd.read_csv(csv)
        print(f"[*] Reusing {csv} ({len(df)} rows)")
    else:
        df = simulate(base_dir)
    audit_ok = audit_results(df, base_dir)
    export_tables(df, tables_dir)
    if not args.skip_figures:
        make_all_figures(df, graphs_dir)
    print("\n" + "=" * 80)
    print("SIMULATION 1 COMPLETED SUCCESSFULLY"
          + ("" if audit_ok else " (with audit warnings)"))
    print(f"All outputs are in: {base_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
