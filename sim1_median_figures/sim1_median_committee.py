#!/usr/bin/env python3
"""
========================================================================================
SIMULATION 1: Median-Based Committee Selection
m = 5 candidates, k <= 4, total electorate = N_HONEST + N_MANIP
========================================================================================

Default configuration (user's revised script):
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
========================================================================================
"""

import os
import platform
import time
import math
import itertools
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
# 1. DIRECTORY CONFIGURATION
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


# ======================================================================================
# 2. SIMULATION PARAMETERS
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

INTEGER_MEDIAN = True


def smart_median(arr, axis=0):
    """Median that returns an integer for even-length inputs when INTEGER_MEDIAN=True.

    For even n the LOWER of the two middle order statistics is used, which is an
    integer for integer-valued input.
    """
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
# 3. VOTING RULES
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
# 4. PREFERENCE GENERATIVE MODELS (STRICT INTEGER SCORES)
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
# 5. MULTIPROCESSING WORKER
# ======================================================================================
def execute_task(args):
    cult_name, k, seed = args
    np.random.seed(seed)
    gen_fn = BEST_MODELS[cult_name]

    stats = {
        'Cutoff': {
            'Evaluative Voting': {'changed': 0, 'overlap': 0.0, 'success': 0},
            'k-Median Rule':     {'changed': 0, 'overlap': 0.0, 'success': 0},
            'k-Median-Shapley':  {'changed': 0, 'overlap': 0.0, 'success': 0},
        },
        'Bottom': {
            'Evaluative Voting': {'changed': 0, 'overlap': 0.0, 'success': 0},
            'k-Median Rule':     {'changed': 0, 'overlap': 0.0, 'success': 0},
            'k-Median-Shapley':  {'changed': 0, 'overlap': 0.0, 'success': 0},
        },
    }

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

        for t_type in ['Cutoff', 'Bottom']:
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
    for t_type in ['Cutoff', 'Bottom']:
        for r_name in ['Evaluative Voting', 'k-Median Rule', 'k-Median-Shapley']:
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
# 6. SANITY CHECKS
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
# 7. MAIN
# ======================================================================================
def main():
    print("=" * 80)
    print(f"SIMULATION 1: m=5, k<=4, total n={TOTAL_N} "
          f"({N_HONEST} truthful + {N_MANIP} manipulators)")
    print(f"Trials per cell = {NUM_ITERATIONS}  |  INTEGER_MEDIAN = {INTEGER_MEDIAN}")
    print(f"Rank -> Score map = {RANK_TO_SCORE.tolist()}")
    print("=" * 80)

    sanity_check_profiles()
    sanity_check_median()

    base_dir, graphs_dir, tables_dir = setup_directories(r"D:\PYTHON\Project - Median")

    tasks = []
    seed = 2026
    for cult_name in BEST_MODELS:
        for k in COMMITTEE_SIZES:
            tasks.append((cult_name, k, seed))
            seed += 1

    num_workers = cpu_count()
    total_tasks = len(tasks)
    print(f"[*] {total_tasks} batches across {num_workers} CPU cores...\n")
    t0 = time.time()

    raw_results = []
    with Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap(execute_task, tasks),
                        total=total_tasks,
                        desc="Simulating Elections", unit="batch"):
            raw_results.append(res)

    flat = [x for sub in raw_results for x in sub]
    df = pd.DataFrame(flat)
    df.to_csv(os.path.join(base_dir, "sim1_raw_results.csv"), index=False)
    print(f"\n[+] Raw results saved ({time.time()-t0:.2f}s).")

    # ---- Summary tables (data only; paper-style figures are rendered separately) ----
    print("[*] Exporting summary tables...")
    for t_type in ['Cutoff', 'Bottom']:
        sub = df[df['Target'] == t_type]
        for metric, nd in [('Change_Rate', 2), ('Success_Rate', 2), ('Avg_Overlap', 3)]:
            sub.pivot_table(index=['Culture', 'Rule'], columns='k', values=metric).round(nd).to_csv(
                os.path.join(tables_dir, f"sim1_{metric.lower()}_{t_type.lower()}.csv"))

    agg = (df.groupby(['Target', 'Rule'])[['Change_Rate', 'Success_Rate', 'Avg_Overlap']]
             .mean().round(3))
    agg.to_csv(os.path.join(tables_dir, "sim1_aggregate_summary.csv"))
    print(agg)

    print("\n" + "=" * 80)
    print("SIMULATION 1 DATA GENERATION COMPLETED SUCCESSFULLY")
    print(f"Run:  python make_figures.py   (renders the paper-style figures)")
    print("=" * 80)


if __name__ == '__main__':
    main()
