#!/usr/bin/env python3
"""Diagnostic: does CR == SR hold for the k-Median rule under the bottom attack?

Compares the lower-median convention (INTEGER_MEDIAN=True) with the standard
np.median on the same attacking profiles, and decomposes the failure:
"committee changed but the target did NOT enter the committee".
"""
import numpy as np
import sim1_median_committee as S

TRIALS = 2000


def run(culture, k, seed, lower_median, n_manip=20, zero_on_others=True):
    np.random.seed(seed)
    gen = S.BEST_MODELS[culture]
    old = S.INTEGER_MEDIAN
    S.INTEGER_MEDIAN = lower_median
    ch = sr = both = 0
    try:
        for _ in range(TRIALS):
            s_h = gen(S.N_HONEST, S.M)
            rules = [S.utilitarian_rule, S.median_rule, S.shapley_median_rule]
            fns = []
            for fn in rules:
                r, c, _ = fn(s_h, k)
                fns.append((fn, r, c))
            for fn, r_h, c_h in fns:
                target = r_h[S.M - 1]                        # bottom target
                m = np.zeros((n_manip, S.M), dtype=int)
                m[:, target] = S.SCALE_MAX
                if not zero_on_others:                       # control: neutral ballot elsewhere
                    m = np.full((n_manip, S.M), np.nan)
                    m[:, target] = S.SCALE_MAX
                    s_m = np.vstack([s_h, np.where(np.isnan(m), S.SCALE_MAX, m)])  # placeholder
                else:
                    s_m = np.vstack([s_h, m])
                _, c_post, _ = fn(s_m, k)
                changed = (c_h != c_post)
                success = (target in c_post)
                if fn is S.median_rule:
                    ch += changed
                    sr += success
                    both += (changed and not success)
    finally:
        S.INTEGER_MEDIAN = old
    return ch / TRIALS * 100, sr / TRIALS * 100, both / TRIALS * 100


print(f"Bottom attack, k-Median rule, {TRIALS} trials per cell")
print(f"{'culture':36s} {'k':>2s} {'conv':>6s} {'change%':>8s} {'success%':>9s} {'gap':>7s} {'chg-no-target%':>15s}")
for culture in ['Mallows (phi=0.7)', 'Polya Urn (alpha=0.2)', 'Impartial Anonymous Culture (IAC)']:
    for k in [1, 2, 3]:
        seed = 2026 + list(S.BEST_MODELS).index(culture) * 4 + (k - 1)
        for lower, name in [(True, 'lower'), (False, 'std')]:
            ch, sr, both = run(culture, k, seed, lower)
            print(f"{culture:36s} {k:2d} {name:>6s} {ch:8.2f} {sr:9.2f} {ch-sr:7.2f} {both:15.2f}")

# also: evaluative rule, to confirm the exact identity there
print("\nControl — Evaluative Voting (bottom attack):")
for k in [1, 2, 3]:
    np.random.seed(2026 + k - 1)
    gen = S.BEST_MODELS['Mallows (phi=0.7)']
    ch = sr = 0
    for _ in range(TRIALS):
        s_h = gen(S.N_HONEST, S.M)
        r_h, c_h, _ = S.utilitarian_rule(s_h, k)
        t = r_h[S.M - 1]
        m = np.zeros((20, S.M), dtype=int); m[:, t] = S.SCALE_MAX
        _, c_post, _ = S.utilitarian_rule(np.vstack([s_h, m]), k)
        ch += (c_h != c_post); sr += (t in c_post)
    print(f"  Mallows k={k}: change={ch/TRIALS*100:.2f}%  success={sr/TRIALS*100:.2f}%  gap={(ch-sr)/TRIALS*100:.2f}pp")
