import csv
import os
from collections import defaultdict
import numpy as np

# Scenario name strings as they appear in the sequential combined CSV
DRAFT_BASELINE = "Llama2 1.1b - Baseline"
TARGET_BASELINE = "Llama2 7b - Baseline"

def load_beta_from_csv(csv_path):
    """
    Computes beta (T_draft / T_target) per N from sequential baseline rows.
    Uses mean_time_per_token averaged across all runs for each N.
    Returns (beta_overall, {N: beta_n}).
    """
    draft_times = defaultdict(list)
    target_times = defaultdict(list)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["Scenario"]
            n = int(row["N"])
            mttp = float(row["mean_time_per_token"])
            if scenario == DRAFT_BASELINE:
                draft_times[n].append(mttp)
            elif scenario == TARGET_BASELINE:
                target_times[n].append(mttp)

    beta_per_n = {}
    for n in sorted(draft_times):
        if n in target_times:
            avg_draft = float(np.mean(draft_times[n]))
            avg_target = float(np.mean(target_times[n]))
            beta_per_n[n] = avg_draft / avg_target

    # Mean beta across all N values
    beta_overall = float(np.mean(list(beta_per_n.values())))
    return beta_overall, beta_per_n

def expected_output_tokens(alpha, k):
    """
    Expected number of tokens produced per speculative round.
    Derived from the geometric series over the acceptance distribution.
    alpha: acceptance rate
    k: speculation depth
    """
    if alpha >= 1.0:
        return k + 1  # all k drafts accepted plus the bonus target token
    return (1 - alpha ** (k + 1)) / (1 - alpha)

def speedup(alpha, k, beta):
    """
    Theoretical speedup of speculative decoding over autoregressive target decoding.
    Numerator: expected output tokens per round.
    Denominator: expected wall time per round normalized to one target step.
    alpha: acceptance rate
    k: speculation depth
    beta: T_draft / T_target (cost ratio)
    """
    return expected_output_tokens(alpha, k) / ((1 - alpha) * (k * beta + 1))

def optimal_k(alpha, beta, k_max=20):
    """
    Grid search over k in [1, k_max] to find the depth that maximizes speedup.
    alpha: acceptance rate
    beta: T_draft / T_target (cost ratio)
    """
    best_k, best_s = 1, 0
    for k in range(1, k_max + 1):
        s = speedup(alpha, k, beta)
        if s > best_s:
            best_s, best_k = s, k
    return best_k, best_s

def speedup_table(alphas, ks, beta):
    # Print a 2-D grid of speedup values: rows = alpha, columns = k
    header = f"{'alpha/k':>10}" + "".join(f"  k={k:2d}" for k in ks)
    print(header)
    print("-" * len(header))
    for alpha in alphas:
        row = f"{alpha:>10.2f}" + "".join(f"  {speedup(alpha, k, beta):5.2f}" for k in ks)
        print(row)

def diminishing_returns(alpha, beta, k_max=20):
    # Show marginal speedup gain for each additional draft token at fixed alpha
    print(f"\nDiminishing returns (alpha={alpha}, beta={beta:.3f}):")
    print(f"{'k':>4}  {'Speedup':>8}  {'Marginal gain':>14}")
    prev = 0
    for k in range(1, k_max + 1):
        s = speedup(alpha, k, beta)
        print(f"{k:>4}  {s:>8.4f}  {s - prev:>+14.4f}")
        prev = s

def validate_against_empirical(empirical_results):
    """
    Compare theoretical speedup to measured results.
    empirical_results: list of dicts with keys:
        label, tokens_per_second, acceptance_rate, total_draft_tokens,
        output_tokens, total_draft_time, total_target_time
    """
    print(f"\n{'Label':<35} {'T.Speedup':>10} {'E.Speedup':>10} {'alpha':>7} {'k_eff':>6}")
    print("-" * 72)

    # First entry must be the 7B baseline — all speedups are relative to it
    baseline_tps = empirical_results[0]["tokens_per_second"]

    for r in empirical_results[1:]:
        alpha = r["acceptance_rate"]
        output = r["output_tokens"]
        drafts = r["total_draft_tokens"]
        k_eff = drafts / max(1, output)  # average draft tokens produced per output token

        t_draft = r.get("total_draft_time", 0)
        t_target = r.get("total_target_time", 0)
        rounds = output - r.get("accepted_tokens", 0)  # approximate number of verification rounds
        beta = (t_draft / max(1, drafts)) / (t_target / max(1, rounds))

        t_speedup = speedup(alpha, round(k_eff), beta)
        e_speedup = r["tokens_per_second"] / baseline_tps

        print(f"{r['label']:<35} {t_speedup:>10.3f} {e_speedup:>10.3f} {alpha:>7.3f} {k_eff:>6.1f}")

def beta_at_n(n, beta, target_uses_kv=False, draft_uses_kv=True):
    # Scale beta by relative attention cost: target grows O(n), draft is O(1) with KV cache
    target_scale = n
    draft_scale = 1 if draft_uses_kv else n
    return beta * (draft_scale / target_scale)

def speedup_vs_n(alpha, k, beta, ns):
    # Return (n, speedup) pairs showing how speedup changes with sequence length
    return [(n, speedup(alpha, k, beta_at_n(n, beta))) for n in ns]


if __name__ == "__main__":
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "sequential", "new data", "sequential combined.csv"
    )
    beta, beta_per_n = load_beta_from_csv(csv_path)

    print("=" * 60)
    print("BETA FROM EMPIRICAL BASELINE DATA")
    print("  beta = mean_time_per_token(draft) / mean_time_per_token(target)")
    print("=" * 60)
    for n, b in beta_per_n.items():
        print(f"  N={n:>4}  beta={b:.4f}")
    print(f"  Overall beta (mean across N) = {beta:.4f}")

    print("\n" + "=" * 60)
    print("Speculative Speedup: k vs alpha")
    print(f"beta (T_draft/T_target) = {beta:.4f}")
    print("Result is speedup ratio of speculative decoding vs target model")
    print("=" * 60)
    alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ks = [2, 3, 4, 5, 6, 8, 10]
    speedup_table(alphas, ks, beta)

    print("\n" + "=" * 60)
    print("OPTIMAL k FOR EACH ACCEPTANCE RATE")
    print("=" * 60)
    for alpha in alphas:
        k_opt, s_opt = optimal_k(alpha, beta)
        print(f"  alpha={alpha:.1f}  ->  optimal k={k_opt}  (speedup={s_opt:.3f}x)")

    diminishing_returns(alpha=0.6, beta=beta)

    print("\n" + "=" * 60)
    print("SPEEDUP VS SEQUENCE LENGTH N")
    print("(alpha=0.6, k=5, empirical beta per N)")
    print("=" * 60)
    empirical_betas = sorted(beta_per_n.items())
    print(f"{'N':>6}  {'beta':>8}  {'Speedup':>10}")
    print("-" * 30)
    for n, b in empirical_betas:
        s = speedup(0.6, 5, b)
        print(f"{n:>6}  {b:>8.4f}  {s:>10.3f}x")

    print("\n" + "=" * 60)
    print("SPEEDUP VS N ACROSS ACCEPTANCE RATES (k=5)")
    print("=" * 60)
    header = f"{'alpha':>7}" + "".join(f"  N={n:>3}" for n, _ in empirical_betas)
    print(header)
    print("-" * len(header))
    for alpha in [0.4, 0.5, 0.6, 0.7, 0.8]:
        row = f"{alpha:>7.1f}" + "".join(f"  {speedup(alpha, 5, b):>6.3f}" for _, b in empirical_betas)
        print(row)
