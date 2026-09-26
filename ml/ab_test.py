"""
A/B TEST ANALYSIS — is model B (XGBoost) really better than model A (Ridge)?

Each live bar was randomly served by A or B (see ml/scorer.py). Once actual volatility
is known we compare the absolute errors of the two groups:

  * Welch's t-test        — are the mean errors different beyond random noise?
  * Bootstrap 95% CI      — plausible range for (MAE_B - MAE_A); below 0 means B is better
  * minimum sample size   — don't decide on too little data (avoids "peeking" too early)

Decision rule: promote B only if p < 0.05 AND the whole CI is below zero.

Run:  python ml/ab_test.py
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_PER_GROUP = 100
ALPHA = 0.05


def compare(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 5000, seed: int = 42) -> dict:
    """Compare absolute errors of two groups. Returns stats and a decision."""
    err_a, err_b = np.asarray(err_a, float), np.asarray(err_b, float)
    rng = np.random.default_rng(seed)
    t_stat, p_value = stats.ttest_ind(err_b, err_a, equal_var=False)
    boot = [rng.choice(err_b, len(err_b)).mean() - rng.choice(err_a, len(err_a)).mean()
            for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    if min(len(err_a), len(err_b)) < MIN_PER_GROUP:
        decision = f"keep collecting (need {MIN_PER_GROUP}+ per group)"
    elif p_value < ALPHA and hi < 0:
        decision = "promote B: significantly lower error"
    elif p_value < ALPHA and lo > 0:
        decision = "keep A: B is significantly worse"
    else:
        decision = "no significant difference: keep A (simpler)"
    return {"n_a": len(err_a), "n_b": len(err_b), "mae_a": err_a.mean(), "mae_b": err_b.mean(),
            "diff_ci_low": lo, "diff_ci_high": hi, "p_value": float(p_value), "decision": decision}


def main():
    from common.db import read_sql, write_rows
    df = read_sql("SELECT variant, served_pred, baseline_pred, actual FROM predictions WHERE actual IS NOT NULL")
    if df.empty:
        raise SystemExit("No predictions with known outcomes yet. Keep the scorer running.")
    df["abs_err"] = (df["served_pred"] - df["actual"]).abs()
    res = compare(df.loc[df.variant == "A", "abs_err"], df.loc[df.variant == "B", "abs_err"])
    res["mae_baseline"] = float((df["baseline_pred"] - df["actual"]).abs().mean())

    print(f"A (Ridge):    n={res['n_a']}  MAE={res['mae_a']:.6f}")
    print(f"B (XGBoost):  n={res['n_b']}  MAE={res['mae_b']:.6f}")
    print(f"Baseline MAE: {res['mae_baseline']:.6f}")
    print(f"MAE_B - MAE_A 95% CI: [{res['diff_ci_low']:.6f}, {res['diff_ci_high']:.6f}]   p={res['p_value']:.4f}")
    print(f"Decision: {res['decision']}")

    write_rows("""INSERT INTO ab_results (n_a, n_b, mae_a, mae_b, mae_baseline, diff_ci_low,
                  diff_ci_high, p_value, decision) VALUES %s""",
               [(res["n_a"], res["n_b"], float(res["mae_a"]), float(res["mae_b"]), res["mae_baseline"],
                 float(res["diff_ci_low"]), float(res["diff_ci_high"]), res["p_value"], res["decision"])])


if __name__ == "__main__":
    main()
