"""Variance extrapolation of the vDBF trajectories, vdbf-paper protocol.

Same rule as corrected_energy_vs_corrected_variance.ipynb in
github.com/chinmay-1008/data-vdbf1, applied to each R and basis:

  * one point per macro-iteration (iterations 1..n, the "*" rows of the log):
        corrected energy   = E   - accumulated truncation error
        corrected variance = Var - accumulated variance error
  * for every final window of num_take >= MIN_FIT_POINTS points, fit
    corrected E against corrected Var with a line and a quadratic, and score
        cost = 0.5 |E0_linear - E0_quadratic| + (1 - R^2_linear)
  * keep the minimum-cost window; E0 = average of the two intercepts,
    uncertainty = half their difference.

Writes results/extrapolation_<basis>.csv (one row per R) and
results/extrapolation_<basis>_scan.csv (every candidate window).  Points
with fewer than MIN_FIT_POINTS iterations get NaN and a note.

    python extrapolate.py [--bases mo,lmo,no,lno] [--min-points 10]
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

MIN_FIT_POINTS = 10
BASES = ("mo", "lmo", "no", "lno")


def fit_polynomials(x, y):
    linear = np.polyfit(x, y, 1)
    quadratic = np.polyfit(x, y, 2)
    residual = float(np.sum((y - np.polyval(linear, x)) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 1.0
    return {"linear_slope": float(linear[0]),
            "E0_linear": float(linear[-1]), "E0_quadratic": float(quadratic[-1]),
            "E0_average": float(0.5 * (linear[-1] + quadratic[-1])),
            "fit_half_difference": float(0.5 * abs(linear[-1] - quadratic[-1])),
            "R_squared": float(r2)}


def corrected_trajectory(record):
    h = record["history"]
    energy = np.asarray(h["energies_per_grad"], dtype=float)
    err = np.asarray(h["accumulated_error_per_grad"], dtype=float)
    var = np.asarray(h["variance_per_grad"], dtype=float)
    var_err = np.asarray(h["accumulated_var_error_per_grad"], dtype=float)
    # Entry 0 is the starting reference; the paper fits iterations 1..n only.
    x, y = (var - var_err)[1:], (energy - err)[1:]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def select_adaptive_tail(x, y, min_points):
    scan = []
    for num_take in range(min_points, len(x) + 1):
        fit = fit_polynomials(x[-num_take:], y[-num_take:])
        cost = fit["fit_half_difference"] + (1.0 - fit["R_squared"])
        scan.append({"num_take": num_take, "selection_cost": cost, **fit})
    best = min(scan, key=lambda row: row["selection_cost"])
    return best, scan


def extrapolate_basis(datadir, basis, min_points):
    rows, scan_rows = [], []
    for path in sorted(glob.glob(os.path.join(datadir, "R_*", f"dbf_{basis}.json"))):
        with open(path) as fh:
            record = json.load(fh)
        x, y = corrected_trajectory(record)
        base = {"R": record["R"], "E_FCI": record["E_FCI"],
                "E_DBF": record["E_DBF"],
                "E_DBF_corrected": record["E_DBF_minus_error"],
                "candidate_points": len(x)}
        if len(x) < min_points:
            rows.append({**base, "n_points": len(x), "start_iteration": None,
                         "E0_linear": np.nan, "E0_quadratic": np.nan,
                         "E0_average": np.nan, "fit_half_difference": np.nan,
                         "R_squared": np.nan, "selection_cost": np.nan,
                         "error_vs_FCI": np.nan,
                         "note": f"only {len(x)} iterations (< {min_points})"})
            continue
        best, scan = select_adaptive_tail(x, y, min_points)
        rows.append({**base, "n_points": best["num_take"],
                     "start_iteration": len(x) - best["num_take"] + 1,
                     **{k: best[k] for k in ("E0_linear", "E0_quadratic",
                                             "E0_average", "fit_half_difference",
                                             "R_squared", "selection_cost")},
                     "error_vs_FCI": best["E0_average"] - record["E_FCI"],
                     "note": ""})
        scan_rows += [{"R": record["R"], **s} for s in scan]
    return rows, scan_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bases", default=",".join(BASES))
    p.add_argument("--datadir", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--min-points", type=int, default=MIN_FIT_POINTS)
    args = p.parse_args()
    for basis in args.bases.split(","):
        rows, scan = extrapolate_basis(args.datadir, basis, args.min_points)
        if not rows:
            print(f"{basis}: no dbf_{basis}.json files yet")
            continue
        write_csv(os.path.join(args.results, f"extrapolation_{basis}.csv"), rows)
        write_csv(os.path.join(args.results, f"extrapolation_{basis}_scan.csv"), scan)
        print(f"\n{basis}:  R    n_pts  start   E0 (avg)          +/-        E0-E_FCI (mHa)")
        for r in rows:
            if r["note"]:
                print(f"     {r['R']:.2f}  {r['note']}")
            else:
                print(f"     {r['R']:.2f}  {r['n_points']:5d}  {r['start_iteration']:5d}  "
                      f"{r['E0_average']:.10f}  {r['fit_half_difference']:.2e}  "
                      f"{1000 * r['error_vs_FCI']:+9.3f}")


if __name__ == "__main__":
    main()
