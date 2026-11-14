"""Gather the PES runs into pes_summary.csv and pes.png.  Run after the jobs
finish, on the cluster or after copying the bundle back:

    python collect_pes.py            # needs numpy and matplotlib
    COEFF=1e-5 python collect_pes.py # the 1e-5 sweep: pes_summary_c1e-5.csv, pes_c1e-5.png

Per R and variant: the reported corrected energy (E - err, of H + λ S^2 for
penalty runs), the tail-fit extrapolation E0 (extrapolate.py, the paper's
adaptive window), and from the exact replay of the final circuit the true <H>,
<S^2> and fidelity with FCI.  Errors are against each point's own frozen-core
FCI (the core orbitals differ by tree; the FCIs by a few uHa).  Where a
reference is already a singlet (empty prep circuit, short R) the <basis>_prep
row reuses <basis>_bare, which is the same run.  Missing runs are left blank.

Figure: columns H, H + λ S^2, singlet-projected start; rows true <H> - FCI,
E0 - FCI, true <S^2>; one colour per basis, canonical/delocalized (MO, NO,
UNO) dashed and localized (LMO, LNO) solid, so localization shows up as the
gap between a dashed line and the solid one of the same family.  Frozen-core
UCCSD(T) (results/fc_cc.csv) in grey for scale."""
import csv
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N2 = os.path.join(HERE, "DBF.jl", "code", "n2_pes")
FC = os.path.join(N2, "frozen_core")
sys.path.insert(0, N2)
from extrapolate import MIN_FIT_POINTS, select_adaptive_tail  # noqa: E402

cfg = dict(line.split("#")[0].strip().split("=", 1) for line in open(os.path.join(HERE, "config.sh"))
           if "=" in line.split("#")[0] and not line.startswith(("export", "JULIA", "EXTRA", "DBF_ENV", " ", "[")))
LAM, PREP = cfg["LAMBDA"], cfg["PREP"]
COEFF = os.environ.get("COEFF") or re.search(r"^COEFF=\$\{COEFF:-([^}]+)\}", open(os.path.join(HERE, "config.sh")).read(), re.M).group(1)
TAG = "" if COEFF == "1e-5" else f"_c{COEFF}"           # run stems, as in run_one.sh
LTAG = "" if COEFF == "1e-6" else f"_c{COEFF}"          # replay labels and output files
TREE = {"mo": "data_uhfcore", "lmo": "data_uhfcore", "no": "data_nocore", "lno": "data_nocore", "uno": "data_unocore"}
BASES = [("mo", "MO", "#2a78d6", "--"), ("lmo", "LMO", "#2a78d6", "-"),
         ("no", "NO", "#eb6834", "--"), ("lno", "LNO", "#eb6834", "-"),
         ("uno", "UNO", "#1baf7a", "-")]
KINDS = [("bare", "H", ""), ("s2pen", f"H + {LAM} S²", f"_s2pen{LAM}"), ("prep", "singlet-projected start, H", f"_prep{PREP}")]
MHA = 1e3


def stem(b, kind):
    s = dict(bare=f"dbf_{b}", s2pen=f"dbf_{b}_s2pen{LAM}", prep=f"dbf_{b}_prep{PREP}")[kind]
    return s + TAG


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


def empty_prep(b, R):
    log = os.path.join(FC, "results", "logs", f"prep_{b}_{R}.out")
    return os.path.isfile(log) and any(l.startswith("prep circuit: 0 rotations") for l in open(log))


def one(R, b, kind):
    name = f"{b}_{kind}"
    run = load(os.path.join(FC, TREE[b], f"R_{R}", f"{stem(b, kind)}.json"))
    if run is None:
        return None
    rep = load(os.path.join(FC, "replay_results", f"{b}_{R}_{name}{LTAG}.json"))
    fci = run["E_FCI"]
    h = run["history"]
    E = np.array(h["energies_per_grad"]) - np.array(h["accumulated_error_per_grad"])
    V = np.array(h["variance_per_grad"]) - np.array(h["accumulated_var_error_per_grad"])
    row = dict(R=float(R), variant=name, E_FCI=fci, E_ref=run["E_ref"], S2_ref=run["S2_ref"],
               n_iter=run["n_iterations"], wall_h=run["wall_time_s"] / 3600, final_terms=run["final_terms"],
               reported=E[-1], reported_var=V[-1], E0=np.nan, E0_unc=np.nan, fit_window=0,
               true_E=np.nan, true_var=np.nan, true_S2=np.nan, fidelity=np.nan)
    if len(E) - 1 >= MIN_FIT_POINTS:
        best, _ = select_adaptive_tail(V[1:], E[1:], MIN_FIT_POINTS)
        row.update(E0=best["E0_average"], E0_unc=best["fit_half_difference"], fit_window=best["num_take"])
    if rep is not None:
        last = rep["rows"][-1]
        row.update(true_E=last["E"], true_var=last["var"], true_S2=last["S2"], fidelity=last["fidelity"])
    return row


def main():
    Rs = sorted(d[2:] for d in os.listdir(os.path.join(FC, "data_unocore")) if d.startswith("R_"))
    rows = {}
    for R in Rs:
        for b, *_ in BASES:
            for kind, *_ in KINDS:
                if kind == "prep" and b == "uno":
                    continue
                r = one(R, b, kind)
                if r is None and kind == "prep" and (R, f"{b}_bare") in rows and empty_prep(b, R):
                    r = dict(rows[(R, f"{b}_bare")], variant=f"{b}_prep")   # empty prep circuit: same run
                if r is not None:
                    rows[(R, r["variant"])] = r
    if not rows:
        sys.exit("no results yet")
    allrows = sorted(rows.values(), key=lambda r: (r["variant"], r["R"]))
    with open(os.path.join(HERE, f"pes_summary{LTAG}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)

    names = [f"{b}_{k}" for k, *_ in KINDS for b, *_ in BASES if not (k == "prep" and b == "uno")]
    for key, title in (("true_E", "true <H> - FCI"), ("E0", "E0 - FCI")):
        print(f"\n{title}, mHa")
        print(f"{'R':>5} " + " ".join(f"{n:>10}" for n in names))
        for R in Rs:
            cells = [rows.get((R, n)) for n in names]
            print(f"{R:>5} " + " ".join(" " * 10 if c is None else f"{(c[key] - c['E_FCI']) * MHA:+10.3f}" for c in cells))
    print(f"\n{len(rows)} of {len(Rs) * len(names)} (R, variant) results")

    cc = {}
    p = os.path.join(FC, "results", "fc_cc.csv")
    if os.path.isfile(p):
        cc = {round(float(r["R"]), 2): float(r["E_UCCSD_T"]) for r in csv.DictReader(open(p))}
    fig, ax = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    for j, (kind, klab, _) in enumerate(KINDS):
        for b, blab, col, ls in BASES:
            sel = sorted((r for (_, n), r in rows.items() if n == f"{b}_{kind}"), key=lambda r: r["R"])
            if not sel:
                continue
            x = [r["R"] for r in sel]
            for i, y in enumerate(([(r["true_E"] - r["E_FCI"]) * MHA for r in sel],
                                   [(r["E0"] - r["E_FCI"]) * MHA for r in sel],
                                   [r["true_S2"] for r in sel])):
                ax[i, j].plot(x, y, ls, marker="o", color=col, ms=3.2, lw=1.3, label=blab)
        if cc:
            ref = sorted((r["R"], (cc[round(r["R"], 2)] - r["E_FCI"]) * MHA)
                         for (_, n), r in rows.items() if n == "uno_s2pen" and round(r["R"], 2) in cc)
            for i in (0, 1):
                if ref:
                    ax[i, j].plot(*zip(*ref), color="#6b6a64", lw=1.0, ls=":", label="UCCSD(T)")
        ax[0, j].set_title(klab, fontsize=11, loc="left")
    for j in range(3):
        for i in (0, 1):
            ax[i, j].set_yscale("symlog", linthresh=0.1, linscale=0.6)
            ax[i, j].axhspan(-1.6, 1.6, color="#efeee9", zorder=0)
            ax[i, j].axhline(0, color="#1f1f1e", lw=0.8, ls=":")
        ax[2, j].set_yscale("symlog", linthresh=1e-3)
        ax[2, j].set_xlabel("R, Å")
        for i in range(3):
            ax[i, j].grid(color="#e4e3de", lw=0.5)
    ax[0, 0].set_ylabel("true ⟨H⟩ of final circuit − FCI, mHa")
    ax[1, 0].set_ylabel("E0 (variance extrapolation) − FCI, mHa")
    ax[2, 0].set_ylabel("true ⟨S²⟩")
    ax[0, 0].legend(frameon=False, fontsize=8.5, ncol=2)
    fig.suptitle(f"N₂ STO-3G, frozen core (16 qubits), vDBF coeff {COEFF}: dashed = canonical/delocalized, "
                 "solid = localized (and UNO)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"pes{LTAG}.png"), dpi=130)
    print(f"wrote pes_summary{LTAG}.csv, pes{LTAG}.png")


if __name__ == "__main__":
    main()
