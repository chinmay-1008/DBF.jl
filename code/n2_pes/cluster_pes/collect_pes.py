"""Gather the PES runs into pes_summary.csv and pes.png.  Run after the jobs
finish, on the cluster or after copying the bundle back:

    python collect_pes.py            # needs numpy and matplotlib

Per R and variant: the reported corrected energy (E - err, of H + λ S^2 for
penalty runs), the tail-fit extrapolation E0 (extrapolate.py, the paper's
adaptive window), and from the exact replay of the final circuit the true <H>,
<S^2> and fidelity with FCI.  Energies are total energies (frozen core,
N2 STO-3G); errors are against each point's own frozen-core FCI.  Where the
reference is already a singlet (no prep circuit, R < 1.2 A) the lno_prep curve
reuses lno_bare, which is the same run.  Missing runs are left blank.

Figure: (1) total energy with RHF, UHF, UCCSD(T) and FCI; (2) true <H> - FCI;
(3) E0 - FCI; (4) true <S^2>."""
import csv
import json
import os
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
           if "=" in line.split("#")[0] and not line.startswith(("export", "JULIA", "EXTRA", " ")))
COEFF, LAM, PREP = cfg["COEFF"], cfg["LAMBDA"], cfg["PREP"]
TAG = "" if COEFF == "1e-5" else f"_c{COEFF}"
VARIANTS = [  # name, tree, basis, run stem, label, colour
    ("lno_bare", "data_nocore", "lno", f"dbf_lno{TAG}", "LNO, UHF-like start, H", "#2a78d6"),
    ("lno_s2pen", "data_nocore", "lno", f"dbf_lno_s2pen{LAM}{TAG}", f"LNO, H + {LAM} S²", "#eb6834"),
    ("lno_prep", "data_nocore", "lno", f"dbf_lno_prep{PREP}{TAG}", "LNO, singlet-projected start, H", "#1baf7a"),
    ("uno_s2pen", "data_unocore", "uno", f"dbf_uno_s2pen{LAM}{TAG}", f"UNO, H + {LAM} S²", "#c03b8a"),
]
MHA = 1e3


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


def empty_prep(R):
    log = os.path.join(FC, "results", "logs", f"prepare_{R}.out")
    return os.path.isfile(log) and any(l.startswith("prep circuit: 0 rotations") for l in open(log))


def one(R, name, tree, basis, stem):
    run = load(os.path.join(FC, tree, f"R_{R}", f"{stem}.json"))
    rep = load(os.path.join(FC, "replay_results", f"{basis}_{R}_{name}.json"))
    if run is None:
        return None
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
    Rs = sorted(d[2:] for d in os.listdir(os.path.join(FC, "data_nocore")) if d.startswith("R_"))
    rows = []
    for R in Rs:
        got = {}
        for name, tree, basis, stem, _, _ in VARIANTS:
            r = one(R, name, tree, basis, stem)
            if r is None and name == "lno_prep" and "lno_bare" in got and empty_prep(R):
                r = dict(got["lno_bare"], variant="lno_prep")   # empty prep circuit: same run
            if r is not None:
                got[name] = r
                rows.append(r)
    if not rows:
        sys.exit("no results yet")
    cols = list(rows[0].keys())
    with open(os.path.join(HERE, "pes_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{'R':>5} " + " ".join(f"{v[0]:>22}" for v in VARIANTS) + "   (true <H> - FCI | E0 - FCI, mHa)")
    for R in Rs:
        cells = []
        for v in VARIANTS:
            r = next((x for x in rows if x["R"] == float(R) and x["variant"] == v[0]), None)
            cells.append(" " * 22 if r is None else
                         f"{(r['true_E'] - r['E_FCI']) * MHA:+9.3f} | {(r['E0'] - r['E_FCI']) * MHA:+9.3f}")
        print(f"{R:>5} " + " ".join(f"{c:>22}" for c in cells))

    ref = {}
    with open(os.path.join(N2, "results", "references.csv")) as fh:
        for r in csv.DictReader(fh):
            ref[float(r["R"])] = r
    fig, ax = plt.subplots(4, 1, figsize=(8, 15), sharex=True)
    xs = sorted(ref)
    for key, lab, st in (("E_RHF", "RHF", ":"), ("E_UHF", "UHF", "--"), ("E_UCCSD_T", "UCCSD(T)", "-."),
                         ("E_FCI", "FCI (all electrons)", "-")):
        ax[0].plot(xs, [float(ref[x][key]) for x in xs], color="#6b6a64", ls=st, lw=1.2, label=lab)
    for name, _, _, _, lab, col in VARIANTS:
        sel = sorted((r for r in rows if r["variant"] == name), key=lambda r: r["R"])
        if not sel:
            continue
        R_ = [r["R"] for r in sel]
        ax[0].plot(R_, [r["true_E"] for r in sel], "o-", color=col, ms=3.5, lw=1.2, label=f"vDBF {lab} (true ⟨H⟩)")
        ax[1].plot(R_, [(r["true_E"] - r["E_FCI"]) * MHA for r in sel], "o-", color=col, ms=3.5, lw=1.2, label=lab)
        ax[2].plot(R_, [(r["E0"] - r["E_FCI"]) * MHA for r in sel], "o-", color=col, ms=3.5, lw=1.2, label=lab)
        ax[3].plot(R_, [r["true_S2"] for r in sel], "o-", color=col, ms=3.5, lw=1.2, label=lab)
    ax[0].set_ylim(min(float(ref[x]["E_FCI"]) for x in xs) - 0.05, max(float(ref[x]["E_UHF"]) for x in xs) + 0.05)
    ax[0].set_ylabel("E, Ha")
    for a, t in ((ax[1], "true ⟨H⟩ of the final circuit − FCI, mHa"), (ax[2], "tail-fit E0 − FCI, mHa")):
        a.set_yscale("symlog", linthresh=0.1, linscale=0.5)
        a.axhspan(-1.6, 1.6, color="#efeee9", zorder=0)
        a.axhline(0, color="#1f1f1e", lw=0.8, ls=":")
        a.set_ylabel(t)
    ax[3].set_ylabel("true ⟨S²⟩")
    ax[3].set_yscale("symlog", linthresh=1e-3)
    ax[3].set_xlabel("R, Å")
    for a in ax:
        a.grid(color="#e4e3de", lw=0.5)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"N₂ STO-3G, frozen core (16 qubits), vDBF coeff {COEFF}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pes.png"), dpi=140)
    print("wrote pes_summary.csv, pes.png")


if __name__ == "__main__":
    main()
