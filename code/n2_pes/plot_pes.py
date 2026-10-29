"""N2 curve: reference methods vs vDBF in each orbital basis.

Reads results/references.csv (n2_pes.py), results/dbf_<basis>.csv
(run_dbf_pes.jl) and results/extrapolation_<basis>.csv (extrapolate.py),
whichever exist, and writes results/n2_pes.png / .pdf:

    top     E(R) for every method (vDBF: final energy solid, variance-
            extrapolated E0 dashed with its uncertainty band)
    middle  E - E_FCI in mHa (log scale), chemical accuracy marked
    bottom  <S^2> of UHF and FCI

It also prints the non-parallelity error (max - min of E - E_FCI over R).
CCSD points that did not converge are drawn as open markers.

    python plot_pes.py [--results results]
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HARTREE_TO_MHA = 1000.0
CHEMICAL_ACCURACY_MHA = 1.6
BASES = ("mo", "lmo", "no", "lno")


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        try:
            out[key] = np.array([float(v) for v in values])
        except ValueError:
            out[key] = np.array([v == "True" or v == "true" for v in values])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()

    ref = read_csv(os.path.join(args.results, "references.csv"))
    R, fci = ref["R"], ref["E_FCI"]

    # (label, R, E, colour, linestyle, converged-mask)
    curves = [
        ("RHF", R, ref["E_RHF"], "0.55", "--", None),
        ("UHF", R, ref["E_UHF"], "0.25", "--", None),
        ("RCCSD", R, ref["E_RCCSD"], "tab:orange", ":", ref["RCCSD_converged"]),
        ("RCCSD(T)", R, ref["E_RCCSD_T"], "tab:orange", "-", ref["RCCSD_converged"]),
        ("UCCSD", R, ref["E_UCCSD"], "tab:green", ":", ref["UCCSD_converged"]),
        ("UCCSD(T)", R, ref["E_UCCSD_T"], "tab:green", "-", ref["UCCSD_converged"]),
    ]
    dbf_colours = {"mo": "tab:blue", "lmo": "tab:cyan", "no": "tab:red",
                   "lno": "tab:purple"}
    dbf, bands = {}, []
    for basis in BASES:
        path = os.path.join(args.results, f"dbf_{basis}.csv")
        if os.path.exists(path):
            dbf[basis] = d = read_csv(path)
            curves.append((f"vDBF {basis.upper()}", d["R"], d["E_DBF"],
                           dbf_colours[basis], "-", None))
        path = os.path.join(args.results, f"extrapolation_{basis}.csv")
        if os.path.exists(path):
            x = read_csv(path)
            ok = np.isfinite(x["E0_average"])
            curves.append((f"vDBF {basis.upper()} extrap.", x["R"][ok],
                           x["E0_average"][ok], dbf_colours[basis], "--", None))
            bands.append((x["R"][ok], x["E0_average"][ok],
                          x["fit_half_difference"][ok], dbf_colours[basis]))

    fig, (ax_e, ax_err, ax_s2) = plt.subplots(
        3, 1, figsize=(7.5, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1.4]})

    ax_e.plot(R, fci, "k-", lw=2.2, label="FCI", zorder=5)
    for label, r, e, colour, ls, conv in curves:
        ax_e.plot(r, e, color=colour, ls=ls, lw=1.4, label=label)
        fci_at = np.interp(r, R, fci)
        err = np.abs(e - fci_at) * HARTREE_TO_MHA
        ax_err.semilogy(r, np.maximum(err, 1e-6), color=colour, ls=ls, lw=1.4,
                        label=label)
        if conv is not None and (~conv).any():
            for ax, y in ((ax_e, e), (ax_err, np.maximum(err, 1e-6))):
                ax.plot(r[~conv], y[~conv], "o", mfc="none", mec=colour, ms=6)
    for r, e0, half, colour in bands:
        ax_e.fill_between(r, e0 - half, e0 + half, color=colour, alpha=0.2, lw=0)
    ax_err.axhline(CHEMICAL_ACCURACY_MHA, color="k", lw=0.8, ls="-.")
    ax_err.text(R[-1], CHEMICAL_ACCURACY_MHA * 1.15, "chemical accuracy",
                ha="right", va="bottom", fontsize=8)

    ax_s2.plot(R, ref["S2_UHF"], "0.25", ls="--", label="UHF")
    ax_s2.plot(R, ref["S2_FCI"], "k-", label="FCI")

    ax_e.set_ylabel("E (Hartree)")
    ax_e.set_ylim(fci.min() - 0.05, max(ref["E_UHF"].max(), fci.max()) + 0.15)
    ax_e.legend(fontsize=7, ncol=2, loc="upper right")
    ax_err.set_ylabel(r"|E $-$ E$_\mathrm{FCI}$| (mHa)")
    ax_s2.set_ylabel(r"$\langle S^2\rangle$")
    ax_s2.set_xlabel("R (Å)")
    ax_s2.legend(fontsize=7)
    ax_e.set_title("N$_2$ / STO-3G (all electrons, 20 qubits); open markers: CCSD not converged",
                   fontsize=9)
    for ax in (ax_e, ax_err, ax_s2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.results, f"n2_pes.{ext}"), dpi=200)
    print(f"Wrote {args.results}/n2_pes.png and .pdf")

    print(f"\n{'method':<20} {'max |err| mHa':>14} {'NPE mHa':>10}  (over R where available)")
    print(f"{'-' * 20} {'-' * 14} {'-' * 10}")
    for label, r, e, _, _, conv in curves:
        err = (e - np.interp(r, R, fci)) * HARTREE_TO_MHA
        note = "" if conv is None or conv.all() else \
            f"  ({(~conv).sum()} unconverged points included)"
        print(f"{label:<20} {np.abs(err).max():14.3f} {err.max() - err.min():10.3f}{note}")


if __name__ == "__main__":
    main()
