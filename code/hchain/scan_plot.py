"""H-chain size scan: stored coefficients vs energy error, Heisenberg vs Schrodinger.

    python scan_plot.py [--target 1.6e-3] [--out results/hchain_scan]

Top row, one panel per spacing: energy error against stored coefficients for
every chain (colour = n).  Heisenberg: the true <H> of the vDBF circuit after
each iteration against the Pauli terms vDBF carried (the tightest cutoff run;
vDBF's own E - err where no exact replay exists, dotted).  Schrodinger: the
best k determinants (ideal selected CI, bestk.json).

Bottom row: coefficients needed to reach the target error against qubit count.
  Heisenberg   fewest Pauli terms at which any cutoff's trajectory reaches it
               (open marker = never reached; the point is the largest operator
               reached, a lower bound)
  best k       fewest determinants (interpolated, log-log)
  same circuit fewest peak amplitudes of the truncated-state replay of any
               run's final circuit that reaches it
plus the sizes of the (n/2, n/2) sector and of the full state space."""
import argparse
import glob
import json
import os
import re
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def crossing(xs, es, target):
    """First x along the path where e <= target, log-log interpolated; None if never."""
    xs, es = np.asarray(xs, float), np.abs(np.asarray(es, float))
    for i, (x, e) in enumerate(zip(xs, es)):
        if e <= target:
            if i == 0:
                return x
            x0, e0 = xs[i - 1], es[i - 1]
            f = (np.log(e0) - np.log(target)) / (np.log(e0) - np.log(max(e, 1e-16)))
            return float(np.exp(np.log(x0) + f * (np.log(x) - np.log(x0))))
    return None


def load():
    pts = {}
    for f in glob.glob(os.path.join(HERE, "results", "sv_H*_c*.json")):
        n, d, c = re.match(r".*sv_H(\d+)_([\d.]+)_c([\de.-]+)\.json", f).groups()
        pts.setdefault((int(n), d), {"runs": {}})["runs"][c] = json.load(open(f))
    for f in glob.glob(os.path.join(HERE, "data", "H*", "R_*", "bestk.json")):
        n, d = re.match(r".*H(\d+)/R_([\d.]+)/bestk\.json", f).groups()
        pts.setdefault((int(n), d), {"runs": {}})["bestk"] = json.load(open(f))
    return pts


def heis_path(run):
    """(len_H, error, exact?) per iteration: true <H> where replayed, else E - err."""
    rows = [r for r in run["heisenberg"] if r["iter"] >= 1 and r["len_H"] > 0]
    if rows:
        return [r["len_H"] for r in rows], [r["E_true_err"] for r in rows], True
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=float, default=1.6e-3)
    p.add_argument("--out", default=os.path.join(HERE, "results", "hchain_scan"))
    a = p.parse_args()
    pts = load()
    ns = sorted({n for n, _ in pts})
    ds = sorted({d for _, d in pts})
    cmap = plt.get_cmap("viridis")
    col = {n: cmap(i / max(1, len(ns) - 1) * 0.9) for i, n in enumerate(ns)}
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9,
                         "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
                         "xtick.major.size": 5, "ytick.major.size": 5, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(2, len(ds), figsize=(3.6 * len(ds), 6.6), squeeze=False)
    table = []
    for j, d in enumerate(ds):
        top, bot = ax[0, j], ax[1, j]
        top.axhspan(1e-7, a.target * 1e3, color="#ececec", lw=0, zorder=0)
        need = {"heis": [], "heis_lb": [], "bestk": [], "circ": []}
        for n in ns:
            pt = pts.get((n, d))
            if pt is None:
                continue
            runs = pt["runs"]
            # Heisenberg: every cutoff's trajectory; the tightest one drawn
            best, reached, largest = None, [], 0
            for c, run in sorted(runs.items(), key=lambda kv: float(kv[0])):
                hp = heis_path(run)
                if hp is None:
                    continue
                xs, es, _ = hp
                best = best or (c, xs, es)
                largest = max(largest, max(xs))
                x = crossing(xs, es, a.target)
                if x is not None:
                    reached.append(x)
            if best is not None:
                c, xs, es = best
                top.plot(xs, np.abs(es) * 1e3, "-o", ms=2.5, lw=1.4, color=col[n])
            if reached:
                need["heis"].append((2 * n, min(reached)))
            elif largest:
                need["heis_lb"].append((2 * n, largest))
            if "bestk" in pt:
                rows = pt["bestk"]["rows"]
                ks, es = [r["k"] for r in rows], [max(r["E_err"], 1e-12) for r in rows]
                top.plot(ks, np.array(es) * 1e3, "--", lw=1.2, color=col[n])
                x = crossing(ks, es, a.target)
                if x is not None:
                    need["bestk"].append((2 * n, x))
            circ = [r["nnz_peak"] for run in runs.values() for r in run["circuit"] if r["E_err"] <= a.target]
            if circ:
                need["circ"].append((2 * n, min(circ)))
            table.append((n, d, dict(need)))
        q = np.array([2 * n for n in ns])
        bot.plot(q, [comb(n, n // 2) ** 2 for n in ns], ":", color="0.45", lw=1)
        bot.plot(q, [2.0 ** (2 * n) for n in ns], "-.", color="0.45", lw=1)
        style = {"heis": ("C0", "o", "-", "Heisenberg: Pauli terms"),
                 "bestk": ("C3", "^", "-", "Schrödinger: best $k$ determinants"),
                 "circ": ("C1", "s", "-", "Schrödinger: same circuit, truncated state")}
        for k, (cc, m, ls, lab) in style.items():
            if need[k]:
                x, y = zip(*need[k])
                bot.plot(x, y, ls, marker=m, color=cc, ms=4.5, lw=1.5, label=lab)
        if need["heis_lb"]:
            x, y = zip(*need["heis_lb"])
            bot.plot(x, y, "o", mfc="white", mec="C0", ms=5, ls="none", label="Heisenberg: not reached (largest operator)")
        for axx in (top, bot):
            axx.set_yscale("log")
            axx.grid(color="#e5e5e5", lw=0.8)
            axx.set_axisbelow(True)
        top.set_xscale("log")
        top.set_ylim(1e-4, 1e3)
        top.set_xlabel("stored coefficients")
        top.text(0.04, 0.95, f"spacing {float(d):.1f} Å", transform=top.transAxes, va="top")
        bot.set_xlabel("qubits")
        bot.set_xticks(q)
        bot.text(0.04, 0.95, f"to reach {a.target * 1e3:g} mHa", transform=bot.transAxes, va="top")
    ax[0, 0].set_ylabel("$E - E_{\\mathrm{FCI}}$ (mHa)")
    ax[1, 0].set_ylabel("coefficients needed")
    from matplotlib.lines import Line2D
    hn = [Line2D([], [], color=col[n], lw=1.6, label=f"H$_{{{n}}}$") for n in ns]
    hn += [Line2D([], [], color="0.3", lw=1.4, ls="-", marker="o", ms=2.5, label="Heisenberg (Pauli terms)"),
           Line2D([], [], color="0.3", lw=1.2, ls="--", label="best $k$ determinants")]
    ax[0, -1].legend(handles=hn, loc="upper right", fancybox=True, framealpha=1, edgecolor="#cccccc", fontsize=7)
    h, l = ax[1, 0].get_legend_handles_labels()
    h += [Line2D([], [], color="0.45", ls=":", lw=1), Line2D([], [], color="0.45", ls="-.", lw=1)]
    l += ["($n/2$, $n/2$) sector", "all $2^{2n}$ states"]
    fig.legend(h, l, loc="lower center", ncol=3, fancybox=True, framealpha=1, edgecolor="#cccccc")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=200)
    print(f"{a.out}.png")
    print(f"{'n':>3} {'d':>5} {'Heisenberg':>12} {'best k':>9} {'circuit':>9}   (coefficients to reach {a.target * 1e3:g} mHa)")
    for n, d, nd in sorted(table):
        get = lambda k: next((f"{y:.3g}" for x, y in nd[k] if x == 2 * n), "-")
        h = get("heis")
        if h == "-" and any(x == 2 * n for x, _ in nd["heis_lb"]):
            h = ">" + next(f"{y:.3g}" for x, y in nd["heis_lb"] if x == 2 * n)
        print(f"{n:3d} {d:>5} {h:>12} {get('bestk'):>9} {get('circ'):>9}")


if __name__ == "__main__":
    main()
