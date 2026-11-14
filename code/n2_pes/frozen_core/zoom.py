"""Zoomed extrapolation panel shared by plot_penalty.py and plot_reference.py.

Shows the extrapolation plane near Var = 0 on linear axes: each run's tail
window and its linear (solid) and quadratic (dashed) fits, E0 (big circle) at
Var = 0, the +-1.6 mHa band around FCI, and for truncated runs the true <H> of
the final circuit at its true variance (hollow star).  E0 values outside the
window are drawn as triangles on its edge.  A table in the corner lists, per
run, E0 - E_FCI and the true (or, for exact runs, final) energy - E_FCI, in mHa.

Every item is a dict: colour, x (variance, Ha^2), y (E - E_FCI, mHa), k (tail
window length, or 0 for no fit), e0 (mHa), and for truncated runs true_var and
true_e (mHa) of the last replayed circuit; for exact runs final_e (mHa)."""
import numpy as np

CHEM = 1.6  # mHa


def zoom_limits(rows, xz=None):
    """Shared window for all rows: y to 1.3 x the largest |E0| that is within
    5 mHa (between 2 and 5 mHa); x as given, or far enough to show the last
    point and the true-energy star of every run whose E0 lands inside."""
    items = [it for r in rows for it in r]
    e0s = [abs(it["e0"]) for it in items if abs(it["e0"]) <= 5]
    yz = min(max(2.0, 1.3 * max(e0s, default=0)), 5.0)
    if xz is None:
        inside = [it for it in items if abs(it["e0"]) <= yz] or items
        xz = 1.15 * max(max(it["x"][-1], it.get("true_var", 0)) for it in inside)
    return xz, yz


def draw_zoom(ax, items, xz, yz, ink, muted):
    ax.axhspan(-CHEM, CHEM, color="#efeee9", lw=0, zorder=0)
    ax.axhline(0, color=ink, lw=0.9, ls=":")
    for it in items:
        x, y, c, k = it["x"], it["y"], it["colour"], it["k"]
        if k:
            ax.plot(x[:-k], y[:-k], "s", color=c, ms=2.4, mfc="none", alpha=0.5)
            ax.plot(x[-k:], y[-k:], "s", color=c, ms=2.6)
            xs = np.linspace(0, x[-k:].max(), 200)
            ax.plot(xs, np.polyval(np.polyfit(x[-k:], y[-k:], 1), xs), color=c, lw=1.3)
            ax.plot(xs, np.polyval(np.polyfit(x[-k:], y[-k:], 2), xs), color=c, lw=1.1, ls="--")
        else:
            ax.plot(x, y, color=c, lw=1.3)
        e0 = it["e0"]
        if abs(e0) <= yz:
            ax.plot([0], [e0], "o", ms=8, color=c, mec=ink, mew=0.8, zorder=5, clip_on=False)
        else:
            edge = np.sign(e0) * yz
            ax.plot([0], [edge], "^" if e0 > 0 else "v", ms=8, color=c, mec=ink, mew=0.8, zorder=5, clip_on=False)
            ax.annotate(f"{e0:+.1f}", (0, edge), xytext=(6, -10 if e0 > 0 else 4), textcoords="offset points",
                        color=c, fontsize=7.5)
        if "true_e" in it and abs(it["true_e"]) <= yz and it["true_var"] <= xz:
            ax.plot([it["true_var"]], [it["true_e"]], "*", ms=11, mfc="white", mec=c, mew=1.4, zorder=6)
    for i, it in enumerate(items):
        if "true_e" in it:
            s = f"E0 {it['e0']:+.2f}   true ⟨H⟩ {it['true_e']:+.2f}"
        else:
            s = f"E0 {it['e0']:+.3f}   final ⟨H⟩ {it['final_e']:+.3f}"
        ax.text(0.98, 0.03 + 0.07 * (len(items) - 1 - i), s, transform=ax.transAxes, ha="right", va="bottom",
                color=it["colour"], fontsize=7.5, family="monospace",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    ax.set_xlim(0, xz)
    ax.set_ylim(-yz, yz)
    ax.set_title("zoom: extrapolation accuracy (mHa vs FCI)", color=ink, fontsize=10, loc="left")
    ax.set_xlabel("variance, Ha²", color=muted, fontsize=8)
    ax.set_ylabel("E − E_FCI, mHa", color=muted, fontsize=8)
