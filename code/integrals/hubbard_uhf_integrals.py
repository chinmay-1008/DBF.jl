"""UHF one- and two-body integrals for the 2D Hubbard model, in a choice of basis.

General-lattice version of BEST_MU/U4_N56_LNO/run_lno.py.  Same pipeline --
UHF -> (UCCSD natural orbitals) -> separate occupied/virtual localization ->
integral transformation -> .npy files -- with the lattice, filling and orbital
basis taken from the command line instead of hardcoded, and a metadata.json
that uhf_integrals_to_pauli.jl reads so nothing has to be patched by hand.

Orbital bases (--basis, any number of them):

    site    lattice sites, no rotation (h_a = h_b = the bare Hubbard model)
    mo      canonical UHF orbitals
    lmo     canonical UHF orbitals, occupied and virtual blocks localized
    no      UCCSD natural orbitals
    lno     UCCSD natural orbitals, occupied-like and virtual-like blocks localized

All the reference script's safeguards are kept: the SCF convergence ladder
(plain DIIS -> Fermi-smearing anneal -> Newton), internal UHF stability
following, explicit convergence checks, multi-start Pipek-Mezey localization
with projector/orthonormality/locality checks, and an energy-reconstruction
check that the integrals in every basis reproduce E_UHF.  Two checks are new:

  * Reference determinant energy.  For each basis the script stores which
    orbitals the reference determinant occupies and that determinant's energy
    from Slater-Condon rules.  The Julia converter takes <ref|H|ref> of its
    PauliSum and must match it.  In the mo and lmo bases this IS E_UHF; in the
    no and lno bases the occupied-like natural orbitals do not span the UHF
    occupied space, so it is a different (higher) number, and comparing the
    PauliSum against E_UHF there would be wrong.
  * FCI invariance (lattices with at most --fci-max-sites sites).  FCI in the
    full space is invariant under any orbital rotation, so every basis must
    give the same energy.  The Julia converter compares the lowest eigenvalue
    of its PauliSum in the (n_alpha, n_beta) sector against this number.

The Hamiltonian carries the -mu * N_hat term, as in the reference.  Add
energy_offset = mu * N (stored in metadata.json) to get physical energies.

Outputs (into --outdir, default data/<tag>):
    <basis>_h1_a.npy  <basis>_h1_b.npy                 one-body, alpha / beta
    <basis>_eri_aa.npy  <basis>_eri_bb.npy  <basis>_eri_ab.npy
                                                       two-body, chemist (pq|rs)
    <basis>_coeff_a.npy  <basis>_coeff_b.npy           site -> orbital coefficients
    metadata.json

Run:
    python -u hubbard_uhf_integrals.py --basis lno > lno.out 2>&1   # 8x8, U=4, N=56
    python -u hubbard_uhf_integrals.py --nx 2 --ny 3 --U 4 --mu 0 --nelec 6 \
        --open-x --open-y --basis all                               # small test
"""

import argparse
import json
import os

import numpy as np
from pyscf import ao2mo, cc, fci, gto, lib, lo, scf

try:
    from pyscf.scf import addons as scf_addons
except ImportError:
    scf_addons = None

try:
    from pyscf.scf.stability import _gen_hop_uhf_external
except ImportError:
    _gen_hop_uhf_external = None


# ---------------------------------------------------------------- parameters

SCF_MAX_CYCLE = 1000
SCF_CONV_TOL = 1.0e-10
SCF_CONV_TOL_GRAD = 1.0e-7
SCF_LEVEL_SHIFT = 0.5
SCF_DAMP = 0.5
SCF_DIIS_START = 12
SMEARING_LADDER = (0.5, 0.2, 0.05)
MAX_STABILITY_CYCLES = 15

CC_MAX_CYCLE = 400
CC_DIIS_SPACE = 15
CC_LEVEL_SHIFT = 0.2

LOCALIZATION_STARTS = 4
LOCALIZATION_GRAD_TOL = 1.0e-5
# Least-localized orbital must carry at least this much weight on one site.
MIN_PEAK_WEIGHT = 0.15

ENERGY_TOL = 1.0e-8
BASES = ("site", "mo", "lmo", "no", "lno")


# ------------------------------------------------------------------- lattice

def snake_index(i, j, nx):
    return j * nx + i if j % 2 == 0 else j * nx + (nx - i - 1)


def build_hubbard_2d(nx, ny, t, u, mu=0.0, periodic_x=True, periodic_y=True):
    n_sites = nx * ny
    h1 = np.zeros((n_sites, n_sites))
    eri = np.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(nx):
        for j in range(ny):
            cur = snake_index(i, j, nx)
            if i + 1 < nx:
                nb = snake_index(i + 1, j, nx)
                h1[cur, nb] = h1[nb, cur] = -t
            elif periodic_x:
                nb = snake_index(0, j, nx)
                h1[cur, nb] = h1[nb, cur] = -t
            if j + 1 < ny:
                nb = snake_index(i, j + 1, nx)
                h1[cur, nb] = h1[nb, cur] = -t
            elif periodic_y:
                nb = snake_index(i, 0, nx)
                h1[cur, nb] = h1[nb, cur] = -t
            h1[cur, cur] = -mu
    for site in range(n_sites):
        eri[site, site, site, site] = u
    return h1, eri


def neel_guess(nx, ny, n_alpha, n_beta):
    """Neel checkerboard filled in i-major order: alpha on (i+j) even, beta on odd.

    For 8x8 with 28 + 28 electrons this is the reference script's guess --
    columns i = 0..6 fill and column i = 7 is left as a vertical stripe of
    eight holes.  A spin with more electrons than its sublattice has sites
    spills onto the other sublattice (double occupancy).
    """
    n_sites = nx * ny
    order = [snake_index(i, j, nx) for i in range(nx) for j in range(ny)]
    parity = {snake_index(i, j, nx): (i + j) % 2
              for i in range(nx) for j in range(ny)}

    def fill(n, preferred):
        first = [s for s in order if parity[s] == preferred]
        second = [s for s in order if parity[s] != preferred]
        return (first + second)[:n]

    occ_alpha, occ_beta = fill(n_alpha, 0), fill(n_beta, 1)
    dm_alpha = np.zeros((n_sites, n_sites))
    dm_beta = np.zeros((n_sites, n_sites))
    dm_alpha[occ_alpha, occ_alpha] = 1.0
    dm_beta[occ_beta, occ_beta] = 1.0

    holes = [s for s in range(n_sites)
             if dm_alpha[s, s] == 0 and dm_beta[s, s] == 0]
    hole_columns = sorted({i for i in range(nx) for j in range(ny)
                           if snake_index(i, j, nx) in holes})
    print(f"Initial guess: N_alpha={len(occ_alpha)}, N_beta={len(occ_beta)}, "
          f"{len(holes)} holes on column(s) {hole_columns}")
    return np.stack((dm_alpha, dm_beta)), (sorted(occ_alpha), sorted(occ_beta))


# -------------------------------------------------------------- SCF machinery

def make_uhf(mol, h1, eri8, level_shift=SCF_LEVEL_SHIFT):
    n_sites = h1.shape[0]
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n_sites)
    mf._eri = eri8
    mf.max_cycle = SCF_MAX_CYCLE
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.level_shift = level_shift
    mf.damp = SCF_DAMP
    mf.diis_start_cycle = SCF_DIIS_START
    return mf


def converge_uhf(mol, h1, eri8, dm0, tag=""):
    """Plain DIIS -> Fermi-smearing anneal -> second-order Newton."""
    mf = make_uhf(mol, h1, eri8)
    mf.kernel(dm0=dm0)
    if mf.converged:
        return mf
    print(f"  {tag}plain SCF failed -- annealing with Fermi smearing")

    dm = dm0
    if scf_addons is not None:
        for sigma in SMEARING_LADDER:
            base = make_uhf(mol, h1, eri8)
            try:
                sm = scf_addons.smearing_(base, sigma=sigma, method="fermi",
                                          fix_spin=True)
            except TypeError:
                sm = scf_addons.smearing_(base, sigma=sigma, method="fermi")
            sm.kernel(dm0=dm)
            dm = np.asarray(sm.make_rdm1())
            print(f"    sigma={sigma}: E={sm.e_tot:.10f} converged={sm.converged}")
        mf = make_uhf(mol, h1, eri8)
        mf.kernel(dm0=dm)
        if mf.converged:
            print(f"  {tag}converged after smearing anneal")
            return mf

    print(f"  {tag}still stuck -- trying second-order (Newton) SCF")
    base = make_uhf(mol, h1, eri8, level_shift=0.0)
    try:
        mf = base.newton()
        mf.max_cycle = 200
        mf.kernel(dm0=dm)
        if mf.converged:
            print(f"  {tag}converged with Newton SCF")
            return mf
    except Exception as exc:
        print(f"  {tag}Newton SCF unavailable: {exc}")

    raise RuntimeError("UHF would not converge")


def external_stability_roots(mf):
    if _gen_hop_uhf_external is None:
        return None
    try:
        hop_rc, diag_rc, hop_ghf, diag_ghf = _gen_hop_uhf_external(
            mf, with_symmetry=True)

        def lowest_roots(hop, diagonal):
            def precondition(residual, eigenvalue, _):
                shifted = diagonal - eigenvalue
                shifted[np.abs(shifted) < 1e-8] = 1e-8
                return residual / shifted
            initial = np.zeros_like(diagonal)
            positive = diagonal > 1e-5
            initial[positive] = 1.0 / diagonal[positive]
            roots, _ = lib.davidson(hop, initial, precondition,
                                    tol=1e-6, nroots=3, verbose=0)
            return np.atleast_1d(roots).astype(float).tolist()

        return {"real_to_complex": lowest_roots(hop_rc, diag_rc),
                "uhf_to_ghf": lowest_roots(hop_ghf, diag_ghf)}
    except Exception as exc:
        print(f"  (external stability unavailable: {exc})")
        return None


def stabilise_uhf(mf, mol, h1, eri8):
    history = []
    for cycle in range(MAX_STABILITY_CYCLES + 1):
        mo_internal, _, stable, _ = mf.stability(
            internal=True, external=False, return_status=True)
        history.append({"cycle": cycle, "energy": float(mf.e_tot),
                        "internal_stable": bool(stable)})
        print(f"  stability cycle {cycle}: E={mf.e_tot:.12f}  stable={stable}")
        if stable:
            break
        if cycle == MAX_STABILITY_CYCLES:
            raise RuntimeError("internal UHF stability was not reached")
        dm = mf.make_rdm1(mo_internal, mf.mo_occ)
        mf = converge_uhf(mol, h1, eri8, dm, tag="[stability follow] ")
    return mf, history


# ------------------------------------------------------------- localization

class HubbardPM(lo.pipek.PM):
    """Pipek-Mezey with each site treated as its own 'atom' (orthonormal basis)."""

    def atomic_pops(self, mol, mo_coeff, method=None, **kwargs):
        return np.einsum("pi,pj->pij", mo_coeff, mo_coeff)


def verify_localization(before, after, label, gradient_norm, min_peak_weight):
    """Hard checks that the localized block is a correct, localized rotation.

    The decisive correctness test is the PROJECTOR: localization must be a
    unitary rotation WITHIN the block's span, so C_after C_after^T must equal
    C_before C_before^T to machine precision.  If it does not, orbitals have
    leaked between the occupied-like and virtual-like subspaces and the
    resulting integrals describe a different reference.

    The quality tests then ask whether the orbitals are actually local: peak
    site weight, inverse participation ratio, how many sites carry 90% of an
    orbital's weight, and whether the orbitals sit on distinct centres.
    """
    n_sites, n_orb = after.shape
    checks = {}

    projector_error = float(np.max(np.abs(after @ after.T - before @ before.T)))
    ortho_error = float(np.max(np.abs(after.T @ after - np.eye(n_orb))))
    checks["projector_error"] = projector_error
    checks["orthogonality_error"] = ortho_error
    checks["gradient_norm"] = float(gradient_norm)

    weights = np.abs(after) ** 2                      # site x orbital
    peak = weights.max(axis=0)
    ipr = (weights ** 2).sum(axis=0)                  # 1 = one site, 1/n = uniform
    ordered = np.sort(weights, axis=0)[::-1]
    cumulative = np.cumsum(ordered, axis=0)
    sites_90 = (cumulative < 0.9).sum(axis=0) + 1
    centres = np.argmax(weights, axis=0)

    checks.update({
        "peak_weight_mean": float(peak.mean()),
        "peak_weight_min": float(peak.min()),
        "ipr_mean": float(ipr.mean()),
        "ipr_min": float(ipr.min()),
        "sites_for_90pct_mean": float(sites_90.mean()),
        "sites_for_90pct_max": int(sites_90.max()),
        "unique_centres": int(len(np.unique(centres))),
        "n_orbitals": int(n_orb),
        "delocalized_baseline_ipr": float(1.0 / n_sites),
    })

    print(f"    {label}: projector err {projector_error:.2e}, "
          f"ortho err {ortho_error:.2e}, grad {gradient_norm:.2e}")
    print(f"    {label}: peak site weight  mean {peak.mean():.4f}  "
          f"min {peak.min():.4f}")
    print(f"    {label}: IPR  mean {ipr.mean():.4f}  min {ipr.min():.4f}  "
          f"(uniform would be {1.0 / n_sites:.4f})")
    print(f"    {label}: sites for 90% of weight  mean {sites_90.mean():.2f}  "
          f"max {sites_90.max()}")
    print(f"    {label}: distinct centres {len(np.unique(centres))} of {n_orb}")

    if projector_error > 1e-10:
        raise RuntimeError(
            f"{label}: localization changed the subspace (projector error "
            f"{projector_error:.3e}) -- orbitals leaked between blocks")
    if ortho_error > 1e-10:
        raise RuntimeError(f"{label}: orbitals not orthonormal "
                           f"({ortho_error:.3e})")
    if gradient_norm > LOCALIZATION_GRAD_TOL:
        raise RuntimeError(f"{label}: localization gradient {gradient_norm:.3e} "
                           f"exceeds {LOCALIZATION_GRAD_TOL:.1e}")
    if peak.min() < min_peak_weight:
        raise RuntimeError(
            f"{label}: least-localized orbital carries only "
            f"{peak.min():.4f} of its weight on one site (need "
            f"{min_peak_weight}) -- the block did not localize")
    if len(np.unique(centres)) < n_orb:
        print(f"    !! {label}: only {len(np.unique(centres))} distinct centres "
              f"for {n_orb} orbitals -- some share a peak site")
    return checks


def localize_block(mol, coefficients, label, n_starts, min_peak_weight):
    """Pipek-Mezey localization from several starts, keeping the best."""
    # Nothing to rotate in an empty or one-orbital block.
    if coefficients.shape[1] <= 1:
        return coefficients, {}

    def score(c):
        return float(np.sum(np.abs(c) ** 4))

    before_score = score(coefficients)
    rng = np.random.default_rng(20260915 + coefficients.shape[1])
    candidates = []
    for start in range(n_starts):
        if start == 0:
            initial = coefficients
        else:
            rotation, _ = np.linalg.qr(
                rng.standard_normal((coefficients.shape[1],) * 2))
            initial = coefficients @ rotation
        localizer = HubbardPM(mol, initial)
        localizer.max_cycle = 2000
        localizer.conv_tol = 1e-10
        localizer.conv_tol_grad = 1e-5
        localizer.verbose = 0
        trial = localizer.kernel()
        grad = float(np.linalg.norm(localizer.get_grad()))
        candidates.append((score(trial), grad, start, trial))
        print(f"    {label} start {start}: score={score(trial):.10f}  "
              f"grad={grad:.3e}")

    converged = [c for c in candidates if c[1] <= LOCALIZATION_GRAD_TOL]
    if not converged:
        raise RuntimeError(f"{label}: localization did not converge from any start")
    after_score, grad, chosen, localized = max(converged, key=lambda c: c[0])

    spread = max(c[0] for c in converged) - min(c[0] for c in converged)
    print(f"    {label}: score {before_score:.8f} -> {after_score:.8f}  "
          f"(ratio {after_score / before_score:.4f}, start {chosen}, "
          f"spread over starts {spread:.2e})")
    if after_score + 1e-8 < before_score:
        raise RuntimeError(f"{label}: localization score decreased")

    checks = verify_localization(coefficients, localized, label, grad,
                                 min_peak_weight)
    checks.update({"score_before": before_score, "score_after": after_score,
                   "selected_start": chosen,
                   "score_spread_over_starts": float(spread)})
    return localized, checks


def localize_spin_blocks(mol, coeffs, nelec, tag, n_starts, min_peak_weight):
    """Localize occupied(-like) and virtual(-like) blocks separately, per spin."""
    localized, checks = [], {}
    for spin_idx, coeff in enumerate(coeffs):
        label = "alpha" if spin_idx == 0 else "beta"
        n_occ = nelec[spin_idx]
        print(f"  --- {tag} {label}: {n_occ} occupied-like, "
              f"{coeff.shape[1] - n_occ} virtual-like ---")
        loc_occ, chk_occ = localize_block(mol, coeff[:, :n_occ],
                                          f"{label}-occ", n_starts,
                                          min_peak_weight)
        loc_vir, chk_vir = localize_block(mol, coeff[:, n_occ:],
                                          f"{label}-vir", n_starts,
                                          min_peak_weight)
        full = np.hstack((loc_occ, loc_vir))

        # Occupied-like and virtual-like blocks must stay mutually orthogonal,
        # and the full set must remain an orthonormal basis of the site space.
        cross = float(np.max(np.abs(loc_occ.T @ loc_vir))) if n_occ else 0.0
        ortho = float(np.max(np.abs(full.T @ full - np.eye(full.shape[1]))))
        print(f"  {label}: occ-vir overlap {cross:.2e}, "
              f"full orthonormality error {ortho:.2e}")
        if cross > 1e-10 or ortho > 1e-10:
            raise RuntimeError(f"{tag} {label}: localized orbitals are not a "
                               f"clean orthonormal basis")
        localized.append(full)
        checks[f"{label}_occupied"] = chk_occ
        checks[f"{label}_virtual"] = chk_vir
        checks[f"{label}_occ_vir_overlap"] = cross
        checks[f"{label}_full_orthonormality_error"] = ortho
    return localized, checks


# ------------------------------------------------- integrals and energies

def transform_integrals(h_core_site, eri8, coeff_a, coeff_b):
    n_mo = coeff_a.shape[1]
    return {
        "h1_alpha": coeff_a.T @ h_core_site @ coeff_a,
        "h1_beta": coeff_b.T @ h_core_site @ coeff_b,
        "eri_aa": ao2mo.restore(1, ao2mo.incore.full(eri8, coeff_a), n_mo),
        "eri_bb": ao2mo.restore(1, ao2mo.incore.full(eri8, coeff_b), n_mo),
        "eri_ab": ao2mo.incore.general(
            eri8, (coeff_a, coeff_a, coeff_b, coeff_b), compact=False
        ).reshape(n_mo, n_mo, n_mo, n_mo),
    }


def energy_from_density(ints, dm_a, dm_b):
    """UHF-form energy of a pair of spin densities, chemist-notation integrals."""
    e_one = np.einsum("pq,pq->", ints["h1_alpha"], dm_a) + \
            np.einsum("pq,pq->", ints["h1_beta"], dm_b)
    e_aa = 0.5 * (np.einsum("pqrs,pq,rs->", ints["eri_aa"], dm_a, dm_a)
                  - np.einsum("pqrs,ps,rq->", ints["eri_aa"], dm_a, dm_a))
    e_bb = 0.5 * (np.einsum("pqrs,pq,rs->", ints["eri_bb"], dm_b, dm_b)
                  - np.einsum("pqrs,ps,rq->", ints["eri_bb"], dm_b, dm_b))
    e_ab = np.einsum("pqrs,pq,rs->", ints["eri_ab"], dm_a, dm_b)
    return float(e_one + e_aa + e_bb + e_ab)


def determinant_energy(ints, occ_alpha, occ_beta):
    """Slater-Condon energy of the determinant occupying these orbitals."""
    n_mo = ints["h1_alpha"].shape[0]
    dm_a, dm_b = np.zeros((n_mo, n_mo)), np.zeros((n_mo, n_mo))
    dm_a[occ_alpha, occ_alpha] = 1.0
    dm_b[occ_beta, occ_beta] = 1.0
    return energy_from_density(ints, dm_a, dm_b)


def fci_energy(ints, nelec):
    """Exact energy in the full space.  Invariant under ANY orbital rotation."""
    n_mo = ints["h1_alpha"].shape[0]
    e, _ = fci.direct_uhf.kernel(
        (ints["h1_alpha"], ints["h1_beta"]),
        (ints["eri_aa"], ints["eri_ab"], ints["eri_bb"]),
        n_mo, nelec, max_cycle=200, conv_tol=1e-12)
    return float(e)


# ----------------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--mu", type=float, default=1.17)
    parser.add_argument("--nelec", type=int, default=56)
    parser.add_argument("--spin", type=int, default=0, help="N_alpha - N_beta")
    parser.add_argument("--open-x", action="store_true",
                        help="open instead of periodic boundary in x")
    parser.add_argument("--open-y", action="store_true",
                        help="open instead of periodic boundary in y")
    parser.add_argument("--basis", nargs="+", default=["lno"],
                        choices=list(BASES) + ["all"])
    parser.add_argument("--loc-starts", type=int, default=LOCALIZATION_STARTS)
    parser.add_argument("--min-peak-weight", type=float, default=MIN_PEAK_WEIGHT)
    parser.add_argument("--fci-max-sites", type=int, default=8,
                        help="run the FCI invariance check at or below this "
                             "many sites")
    parser.add_argument("--outdir", default=None,
                        help="default: data/<lattice tag> next to this script")
    args = parser.parse_args()
    args.basis = list(BASES) if "all" in args.basis else \
        [b for b in BASES if b in args.basis]
    return args


def main():
    args = parse_args()
    nx, ny, n_sites = args.nx, args.ny, args.nx * args.ny
    periodic_x, periodic_y = not args.open_x, not args.open_y
    bc = ("p" if periodic_x else "o") + ("p" if periodic_y else "o")
    tag = f"{nx}x{ny}_{bc}_N{args.nelec}_U{args.U:g}_mu{args.mu:g}"
    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", tag)
    os.makedirs(outdir, exist_ok=True)

    h1, eri = build_hubbard_2d(nx, ny, args.t, args.U, args.mu,
                               periodic_x=periodic_x, periodic_y=periodic_y)
    eri8 = ao2mo.restore(8, eri, n_sites)
    del eri

    print(f"Hubbard {nx}x{ny}, t={args.t}, U={args.U}, mu={args.mu}, "
          f"N={args.nelec}, 2S={args.spin}, "
          f"{'PBC' if periodic_x else 'OBC'} in x, "
          f"{'PBC' if periodic_y else 'OBC'} in y")
    print(f"bases: {', '.join(args.basis)}")
    print(f"mu*N = {args.mu * args.nelec:.6f}  "
          f"(added back to get the physical energy)\n")

    mol = gto.M()
    mol.nelectron = args.nelec
    mol.spin = args.spin
    mol.nao = n_sites
    mol.incore_anyway = True
    mol.build()
    nelec = tuple(int(n) for n in mol.nelec)

    # ---- UHF ----
    print("=== UHF ===")
    dm0, neel_occ = neel_guess(nx, ny, *nelec)
    mf = converge_uhf(mol, h1, eri8, dm0)
    mf, stability_history = stabilise_uhf(mf, mol, h1, eri8)

    density = mf.make_rdm1()
    n_check = np.trace(density[0]) + np.trace(density[1])
    if not np.isclose(n_check, args.nelec, atol=1e-7):
        raise RuntimeError(f"density has {n_check} electrons")
    for spin_idx in range(2):
        if not np.allclose(mf.mo_occ[spin_idx][:nelec[spin_idx]], 1.0):
            raise RuntimeError("UHF occupations are not aufbau -- the first "
                               "n_occ canonical orbitals are not the occupied ones")

    s2, mult = mf.spin_square()
    grad_norm = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    e_uhf_physical = mf.e_tot + args.mu * args.nelec
    print(f"\nE_UHF(H_mu)        = {mf.e_tot:.12f}")
    print(f"E_UHF(mu=0)        = {e_uhf_physical:.12f}")
    print(f"E_UHF per site     = {e_uhf_physical / n_sites:.12f}")
    print(f"<S^2>              = {s2:.6f}   2S+1 = {mult:.6f}")
    print(f"orbital gradient   = {grad_norm:.3e}")
    roots = external_stability_roots(mf)
    if roots is not None:
        rc = min(roots["real_to_complex"])
        ghf = min(roots["uhf_to_ghf"])
        print(f"real->complex lowest root = {rc:.6f}")
        print(f"UHF->GHF      lowest root = {ghf:.6f}")
        if rc < -1e-5:
            raise RuntimeError(
                "UHF has a real-to-complex instability: the collinear real "
                "reference is not even a local minimum in the wider space")
        if ghf < -1e-5:
            print("  (UHF->GHF instability present and NOT followed -- following "
                  "it would give a noncollinear GHF spinor rather than the "
                  "collinear UHF reference this project uses)")
    else:
        print("  !! external stability could not be evaluated")

    metadata = {
        "model": "hubbard_2d",
        "reference": "UHF",
        "lattice": [nx, ny], "t": args.t, "U": args.U, "mu": args.mu,
        "periodic": [periodic_x, periodic_y],
        "site_ordering": "snake",
        "n_sites": n_sites, "n_orbitals": n_sites,
        "n_electrons": args.nelec, "n_alpha": nelec[0], "n_beta": nelec[1],
        "spin_orbital_ordering": "alpha block (1..n) then beta block (n+1..2n)",
        "integral_notation": "chemist (pq|rs)",
        "constant": 0.0,
        "energy_offset": args.mu * args.nelec,
        "energy_offset_note": "H contains -mu*N_hat; add energy_offset for "
                              "the mu = 0 energy",
        "E_UHF_with_mu": float(mf.e_tot),
        "E_UHF_zero_mu": float(e_uhf_physical),
        "E_UHF_per_site": float(e_uhf_physical / n_sites),
        "S_squared": float(s2),
        "scf_gradient_norm": grad_norm,
        "stability_history": stability_history,
        "external_stability_roots": roots,
        "bases": {},
    }

    # ---- orbital bases ----
    first_n = (list(range(nelec[0])), list(range(nelec[1])))
    coeffs = {}
    if "site" in args.basis:
        coeffs["site"] = ((np.eye(n_sites), np.eye(n_sites)), neel_occ)
    if "mo" in args.basis:
        coeffs["mo"] = (tuple(mf.mo_coeff), first_n)
    if "lmo" in args.basis:
        print("\n=== localized canonical orbitals ===")
        loc, checks = localize_spin_blocks(mol, mf.mo_coeff, nelec, "lmo",
                                           args.loc_starts, args.min_peak_weight)
        coeffs["lmo"] = (tuple(loc), first_n)
        metadata["localization_lmo"] = checks

    if "no" in args.basis or "lno" in args.basis:
        print("\n=== UCCSD ===")
        mycc = cc.UCCSD(mf)
        mycc.max_cycle = CC_MAX_CYCLE
        mycc.diis_space = CC_DIIS_SPACE      # NOT diis_space_size, which is ignored
        mycc.level_shift = CC_LEVEL_SHIFT
        mycc.kernel()
        if not mycc.converged:
            raise RuntimeError("UCCSD did not converge -- do not build integrals "
                               "from these natural orbitals")
        e_ccsd_physical = mycc.e_tot + args.mu * args.nelec
        print(f"E_UCCSD(H_mu)      = {mycc.e_tot:.12f}")
        print(f"E_UCCSD per site   = {e_ccsd_physical / n_sites:.12f}")

        mycc.solve_lambda()
        rdm1 = mycc.make_rdm1()

        no_coeffs, no_occs = [], []
        for spin_idx, (dm, mo_coeff) in enumerate(zip(rdm1, mf.mo_coeff)):
            label = "Alpha" if spin_idx == 0 else "Beta"
            occ_vals, vecs = np.linalg.eigh(dm)
            order = occ_vals.argsort()[::-1]
            occ_vals, vecs = occ_vals[order], vecs[:, order]
            no_coeffs.append(mo_coeff @ vecs)
            no_occs.append(occ_vals)
            n_occ = nelec[spin_idx]
            gap = (occ_vals[n_occ - 1] - occ_vals[n_occ]
                   if 0 < n_occ < n_sites else float("nan"))
            print(f"{label} natural occupations: "
                  f"max {occ_vals[0]:.8f}, min {occ_vals[-1]:.8f}, "
                  f"gap at orbital {n_occ} = {gap:.8f}")
            print(f"  sum = {occ_vals.sum():.8f} (expect {n_occ})")

        metadata.update({
            "E_UCCSD_with_mu": float(mycc.e_tot),
            "E_UCCSD_per_site": float(e_ccsd_physical / n_sites),
            "natural_occupations_alpha": no_occs[0].tolist(),
            "natural_occupations_beta": no_occs[1].tolist(),
        })
        if "no" in args.basis:
            coeffs["no"] = (tuple(no_coeffs), first_n)
        if "lno" in args.basis:
            print("\n=== localized natural orbitals ===")
            loc, checks = localize_spin_blocks(mol, no_coeffs, nelec, "lno",
                                               args.loc_starts,
                                               args.min_peak_weight)
            coeffs["lno"] = (tuple(loc), first_n)
            metadata["localization_lno"] = checks

    # ---- integral transformation, checks, save ----
    run_fci = n_sites <= args.fci_max_sites
    fci_reference = None
    h_core_site = mf.get_hcore()
    for basis in args.basis:
        (coeff_a, coeff_b), (occ_alpha, occ_beta) = coeffs[basis]
        print(f"\n=== integrals: {basis} ===")
        for coeff, label in ((coeff_a, "alpha"), (coeff_b, "beta")):
            ortho = float(np.max(np.abs(coeff.T @ coeff - np.eye(n_sites))))
            if ortho > 1e-10:
                raise RuntimeError(f"{basis} {label}: orbitals not orthonormal "
                                   f"({ortho:.3e})")
        ints = transform_integrals(h_core_site, eri8, coeff_a, coeff_b)

        # The UHF density, rotated into this basis, must give back E_UHF.
        dm_a = coeff_a.T @ density[0] @ coeff_a
        dm_b = coeff_b.T @ density[1] @ coeff_b
        e_rebuilt = energy_from_density(ints, dm_a, dm_b)
        error = abs(e_rebuilt - mf.e_tot)
        print(f"  UHF energy rebuilt from the integrals: {e_rebuilt:.12f}  "
              f"(error {error:.3e})")
        if error > ENERGY_TOL:
            raise RuntimeError(f"{basis}: integrals do not reproduce the UHF "
                               f"energy")

        e_det = determinant_energy(ints, occ_alpha, occ_beta)
        print(f"  reference determinant energy:          {e_det:.12f}  "
              f"(E_det - E_UHF = {e_det - mf.e_tot:+.3e})")
        if basis in ("mo", "lmo") and abs(e_det - mf.e_tot) > ENERGY_TOL:
            raise RuntimeError(f"{basis}: the first n_occ orbitals do not "
                               f"reproduce the UHF determinant")

        entry = {
            "files": {key: f"{basis}_{key.replace('alpha', 'a').replace('beta', 'b')}.npy"
                      for key in ints},
            "coeff_files": {"alpha": f"{basis}_coeff_a.npy",
                            "beta": f"{basis}_coeff_b.npy"},
            "occupied_alpha": [int(i) for i in occ_alpha],
            "occupied_beta": [int(i) for i in occ_beta],
            "occupied_indexing": "0-based orbital index within each spin block",
            "reference_determinant_energy": e_det,
            "uhf_energy_rebuilt_error": float(error),
        }

        if run_fci:
            e_fci = fci_energy(ints, nelec)
            fci_reference = e_fci if fci_reference is None else fci_reference
            print(f"  FCI energy:                            {e_fci:.12f}  "
                  f"(delta vs first basis {e_fci - fci_reference:+.2e})")
            if abs(e_fci - fci_reference) > ENERGY_TOL:
                raise RuntimeError(f"{basis}: FCI energy changed under rotation "
                                   f"-- the integral transformation is wrong")
            entry["fci_energy"] = e_fci

        for key, name in entry["files"].items():
            np.save(os.path.join(outdir, name), ints[key])
        np.save(os.path.join(outdir, entry["coeff_files"]["alpha"]), coeff_a)
        np.save(os.path.join(outdir, entry["coeff_files"]["beta"]), coeff_b)
        print(f"  wrote {basis}_*.npy  (eri shape {ints['eri_aa'].shape})")
        metadata["bases"][basis] = entry

    if run_fci:
        metadata["E_FCI_with_mu"] = fci_reference
        metadata["E_FCI_per_site"] = (fci_reference + args.mu * args.nelec) / n_sites

    meta_path = os.path.join(outdir, "metadata.json")
    with open(meta_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\n  wrote {meta_path}")

    here = os.path.dirname(os.path.abspath(__file__))
    print("\n=== next step ===")
    for basis in args.basis:
        print(f"julia --project={here} {os.path.join(here, 'uhf_integrals_to_pauli.jl')} "
              f"{outdir} --basis {basis}")


if __name__ == "__main__":
    main()
