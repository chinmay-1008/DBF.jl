"""N2 potential energy curve: RHF, UHF, RCCSD(T), UCCSD(T), FCI references and
integrals in four orbital bases for the DBF comparison.

Pipeline, for R on a grid (default 0.8 .. 3.0 A, STO-3G, all electrons):

  1. RHF sweep, ascending R.  Each point starts from the previous geometry's
     converged density matrix and follows internal instabilities until the
     solution is internally stable.  The RHF -> UHF (external) instability is
     recorded so the Coulson-Fischer point is visible.
  2. UHF sweeps.  Forward (ascending R, first point seeded from the RHF density)
     and backward (descending R, seeded from the forward end point); both use
     the previous geometry's density as the guess and follow internal UHF
     instabilities.  Where the solution is still spin-symmetric but the RHF is
     RHF -> UHF unstable, the external eigenvector is followed to break the
     spin symmetry (PySCF's UHF internal check cannot see that direction from
     an alpha = beta start) -- no ad-hoc density kick.
     At each R the lower of the two stable solutions is kept, and the sweeps
     disagreeing (two UHF solutions) is reported.
  3. Per R: RCCSD/RCCSD(T) on RHF, UCCSD/UCCSD(T) on UHF, FCI (lowest state in
     the (n_alpha, n_beta) sector, the same sector DBF works in), UCCSD natural
     orbitals, Boys localization, and integrals in the bases

         mo    canonical UHF orbitals
         lmo   UHF orbitals; core kept canonical, valence-occupied and virtual
               blocks Boys-localized separately
         no    UCCSD natural orbitals (descending occupation)
         lno   natural orbitals; core-like kept, the rest of the occupied-like
               block and the virtual-like block Boys-localized separately

SCF convergence is never assumed: every SCF has to report converged, have an
orbital-gradient norm below SCF_GRAD_CHECK and aufbau occupations.  A failed
attempt is retried from the previous geometry's density through the ladder
DIIS -> level shift -> Fermi-smearing anneal -> Newton, and if nothing works
the script stops naming R.  No unconverged point is ever written.

Outputs (default data/R_<R>/ and results/):
    <basis>_h1_a.npy ... <basis>_eri_ab.npy, <basis>_coeff_a/b.npy, metadata.json
        same layout as code/integrals, so ../integrals/uhf_integrals_to_pauli.jl
        converts them unchanged (constant = E_nuc, mu = 0)
    results/references.csv      one row per R with every reference energy

Run:
    python -u n2_pes.py > results/logs/n2_pes.out 2>&1
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from pyscf import cc, fci, gto, lo, scf
from pyscf.scf import addons as scf_addons

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "integrals"))
from hubbard_uhf_integrals import (determinant_energy,  # noqa: E402
                                   energy_from_density, transform_integrals)


# ---------------------------------------------------------------- parameters

SCF_MAX_CYCLE = 500
SCF_CONV_TOL = 1.0e-10
SCF_CONV_TOL_GRAD = 1.0e-7
SCF_GRAD_CHECK = 1.0e-6
LEVEL_SHIFTS = (0.2, 0.5)
SMEARING_LADDER = (0.3, 0.1, 0.02)
MAX_STABILITY_CYCLES = 20

CC_CONV_TOL = 1.0e-9
CC_MAX_CYCLE = 300

N_CORE = 2                      # N 1s orbitals, kept canonical when localizing
LOCALIZATION_STARTS = 4
# PySCF's Boys gradient can plateau near 1e-5 at a converged optimum (every
# start reaching the same score), so convergence is judged by the gradient
# AND by the best starts agreeing on the objective.
LOCALIZATION_GRAD_TOL = 1.0e-4
LOCALIZATION_SCORE_SPREAD = 1.0e-6

ENERGY_TOL = 1.0e-8
SWEEP_TOL = 1.0e-6
BASES = ("mo", "lmo", "no", "lno")


# ------------------------------------------------------------------ molecule

def build_mol(r, basis):
    return gto.M(atom=f"N 0 0 0; N 0 0 {r:.6f}", basis=basis, unit="Angstrom",
                 symmetry=False, spin=0, charge=0, verbose=0)


# -------------------------------------------------------------- SCF machinery

def make_mf(mol, kind, level_shift=0.0):
    mf = scf.RHF(mol) if kind == "rhf" else scf.UHF(mol)
    mf.max_cycle = SCF_MAX_CYCLE
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.level_shift = level_shift
    mf.verbose = 0
    return mf


def scf_problem(mf):
    """None if mf is a genuinely converged aufbau solution, else the reason."""
    if not mf.converged:
        return "not converged"
    grad = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if grad > SCF_GRAD_CHECK:
        return f"orbital gradient {grad:.2e}"
    energies = mf.mo_energy if isinstance(mf.mo_energy, tuple) or \
        np.ndim(mf.mo_energy) == 2 else [mf.mo_energy]
    occs = mf.mo_occ if np.ndim(mf.mo_occ) == 2 else [mf.mo_occ]
    for e, occ in zip(energies, occs):
        occ = np.asarray(occ)
        if (occ > 0).all() or (occ == 0).all():
            continue
        if e[occ > 0].max() > e[occ == 0].min() + 1e-8:
            return "non-aufbau occupations"
    return None


def converge_scf(mol, kind, dm_guess, tag):
    """Converge from dm_guess; every fallback restarts from dm_guess too."""
    attempts = [("DIIS", lambda: _plain(mol, kind, dm_guess, 0.0))]
    attempts += [(f"level shift {s}", lambda s=s: _plain(mol, kind, dm_guess, s))
                 for s in LEVEL_SHIFTS]
    attempts += [("Fermi smearing anneal", lambda: _smeared(mol, kind, dm_guess)),
                 ("Newton", lambda: _newton(mol, kind, dm_guess))]
    for name, attempt in attempts:
        try:
            mf = attempt()
        except Exception as exc:          # a fallback can itself blow up
            print(f"    {tag}: {name} raised {exc!r}")
            continue
        problem = scf_problem(mf)
        if problem is None:
            if name != "DIIS":
                print(f"    {tag}: converged with {name}")
            return mf
        print(f"    {tag}: {name} failed ({problem})")
    raise RuntimeError(f"{tag}: SCF would not converge from the previous "
                       f"density with any fallback")


def _plain(mol, kind, dm, shift):
    mf = make_mf(mol, kind, level_shift=shift)
    mf.kernel(dm0=dm)
    if shift and mf.converged:
        # A converged level-shifted run is re-polished without the shift so
        # the reported energy / orbitals are those of the true SCF solution.
        polish = make_mf(mol, kind)
        polish.kernel(dm0=mf.make_rdm1())
        return polish
    return mf


def _smeared(mol, kind, dm):
    for sigma in SMEARING_LADDER:
        base = make_mf(mol, kind)
        if kind == "uhf":
            sm = scf_addons.smearing_(base, sigma=sigma, method="fermi",
                                      fix_spin=True)
        else:
            sm = scf_addons.smearing_(base, sigma=sigma, method="fermi")
        sm.kernel(dm0=dm)
        dm = np.asarray(sm.make_rdm1())
    mf = make_mf(mol, kind)
    mf.kernel(dm0=dm)
    return mf


def _newton(mol, kind, dm):
    mf = make_mf(mol, kind).newton()
    mf.max_cycle = 200
    mf.kernel(dm0=dm)
    return mf


def stabilise(mf, mol, kind, tag):
    """Follow internal instabilities until the SCF solution is stable."""
    for cycle in range(MAX_STABILITY_CYCLES + 1):
        mo_int, _, stable, _ = mf.stability(internal=True, external=False,
                                            return_status=True)
        if stable:
            if cycle:
                print(f"    {tag}: stable after {cycle} rotation(s), "
                      f"E = {mf.e_tot:.10f}")
            return mf
        e_before = mf.e_tot
        dm = mf.make_rdm1(mo_int, mf.mo_occ)
        mf = converge_scf(mol, kind, dm, f"{tag} [stability {cycle + 1}]")
        print(f"    {tag}: internal instability, rotated  "
              f"E {e_before:.10f} -> {mf.e_tot:.10f}")
        if mf.e_tot > e_before + 1e-9:
            raise RuntimeError(f"{tag}: stability rotation raised the energy")
    raise RuntimeError(f"{tag}: internal stability not reached in "
                       f"{MAX_STABILITY_CYCLES} rotations")


def rhf_external_unstable(mf):
    _, _, _, stable_ext = mf.stability(internal=False, external=True,
                                       return_status=True)
    return not stable_ext


def uhf_spin_square(mf):
    return float(mf.spin_square()[0])


def rhf_sweep(grid, basis):
    print("=" * 72 + "\nRHF sweep (ascending R)\n" + "=" * 72)
    results, dm_prev = {}, None
    for r in grid:
        mol = build_mol(r, basis)
        dm0 = dm_prev if dm_prev is not None else make_mf(mol, "rhf").get_init_guess()
        tag = f"RHF R={r:.2f}"
        mf = stabilise(converge_scf(mol, "rhf", dm0, tag), mol, "rhf", tag)
        ext = rhf_external_unstable(mf)
        print(f"  R={r:.2f}  E_RHF = {mf.e_tot:.10f}   "
              f"RHF->UHF unstable: {ext}")
        results[r] = (mf, ext)
        dm_prev = mf.make_rdm1()
    return results


def break_spin_symmetry(mf, mf_rhf, mol, tag):
    """A spin-symmetric UHF (alpha = beta) can look internally stable even when
    a broken-symmetry solution exists: PySCF's UHF stability Davidson starts
    from alpha/beta-symmetric vectors and never sees the triplet direction.
    If the RHF at this geometry is RHF -> UHF unstable, follow that external
    eigenvector instead and keep the result if it is lower."""
    if uhf_spin_square(mf) > 1e-6:
        return mf
    mo_ext = mf_rhf.stability(internal=False, external=True,
                              return_status=True)[1]
    occ = np.array([mf_rhf.mo_occ / 2, mf_rhf.mo_occ / 2])
    dm = make_mf(mol, "uhf").make_rdm1(mo_ext, occ)
    broken = stabilise(converge_scf(mol, "uhf", dm, f"{tag} [RHF->UHF]"),
                       mol, "uhf", f"{tag} [RHF->UHF]")
    if broken.e_tot < mf.e_tot - 1e-9:
        print(f"    {tag}: followed the RHF->UHF instability  "
              f"E {mf.e_tot:.10f} -> {broken.e_tot:.10f}")
        return broken
    return mf


def uhf_sweep(grid, basis, dm_start, label, rhf):
    print("=" * 72 + f"\nUHF sweep ({label})\n" + "=" * 72)
    results, dm_prev = {}, dm_start
    for r in grid:
        mol = build_mol(r, basis)
        tag = f"UHF[{label}] R={r:.2f}"
        mf = stabilise(converge_scf(mol, "uhf", dm_prev, tag), mol, "uhf", tag)
        mf_rhf, rhf_ext = rhf[r]
        if rhf_ext:
            mf = break_spin_symmetry(mf, mf_rhf, mol, tag)
        print(f"  R={r:.2f}  E_UHF = {mf.e_tot:.10f}   <S^2> = "
              f"{uhf_spin_square(mf):.6f}")
        results[r] = mf
        dm_prev = mf.make_rdm1()
    return results


# ------------------------------------------------------- correlated methods

def run_cc(mf, label):
    """CCSD + (T).  Non-convergence is recorded, not hidden; one retry with
    level shift and a bigger DIIS space."""
    mycc = None
    for shift, space in ((0.0, 8), (0.2, 15)):
        mycc = cc.CCSD(mf)
        mycc.conv_tol = CC_CONV_TOL
        mycc.max_cycle = CC_MAX_CYCLE
        mycc.level_shift = shift
        mycc.diis_space = space
        mycc.verbose = 0
        mycc.kernel()
        if mycc.converged:
            break
        print(f"    {label}: CCSD not converged (level shift {shift})")
    e_t = mycc.ccsd_t()
    return mycc, float(mycc.e_tot), float(mycc.e_tot + e_t), bool(mycc.converged)


def natural_orbitals(mycc, mf_uhf):
    """UCCSD (unrelaxed) 1-RDM -> natural orbitals per spin."""
    mycc.solve_lambda()
    if not mycc.converged_lambda:
        raise RuntimeError("UCCSD lambda equations did not converge")
    dm_a, dm_b = mycc.make_rdm1()
    coeffs, occs = [], []
    for dm, c_mo in zip((dm_a, dm_b), mf_uhf.mo_coeff):
        occ, vec = np.linalg.eigh(dm)
        order = np.argsort(occ)[::-1]
        coeffs.append(c_mo @ vec[:, order])
        occs.append(occ[order])
    return coeffs, occs


def fci_lowest(ints, nelec, nroots=3):
    """Lowest eigenvalue in the (n_alpha, n_beta) sector, from a few roots so
    a near-degenerate spin multiplet at dissociation cannot hide it."""
    n_mo = ints["h1_alpha"].shape[0]
    e, _ = fci.direct_uhf.kernel(
        (ints["h1_alpha"], ints["h1_beta"]),
        (ints["eri_aa"], ints["eri_ab"], ints["eri_bb"]),
        n_mo, nelec, nroots=nroots, max_cycle=300, conv_tol=1e-12)
    return float(np.min(np.atleast_1d(e)))


def fci_reference(mf_rhf, nelec, nroots=3):
    """PySCF FCI (spin-unconstrained) on the RHF orbitals, with <S^2>."""
    solver = fci.direct_spin1.FCI(mf_rhf.mol)
    solver.conv_tol = 1e-12
    solver.max_cycle = 300
    solver.nroots = nroots
    c = mf_rhf.mo_coeff
    h1 = c.T @ mf_rhf.get_hcore() @ c
    from pyscf import ao2mo
    eri = ao2mo.full(mf_rhf.mol, c)
    e, vecs = solver.kernel(h1, eri, c.shape[1], nelec,
                            ecore=mf_rhf.energy_nuc())
    e = np.atleast_1d(e)
    vecs = vecs if isinstance(vecs, list) else [vecs]
    k = int(np.argmin(e))
    s2 = float(solver.spin_square(vecs[k], c.shape[1], nelec)[0])
    return float(e[k]), s2, [float(x) for x in e]


# ------------------------------------------------------------- localization

def boys_score(mol, c):
    """Boys objective sum_i |<i|r|i>|^2 (bigger = more localized)."""
    r = mol.intor_symmetric("int1e_r", comp=3)
    centres = np.einsum("xpq,pi,qi->ix", r, c, c)
    return float(np.sum(centres ** 2))


def localize_block(mol, c, label, n_starts):
    """Boys localization from several starts, keeping the best."""
    if c.shape[1] <= 1:
        return c, {}
    s = mol.intor_symmetric("int1e_ovlp")
    rng = np.random.default_rng(20261025 + c.shape[1])
    candidates = []
    for start in range(n_starts):
        if start == 0:
            initial = c
        else:
            rot, _ = np.linalg.qr(rng.standard_normal((c.shape[1],) * 2))
            initial = c @ rot
        loc = lo.Boys(mol, initial)
        loc.max_cycle = 2000
        loc.conv_tol = 1e-10
        loc.conv_tol_grad = 1e-6
        loc.verbose = 0
        trial = loc.kernel()
        grad = float(np.linalg.norm(loc.get_grad()))
        candidates.append((boys_score(mol, trial), grad, start, trial))
    converged = [x for x in candidates if x[1] <= LOCALIZATION_GRAD_TOL]
    if not converged:
        raise RuntimeError(f"{label}: Boys localization did not converge")
    score, grad, chosen, after = max(converged, key=lambda x: x[0])
    agreeing = [x for x in converged if score - x[0] < LOCALIZATION_SCORE_SPREAD]
    if len(agreeing) < 2:
        raise RuntimeError(f"{label}: only one start reached the best Boys "
                           f"score {score:.10f}; optimum not confirmed")

    # Localization must be a rotation within the block: same AO-basis
    # projector (density), orthonormal in the AO metric.
    proj_err = float(np.max(np.abs(after @ after.T - c @ c.T)))
    ortho_err = float(np.max(np.abs(after.T @ s @ after - np.eye(c.shape[1]))))
    before = boys_score(mol, c)
    print(f"    {label}: Boys score {before:.6f} -> {score:.6f} (start "
          f"{chosen}, {len(agreeing)}/{n_starts} starts agree), grad "
          f"{grad:.1e}, projector err {proj_err:.1e}, "
          f"ortho err {ortho_err:.1e}")
    if proj_err > 1e-10 or ortho_err > 1e-10:
        raise RuntimeError(f"{label}: localization left the block's span")
    if score + 1e-8 < before:
        raise RuntimeError(f"{label}: Boys score decreased")
    return after, {"score_before": before, "score_after": score,
                   "gradient_norm": grad, "projector_error": proj_err,
                   "orthogonality_error": ortho_err, "selected_start": chosen,
                   "starts_agreeing": len(agreeing)}


def localize_spins(mol, coeffs, nelec, tag, n_starts):
    """Per spin: core kept, [core, n_occ) and [n_occ, n) localized apart."""
    out, checks = [], {}
    for spin, (c, n_occ) in enumerate(zip(coeffs, nelec)):
        name = "alpha" if spin == 0 else "beta"
        core = c[:, :N_CORE]
        occ, chk_o = localize_block(mol, c[:, N_CORE:n_occ],
                                    f"{tag} {name}-occ", n_starts)
        vir, chk_v = localize_block(mol, c[:, n_occ:],
                                    f"{tag} {name}-vir", n_starts)
        out.append(np.hstack((core, occ, vir)))
        checks[f"{name}_occupied"] = chk_o
        checks[f"{name}_virtual"] = chk_v
    return out, checks


# ------------------------------------------------------------------- output

def write_point(outdir, r, mol, mf_uhf, bases_coeffs, nelec, e_fci, refs,
                extra):
    os.makedirs(outdir, exist_ok=True)
    h_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    eri8 = mol.intor("int2e", aosym="s8")
    e_nuc = float(mol.energy_nuc())
    e_uhf_el = float(mf_uhf.e_tot - e_nuc)
    s = mol.intor_symmetric("int1e_ovlp")
    dm_a, dm_b = mf_uhf.make_rdm1()

    meta = {
        "model": f"N2 {mol.basis} R={r:.2f} A",
        "R_angstrom": r,
        "basis_set": mol.basis,
        "n_orbitals": int(mol.nao),
        "n_alpha": int(nelec[0]), "n_beta": int(nelec[1]),
        "constant": e_nuc, "constant_meaning": "nuclear repulsion",
        "mu": 0.0, "energy_offset": 0.0,
        "E_UHF_with_mu": e_uhf_el,
        "E_FCI_with_mu": e_fci - e_nuc,
        "references": refs,
        "bases": {},
    }
    meta.update(extra)
    fci_values = {}
    for basis, (ca, cb) in bases_coeffs.items():
        ints = transform_integrals(h_ao, eri8, ca, cb)
        # Density of the UHF solution expressed in this orbital basis.
        da = ca.T @ s @ dm_a @ s @ ca
        db = cb.T @ s @ dm_b @ s @ cb
        e_rebuilt = energy_from_density(ints, da, db)
        if abs(e_rebuilt - e_uhf_el) > ENERGY_TOL:
            raise RuntimeError(f"R={r:.2f} {basis}: integrals give E_UHF "
                               f"{e_rebuilt + e_nuc:.10f}, SCF "
                               f"{mf_uhf.e_tot:.10f}")
        occ_a, occ_b = list(range(nelec[0])), list(range(nelec[1]))
        e_det = determinant_energy(ints, occ_a, occ_b)
        if basis in ("mo", "lmo") and abs(e_det - e_uhf_el) > ENERGY_TOL:
            raise RuntimeError(f"R={r:.2f} {basis}: reference determinant "
                               f"{e_det:.10f} != E_UHF {e_uhf_el:.10f}")
        e_fci_basis = fci_lowest(ints, nelec)
        fci_values[basis] = e_fci_basis
        if abs(e_fci_basis + e_nuc - e_fci) > ENERGY_TOL:
            raise RuntimeError(f"R={r:.2f} {basis}: FCI {e_fci_basis + e_nuc:.10f}"
                               f" != reference FCI {e_fci:.10f}")
        files = {}
        for key, value in ints.items():
            name = f"{basis}_{key.replace('alpha', 'a').replace('beta', 'b')}.npy"
            np.save(os.path.join(outdir, name), value)
            files[key] = name
        np.save(os.path.join(outdir, f"{basis}_coeff_a.npy"), ca)
        np.save(os.path.join(outdir, f"{basis}_coeff_b.npy"), cb)
        meta["bases"][basis] = {
            "files": files,
            "coefficients": [f"{basis}_coeff_a.npy", f"{basis}_coeff_b.npy"],
            "occupied_alpha": occ_a, "occupied_beta": occ_b,
            "occupied_indexing": "0-based orbital index within each spin block",
            "reference_determinant_energy": e_det,
            "fci_energy": e_fci_basis,
        }
        print(f"    {basis:>3}: E_ref = {e_det + e_nuc:.10f}  "
              f"E_FCI = {e_fci_basis + e_nuc:.10f}  (rebuild err "
              f"{abs(e_rebuilt - e_uhf_el):.1e})")
    with open(os.path.join(outdir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return fci_values


# ---------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--r-min", type=float, default=0.8)
    p.add_argument("--r-max", type=float, default=3.0)
    p.add_argument("--dr", type=float, default=0.1)
    p.add_argument("--basis-set", default="sto-3g")
    p.add_argument("--datadir", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--loc-starts", type=int, default=LOCALIZATION_STARTS)
    return p.parse_args()


def main():
    args = parse_args()
    n_pts = int(round((args.r_max - args.r_min) / args.dr)) + 1
    grid = [round(args.r_min + i * args.dr, 6) for i in range(n_pts)]
    os.makedirs(args.results, exist_ok=True)
    print(f"N2 / {args.basis_set}, {len(grid)} points R = {grid[0]} .. {grid[-1]} A")

    rhf = rhf_sweep(grid, args.basis_set)

    dm_rhf = rhf[grid[0]][0].make_rdm1()
    forward = uhf_sweep(grid, args.basis_set, (dm_rhf / 2, dm_rhf / 2),
                        "forward", rhf)
    backward = uhf_sweep(grid[::-1], args.basis_set,
                         forward[grid[-1]].make_rdm1(), "backward", rhf)
    uhf, uhf_from = {}, {}
    print("=" * 72 + "\nUHF solution choice\n" + "=" * 72)
    for r in grid:
        ef, eb = forward[r].e_tot, backward[r].e_tot
        uhf[r], uhf_from[r] = (forward[r], "forward") if ef <= eb else \
            (backward[r], "backward")
        flag = "  <-- two UHF solutions" if abs(ef - eb) > SWEEP_TOL else ""
        print(f"  R={r:.2f}  forward {ef:.10f}  backward {eb:.10f}  "
              f"keep {uhf_from[r]}{flag}")

    rows = []
    for r in grid:
        print("=" * 72 + f"\nR = {r:.2f} A\n" + "=" * 72)
        mf_rhf, rhf_ext = rhf[r]
        mf_uhf = uhf[r]
        mol = mf_uhf.mol
        nelec = (int(mol.nelec[0]), int(mol.nelec[1]))

        rcc, e_rccsd, e_rccsd_t, rcc_conv = run_cc(mf_rhf, "RCCSD")
        ucc, e_uccsd, e_uccsd_t, ucc_conv = run_cc(mf_uhf, "UCCSD")
        if not ucc_conv:
            raise RuntimeError(f"R={r:.2f}: UCCSD did not converge, so the "
                               f"natural orbitals are undefined")
        e_fci, s2_fci, fci_roots = fci_reference(mf_rhf, nelec)
        print(f"  RHF {mf_rhf.e_tot:.10f}  UHF {mf_uhf.e_tot:.10f} "
              f"(<S^2> {uhf_spin_square(mf_uhf):.4f})")
        print(f"  RCCSD {e_rccsd:.10f}  RCCSD(T) {e_rccsd_t:.10f}  "
              f"converged {rcc_conv}")
        print(f"  UCCSD {e_uccsd:.10f}  UCCSD(T) {e_uccsd_t:.10f}  "
              f"converged {ucc_conv}")
        print(f"  FCI   {e_fci:.10f}  <S^2> {s2_fci:.4f}  roots "
              + " ".join(f"{x:.8f}" for x in fci_roots))

        no_coeffs, no_occ = natural_orbitals(ucc, mf_uhf)
        lmo, lmo_chk = localize_spins(mol, list(mf_uhf.mo_coeff), nelec,
                                      "lmo", args.loc_starts)
        lno, lno_chk = localize_spins(mol, no_coeffs, nelec, "lno",
                                      args.loc_starts)
        bases_coeffs = {"mo": tuple(mf_uhf.mo_coeff), "lmo": tuple(lmo),
                        "no": tuple(no_coeffs), "lno": tuple(lno)}

        refs = {"E_RHF": float(mf_rhf.e_tot), "RHF_external_unstable": rhf_ext,
                "E_UHF": float(mf_uhf.e_tot), "S2_UHF": uhf_spin_square(mf_uhf),
                "UHF_sweep": uhf_from[r],
                "E_RCCSD": e_rccsd, "E_RCCSD_T": e_rccsd_t,
                "RCCSD_converged": rcc_conv,
                "E_UCCSD": e_uccsd, "E_UCCSD_T": e_uccsd_t,
                "UCCSD_converged": ucc_conv,
                "E_FCI": e_fci, "S2_FCI": s2_fci}
        extra = {"no_occupations_alpha": no_occ[0].tolist(),
                 "no_occupations_beta": no_occ[1].tolist(),
                 "localization_lmo": lmo_chk, "localization_lno": lno_chk}
        write_point(os.path.join(args.datadir, f"R_{r:.2f}"), r, mol, mf_uhf,
                    bases_coeffs, nelec, e_fci, refs, extra)
        rows.append({"R": r, "E_nuc": float(mol.energy_nuc()), **refs})

    path = os.path.join(args.results, "references.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path} and {len(grid)} points under {args.datadir}/")
    print("Next:  for d in data/R_*; do for b in " + " ".join(BASES) +
          "; do julia --project=../integrals ../integrals/"
          "uhf_integrals_to_pauli.jl $d --basis $b; done; done")


if __name__ == "__main__":
    main()
