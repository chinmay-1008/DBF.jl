"""Linear H_n chains (STO-3G, evenly spaced) for the system-size scan: integrals
in the UHF natural orbitals (UNO), in the layout of n2_pes/frozen_core, so the
same converter and vDBF driver run on them.

    python make_hchain.py [--n 4,6,8,10,12] [--d 1.00,2.00]

Per chain H_n and spacing d it writes data/H<n>/R_<d>/:
    uno_h1_a.npy ... uno_eri_ab.npy   integrals (restricted: alpha = beta)
    uno_ab_overlap.npy                identity (for build_s2.jl's S^2)
    uno_coeff.npy                     orbitals in the AO basis
    fci_vector.npy                    FCI ground state, PySCF string order
    metadata.json                     converter metadata + references

Orbitals.  RHF, then UHF from the RHF density, following the RHF -> UHF
instability when there is one (n2_pes.py's SCF machinery: every SCF must
converge, stability is followed to a stable solution).  The UNOs diagonalize
the spin-summed UHF density, so the closed-shell determinant on the n/2 highest
is an exact singlet.  Where UHF = RHF (short spacings) the occupations are all
2 or 0 and the UNOs are only defined up to rotations inside those blocks; those
blocks are semicanonicalized with the spin-averaged Fock, which makes them the
canonical RHF orbitals.

FCI: lowest of 3 roots of the (n/2, n/2) sector (direct_spin1), with <S^2>.
Checks (each raises): orbitals orthonormal; the UNO-basis FCI equals the
RHF-basis FCI (n <= 10); <ref|H|ref> from the integrals equals the AO-basis
energy of the closed-shell determinant.
"""
import argparse
import json
import os
import sys

import numpy as np
from pyscf import ao2mo, fci, gto, scf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "n2_pes"))
sys.path.insert(0, os.path.join(HERE, "..", "integrals"))
from hubbard_uhf_integrals import determinant_energy, transform_integrals  # noqa: E402
from n2_pes import (break_spin_symmetry, converge_scf, make_mf, rhf_external_unstable,  # noqa: E402
                    stabilise, uhf_spin_square)

SHORT = {"h1_alpha": "h1_a", "h1_beta": "h1_b", "eri_aa": "eri_aa", "eri_bb": "eri_bb", "eri_ab": "eri_ab"}
EDGE = 1e-8          # occupations within this of 2 or 0 are semicanonicalized


def chain(n, d):
    return gto.M(atom="; ".join(f"H 0 0 {i * d:.6f}" for i in range(n)), basis="sto-3g", unit="Angstrom",
                 symmetry=False, spin=0, charge=0, verbose=0)


def scf_pair(mol, tag):
    mf_rhf = stabilise(converge_scf(mol, "rhf", make_mf(mol, "rhf").get_init_guess(), f"{tag} RHF"), mol, "rhf", f"{tag} RHF")
    dm = mf_rhf.make_rdm1()
    mf_uhf = stabilise(converge_scf(mol, "uhf", np.array((dm / 2, dm / 2)), f"{tag} UHF"), mol, "uhf", f"{tag} UHF")
    if rhf_external_unstable(mf_rhf):
        mf_uhf = break_spin_symmetry(mf_uhf, mf_rhf, mol, f"{tag} UHF")
    return mf_rhf, mf_uhf


def uhf_natural_orbitals(mol, mf_uhf):
    s = mol.intor_symmetric("int1e_ovlp")
    pa, pb = mf_uhf.make_rdm1()
    w, v = np.linalg.eigh(s)
    sh, sih = v @ np.diag(w ** 0.5) @ v.T, v @ np.diag(w ** -0.5) @ v.T
    occ, u = np.linalg.eigh(sh @ (pa + pb) @ sh)
    order = np.argsort(occ)[::-1]
    occ, c = occ[order], sih @ u[:, order]
    fa, fb = mf_uhf.get_fock(dm=np.array((pa, pb)))
    for blk in (np.where(occ > 2 - EDGE)[0], np.where(occ < EDGE)[0]):
        if len(blk) > 1:
            e, r = np.linalg.eigh(c[:, blk].T @ (0.5 * (fa + fb)) @ c[:, blk])
            c[:, blk] = c[:, blk] @ r
    return occ, c


def fci_ground(h1, eri, n, nelec):
    solver = fci.direct_spin1.FCI()
    solver.conv_tol, solver.max_cycle, solver.nroots = 1e-12, 300, 3
    e, vecs = solver.kernel(h1, eri, n, nelec)
    k = int(np.argmin(e))
    return float(e[k]), vecs[k], float(solver.spin_square(vecs[k], n, nelec)[0]), [float(x) for x in e]


def point(n, d):
    tag = f"H{n} d={d:.2f}"
    mol = chain(n, d)
    na = nb = n // 2
    mf_rhf, mf_uhf = scf_pair(mol, tag)
    occ, c = uhf_natural_orbitals(mol, mf_uhf)
    s = mol.intor_symmetric("int1e_ovlp")
    if np.max(np.abs(c.T @ s @ c - np.eye(n))) > 1e-10:
        raise RuntimeError(f"{tag}: UNOs not orthonormal")
    h_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    eri8 = mol.intor("int2e", aosym="s8")
    e_nuc = float(mol.energy_nuc())
    ints = transform_integrals(h_ao, eri8, c, c)
    e_ref = determinant_energy(ints, list(range(na)), list(range(nb)))
    dm = 2 * c[:, :na] @ c[:, :na].T
    e_ref_ao = scf.RHF(mol).energy_elec(dm=dm)[0]
    if abs(e_ref - e_ref_ao) > 1e-9:
        raise RuntimeError(f"{tag}: <ref|H|ref> {e_ref} != AO energy {e_ref_ao}")
    e_fci, vec, s2_fci, roots = fci_ground(ints["h1_alpha"], ints["eri_aa"], n, (na, nb))
    if n <= 10:
        cr = mf_rhf.mo_coeff
        e_chk = fci_ground(cr.T @ h_ao @ cr, ao2mo.restore(1, ao2mo.full(eri8, cr), n), n, (na, nb))[0]
        if abs(e_chk - e_fci) > 1e-8:
            raise RuntimeError(f"{tag}: FCI in UNO {e_fci} != FCI in RHF orbitals {e_chk}")
    dst = os.path.join(HERE, "data", f"H{n}", f"R_{d:.2f}")
    os.makedirs(dst, exist_ok=True)
    for k, sh in SHORT.items():
        np.save(os.path.join(dst, f"uno_{sh}.npy"), ints[k])
    np.save(os.path.join(dst, "uno_ab_overlap.npy"), np.eye(n))
    np.save(os.path.join(dst, "uno_coeff.npy"), c)
    np.save(os.path.join(dst, "fci_vector.npy"), vec)
    refs = {"E_RHF": float(mf_rhf.e_tot), "E_UHF": float(mf_uhf.e_tot), "S2_UHF": uhf_spin_square(mf_uhf),
            "E_FCI": e_fci + e_nuc, "S2_FCI": s2_fci, "FCI_roots": [x + e_nuc for x in roots]}
    meta = {
        "model": f"H{n} chain sto-3g, spacing {d:.2f} A, UHF natural orbitals", "R_angstrom": d,
        "n_atoms": n, "basis_set": "sto-3g", "n_orbitals": n, "n_alpha": na, "n_beta": nb,
        "constant": e_nuc, "constant_meaning": "nuclear repulsion", "E_nuc": e_nuc, "mu": 0.0, "energy_offset": 0.0,
        "E_UHF_with_mu": refs["E_UHF"] - e_nuc, "E_FCI_with_mu": e_fci, "references": refs,
        "uno_occupations": [float(x) for x in occ], "fci_dimension": int(vec.size),
        "bases": {"uno": {
            "files": {k: f"uno_{sh}.npy" for k, sh in SHORT.items()},
            "occupied_alpha": list(range(na)), "occupied_beta": list(range(nb)),
            "occupied_indexing": "0-based orbital index within each spin block",
            "reference_determinant_energy": e_ref, "fci_energy": e_fci}},
    }
    with open(os.path.join(dst, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"{tag}: E_RHF {refs['E_RHF']:.8f}  E_UHF {refs['E_UHF']:.8f} (<S^2> {refs['S2_UHF']:.3f})  "
          f"E_ref {e_ref + e_nuc:.8f}  E_FCI {e_fci + e_nuc:.8f} (<S^2> {s2_fci:.4f}, dim {vec.size})  "
          f"ref - FCI {1e3 * (e_ref - e_fci):.1f} mHa", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", default="4,6,8,10,12")
    p.add_argument("--d", default="1.00,2.00")
    a = p.parse_args()
    for n in (int(x) for x in a.n.split(",")):
        if n % 2:
            raise ValueError("even n only (closed-shell singlet)")
        for d in (float(x) for x in a.d.split(",")):
            point(n, d)


if __name__ == "__main__":
    main()
