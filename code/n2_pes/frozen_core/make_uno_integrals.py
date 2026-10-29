"""Frozen-core integrals in the UHF natural orbitals (UNO): restricted orbitals,
one set for both spins, whose closed-shell determinant is an exact singlet.

The UNOs diagonalize the spin-summed UHF density, S^1/2 (P_a + P_b) S^1/2,
with occupations between 0 and 2.  Paired alpha/beta UHF orbitals with overlap
cos θ give UNOs at 1 +- cos θ, so the stretched bonds show up as occupations
near 1.  The reference doubly occupies the 7 highest.  Inside the occupied
block the orbitals near occupation 2 (1s cores and lone pairs, nearly
degenerate) are semicanonicalized with the spin-averaged UHF Fock so the two
1s orbitals come out cleanly as the frozen core.  Bases:

    uno    occupied block: core + the rest by descending occupation; virtual
           block by descending occupation
    luno   uno with the 5 active occupied and 3 virtual orbitals Boys-localized
           apart (same determinant)

written to data_unocore/R_<R>/ in make_fc_integrals.py's layout.

Checks (each raises): full-space FCI in the UNO basis equals the reference FCI;
the two bases give the same active-space FCI; E_core + <ref|H_act|ref> equals
the full closed-shell determinant energy; alpha-beta overlap is the identity.

    python make_uno_integrals.py [--R 2.30] [--bases uno,luno]
"""
import argparse
import json
import os
import sys

import numpy as np
from pyscf import gto, scf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "integrals"))
from hubbard_uhf_integrals import determinant_energy, fci_energy, transform_integrals  # noqa: E402
from make_fc_integrals import NCORE, SHORT, frozen_core  # noqa: E402
from n2_pes import LOCALIZATION_STARTS, fci_lowest, localize_block  # noqa: E402

NEAR_TWO = 1.99


def uhf_natural_orbitals(mol, ca, cb, na, nb):
    s = mol.intor_symmetric("int1e_ovlp")
    pa, pb = ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T
    w, v = np.linalg.eigh(s)
    sh, sih = v @ np.diag(w ** 0.5) @ v.T, v @ np.diag(w ** -0.5) @ v.T
    occ, u = np.linalg.eigh(sh @ (pa + pb) @ sh)
    order = np.argsort(occ)[::-1]
    occ, c = occ[order], sih @ u[:, order]
    # semicanonicalize the near-doubly-occupied block with the spin-averaged Fock
    mf = scf.UHF(mol)
    fa, fb = mf.get_fock(dm=np.array((pa, pb)))
    two = np.where(occ > NEAR_TWO)[0]
    if len(two) < NCORE or two[-1] != len(two) - 1:
        raise RuntimeError(f"occupations {occ}: fewer than {NCORE} near 2")
    blk = c[:, two]
    e, r = np.linalg.eigh(blk.T @ (0.5 * (fa + fb)) @ blk)
    c[:, two] = blk @ r
    return occ, c, e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", default="2.30")
    p.add_argument("--bases", default="uno,luno")
    args = p.parse_args()
    want = args.bases.split(",")
    for R in args.R.split(","):
        src = os.path.join(HERE, "..", "data", f"R_{R}")
        meta = json.load(open(os.path.join(src, "metadata.json")))
        na, nb = meta["n_alpha"], meta["n_beta"]
        mol = gto.M(atom=f"N 0 0 0; N 0 0 {float(R):.6f}", basis="sto-3g", unit="Angstrom",
                    symmetry=False, spin=0, charge=0, verbose=0)
        ca, cb = (np.load(os.path.join(src, f"mo_coeff_{s}.npy")) for s in "ab")
        occ, c, e_two = uhf_natural_orbitals(mol, ca, cb, na, nb)
        print(f"R={R}  UNO occupations: " + " ".join(f"{x:.4f}" for x in occ))
        print(f"        semicanonical energies of the near-2 block: " + " ".join(f"{x:.3f}" for x in e_two))
        if occ[na - 1] - occ[na] < 1e-6:
            raise RuntimeError("no gap between occupied and virtual UNOs")
        if not np.all(e_two[:NCORE] < -10) or np.any(e_two[NCORE:] < -10):
            raise RuntimeError(f"core orbitals not separated: {e_two}")
        luno = None if "luno" not in want else np.hstack((
            c[:, :NCORE],
            localize_block(mol, c[:, NCORE:na], "luno occ", LOCALIZATION_STARTS)[0],
            localize_block(mol, c[:, na:], "luno vir", LOCALIZATION_STARTS)[0]))

        h_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
        eri8 = mol.intor("int2e", aosym="s8")
        e_nuc = float(mol.energy_nuc())
        e_fci_full = meta["E_FCI_with_mu"] + meta["constant"]
        s = mol.intor_symmetric("int1e_ovlp")
        dst = os.path.join(HERE, "data_unocore", f"R_{R}")
        os.makedirs(dst, exist_ok=True)
        bases, g = {}, None
        for basis, cc in (("uno", c), ("luno", luno)):
            if basis not in want:
                continue
            ints = transform_integrals(h_ao, eri8, cc, cc)
            e_det_full = determinant_energy(ints, list(range(na)), list(range(nb)))
            e_fci = fci_lowest(ints, (na, nb)) + e_nuc
            if abs(e_fci - e_fci_full) > 1e-8:
                raise RuntimeError(f"{basis}: full FCI {e_fci:.10f} != {e_fci_full:.10f}")
            O = cc.T @ s @ cc
            if np.max(np.abs(O - np.eye(len(O)))) > 1e-10:
                raise RuntimeError(f"{basis}: orbitals not orthonormal")
            e_core, act = frozen_core(ints, NCORE)
            n_a, n_b = na - NCORE, nb - NCORE
            e_ref = determinant_energy(act, list(range(n_a)), list(range(n_b)))
            if abs(e_core + e_ref - e_det_full) > 1e-9:
                raise RuntimeError(f"{basis}: E_core + E_ref(active) != full determinant energy")
            e_fc = fci_energy(act, (n_a, n_b))
            if g is None:
                g = (e_core, e_fc)
            elif abs(e_core + e_fc - sum(g)) > 1e-8 or abs(e_core - g[0]) > 1e-9:
                raise RuntimeError(f"{basis}: frozen-core energies differ from uno")
            for k, sh in SHORT.items():
                np.save(os.path.join(dst, f"{basis}_{sh}.npy"), act[k])
            np.save(os.path.join(dst, f"{basis}_ab_overlap.npy"), O[NCORE:, NCORE:])
            np.save(os.path.join(dst, f"{basis}_coeff.npy"), cc)
            bases[basis] = {
                "files": {k: f"{basis}_{sh}.npy" for k, sh in SHORT.items()},
                "occupied_alpha": list(range(n_a)), "occupied_beta": list(range(n_b)),
                "occupied_indexing": "0-based orbital index within each spin block (active space)",
                "reference_determinant_energy": e_ref, "fci_energy": e_fc,
            }
            print(f"  {basis:4s} E_ref {e_det_full + e_nuc:.8f} ({1e3 * (e_det_full + e_nuc - e_fci_full):+.2f} mHa"
                  f" vs FCI; UHF {1e3 * (meta['E_UHF_with_mu'] + meta['constant'] - e_fci_full):+.2f})"
                  f"  FCI(fc) {e_core + e_fc + e_nuc:.8f}")
        constant = e_nuc + g[0]
        out = {k: meta[k] for k in ("R_angstrom", "basis_set", "mu", "energy_offset", "references")}
        out.update({
            "model": f"N2 sto-3g R={R} A, frozen 1s core (unocore: {NCORE} orbitals per spin)",
            "n_orbitals": meta["n_orbitals"] - NCORE, "n_alpha": na - NCORE, "n_beta": nb - NCORE,
            "constant": constant, "constant_meaning": "nuclear repulsion + frozen-core energy",
            "E_nuc": e_nuc, "E_core": g[0],
            "E_UHF_with_mu": meta["E_UHF_with_mu"] + meta["constant"] - constant,
            "E_FCI_with_mu": g[1], "E_FCI_full_electrons": e_fci_full,
            "uno_occupations": [float(x) for x in occ],
            "bases": bases,
        })
        with open(os.path.join(dst, "metadata.json"), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
