"""Frozen-core N2 integrals: freeze the two N 1s orbitals (indices 0, 1 in every
basis, h1 ~ -26 Ha) and write an 8-orbital, (5a, 5b), 16-qubit problem in the
same layout as ../data/R_<R>/, so ../../integrals/uhf_integrals_to_pauli.jl and
../run_dbf_pes.jl work on it unchanged.

The frozen orbitals differ between bases that share a reference (mo/lmo freeze
the UHF 1s orbitals, no/lno the two core-like NOs), so E_core and the
frozen-core FCI differ too.  run_dbf_pes.jl takes one constant and one FCI per
point directory, so each core group gets its own tree:

    data_uhfcore/R_<R>/   mo, lmo
    data_nocore/R_<R>/    no, lno

with
    constant  = E_nuc + E_core
    h_eff^a   = h^a + J^a[core a] - K^a[core a] + J^ab[core b]   (active block)
    E_core    = 1/2 tr_core (h + h_eff), both spins

Each basis also gets <basis>_ab_overlap.npy, the active block of
O_pq = <phi^alpha_p | phi^beta_q>: alpha and beta spatial orbitals differ in
all four bases, so the physical S+ = sum_pq O_pq a+_p,alpha a_q,beta.

Checks (each raises):
  * E_core + <ref|H_active|ref> equals the full reference determinant energy
    (the frozen orbitals are occupied in the reference) to 1e-9;
  * both bases of a core group give the same active-space FCI to 1e-8;
  * in the mo basis, <S^2> of the UHF determinant from O reproduces
    references.csv's S2_UHF to 1e-8.

    python make_fc_integrals.py [--R 1.10,1.70,2.00,2.30] [--bases mo,lmo,no,lno]
"""
import argparse
import json
import os
import sys

import numpy as np
from pyscf import gto

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data")
sys.path.insert(0, os.path.join(HERE, "..", "..", "integrals"))
from hubbard_uhf_integrals import determinant_energy, fci_energy  # noqa: E402

NCORE = 2
GROUP = {"mo": "uhfcore", "lmo": "uhfcore", "no": "nocore", "lno": "nocore"}
SHORT = {"h1_alpha": "h1_a", "h1_beta": "h1_b", "eri_aa": "eri_aa",
         "eri_ab": "eri_ab", "eri_bb": "eri_bb"}


def frozen_core(ints, nc):
    h_a, h_b = ints["h1_alpha"], ints["h1_beta"]
    aa, ab, bb = ints["eri_aa"], ints["eri_ab"], ints["eri_bb"]
    c = slice(0, nc)
    ja = np.einsum("pqcc->pq", aa[:, :, c, c])
    ka = np.einsum("pccq->pq", aa[:, c, c, :])
    jb = np.einsum("pqcc->pq", bb[:, :, c, c])
    kb = np.einsum("pccq->pq", bb[:, c, c, :])
    jab = np.einsum("pqcc->pq", ab[:, :, c, c])   # beta core seen by alpha
    jba = np.einsum("ccpq->pq", ab[c, c, :, :])   # alpha core seen by beta
    fa, fb = h_a + ja - ka + jab, h_b + jb - kb + jba
    e_core = 0.5 * (np.trace((h_a + fa)[c, c]) + np.trace((h_b + fb)[c, c]))
    a = slice(nc, None)
    return float(e_core), {"h1_alpha": fa[a, a], "h1_beta": fb[a, a],
                           "eri_aa": aa[a, a, a, a], "eri_ab": ab[a, a, a, a],
                           "eri_bb": bb[a, a, a, a]}


def ab_overlap(R, src, basis):
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {float(R):.6f}", basis="sto-3g", unit="Angstrom",
                symmetry=False, spin=0, charge=0, verbose=0)
    ca = np.load(os.path.join(src, f"{basis}_coeff_a.npy"))
    cb = np.load(os.path.join(src, f"{basis}_coeff_b.npy"))
    return ca.T @ mol.intor("int1e_ovlp") @ cb


def determinant_s2(O, occ_a, occ_b):
    na, nb = len(occ_a), len(occ_b)
    sz = 0.5 * (na - nb)
    return sz * sz + sz + nb - float(np.sum(np.abs(O[np.ix_(occ_a, occ_b)]) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", default="1.10,1.70,2.00,2.30")
    p.add_argument("--bases", default="mo,lmo,no,lno")
    args = p.parse_args()
    for R in args.R.split(","):
        src = os.path.join(SRC, f"R_{R}")
        meta = json.load(open(os.path.join(src, "metadata.json")))
        na, nb = meta["n_alpha"] - NCORE, meta["n_beta"] - NCORE
        n_act = meta["n_orbitals"] - NCORE
        groups = {}
        for basis in args.bases.split(","):
            full = {k: np.load(os.path.join(src, f"{basis}_{s}.npy")) for k, s in SHORT.items()}
            e_core, act = frozen_core(full, NCORE)
            occ_a, occ_b = list(range(na)), list(range(nb))
            e_ref = determinant_energy(act, occ_a, occ_b)
            e_ref_full = meta["bases"][basis]["reference_determinant_energy"]
            if abs(e_core + e_ref - e_ref_full) > 1e-9:
                raise RuntimeError(f"R={R} {basis}: E_core + E_ref(active) = "
                                   f"{e_core + e_ref:.12f} != {e_ref_full:.12f}")
            e_fci = fci_energy(act, (na, nb))
            g = groups.setdefault(GROUP[basis], {"E_core": e_core, "E_fci": e_fci, "bases": {}})
            if abs((e_core + e_fci) - (g["E_core"] + g["E_fci"])) > 1e-8:
                raise RuntimeError(f"R={R} {basis}: frozen-core FCI differs within {GROUP[basis]}")
            dst = os.path.join(HERE, f"data_{GROUP[basis]}", f"R_{R}")
            os.makedirs(dst, exist_ok=True)
            for k, s in SHORT.items():
                np.save(os.path.join(dst, f"{basis}_{s}.npy"), act[k])
            O = ab_overlap(R, src, basis)
            occ_full = list(range(meta["n_alpha"]))
            s2_ref = determinant_s2(O, occ_full, list(range(meta["n_beta"])))
            if basis == "mo":
                s2_uhf = float(meta["references"]["S2_UHF"])
                if abs(s2_ref - s2_uhf) > 1e-8:
                    raise RuntimeError(f"R={R}: <S^2>(UHF det) {s2_ref} != S2_UHF {s2_uhf}")
            np.save(os.path.join(dst, f"{basis}_ab_overlap.npy"), O[NCORE:, NCORE:])
            # Electronic energies relative to this group's constant
            shift = e_core - g["E_core"]
            g["bases"][basis] = {
                "files": {k: f"{basis}_{s}.npy" for k, s in SHORT.items()},
                "occupied_alpha": occ_a, "occupied_beta": occ_b,
                "occupied_indexing": "0-based orbital index within each spin block (active space)",
                "reference_determinant_energy": e_ref + shift,
                "fci_energy": e_fci + shift,
            }
            if abs(shift) > 1e-9:
                raise RuntimeError(f"R={R} {basis}: E_core differs within {GROUP[basis]} by {shift}")
            print(f"R={R} {basis:3s}  E_core {e_core:.8f}  E_ref {e_core + e_ref + meta['constant']:.8f}"
                  f"  FCI(fc) {e_core + e_fci + meta['constant']:.8f}"
                  f"  FCI(full) {meta['E_FCI_with_mu'] + meta['constant']:.8f}"
                  f"  core corr {1e3 * (e_core + e_fci - meta['E_FCI_with_mu']):.3f} mHa", flush=True)
        for name, g in groups.items():
            constant = meta["constant"] + g["E_core"]
            out = {k: meta[k] for k in ("R_angstrom", "basis_set", "mu", "energy_offset", "references")}
            out.update({
                "model": f"N2 sto-3g R={R} A, frozen 1s core ({name}: {NCORE} orbitals per spin)",
                "n_orbitals": n_act, "n_alpha": na, "n_beta": nb,
                "constant": constant, "constant_meaning": "nuclear repulsion + frozen-core energy",
                "E_nuc": meta["constant"], "E_core": g["E_core"],
                "E_UHF_with_mu": meta["E_UHF_with_mu"] + meta["constant"] - constant,
                "E_FCI_with_mu": g["E_fci"],
                "E_FCI_full_electrons": meta["E_FCI_with_mu"] + meta["constant"],
                "bases": g["bases"],
            })
            path = os.path.join(HERE, f"data_{name}", f"R_{R}", "metadata.json")
            with open(path, "w") as fh:
                json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
