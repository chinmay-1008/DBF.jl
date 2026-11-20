"""Frozen-core (two 1s orbitals) RCCSD/RCCSD(T) and UCCSD/UCCSD(T) for N2/STO-3G
at every R, on the same RHF and UHF solutions as results/references.csv (the
UHF restarts from the stored canonical UHF orbitals in data/R_*/).
    python frozen_core/fc_cc.py frozen_core/results/fc_cc.csv     # from code/n2_pes/, needs PySCF
The frozen 1s orbitals are the canonical RHF/UHF ones (vDBF freezes UNO/NO ones;
the frozen-core energies differ by a few uHa)."""
import csv, os, sys
import numpy as np
sys.path.insert(0, ".")
from n2_pes import build_mol, run_cc, rhf_sweep
from pyscf import scf, cc

refs = {round(float(r["R"]), 2): r for r in csv.DictReader(open("results/references.csv"))}
rhfs = rhf_sweep(sorted(refs), "sto-3g")
rows = []
for R in sorted(refs):
    ref = refs[R]
    mol = build_mol(R, "sto-3g")
    rhf = rhfs[R][0]
    assert abs(rhf.e_tot - float(ref["E_RHF"])) < 1e-7, (R, "RHF", rhf.e_tot, ref["E_RHF"])

    d = f"data/R_{R:.2f}"
    ca, cb = np.load(f"{d}/mo_coeff_a.npy"), np.load(f"{d}/mo_coeff_b.npy")
    na, nb = mol.nelec
    dm = np.array([ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T])
    uhf = scf.UHF(mol); uhf.conv_tol = 1e-11; uhf.verbose = 0
    uhf.kernel(dm0=dm)
    assert abs(uhf.e_tot - float(ref["E_UHF"])) < 1e-7, (R, "UHF", uhf.e_tot, ref["E_UHF"])

    row = {"R": R, "E_RHF": rhf.e_tot, "E_UHF": uhf.e_tot}
    for tag, mf in (("R", rhf), ("U", uhf)):
        orig = cc.CCSD
        cc.CCSD = lambda m, _o=orig: _o(m, frozen=2)   # run_cc builds cc.CCSD(mf)
        try:
            mycc, e, et, ok = run_cc(mf, f"{tag}CCSD R={R}")
        finally:
            cc.CCSD = orig
        row[f"E_{tag}CCSD"], row[f"E_{tag}CCSD_T"], row[f"{tag}CCSD_converged"] = e, et, ok
    rows.append(row)
    print(f"R={R:.2f}  RCCSD {row['E_RCCSD']:.8f} (T) {row['E_RCCSD_T']:.8f} {row['RCCSD_converged']}   "
          f"UCCSD {row['E_UCCSD']:.8f} (T) {row['E_UCCSD_T']:.8f} {row['UCCSD_converged']}", flush=True)

with open(sys.argv[1], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(sys.argv[1])
