"""Best energy with k determinants for an H-chain point: H diagonalized in the
space of the k largest determinants of the FCI ground state (ideal selected CI),
for k on a log grid up to the full (n/2, n/2) sector.

    python schrodinger_bestk.py data/H8/R_2.00 [...]      # writes bestk.json there

Matrix-vector products use PySCF's FCI contraction on the full sector with the
other determinants zeroed, so nothing larger than the CI vector is stored.
Small spaces (k <= 60) are diagonalized densely, larger ones with eigsh started
from the truncated FCI vector.  The k = full point must reproduce E_FCI."""
import json
import os
import sys

import numpy as np
from pyscf import fci
from scipy.sparse.linalg import LinearOperator, eigsh


def bestk(d):
    meta = json.load(open(os.path.join(d, "metadata.json")))
    n, nelec = meta["n_orbitals"], (meta["n_alpha"], meta["n_beta"])
    h1, eri = np.load(os.path.join(d, "uno_h1_a.npy")), np.load(os.path.join(d, "uno_eri_aa.npy"))
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, n, nelec, 0.5)
    c = np.load(os.path.join(d, "fci_vector.npy"))
    shape, flat = c.shape, c.ravel()
    order = np.argsort(-np.abs(flat))
    e_fci = meta["E_FCI_with_mu"]

    def hv(idx, v):
        full = np.zeros(flat.size)
        full[idx] = v
        return fci.direct_spin1.contract_2e(h2e, full.reshape(shape), n, nelec).ravel()[idx]

    ks = np.unique(np.round(np.logspace(0, np.log10(flat.size), 36)).astype(int))
    rows = []
    for k in ks:
        idx = order[:k]
        if k <= 60:
            M = np.column_stack([hv(idx, e) for e in np.eye(k)])
            e = float(np.linalg.eigvalsh(0.5 * (M + M.T))[0])
        else:
            op = LinearOperator((k, k), matvec=lambda v, idx=idx: hv(idx, v), dtype=float)
            e = float(eigsh(op, k=1, which="SA", v0=flat[idx], tol=1e-10)[0][0])
        rows.append({"k": int(k), "E_err": e - e_fci, "weight": float(np.sum(flat[idx] ** 2))})
        print(f"{d}  k {k:8d}  E-FCI {e - e_fci:.3e}  weight {rows[-1]['weight']:.8f}", flush=True)
    if abs(rows[-1]["E_err"]) > 1e-8:
        raise RuntimeError(f"{d}: full space gives {rows[-1]['E_err']:.2e}, not FCI")
    with open(os.path.join(d, "bestk.json"), "w") as fh:
        json.dump({"dir": d, "n": n, "dim": int(flat.size), "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        bestk(d)
