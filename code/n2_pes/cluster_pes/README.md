# N₂ PES with vDBF on Big Red 200

This bundle runs the full N₂ STO-3G potential energy surface: frozen core, 16 qubits, R = 0.8–3.0 Å in steps of 0.1 Å (23 points). Four vDBF variants run at every point, with the same settings as the 2.3 Å study:

| Variant | Orbitals and start | Objective |
|---|---|---|
| `lno_bare` | LNO, UHF-like reference determinant | H |
| `lno_s2pen` | LNO, same reference | H + 0.05 S² |
| `lno_prep` | LNO, singlet projection of the reference (prep circuit with 1 − F = 1e-5, folded into H with cutoff 1e-6) | H |
| `uno_s2pen` | UNO, closed-shell reference (an exact singlet) | H + 0.05 S² |

All variants use operator truncation `CoeffTruncation(1e-6)`, gradient truncation 1e-6, 200 iterations, and 50 rotations per gradient. After each run, `replay_fc.jl` applies the stored rotations to |0⟩ with no truncation. That gives the true ⟨H⟩, Var, ⟨S²⟩ and fidelity at every checkpoint.

## What runs where

- **Already done locally:** the integrals, because they need PySCF. They are shipped as small `.npy` files with `metadata.json`, one directory per point.
- **On the cluster, two job arrays:**

  | Array | Script | One task per | Builds or runs |
  |---|---|---|---|
  | prepare | `prepare.slurm` → `prepare_one.sh` | R (`points.txt`, 23 lines) | Pauli Hamiltonians for LNO and UNO; S² operators; the singlet prep circuit for LNO |
  | vDBF | `run_task.slurm` → `run_one.sh` | (R, variant) (`tasks.txt`, 92 lines) | vDBF, then the exact replay |

  `submit.sh` submits the prepare array, then the vDBF tasks as two arrays split by memory, `small` and `big`. Both wait for every prepare task to succeed (`afterok`):

  | vDBF array | Variants | Memory | Cores |
  |---|---|---|---|
  | `n2pes_small` | `lno_bare`, `uno_s2pen` | `MEM_SMALL`, 32G | 2 |
  | `n2pes_big` | `lno_s2pen`, `lno_prep` | `MEM_BIG`, 64G | 2 |

**Checks inside the prepare step.** Each one stops the task on failure:
- the converter's exact diagonalization of the (5α, 5β) sector equals PySCF FCI;
- S² matches the state-vector S²;
- each prep circuit's replayed fidelity matches its build.

## Contents

```
config.sh          Julia (juliaup on PATH), vDBF settings (COEFF, LAMBDA, PREP, PREP_TOL)
setup.sh           one-time: install and precompile the Julia environments (login node, needs internet)
points.txt         the 23 R values (prepare array)
tasks.txt          one "R variant" per line (vDBF array)
prepare.slurm      SLURM array: prepare_one.sh for one R
run_task.slurm     SLURM array: run_one.sh for one line of tasks.txt
submit.sh          submits the prepare array, then the small- and big-memory vDBF arrays
prepare_one.sh     Hamiltonians, S², prep circuit for one R (skips files that exist)
run_one.sh         vDBF + replay for one R and variant (skips outputs that exist)
collect_pes.py     afterwards: pes_summary.csv and pes.png
stage.sh           LOCAL only: rebuilds DBF.jl/, points.txt and tasks.txt from the repo
DBF.jl/            the code and inputs, laid out as in the repo
  src/ ext/ Project.toml                     DBF package
  code/integrals/                            converter + env for prepare and replay
  code/n2_pes/run_dbf_pes.jl, extrapolate.py, Project.toml, Manifest.toml (vDBF env)
  code/n2_pes/frozen_core/                   replay_fc.jl, sv_dbf.jl, build_s2.jl, build_singlet_prep.jl
  code/n2_pes/frozen_core/data_nocore/R_*/   LNO integrals + metadata
  code/n2_pes/frozen_core/data_unocore/R_*/  UNO integrals + metadata
```

## Steps

1. **Copy the folder to the cluster**, to `/N/u/cshrikh/BigRed200/v_DBF/n2_pes`:
   ```bash
   ssh cshrikh@bigred200.uits.iu.edu 'mkdir -p /N/u/cshrikh/BigRed200/v_DBF/n2_pes'
   scp -r cluster_pes/* cshrikh@bigred200.uits.iu.edu:/N/u/cshrikh/BigRed200/v_DBF/n2_pes/
   ```
2. **Check the job settings.** The job scripts already follow the SPP showcase (`spp_showcase/cluster/submit_spp.sh`): partition `general`, account `r01859`, `module reset`, then Julia from `~/.juliaup/bin`.
   - **vDBF Julia project:** `run_task.slurm` sets `DBF_ENV=/N/u/cshrikh/BigRed200/Projects/n2_dbf/DBF.jl`, a clone of the `qual_exam` branch (see step 3). Comment that line out to use the bundled DBF copy instead. Both are the same DBF code.
   - **Prepare and replay:** these always use the bundled `DBF.jl/code/integrals` environment, because they need NPZ.
   - **Time:** `--time` is in `prepare.slurm` and `run_task.slurm`. Memory is set per array in `submit.sh` (see Resources below).
3. **Clone DBF and install the environments, once, on a login node.** PauliOperators must sit next to `DBF.jl`, because DBF's `Project.toml` points to it at `../PauliOperators.jl`:
   ```bash
   mkdir -p /N/u/cshrikh/BigRed200/Projects/n2_dbf && cd /N/u/cshrikh/BigRed200/Projects/n2_dbf
   git clone -b qual_exam https://github.com/chinmay-1008/DBF.jl.git
   git clone https://github.com/nmayhall/PauliOperators.jl.git
   cd /N/u/cshrikh/BigRed200/v_DBF/n2_pes && bash setup.sh
   ```
   It also checks the `DBF_ENV` from `run_task.slurm`. That environment needs JSON, `SparsePauliVector`, and the `dbf_groundstate` keywords the script uses. If anything is missing, it says what. Packages go to the default depot, `~/.julia`, as for the SPP runs. PauliOperators is pinned to a commit of `nmayhall/PauliOperators.jl` `main`, so the login node needs GitHub access. The environments were resolved with Julia 1.12.7; check `julia --version` there.
4. **Smoke test, a few minutes, in an interactive session.**
   ```bash
   bash prepare_one.sh 1.20
   DBF_ENV=/N/u/cshrikh/BigRed200/Projects/n2_dbf/DBF.jl EXTRA_ARGS="--max-iter 3" bash run_one.sh 1.20 uno_s2pen
   rm DBF.jl/code/n2_pes/frozen_core/data_unocore/R_1.20/dbf_uno_* DBF.jl/code/n2_pes/frozen_core/replay_results/uno_1.20_*
   ```
   Keep the prepared inputs: the prepare job skips files that already exist. Delete the 3-iteration run outputs as shown, or the real run will skip that point.
5. **Submit.** Optionally start with a test at one geometry, then submit the rest.
   ```bash
   bash submit.sh '^2\.30 '            # prepare array, then the 4 tasks at 2.3 A
   NOPREP=1 bash submit.sh             # later: everything (finished tasks skip)
   bash submit.sh                      # or all at once
   MEM_BIG=96G NOPREP=1 bash submit.sh # resubmit after an out-of-memory kill
   ```
   The optional argument is a `grep -E` pattern on `tasks.txt` lines. `MAXPAR` caps how many tasks of each array run at once (default 20).
6. **Collect.** When the jobs are done, run this (needs numpy and matplotlib):
   ```bash
   python collect_pes.py
   ```
   It prints a table of true ⟨H⟩ − FCI and E0 − FCI, and writes `pes_summary.csv` and `pes.png`.

**Where outputs go**, all under `DBF.jl/code/n2_pes/frozen_core/`:

| Output | Location |
|---|---|
| Prepared inputs | `data_*/R_<R>/hamiltonian_<b>.jld2`, `s2_<b>.jld2`, `prep_singlet5_lno.jld2` |
| vDBF result | `data_*/R_<R>/<stem>.json` (final numbers + per-iteration history) |
| vDBF checkpoint | `data_*/R_<R>/<stem>_ckpt.jld2` (generators and angles; the replay needs it) |
| Replay | `replay_results/<basis>_<R>_<variant>.json` |
| Logs | `results/logs/prepare_<R>.out`, `dbf_<label>.out`, `replay_<label>.out`, plus `slurm_logs/` |
| Peak memory | `results/logs/maxrss.txt`: one line per vDBF task, from `/usr/bin/time -v`, as in the SPP showcase |

To bring results back for plotting:
```bash
rsync -av --include='*/' --include='*.json' --include='*.out' --include='*.txt' --include='*.sh' --exclude='*' <cluster>:<path>/ cluster_pes/
```
Add `--include='*_ckpt.jld2'` to that command if you also want the circuits.

## Resources

- **Prepare:** about 5–10 minutes per R and under 2 GB. The two converter runs dominate.
- **vDBF:** these timings come from the 2.3 Å runs on an M-series laptop, single-threaded, sharing the CPU:

  | Variant at 2.3 Å | Wall time | Largest operator |
  |---|---|---|
  | `lno_bare` | ~13 min | ~10M terms |
  | `lno_s2pen` | 3.4 h | ~29M terms (54M in [H, Z]) |
  | `lno_prep` | 2.8 h, including a 50 min fold of the 2219-rotation prep | 9.4M terms after the fold |
  | `uno_s2pen` | 1.75 h | ~11M terms (29M in [H, Z]); 22 GB peak footprint |

  - **Time:** other geometries can be slower. The 1.4–2.0 Å region has the smallest spin gaps. The default `--time=48:00:00` is generous. A run that hits the wall-time restarts from scratch, because the checkpoints can't be resumed.
  - **Memory:** the general partition has 256 GB nodes with no per-node cap. 1920 MB per core is only the default when no `--mem` is given. A `SparsePauliVector` term costs 256 bytes, counting the scratch buffers. Without a limit, Julia's garbage collector lets the heap grow to about twice the live data: `uno_s2pen` at 2.3 Å (11M terms in H, 29M in [H, Z]) peaked at 22 GB. Under Slurm, `run_one.sh` therefore passes `--heap-size-hint` at 80% of the job's memory. `lno_s2pen` and `lno_prep` reach about 29M terms in H and 68M in [H, Z], so they get 64G. Check `maxrss.txt` after the first tasks; an out-of-memory task can be resubmitted with a larger `MEM_BIG` or `MEM_SMALL`.
  - **Cores:** the vDBF flow is single-threaded (neither DBF nor PauliOperators uses threads there). Each task gets 2 cores, and the second helps the replay. Memory dominates the billing: 0.512 units per GB against 1 per core.

## Notes

- **`lno_prep` at R ≤ 1.2 Å exits immediately.** There the LNO reference is already within 1 − F < 1e-5 of its singlet projection: an exact singlet below 1.2 Å, and ⟨S²⟩ = 1e-5 at 1.2 Å. So the prep circuit is empty and the run would repeat `lno_bare`. The task exits right away, and `collect_pes.py` reuses `lno_bare` for that curve.
- **Penalty runs report H + 0.05 S².** For `*_s2pen`, the logged energy and E0 are of H + 0.05 S². The replay gives the true ⟨H⟩ and ⟨S²⟩ separately.
- **Each run is measured from its own FCI.** The UNO tree freezes the UNO 1s orbitals, so its frozen-core FCI differs from LNO's by a few µHa.
- **Changing settings.** Change λ or the cutoff in `config.sh`. Output names include them (`_s2pen<λ>`, `_c<coeff>`), so runs with different settings don't overwrite each other.
