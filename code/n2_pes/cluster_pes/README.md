# N₂ PES with vDBF on Big Red 200: five orbital bases

This bundle runs the full N₂ STO-3G potential energy surface: frozen core, 16 qubits, R = 0.8–3.0 Å in steps of 0.1 Å (23 points). It runs every orbital basis with and without the S² penalty, plus a singlet-projected start for the four spin-broken references. The point is to see whether localization causes problems.

The five bases:

| Basis | Orbitals | Reference determinant |
|---|---|---|
| `mo` | canonical UHF | UHF (spin-broken) |
| `lmo` | `mo`, Boys-localized within the occupied and within the virtual block | same as `mo` |
| `no` | UCCSD natural orbitals, per spin | UHF-like (spin-broken) |
| `lno` | `no`, Boys-localized the same way | same as `no` |
| `uno` | UHF natural orbitals: the spin-summed UHF density, one set for both spins | closed shell, an exact singlet |

Each basis runs with up to three kinds:

| Kind | Objective and start |
|---|---|
| `bare` | H |
| `s2pen` | H + 0.05 S² |
| `prep` | H, from the singlet projection of the reference: a prep circuit with 1 − F = 1e-5, folded into H with cutoff 1e-6. Not for `uno`, which is already a singlet. |

That gives 14 variants per point (5 + 5 + 4) and **322 tasks** (`tasks.txt`, one `R variant` per line).

All variants use:
- operator truncation `CoeffTruncation(1e-6)`
- gradient truncation 1e-6
- 200 iterations, 50 rotations per gradient

After each run, `replay_fc.jl` applies the stored rotations to |0⟩ with no truncation. That gives the true ⟨H⟩, Var, ⟨S²⟩ and fidelity at every checkpoint.

## What runs where

- **Already done locally:** the integrals, because they need PySCF. They are shipped as small `.npy` files with `metadata.json`, in three trees, one per choice of frozen core:
  - `data_uhfcore` (mo, lmo)
  - `data_nocore` (no, lno)
  - `data_unocore` (uno)
- **On the cluster:**
  1. **prepare array** (`prepare.slurm` → `prepare_one.sh`), one task per R. It builds the 5 Pauli Hamiltonians, the S² operators, and the 4 singlet prep circuits: about 2 minutes and under 2 GB per R.
  2. **node jobs** (`node.slurm` → `pool.py`). Each takes a whole node (128 cores, all its memory) and keeps as many vDBF tasks running as fit in memory. Several node jobs share one task list. They start only after every prepare task has succeeded.

**How `pool.py` packs a node.** vDBF is single-threaded and its memory grows with the operator, so memory sets how many tasks fit, not cores.
- Each task reserves the amount listed in `memory.txt` for its variant, halved for R ≤ 1.2 Å. A task starts only when its reservation fits in 90% of the node's memory.
- A running task counts as the larger of its reservation and its actual resident memory.
- The biggest tasks go first and smaller ones fill the gaps.
- Julia's `--heap-size-hint` is set to 85% of the task's reservation.
- If the running tasks together pass 95% of the budget, the most recently started one is killed (the least work lost) and requeued with a bigger reservation.
- A task killed for memory by the system is retried with twice its reservation, up to 3 tries in all.
- No task starts with less than 10 h of wall time left. If a job is stopped, its running tasks' claims are released and the next job picks them up.

**Checks inside the prepare step.** Each one stops the task on failure:
- the converter's exact diagonalization of the (5α, 5β) sector equals PySCF FCI;
- S² matches the state-vector S²;
- each prep circuit's replayed fidelity matches its build.

## Contents

```
config.sh          Julia (juliaup on PATH), DBF_ENV, vDBF settings (COEFF, LAMBDA, PREP, PREP_TOL)
setup.sh           one-time: install and precompile the Julia environments (login node, needs internet)
points.txt         the 23 R values (prepare array)
tasks.txt          one "R variant" per line (322)
memory.txt         memory reservation per variant, GB (pool.py)
prepare.slurm      SLURM array: prepare_one.sh for one R
node.slurm         SLURM whole-node job: pool.py
submit.sh          submits the prepare array, then NODES node jobs
prepare_one.sh     Hamiltonians, S², prep circuits for one R (skips files that exist)
run_one.sh         vDBF + replay for one R and variant (skips outputs that exist)
pool.py            fills a node with run_one.sh tasks (python3, standard library only)
collect_pes.py     afterwards: pes_summary.csv and pes.png
stage.sh           LOCAL only: rebuilds DBF.jl/, points.txt and tasks.txt from the repo
DBF.jl/            the code and inputs, laid out as in the repo
```

## Steps

1. **Copy the folder to the cluster**, to `/N/u/cshrikh/BigRed200/v_DBF/n2_pes`. If the old bundle is already there, delete it first: its `run_task.slurm` and old logs are no longer used.
   ```bash
   ssh cshrikh@bigred200.uits.iu.edu 'mkdir -p /N/u/cshrikh/BigRed200/v_DBF/n2_pes'
   scp -r cluster_pes/* cshrikh@bigred200.uits.iu.edu:/N/u/cshrikh/BigRed200/v_DBF/n2_pes/
   ```
2. **Check the job settings.** The jobs follow the SPP showcase: partition `general`, account `r01859`, `module reset`, then Julia from `~/.juliaup/bin`.
   - **vDBF Julia project:** `config.sh` sets `DBF_ENV=/N/u/cshrikh/BigRed200/Projects/n2_dbf/DBF.jl`, a clone of the `qual_exam` branch (step 3). If that directory doesn't exist, the bundled copy of the same code is used.
   - **Time:** `--time` is in `prepare.slurm` (6 h) and `node.slurm` (48 h).
3. **Clone DBF and install the environments, once, on a login node.** PauliOperators must sit next to `DBF.jl`, because DBF's `Project.toml` points to it at `../PauliOperators.jl`.
   ```bash
   mkdir -p /N/u/cshrikh/BigRed200/Projects/n2_dbf && cd /N/u/cshrikh/BigRed200/Projects/n2_dbf
   git clone -b qual_exam https://github.com/chinmay-1008/DBF.jl.git
   git clone https://github.com/nmayhall/PauliOperators.jl.git
   cd /N/u/cshrikh/BigRed200/v_DBF/n2_pes && bash setup.sh
   ```
   If the clone already exists, run `git pull` in it instead. `setup.sh` checks that `DBF_ENV` has JSON, `SparsePauliVector` and the `dbf_groundstate` keywords the script uses, and says what's missing.
4. **Submit.**
   ```bash
   bash submit.sh                       # prepare array, then 4 node jobs on all 322 tasks
   NODES=8 bash submit.sh               # more node jobs
   bash submit.sh '^2\.30 '             # only the 14 tasks at 2.3 A (grep -E pattern on tasks.txt)
   ```
   After a job ends (time limit or done), resubmit with `NOPREP=1 bash submit.sh`. Finished tasks are skipped, and unfinished ones are picked up again. To also retry tasks marked failed, add `RETRY_FAILED=1`.
5. **Follow progress:**
   ```bash
   tail -f queue/events.log                      # start / done / retry / FAILED, every node
   ls queue/done | wc -l; ls queue/failed        # how many finished, which gave up
   cat DBF.jl/code/n2_pes/frozen_core/results/logs/maxrss.txt   # peak memory per task
   ```
   Use `maxrss.txt` to tune `memory.txt`. Smaller reservations pack more tasks per node.
6. **Collect.** When the jobs are done, run this (needs numpy and matplotlib):
   ```bash
   python collect_pes.py
   ```
   It prints true ⟨H⟩ − FCI and E0 − FCI for every R and variant, and writes:
   - `pes_summary.csv`
   - `pes.png`: columns H, H + 0.05 S², singlet-projected start; rows true ⟨H⟩ − FCI, E0 − FCI, ⟨S²⟩. Dashed lines are canonical bases and solid lines localized ones, so the effect of localization is the gap between the two lines of the same colour.
7. **Another truncation.** The default sweep is `COEFF=1e-6`. A second sweep at another coefficient can run alongside it:
   ```bash
   COEFF=1e-5 NODES=1 NOPREP=1 bash submit.sh
   tail -f queue_c1e-5/events.log
   COEFF=1e-5 python collect_pes.py              # pes_summary_c1e-5.csv, pes_c1e-5.png
   ```
   It needs no new inputs: the Hamiltonians, S² operators and prep circuits do not depend on the coefficient. It keeps everything apart from the 1e-6 sweep:
   - its own queue, `queue_c1e-5/`, so the 1e-6 done markers do not skip its tasks;
   - its own run files: stems without the `_c` suffix, `dbf_<b>[...].json`;
   - its own replays and logs: `<b>_<R>_<variant>_c1e-5`.

   Memory reservations are `memory.txt` × `MEM_SCALE`, 0.3 by default at 1e-5. On a laptop the UNO + S² runs at 1e-5 had 4–5× fewer terms and finished 4–6× faster than at 1e-6.

**Where outputs go**, all under `DBF.jl/code/n2_pes/frozen_core/`:

| Output | Location |
|---|---|
| Prepared inputs | `data_*/R_<R>/hamiltonian_<b>.jld2`, `s2_<b>.jld2`, `prep_singlet5_<b>.jld2` |
| vDBF result | `data_*/R_<R>/<stem>.json` (final numbers + per-iteration history) |
| vDBF checkpoint | `data_*/R_<R>/<stem>_ckpt.jld2` (generators and angles; the replay needs it) |
| Replay | `replay_results/<b>_<R>_<variant>.json` |
| Logs | `results/logs/prepare_<R>.out`, `prep_<b>_<R>.out`, `dbf_<label>.out`, `replay_<label>.out`; plus `queue/logs/` and `slurm_logs/` |
| Peak memory | `results/logs/maxrss.txt` |

To bring results back for plotting:
```bash
rsync -av --include='*/' --include='*.json' --include='*.out' --include='*.txt' --include='*.csv' --exclude='*' <cluster>:<path>/ cluster_pes/
```
Add `--include='*_ckpt.jld2'` to that command if you also want the circuits.

## Resources

- **Storage.** Far below the 50 GB quota.
  - The bundle is about 15 MB.
  - Each task writes about 1 MB: a 0.8 MB checkpoint, plus the result JSON, replay JSON and logs. 322 tasks come to about 0.3 GB.
  - The prepared inputs are about 50 MB.
  - The Julia depot in `~/.julia` takes a few GB, and is shared with the SPP runs.
- **Memory and time.** From the 2.3 Å runs on a laptop at cutoff 1e-6:

  | Variant at 2.3 Å | Time | Largest operator |
  |---|---|---|
  | `lno_bare` | 0.2 h | 3.4M terms |
  | `uno_s2pen` | 1.8 h | 11M terms, 22 GB peak (no heap hint) |
  | `lmo_s2pen` | 2.7 h | 24M terms |
  | `lno_prep` | 2.8 h | 23M terms, including a 50 min fold of the 2219-rotation prep |
  | `lno_s2pen` | 3.4 h | 29M terms |

  - `mo` and `no` haven't been run at 1e-6. At 1e-5 they reached about 1.5–2× the terms of the localized bases, so `memory.txt` gives them more (48 GB bare; 96 GB for s2pen and prep).
  - Along R, `uno_s2pen` ranged from 1.5M terms (0.8 Å) to 20M (2.1 Å).
  - With the default reservations, a node runs about 3–6 tasks at once. 322 tasks take very roughly 25–40 h on 4 nodes. More nodes (`NODES=8`), or smaller reservations once `maxrss.txt` shows real peaks, shorten that.
  - A task that hits the job's time limit restarts from scratch, because the checkpoints can't be resumed.
- **Cores.** vDBF is single-threaded, so most of a node's 128 cores sit idle; memory is what limits how many tasks fit. The replay step uses 4 threads. A whole node is billed as 128 cores plus 256 GB × 0.512 for as long as the job runs.

## Notes

- **Empty prep circuits exit immediately.** At short R the UHF reference equals the RHF one, so it is already a singlet and its prep circuit is empty. The `<b>_prep` task exits right away, and `collect_pes.py` reuses `<b>_bare` for that point.
- **Penalty runs report H + 0.05 S².** For `*_s2pen`, the logged energy and E0 are of H + 0.05 S². The replay gives the true ⟨H⟩ and ⟨S²⟩ separately.
- **Each run is measured from its own FCI.** The three trees freeze different 1s orbitals, so their frozen-core FCIs differ by a few µHa.
- **`luno` isn't included.** Localizing the UNOs fails at 1.1 Å: the Boys optimum isn't unique where the UNOs are the RHF orbitals.
- **Changing settings.** Change λ or the cutoff in `config.sh`. Output names include them (`_s2pen<λ>`, `_c<coeff>`), so runs with different settings don't overwrite each other.
