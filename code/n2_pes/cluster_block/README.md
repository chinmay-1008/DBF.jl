# Block vDBF at 2.3 Å on Big Red 200

This bundle runs the block vDBF with Pauli propagation (`run_block_pp.jl`) for N₂ at R = 2.30 Å, in the UNO basis, at operator cutoff **10⁻⁶**, for 500 iterations. It does three runs side by side on one node:

| Run | Flow operator | Label |
|---|---|---|
| bare H | H | `uno_2.30_blockpp_c1e-6` |
| λ = 0.02 | H + 0.02 S² | `uno_2.30_blockpp_c1e-6_s2pen0.02` |
| λ = 0.05 | H + 0.05 S² | `uno_2.30_blockpp_c1e-6_s2pen0.05` |

After each run, `replay_block.jl` applies the stored rotations to the three block states with no truncation. That gives the true energies, variances and ⟨S²⟩ at every iteration.

These are the runs of `code/vdbf_excited_states_runs.md` (section 10, "With Pauli propagation (2.3 Å, cutoff 10⁻⁵)") at a 10× smaller cutoff. They don't fit on a 32 GB laptop. They ran as job 9069990 (results in section 10, "cutoff 10⁻⁶").

**The new kets.** The same bundle runs the new 2.3 Å kets (section 11: |0⟩, 3,4→5,6 and 3→5), bare H, with each of the three ranking operators of section 11. Like the old-kets runs, they go side by side on one node (step 5):

| Run | Ranking operator D | Label |
|---|---|---|
| projector | the block projector (exact gradient) | `uno_2.30_blockpp_c1e-6_k3456_35` |
| D_S | the projector written in Z strings | `uno_2.30_blockpp_c1e-6_k3456_35_ds` |
| dz | the block's 1- and 2-body Z operator | `uno_2.30_blockpp_c1e-6_k3456_35_dz` |

They ran as job 9093882 (results in `code/vdbf_excited_states_runs.md`, section 11, "The new kets at 2.3 Å, cutoff 10⁻⁶").

## Contents

```
config.sh     settings: DBF_ENV (the Julia project for the runs), R, COEFF, MAX_ITER, SAVE_EVERY, LAMBDAS, KETS, KTAG, RUNS,
              HEAP, RUN_LIMIT (all overridable from the environment)
stage.sh      LOCAL only: builds DBF.jl/ (block scripts, 2.3 Å inputs, the replay environment, a fallback copy of the package,
              checks/: 3 iterations of each local 2.3 Å run at 1e-5, for setup.sh)
setup.sh      on a login node: precompile the environments and check short runs against the local ones
run_one.sh    one run and its replay: bash run_one.sh <λ> or <λ>/<rank>   (λ = 0 is bare H)
block.slurm   one whole node: the runs of RUNS (default: one per λ in LAMBDAS) side by side
DBF.jl/       made by stage.sh; laid out as in the repo
```

## Steps

1. **Stage, locally**, from this folder:
   ```bash
   bash stage.sh
   ```
   It copies `hamiltonian_uno.jld2` and `s2_uno.jld2` as built locally, so the cluster runs start from the same H and S² as the local ones.

2. **Copy the bundle to the cluster**, still from this folder. It goes in its own folder, next to the PES bundle (`v_DBF/n2_pes`), which is not touched:
   ```bash
   ssh cshrikh@bigred200.uits.iu.edu 'mkdir -p /N/u/cshrikh/BigRed200/v_DBF/n2_block'
   rsync -av --exclude=slurm_logs ./ cshrikh@bigred200.uits.iu.edu:/N/u/cshrikh/BigRed200/v_DBF/n2_block/
   ```
   That sends the scripts and `DBF.jl/`, about 0.4 MB.

3. **The DBF environment: the clone the PES runs already use.** The runs use `--project=/N/u/cshrikh/BigRed200/Projects/n2_dbf/DBF.jl`, set by `DBF_ENV` in `config.sh`. That is the `qual_exam` clone of `chinmay-1008/DBF.jl`, with `nmayhall/PauliOperators.jl` next to it. No new clone is needed:
   - `src/`, `Project.toml` and `ext/` are identical on `qual_exam` and `excited_state`;
   - the block scripts (`run_block_pp.jl`, `replay_block.jl`) come with this bundle, like `run_dbf_pes.jl` in the PES bundle.

   If that clone doesn't exist yet, make it as in `../cluster_pes/README.md`, step 3. If `src/` on `excited_state` ever changes, clone `excited_state` next to it instead and point `DBF_ENV` at that clone. The replay uses the bundled `DBF.jl/code/integrals` environment, as the PES replays do.

4. **Install and check on a login node** (again after each restage):
   ```bash
   cd /N/u/cshrikh/BigRed200/v_DBF/n2_block && bash setup.sh
   ```
   - It precompiles both environments and checks that JSON is in the default environment.
   - Then, for each local 2.3 Å run in `DBF.jl/checks/` (the old kets with D1c; the new kets with the projector, D_S and dz), it runs 3 iterations at cutoff 10⁻⁵ with that run's options and compares them with the local run.
   - Each check should end with `ok: the cluster run reproduces the local run`. If one says `DIFFERENT`, the cluster's PauliOperators.jl clone or DBF `src/` computes differently from the local one.

5. **Submit:**
   - The old kets, bare H and the two penalties (job 9069990, done):
     ```bash
     mkdir -p slurm_logs && sbatch block.slurm
     ```
   - The new kets, bare H, with the projector, D_S and dz:
     ```bash
     mkdir -p slurm_logs && KETS=ref,3.4.5.6,3.5 KTAG=k3456_35 RUNS="0/exact 0/ds 0/dz" sbatch -J n2kets block.slurm
     ```
     For the projector alone (about 8 h of node time instead of 48 h), use `RUNS="0/exact"`.

   The job settings follow `../cluster_pes/node.slurm`: partition `general`, account `r01859`, one whole node, 48 h. A finished run is never rerun: resubmitting skips it.

6. **Follow progress.** Each run prints one line per iteration:
   ```bash
   tail -f DBF.jl/code/n2_pes/frozen_core/logs/uno_2.30_blockpp_c1e-6_k3456_35_ds.out
   ```
   `slurm_logs/run_[<KTAG>_]<λ>[_<rank>]_<job>.log` has each run's start, end and replay summary. `DBF.jl/code/n2_pes/frozen_core/logs/maxrss.txt` has the peak memory of every step.

7. **Runs longer than the job.** Each run writes its result JSON every 50 iterations, with `"finished": false`, so a stopped run keeps everything up to its last save.
   - A run still going after `RUN_LIMIT` (46 h) is stopped, and its last save is replayed inside the job. So every run comes back with a replay.
   - If a job is killed anyway, replay what was saved by hand:
     ```bash
     sbatch -p general -A r01859 --time=2:00:00 -c 4 --mem=32G -o slurm_logs/replay_%j.log --wrap "REPLAY_ONLY=1 bash run_one.sh 0.05"
     ```
     (with `KETS=... KTAG=...` and `0/ds` for a new-kets run).
   - Resubmitting `block.slurm` skips finished runs. An unfinished run starts again from iteration 0, because runs can't be resumed. To rerun only some: `LAMBDAS="0.05" sbatch block.slurm`.

8. **Bring the results back**, from `code/n2_pes/frozen_core/` in the local repo:
   ```bash
   rsync -av --include='*c1e-6*.json' --exclude='*' cshrikh@bigred200.uits.iu.edu:/N/u/cshrikh/BigRed200/v_DBF/n2_block/DBF.jl/code/n2_pes/frozen_core/sv_results/ sv_results/
   rsync -av cshrikh@bigred200.uits.iu.edu:/N/u/cshrikh/BigRed200/v_DBF/n2_block/DBF.jl/code/n2_pes/frozen_core/logs/ logs/
   ```
   The labels contain `c1e-6`, so they sit next to the local `c1e-5` files without overwriting them. The local 1.1 Å runs at 1e-6 are not on the cluster, so they are not touched either.

## Resources

**Measured (job 9069990, November 2026).** All three runs finished their 500 iterations, and each replay took about 5 min.

| run | time | peak memory | Pauli terms at the end |
|---|---|---|---|
| bare H | 21.0 h | 34 GB | 23 M |
| λ = 0.02 | 28.6 h | 43 GB | 33 M |
| λ = 0.05 | 21.1 h | 38 GB | 25 M |

`HEAP=50G` would be enough.

**The new kets, measured (job 9093882, November 2026).** All three runs finished their 500 iterations; the job took 13.1 h, and each replay about 5 min.

| run | time | peak memory | Pauli terms at the end |
|---|---|---|---|
| projector | 4.8 h | 6.0 GB | 7.3 M |
| D_S | 12.2 h | 19 GB | 9.0 M |
| dz | 13.0 h | 15 GB | 10.3 M |

The estimates made before the job (8, 55 and 80 h; 25, 35 and 55 GB) were far too high: at 1e-6 these runs ended with only 1.7–2.8× the terms of the 1e-5 runs, not 6×.

The estimates written before the old-kets runs:
- **Terms:** the 10⁻⁵ runs ended at 3.6M (bare H), 3.9M (λ = 0.02) and 6.6M (λ = 0.05) terms. At 1.1 Å, going from 10⁻⁵ to 10⁻⁶ gave 6× the terms, so expect roughly 20–40M.
- **Memory:** each run gets `--heap-size-hint=70G`, and three of those fit a 256 GB node. The peaks in `maxrss.txt` show whether that can be lowered (`HEAP=40G sbatch block.slurm`).
- **Time:** the 10⁻⁵ runs took 1.7, 2.7 and 3.0 h for 500 iterations on a laptop. At 1.1 Å, 10⁻⁶ took 5× longer, so expect about 8–15 h per run on a laptop, and probably more on the cluster's slower cores. That fits in 48 h with some margin.
- **Cores:** each run is single-threaded, and the replays use 4 threads. The node is there for its memory. It is billed whole for as long as the job runs.
