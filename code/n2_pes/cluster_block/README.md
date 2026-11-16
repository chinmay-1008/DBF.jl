# Block vDBF at 2.3 Å on Big Red 200

This bundle runs the block vDBF with Pauli propagation (`run_block_pp.jl`) for N₂ at R = 2.30 Å, in the UNO basis, at operator cutoff **10⁻⁶**, for 500 iterations. It does three runs side by side on one node:

| Run | Flow operator | Label |
|---|---|---|
| bare H | H | `uno_2.30_blockpp_c1e-6` |
| λ = 0.02 | H + 0.02 S² | `uno_2.30_blockpp_c1e-6_s2pen0.02` |
| λ = 0.05 | H + 0.05 S² | `uno_2.30_blockpp_c1e-6_s2pen0.05` |

After each run, `replay_block.jl` applies the stored rotations to the three block states with no truncation. That gives the true energies, variances and ⟨S²⟩ at every iteration.

These are the runs of `code/vdbf_excited_states_runs.md` (section 10, "With Pauli propagation (2.3 Å, cutoff 10⁻⁵)") at a 10× smaller cutoff. They don't fit on a 32 GB laptop.

## Contents

```
config.sh     settings: DBF_ENV (the Julia project for the runs), R, COEFF, MAX_ITER, SAVE_EVERY, LAMBDAS, HEAP (all overridable from the environment)
stage.sh      LOCAL only: builds DBF.jl/ (block scripts, 2.3 Å inputs, the replay environment, a fallback copy of the package, check_reference.json)
setup.sh      once, on a login node: precompile the environments and check a short run against the local one
run_one.sh    one run and its replay: bash run_one.sh <λ>   (λ = 0 is bare H)
block.slurm   one whole node: the three runs side by side
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

4. **Install and check, once, on a login node:**
   ```bash
   cd /N/u/cshrikh/BigRed200/v_DBF/n2_block && bash setup.sh
   ```
   It precompiles both environments and checks that JSON is in the default environment. Then it runs 3 iterations at cutoff 10⁻⁵ and compares them with your local run. It should end with `ok: the cluster run reproduces the local run`. If it says `DIFFERENT`, the cluster's PauliOperators.jl clone computes differently from the local one.

5. **Submit:**
   ```bash
   mkdir -p slurm_logs && sbatch block.slurm
   ```
   The job settings follow `../cluster_pes/node.slurm`: partition `general`, account `r01859`, one whole node, 48 h.

6. **Follow progress.** Each run prints one line per iteration:
   ```bash
   tail -f DBF.jl/code/n2_pes/frozen_core/logs/uno_2.30_blockpp_c1e-6_s2pen0.05.out
   ```
   `slurm_logs/run_<λ>_<job>.log` has each run's start, end and replay summary. `DBF.jl/code/n2_pes/frozen_core/logs/maxrss.txt` has the peak memory of every step.

7. **If the job hits the 48 h limit.** Each run writes its result JSON every 50 iterations, with `"finished": false`, so a stopped run keeps everything up to its last save.
   - Replay what was saved:
     ```bash
     sbatch -p general -A r01859 --time=2:00:00 -c 4 --mem=32G -o slurm_logs/replay_%j.log --wrap "REPLAY_ONLY=1 bash run_one.sh 0.05"
     ```
   - Resubmitting `block.slurm` skips finished runs. An unfinished run starts again from iteration 0, because runs can't be resumed. To rerun only some: `LAMBDAS="0.05" sbatch block.slurm`.

8. **Bring the results back**, from `code/n2_pes/frozen_core/` in the local repo:
   ```bash
   rsync -av --include='*c1e-6*.json' --exclude='*' cshrikh@bigred200.uits.iu.edu:/N/u/cshrikh/BigRed200/v_DBF/n2_block/DBF.jl/code/n2_pes/frozen_core/sv_results/ sv_results/
   rsync -av cshrikh@bigred200.uits.iu.edu:/N/u/cshrikh/BigRed200/v_DBF/n2_block/DBF.jl/code/n2_pes/frozen_core/logs/ logs/
   ```
   The labels end in `c1e-6`, so they sit next to the local `c1e-5` files without overwriting them.

## Resources

These are estimates from the local 10⁻⁵ runs; nothing has run at 10⁻⁶ yet.
- **Terms:** the 10⁻⁵ runs ended at 3.6M (bare H), 3.9M (λ = 0.02) and 6.6M (λ = 0.05) terms. At 1.1 Å, going from 10⁻⁵ to 10⁻⁶ gave 6× the terms, so expect roughly 20–40M.
- **Memory:** each run gets `--heap-size-hint=70G`, and three of those fit a 256 GB node. The peaks in `maxrss.txt` show whether that can be lowered (`HEAP=40G sbatch block.slurm`).
- **Time:** the 10⁻⁵ runs took 1.7, 2.7 and 3.0 h for 500 iterations on a laptop. At 1.1 Å, 10⁻⁶ took 5× longer, so expect about 8–15 h per run on a laptop, and probably more on the cluster's slower cores. That fits in 48 h with some margin.
- **Cores:** each run is single-threaded, and the replays use 4 threads. The node is there for its memory. It is billed whole for as long as the job runs.
