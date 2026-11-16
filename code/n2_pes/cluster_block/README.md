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
config.sh     settings: R, COEFF, MAX_ITER, SAVE_EVERY, LAMBDAS, HEAP (all overridable from the environment)
stage.sh      LOCAL only: builds DBF.jl/ (package, the two pinned Julia environments, the block scripts, the 2.3 Å inputs)
setup.sh      once, on a login node: install and precompile the environments
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

2. **Copy the bundle to the cluster:**
   ```bash
   ssh cshrikh@bigred200.uits.iu.edu 'mkdir -p /N/u/cshrikh/BigRed200/v_DBF/n2_block'
   scp -r ./* cshrikh@bigred200.uits.iu.edu:/N/u/cshrikh/BigRed200/v_DBF/n2_block/
   ```
   It is a separate folder from the PES bundle (`v_DBF/n2_pes`), so nothing there is touched.

3. **Install, once, on a login node:**
   ```bash
   cd /N/u/cshrikh/BigRed200/v_DBF/n2_block && bash setup.sh
   ```
   It uses juliaup's Julia, as the PES bundle does. To check the whole chain quickly, do a 2-iteration run at cutoff 10⁻⁵, which takes a minute or two:
   ```bash
   COEFF=1e-5 MAX_ITER=2 bash run_one.sh 0.05
   ```
   Its files are named `..._c1e-5_...`, so they don't clash with the real runs.

4. **Submit:**
   ```bash
   mkdir -p slurm_logs && sbatch block.slurm
   ```
   The job settings follow `../cluster_pes/node.slurm`: partition `general`, account `r01859`, one whole node, 48 h.

5. **Follow progress.** Each run prints one line per iteration:
   ```bash
   tail -f DBF.jl/code/n2_pes/frozen_core/logs/uno_2.30_blockpp_c1e-6_s2pen0.05.out
   ```
   `slurm_logs/run_<λ>_<job>.log` has each run's start, end and replay summary. `DBF.jl/code/n2_pes/frozen_core/logs/maxrss.txt` has the peak memory of every step.

6. **If the job hits the 48 h limit.** Each run writes its result JSON every 50 iterations, with `"finished": false`, so a stopped run keeps everything up to its last save.
   - Replay what was saved:
     ```bash
     sbatch -p general -A r01859 --time=2:00:00 -c 4 --mem=32G -o slurm_logs/replay_%j.log --wrap "REPLAY_ONLY=1 bash run_one.sh 0.05"
     ```
   - Resubmitting `block.slurm` skips finished runs. An unfinished run starts again from iteration 0, because runs can't be resumed. To rerun only some: `LAMBDAS="0.05" sbatch block.slurm`.

7. **Bring the results back**, from `code/n2_pes/frozen_core/` in the local repo:
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
