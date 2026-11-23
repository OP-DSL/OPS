# TGV (tgvstorenone) MPI-decomposition sweep on LUMI-G

This package builds OPS `apps/c/TGV_StoreNone` (the OpenSBLI Taylor-Green vortex case, OPS C/C++,
MPI + HIP; make target `opensbli_mpi_hip`, frozen as `tgvstorenone_mpi_hip`) on LUMI and measures one
cubic problem under a fixed list of MPI decompositions, with three ways of choosing the GPU block size
of each kernel. It is a separate copy of `scripts/lumi_cloverleaf_decomp` (itself a copy of
`scripts/lumi_senga2_decomp`) with the same commands, the same campaign and dataset layout and the same
run design, so that TGV is run exactly as CloverLeaf_3D and SENGA2 are. The older launcher
`scripts/lumi_tgv_top3` (replay of a fixed dataset at 2160^3) is unchanged and independent.

| Item | Default |
|---|---|
| Application | `apps/c/TGV_StoreNone` of this OPS checkout (no other repository); no input file: the arguments are `NX NY NZ NITER` |
| Scale | 20 LUMI-G nodes, 160 MPI ranks, one GCD (MI250X, `gfx90a`) per rank, GPU direct |
| Problem | 1440^3 grid points, `NITER` = `--steps` (100 is a placeholder: section 1.5 measures the time per step; first estimate for 5-minute runs: about 2800) |
| Decompositions | 16, fixed: `5x8x4 4x10x4 5x16x2 4x20x2 8x5x4 5x32x1 8x4x5 8x10x2 4x40x1 8x20x1 10x4x4 2x40x2 10x8x2 10x2x8 10x16x1 16x10x1` |
| Per decomposition | 1 autotune run (A), then 1 pair of default (D) and precomputed (P) runs (`--repetitions N` gives N pairs) |
| Extra runs | 1 unforced baseline run (one per repetition), and sentinel runs that measure the noise floor |
| Output | the dataset `OPS_LOGS_<stamp>_n020/` in the structure of the TGV and SENGA2 decomposition campaigns |

Nothing here needs an assistant: every step is a command of `run.sh`, each prints what it did and what
to run next, and every failure names the file or log to read. No project, account or user name is
stored in the package: you give the Slurm account with `--account` (or `SLURM_ACCOUNT`).

## Shortest path: `scripts/campaign.sh`

`campaign.sh` runs the steps below for you (it creates the campaign, builds or reuses an identical
earlier build, freezes the plan and submits), exactly as it does for `cloverleaf` and `senga2`:

```bash
cd ~/OPS && git checkout feature/gpu_block_size && git pull
echo ACCOUNT=<your_project> > ~/.ops_campaign.conf          # once
bash scripts/campaign.sh tgv new                             # 16 decompositions, 1440^3, 20 nodes, 2800 steps
bash scripts/campaign.sh tgv new --decompositions 8x5x4,4x20x2 --steps 2800
bash scripts/campaign.sh tgv status
bash scripts/campaign.sh tgv continue                        # what is still missing; retry also repeats failed runs
```

Add the option `--dry-run` to any of them to print the commands without running them. `campaign.sh`
submits with an explicit go/no-go override and does not run the qualification stage: for the first TGV
campaign follow the quick path below once (it measures the time per step and checks the build on LUMI).
The 2800 steps are an estimate, not a measurement: 500 steps at 2160^3 on 20 nodes took about 181 s, so
one step at 1440^3 is expected to take about 0.107 s (scaled by the number of grid points), i.e. about
300 s for 2800 steps. `gonogo` (section 1.5) prints the measured value.

## Quick path (copy and paste)

```bash
cd ~/OPS && git checkout feature/gpu_block_size && git pull
cd ~/OPS/scripts/lumi_tgv_decomp
bash run.sh plan --output-dir ~/tgv_plan_preview          # look at the plan (nothing is submitted)
bash run.sh prepare --account <your_project>                 # prints the campaign directory C
C=/scratch/<your_project>/ops-tgv-decomp/campaigns/TGV_DECOMP_<date>
bash $C/launcher/run.sh build --campaign $C --submit         # or without --submit, in this shell
bash $C/launcher/run.sh smoke --campaign $C --submit         # qualification, two jobs
bash $C/launcher/run.sh gonogo --campaign $C --gpu-hours-per-node-hour <rate> --remaining-gpu-hours <hours> --disk-available-gb <GB>
bash $C/launcher/run.sh plan --campaign $C --steps <N>       # N: the value gonogo suggests for ~5 min runs
bash $C/launcher/run.sh gonogo --campaign $C --steps <N> --gpu-hours-per-node-hour <rate> --remaining-gpu-hours <hours> --disk-available-gb <GB>
bash $C/launcher/run.sh submit --campaign $C --jobs 2
bash $C/launcher/run.sh status --campaign $C
bash $C/launcher/run.sh collect --campaign $C
```

Contents: [1 Steps](#1-steps-in-order) · [2 Build variables](#2-build-variables) ·
[3 What a run must satisfy](#3-what-a-valid-run-is) · [4 Resume and failures](#4-allocations-resume-and-failures) ·
[5 Results](#5-results) · [6 Campaign directory and dataset](#6-campaign-directory) · [7 Exit statuses](#7-exit-statuses) ·
[8 Limitations](#8-limitations-to-carry-into-every-result) · [9 Only LUMI can verify](#9-what-only-lumi-can-verify) ·
[10 First-run checks](#10-checks-to-do-by-eye-in-the-first-lumi-runs) · [11 Troubleshooting](#11-troubleshooting) ·
[12 Local tests](#12-local-tests)

## 1. Steps, in order

All commands are run from a LUMI login node. `$C` is the campaign directory printed by `prepare`.
After `prepare`, always use the frozen copy `$C/launcher/run.sh`, so that the code of a campaign cannot
change while it runs.

### 1.1 Look at the plan (no Slurm; nothing is written outside the directory you name)

```bash
bash run.sh plan --output-dir ~/tgv_plan_preview
```

It writes `plan.json`, `plan.csv` (one line per run: position, stage, decomposition, mode, repetition,
pair, full argument list, seed) and the qualification list `smoke_plan.csv`. With the defaults it
reports 16 autotune, 16 default and 16 precomputed sweep runs, 1 baseline run and 2 sentinel runs.
A decomposition that cannot run (a local width that is not an integer, or below the halo depth 5)
is skipped with a `NOTICE` line; for 1440^3 on 160 ranks nothing is skipped.

Options: `--nodes`, `--grid`, `--steps`, `--repetitions`, `--decompositions`, `--seed`, `--fixed-order`,
`--sentinel-every`, `--exclude PxxPyxPz`. `--decompositions` is `default` (the 16 above), `all` (every
valid ordered triple) or an explicit list such as `8x5x4,4x10x4`.

### 1.2 Create the campaign

```bash
bash run.sh prepare --account <your_project>
```

This creates `/scratch/<your_project>/ops-tgv-decomp/campaigns/TGV_DECOMP_<date>` (change
with `--output-root DIR` or `--campaign DIR`) and puts into it a frozen copy of these scripts, the
environment file and a snapshot of the tracked OPS files that are needed (`ops/c`, `makefiles`,
`ops_translator`, `apps/c/TGV_StoreNone`). Your OPS checkout is never modified; uncommitted changes in
those folders are saved as `source/ops_local_changes.patch` and flagged in `campaign.json`.

Other options: `--env-file FILE` (section 2), `--translator-python /path/to/python3`, `--partition`,
`--smoke-partition`, `--build-partition`, `--cpu-bind`, `--consistency-tolerance`, and the plan options
of 1.1 (the campaign fixes nodes, grid, qualification steps, repetitions, decompositions and seed).

### 1.3 Build the executable (qualification stage Q0)

```bash
bash $C/launcher/run.sh build --campaign $C --print-toolchain    # what will be used; changes nothing
bash $C/launcher/run.sh build --campaign $C                      # in this shell (login node)
bash $C/launcher/run.sh build --campaign $C --submit             # or in a 1-node Slurm job (2 hours)
bash $C/launcher/run.sh build --campaign $C --submit --dry-run   # show sbatch only
```

The grid and the number of time steps are command-line arguments of the application, so one
executable serves every run of the campaign. It is built in `$C/build/hip/` (own copy of the sources, `build.log`, `build_manifest.json`, hash):

1. the OPS translator generates the sources for all targets of the application, exactly as the
   `.generated` rule of `makefiles/Makefile.c_app` does (about one minute), and the launcher checks that
   **every** generated HIP kernel (`hip/*_kernel.cpp`) calls `ops_get_kernel_block_size` and
   `ops_record_kernel_performance`. If one does not, the build stops: an executable without the hooks
   would silently ignore the tuned blocks;
2. `make -C ops/c mpi_hip` builds the OPS C MPI+HIP library (`libops_mpi_hip.a`);
3. `make opensbli_mpi_hip` in the application folder compiles and links the executable with the rule of
   `makefiles/Makefile.c_app` (one `hipcc` call with every generated kernel; several minutes); the result
   is frozen as `$C/build/hip/tgvstorenone_mpi_hip`;
4. `ldd` checks that the executable links `libmpi_gtl_hsa` (GPU-aware Cray MPICH).

The translator needs Python >= 3.8 and its packages, including `libclang`. By default a virtual
environment is created in `$C/build/translator-env` with `pip install -r ops_translator/requirements.txt`
(network access to PyPI). If that is not possible, prepare with `--translator-python`.

If a step fails, the message gives the step, the exact command and the log. Fix the cause (usually a
variable of section 2) and run the same `build` command again: finished objects are reused; if you
changed a compiler or flag, the objects are discarded automatically. Because the environment file is
frozen in the campaign, either edit `$C/environment.sh` before the first successful build, or export
the `TGV_*` build variable in your shell for a local build. Once a build is complete it is never rebuilt.

### 1.4 Qualification (Q1 to Q5)

```bash
bash $C/launcher/run.sh smoke --campaign $C --dry-run     # list the runs, submit nothing
bash $C/launcher/run.sh smoke --campaign $C --submit      # two jobs: Q1-Q3, then Q4-Q5
```

| Stage | Grid | Resources | Runs | What it proves |
|---|---|---|---|---|
| Q0 | - | build | - | toolchain, hooks, `libmpi_gtl_hsa` |
| Q1 | small 256^3 | 1 node, 1 rank | default | the HIP build runs on MI250X; kernel call counts; time |
| Q2 | small | 1 rank | autotune, then replay | hooks fire, catalog not empty, replay loads exactly the catalog |
| Q3 | small | 1 node, 8 ranks | 2x2x2, 8x1x1, 1x8x1, 1x1x8, unforced, 2x2x2 without `-gpudirect` | multi-rank execution (kernel call counts agree), forced grid is the printed grid, one GCD per rank, GPU direct on/off |
| Q4 | full 1440^3 | 20 nodes, 160 ranks, 10 steps | unforced, and the thinnest planned triple (`2x40x2`) | memory fits, identifies `d0` |
| Q5 | full | 160 ranks, `--steps` | forced `d0`: A, then D P P D | the whole per-decomposition pipeline; **time per step** |

The first job (Q1 to Q3) uses one whole node of `--smoke-partition` (default `standard-g`); the second
(Q4, Q5) uses the 20 nodes and starts only if the first ended with status 0. Use `--stage small` or
`--stage full` to submit one of them, `--smoke-partition P` to send the one-node job to another
partition for this submission only, `--time-small` / `--time-full` for the wall times (1 and 2 hours),
and `--retry-failed` to rerun failed qualification runs (old attempts are kept). The time guard of
section 1.7 also applies here, with fixed expected durations (small grid: 600 s autotune, 300 s
default/precomputed; full grid: 1800 s and 900 s); `--no-time-guard` switches it off. Qualification runs
never enter the sweep statistics.

### 1.5 Go / no-go, and the number of steps

```bash
bash $C/launcher/run.sh gonogo --campaign $C \
    --gpu-hours-per-node-hour <rate> --remaining-gpu-hours <hours> --disk-available-gb <GB>
```

It prints the **time per step** measured in Q5 (from `Total Wall time` of the D, P and A runs), the
start-up overhead per run, and the step count whose default-block time-step loop lasts about 5 minutes:

```text
     Time per step (Q5, 1440^3 on 160 ranks, 100 steps): D 0.1070 s, P 0.0990 s, A 0.2100 s; ...
     Suggested --steps for a time-step loop of about 300 s (default blocks): 2804
```

`--target-seconds` changes the 300 s. Then one `PASS`/`FAIL` line per check and `VERDICT: GO` or
`VERDICT: NO-GO`; the exit status is 0 for GO and 6 for NO-GO. The checks:

1. every qualification run is valid (section 3);
2. in Q3 and Q4 the printed processor grid equals the forced triple; unforced grids are shown;
3. the full grid fits on every GCD (both Q4 runs valid);
4. Q5: catalog not empty, replay complete, the two default runs within 5 % of each other;
5. budget: the cost of the sweep projected from the measured Q5 times, at the sweep step count
   (`--steps`, default: the frozen plan, else the campaign value), is at most 80 % of what is left.
   The projection is always printed. The conversion to GPU-hours is **your** input, because the billing
   rule must be confirmed for your project: `--gpu-hours-per-node-hour` (check the LUMI documentation and
   `lumi-allocations`; a LUMI-G node has 4 GPU cards) and `--remaining-gpu-hours`;
6. disk: the projected log volume fits in `--disk-available-gb`;
7. the kernel call counts (number of kernels and total number of loop calls in the OPS kernel table) of
   the qualification runs with the same grid and step count agree within `--consistency-tolerance`
   (default 0: they are integers) across decompositions, ranks and block modes.

Without the three inputs, checks 5 and 6 fail and the verdict is NO-GO. `gonogo.json` keeps the result
and the step count it projected; `measured_times.json` keeps the measured times, which the sweep uses
for its time guard (scaled to the sweep step count).

### 1.6 Freeze the sweep plan, with the chosen number of steps

```bash
bash $C/launcher/run.sh plan --campaign $C --steps <N>
bash $C/launcher/run.sh gonogo --campaign $C --steps <N> --gpu-hours-per-node-hour <rate> --remaining-gpu-hours <hours> --disk-available-gb <GB>
```

`--steps` is the `NITER` argument of every sweep run (without it, the campaign value). The plan (`$C/plan.json`
and its sha256) is written once and never regenerated: resume reads it, and a changed file is refused.
`submit` refuses a plan whose step count differs from the one the last `gonogo` projected, so run
`gonogo --steps <N>` again after freezing.

### 1.7 Submit, and continue over several allocations

```bash
bash $C/launcher/run.sh submit --campaign $C --dry-run                 # show sbatch and the job script
bash $C/launcher/run.sh submit --campaign $C --time 24:00:00 --jobs 2  # two chained 20-node jobs
bash $C/launcher/run.sh status --campaign $C
```

`submit` refuses to start unless the verdict is GO (`--override-gonogo "reason"` records an explicit
override). `--jobs N` chains N allocations; each one continues where the previous stopped, and a job
that finds nothing left to do ends at once. To continue later:

```bash
bash $C/launcher/run.sh submit --campaign $C --resume
bash $C/launcher/run.sh submit --campaign $C --resume --retry-failed   # also rerun failed runs
```

`--resume` refuses while an earlier sweep job is still pending or running, so that no run is duplicated.

Time guard: before each run (before each D/P pair as a whole) the job compares the time left in the
allocation with the expected duration of that run times 1.2 plus 60 s, and stops cleanly (exit status
3) instead of starting something it cannot finish. Expected durations come from `measured_times.json`
(1.5 x the Q5 elapsed time scaled to the sweep step count, + 60 s); before that they are 1800 s (A) and
900 s (D, P). Override with `--est-autotune`, `--est-default`, `--est-precomputed` (seconds), or disable
with `--no-time-guard`. Each run is also killed after 3 x its expected duration and recorded as `TIMEOUT`.

### 1.8 Collect

```bash
bash $C/launcher/run.sh collect --campaign $C
```

It can be run at any time, also on a partial campaign. See section 5.

### 1.9 Upgrade a campaign whose sweep has not started

`upgrade` replaces the frozen launcher `$C/launcher` by the package of your checkout (the old copy is kept
in `$C/launcher_previous/<utc>/`, hashes in `$C/launcher_upgrades.json`). It is refused while any job of
the campaign is pending or running, once a sweep job was submitted, or once a sweep run exists.
`--repetitions N` and `--decompositions ...` change `campaign.json` while the sweep plan is not frozen;
`--restart-sweep` moves a started sweep into `$C/archive/sweep_<utc>/` (nothing is deleted;
qualification and the executable stay) so that it can be planned again.

```bash
bash ~/OPS/scripts/lumi_tgv_decomp/run.sh upgrade --campaign $C
bash ~/OPS/scripts/lumi_tgv_decomp/run.sh upgrade --campaign $C --decompositions default --restart-sweep
```

The upgrade must be run with the `run.sh` of the checkout (the command refuses to run from `$C/launcher`).

## 2. Build variables

TGV_StoreNone has not been built with this package on LUMI before. The default recipe is the one of
`scripts/lumi_tgv_top3`, which built this application on LUMI (ROCm `clang`/`clang++` for the OPS
library, `hipcc` for the HIP sources and the application, `-std=c++11` everywhere, `MPI_LINK=-lmpi`,
`TARGETS=mpi_hip`), driven the way the CloverLeaf_3D package drives its build; every compiler and flag
can be changed without editing code. `environment.sh` loads LUMI, `partition/G`, `PrgEnv-gnu` (a login shell that still
has the default `PrgEnv-cray` is switched), `rocm`, `craype-accel-amd-gfx90a`, and exports
`MPICH_GPU_SUPPORT_ENABLED=1`. To pin module versions or change anything, copy the file, edit it and give
it to `prepare --env-file`. `bash run.sh build --print-toolchain` (after `source environment.sh`) lists
every variable, its default and the value found in your environment:

| Variable | Meaning | Default |
|---|---|---|
| `TGV_HIPCC` | HIP compiler driver (OPS HIP sources; the application in one call) | `hipcc` |
| `TGV_ROCM_ROOT`, `TGV_MPI_ROOT`, `TGV_GTL_DIR` | installation roots | `HIP_INSTALL_PATH`/`ROCM_PATH`, `CRAY_MPICH_DIR`, `PE_MPICH_GTL_DIR_amd_gfx90a` |
| `TGV_CC`, `TGV_CXX`, `TGV_MPICXX` | host compilers of the OPS library | ROCm `clang`, `clang++` (`HIP_CLANG_PATH` or `<rocm>/llvm/bin`) |
| `TGV_CXXFLAGS` | host C++ flags | `-O3 -fPIC -std=c++11` |
| `TGV_CFLAGS` | host C flags of the OPS library | `-O3 -fPIC` |
| `TGV_HIPFLAGS` | HIP flags (OPS HIP sources, the application and its kernels) | `-O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__` |
| `TGV_MPI_HIP_LINK` | MPI and GPU transport libraries | `-L<mpi>/lib -L<gtl> -lmpi -lmpi_gtl_hsa` |
| `TGV_HIP_LINK`, `TGV_HIPRAND_LINK` | HIP runtime, hipRAND | `-L<rocm>/lib -L<rocm>/lib64 -lamdhip64 -lrocm_smi64`, `-L<rocm>/lib -lhiprand` |
| `TGV_LINK_EXTRA` | extra link flags | empty |
| `TGV_BUILD_JOBS` | parallel make jobs for the library | 8 |
| `TGV_REQUIRE_GTL` | `0` skips the `libmpi_gtl_hsa` check (never for LUMI runs) | 1 |

The build uses `OPS_COMPILER=hip` and passes every compiler and flag on the `make` command line; HDF5 is
not used (`HDF5_INSTALL_PATH` is empty). The compiler is part of the result: results of two campaigns
must not be mixed without saying so.

## 3. What a valid run is

Each run is one `srun` in a fresh attempt directory (qualification:
`runs/qualification/<run_id>/attempt_NNN/`; sweep: `<dataset>/_attempts/<CASE_ID>/iter_<r>/<variant>/attempt_NNN/`,
section 6.1) with an empty working directory `work/`: the application reads no input file and writes
no output file. Its arguments are `<grid> <grid> <grid> <steps>` (`NX NY NZ NITER` of
`apps/c/TGV_StoreNone/opensbli.cpp`) followed by the OPS options of the table below. A run is `VALID`
only if all of this holds:

- `srun` and every rank exit with status 0;
- exactly one positive `Total Wall time` in the root log (`ranks/rank_0.log`) and a `Total kernel time`;
- `GPU Direct enabled` printed by the root rank (and absent in the one run without `-gpudirect`);
- the root rank reports the expected `OPS_autotune_mode`;
- the processor grid printed by OPS equals the forced triple (unforced: it uses all ranks);
- one status record per rank, each with a different (host, GPU id);
- no `overpartitioning`, `out of memory`, `nan` or `Error`/`ERROR` text in any rank log (`nan` is
  ignored inside the OPS kernel table, whose derived columns, deviation and bandwidth, are not results);
- the OPS kernel table is present in the root log and every time in it is a finite number;
- the loop that runs once per time step (`opensbliblock00Kernel033`, "Save equations") was called
  exactly `<steps>` times: this is the step counter, because the application prints no step history.
  The diagnostics (steps done, number of kernels, total number of loop calls) are stored in
  `record.json` and `config.txt`;
- precomputed runs: every rank prints `Loaded N precomputed block sizes from <path>` with N equal to the
  catalog rows and the path of that decomposition's own catalog; rank 0 falls back to the default block
  for exactly the kernels that the autotune run executed but did not finish tuning; no other rank reports
  a catalog kernel as missing;
- default and autotune runs: no rank prints a `Loaded ... precomputed` line.

**What is not checked: the numerical solution.** `tgvstorenone` prints no physical quantity at all (no
kinetic energy, no residual, no field summary; only the OPS timing table and `Total Wall time`), and
there is no reference solution for 1440^3. "Finite outputs" therefore means finite times, and the
cross-decomposition check compares what the output does contain independently of the decomposition:
the number of kernels and the total number of loop calls on rank 0 (for N steps: 1 + 79 N). It proves
that every decomposition and block mode executed the same loops the same number of times; it cannot
detect a wrong or non-finite field. The agreement is checked by `gonogo` (qualification) and reported
by `collect` (sweep); it never makes a single run invalid.

Failure categories: `OOM`, `OVERPARTITION`, `TIMEOUT`, `OTHER`. The reason is in `record.json`
(`reason`) and in `state/events.log`, with the directory of the logs.

| Mode | Arguments | Environment |
|---|---|---|
| D default | all ranks `<grid> <grid> <grid> <steps> -OPS_AUTOTUNE_MODE=0 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 OPS_BLOCK_SIZE_Z=1` | CSV logs off on every rank |
| A autotune | rank 0 `-OPS_AUTOTUNE_MODE=1`; ranks 1..159 mode 0 with the default block | full tuning CSVs on rank 0 only |
| P precomputed | all ranks `-OPS_AUTOTUNE_MODE=2` | `OPS_BLOCKSIZE_CSV=<catalog of this decomposition>`; CSV logs off, as in D |

All runs also get `OPS_FORCE_DECOMP_X/Y/Z=` (except the unforced baseline), `-gpudirect` and
`-OPS_DIAGS=2`. OPS copies each argument through a 64-byte buffer; the launcher refuses longer ones.
`OPS_LOGS_DIR` always points inside the run directory. In precomputed runs the default block given on
the command line is used only for kernels missing from the catalog (`ops_util.cpp`, mode 2).

The catalog of a decomposition is built once, from rank 0's `ops_blocksize_best_logmod.csv` of that
decomposition's own autotune run: footer lines are skipped, one row per kernel is kept (lowest
`best_time`), every block must be positive with `bx*by*bz <= min(1024, max_threads_per_block)`. It is
written as `kernel_id,bx,by,bz`, made read-only and hashed (sweep:
`<dataset>/precomputed_blocks/tgvstorenone_hip_<grid>_cfg1/mode_1/decomp<PxPyPz>/`, with a `set_id` in its
`manifest.json`; qualification: `catalogs/qualification/<size>_<decomposition>/`). If the CSV is missing,
empty, has only footers, has an invalid row or two different blocks with the same best time for one
kernel, or if the autotune run is invalid, the decomposition is marked `AUTOTUNE_FAILED`: no catalog is
written, no precomputed run is launched, and no substitute is ever fabricated. Its default runs are still
executed.

## 4. Allocations, resume and failures

Order inside a sweep job:

1. the unforced baseline (default runs without `OPS_FORCE_DECOMP_*`). The grid OPS prints defines `d0`
   (`state/d0.json`; `MPI_Dims_create` is expected to give `8x5x4` for 160 ranks, but it is read, never
   assumed);
2. one sentinel run (forced `d0`, default blocks) at the start of **every** allocation, and one after
   every 8 decompositions;
3. the decompositions in the seeded random order of the plan. For each: A first, then the pair(s); the
   first mode alternates with the plan position (D P at even positions, P D at odd ones).

Rules:

- A run with a valid record is never repeated. Resume aborts if the executable hash or the plan hash
  differs from the records.
- A D/P pair is atomic within one allocation. If only one member finished, it is kept, marked superseded
  (`SUPERSEDED.json`, moved back to `_attempts/` with `COMMIT_STATUS=SUPERSEDED`) and the whole pair is rerun.
- A run enters the dataset (its final folder) only when it satisfies the commit rule; failed and
  interrupted attempts stay in `_attempts/`. A valid run whose move was cut off by the end of an
  allocation is committed by the next job without being rerun.
- A frozen catalog is reused in a later allocation only if its sha256 still matches; otherwise
  `CATALOG REJECTED` is logged and no precomputed run is launched for it.
- A failed run is never retried automatically. `--retry-failed` reruns failed runs; the failed attempts
  are kept and marked superseded.
- If a default run fails with `OOM` or `OVERPARTITION`, the decomposition is marked `INVALID_AT_RUNTIME`
  and its remaining runs are skipped; the sweep continues.
- Three consecutive failed decompositions abort the job (exit status 4) and write `state/ABORTED.json`.
  Later jobs of a chain then do nothing. After fixing the cause, continue with `submit --resume --retry-failed`.

## 5. Results

`collect` writes into `$C/results/`:

| File | Content |
|---|---|
| `runs.csv` | one row per run record, including failed, superseded and qualification runs (flagged; `used_in_statistics`), with the diagnostics (`diag_*`) and `consistency_rel_diff` |
| `decompositions.csv` | one row per planned decomposition |
| `incomplete_decompositions.csv` | decompositions without a complete set, with what is missing and why |
| `sentinels.csv` | sentinel runs per allocation and overall: n, min, median, max, relative half-range |
| `collection_manifest.json` | input records with hashes, exclusions, code version, grouping key, estimator definitions, consistency summary |

Estimators (also in the manifest), with `T` = `Total Wall time` printed by the root rank (the time-step
loop only):

- `D_median_wall_time_s`, `D_min_...`, `D_max_...` (and `P_...`): median, minimum and maximum over the
  repetitions, given only when all are valid. With the default single repetition the three are the one
  measured value. No confidence interval.
- `S_pair1..N` = `T(D, pair i) / T(P, pair i)`; `S_median_of_paired_ratios` is their median.
  `S_ratio_of_medians` is a different estimator and is tabulated beside it.
- `A_wall_time_s`: the single autotune run; it is never used as a D or P sample.
  `search_cost_A_over_D_median` has one sample and no uncertainty.
- `catalog_rows`, `executed_kernels`, `kernel_coverage`, and `catalog_default_time_share` (flagged below 0.90).
- Consistency: `consistency_rel_diff` is the largest relative difference of the number of kernels and
  of the total number of loop calls from the first valid unforced baseline run (0 when they agree);
  `max_consistency_rel_diff` and `consistency_ok` per decomposition. It is reported, never a reason to
  exclude a run.
- Covariates: `slurm_job_ids`, node lists, `permutation_class`, `is_d0`, `mpi_comm_volume_mb`.

A decomposition is complete with one valid autotune run and frozen catalog and all D and P repetitions
valid in same-job pairs. Incomplete ones are listed and never imputed: their missing values stay empty.
The collector does not rank decompositions and does not name a best one.

## 6. Campaign directory

```text
$C/
  campaign.json  source_snapshot.json  environment.sh  launcher/     frozen configuration and code
  source/OPS                                                          frozen OPS snapshot (never edited)
  build/hip/  OPS/ build.log build_manifest.json tgvstorenone_mpi_hip   the one executable
  smoke_plan.json  plan.json  *.sha256  *.csv                         frozen plans
  OPS_LOGS_<stamp>_n<NNN>/                   the sweep dataset (section 6.1)
  runs/qualification/<run_id>/attempt_NNN/   qualification runs (not part of the dataset)
      record.json  srun.log  ranks/rank_N.log  ranks/status_N.json  execution.json
      work/                      empty working directory of the run
      rank_0/autotune_on/*.csv   only in autotune runs
  catalogs/qualification/<size>_<PxxPyxPz>/blocks.csv  manifest.json
  state/events.log  state/d0.json  state/decompositions/  state/ABORTED.json
  gonogo.json  measured_times.json  submissions.json  slurm/  results/
```

### 6.1 Dataset

The sweep writes its results directly in the structure of the earlier TGV and SENGA2 decomposition
campaigns (`OPS_LOGS_<stamp>_n<NNN>`), so that their readers work on it. Copy this folder to keep or
analyse the results.

```text
$C/OPS_LOGS_<stamp>_n<NNN>/
  campaign_manifest.json        campaign id <stamp>_n<NNN>, case_count, commit_rule, plan_sha256 (of plan.jsonl), niter
  plan.jsonl                    one sweep case per line
  precomputed_blocks/tgvstorenone_hip_<G>x<G>x<G>_cfg1/mode_1/decomp<PxPyPz>/{blocks.csv, manifest.json}
  _attempts/<CASE_ID>/iter_<r>/<variant>/attempt_NNN/      failed, interrupted, superseded attempts
  tgvstorenone_hip_<G>x<G>x<G>_cfg1_comparison_<stamp>_n<NNN>/
    config.txt  autotune_off_summary.txt  autotune_on_summary.txt  autotune_precomputed_summary.txt
    mode_1/decomp<PxPyPz>/iter_<r>/{config.txt, autotune_off/, autotune_on/ (iter_1 only), autotune_precomputed/}
    baseline_unforced/iter_<r>/autotune_off/      unforced baseline (RUN_CLASS=BASELINE_UNFORCED)
    sentinels/<label>/autotune_off/               sentinels (RUN_CLASS=SENTINEL)
```

Every run folder holds `config.txt`, `run.log` (all rank logs, each under a `===== rank_N.log =====` line;
it keeps the OPS lines `Total kernel time:`, `Total halo exchange time:` and `Average volume of MPI
communications per process:`), `mpi_rank_metrics.csv`, `ranks/`, `record.json`, `srun.log`,
`execution.json` and `work/`. A run is in its final folder if and only if its `config.txt` has
`RUN_STATUS=OK`, `RUN_EXIT_CODE=0`, `VALIDATION_STATUS=PASS` and `COMMIT_STATUS=COMMITTED`.

`config.txt` of a run has the keys of the TGV files, in their order, then extras (GPU direct evidence,
OPS kernel and halo totals, `MPI_COMM_VOLUME_MB`, the diagnostics `DIAG_STEPS_DONE`, `DIAG_KERNELS`
and `DIAG_KERNEL_CALLS`, Slurm job and node list, pair index and order, plan position and seed, catalog
rows and fallback kernels, failure reason). `APP_WALL_TIME` is `Total Wall time` of the root rank (digits
as printed), `NITER` the steps, `APP_ARGS` the positional arguments `<grid> <grid> <grid> <steps>`,
`USER_HALO_TOTAL_TIME` empty (OPS prints that line only for user halo exchanges, which TGV has none of),
`TIMING_PARSER_VERSION`
`tgvstorenone_total_wall_time_v1`, `COORD_SOURCE` `not_measured`.

## 7. Exit statuses

| Status | Meaning |
|---|---|
| 0 | done; for `gonogo`: GO |
| 2 | usage, configuration or environment error (message starts with `ERROR:`) |
| 3 | time guard: stopped cleanly before a run that would not fit; resubmit with `submit --resume` |
| 4 | aborted after three consecutive failed decompositions (or an earlier abort is still in place) |
| 5 | the pass finished but at least one run or decomposition failed; nothing is retried automatically |
| 6 | `gonogo`: NO-GO |

The last line of every Slurm output file (`$C/slurm/*.out`) states the exit status.

## 8. Limitations to carry into every result

- **Tuning on rank 0 only, one shared catalog per decomposition.** 1440 is divisible by every factor of
  the 16 decompositions, so all ranks of a decomposition have the same local extents. But: (a) a kernel
  that rank 0 does not execute is never tuned and uses the default block on the ranks that do execute it
  (the six symmetry-boundary loops of TGV do work only on ranks that own that physical boundary; for the
  faces rank 0 does not own, the "tuned" block is the minimum of timings of an empty range: in the 8-rank
  LUMI CSV used as test fixture their best times are about 1e-6 s); (b) each candidate
  block is timed once, in one run: the catalog is "the minimum observed once", not an optimum; (c) rank 0
  is slower than its neighbours during the autotune run, so that run is not a measure of a uniformly
  tuned job; (d) no per-rank or per-node variation is captured.
- **A fixed list of 16 decompositions** (the list of the CloverLeaf_3D package: the 16 fastest of the
  SENGA2 sweep, not chosen for TGV). Nothing is said about the other 47 ordered triples of 160.
- **No confirmation stage.** The decomposition with the lowest time among the single-run measurements is
  optimistically biased: any "best decomposition" figure taken from the tables is selection-biased until
  fresh runs confirm it.
- **Catalog stability is not measured.** Each decomposition is tuned once.
- **One repetition (default).** A per-decomposition difference is only "resolved" when it exceeds the
  noise floor measured by the sentinels (`sentinels.csv`).
- **No physical validation.** The application prints no physical quantity, so neither a run nor the
  consistency check can tell a correct solution from a wrong or non-finite one (section 3). Adding a
  printed diagnostic (for example a global kinetic-energy reduction after the time loop) needs a change
  of the application and is a separate decision.
- **One scale.** No strong scaling, other sizes, shapes or applications. The step count is chosen from
  Q5 (default blocks, forced `d0`); other decompositions and modes run longer or shorter.
- `Total Wall time` covers the time-step loop only; start-up (allocation, initialisation) is in `JOB_WALL_TIME`.

## 9. What only LUMI can verify

None of the following was or could be checked before the first LUMI run:

- that TGV_StoreNone builds and links with this build driver on LUMI (the flags are those of
  `lumi_tgv_top3`, but the translator is called directly and `make` gets the variables of this package:
  ROCm clang/hipcc, `-std` levels, `libmpi_gtl_hsa`, `-lrocm_smi64`, the translator Python with `libclang`);
- that the hooks produce a non-empty best-block CSV for TGV on `gfx90a` at 160 ranks with rank-0-only
  tuning, and its size (8 ranks, 800^3: 25 of 26 kernels);
- that TGV under OPS MPI+HIP is numerically consistent across the 16 decompositions: this package cannot
  verify it at all, because the application prints no physical quantity;
- the device memory per GCD at 1440^3 on 160 ranks;
- that GPU direct is active and that each rank has its own GCD with the CPU map used;
- that the OPS axis order matches `X,Y,Z` of the arguments, and which grid `MPI_Dims_create` returns for
  160 ranks (expected `8x5x4`);
- the exact text of the log lines the validator reads at 160 ranks (processor grid, `Loaded`, warnings,
  the kernel table with `opensbliblock00Kernel033`, `Total Wall time`; taken from an 8-rank LUMI log),
  and whether the fallback warning is printed once per kernel;
- the time per step, hence the step count for 5-minute runs; wall times, budget consumption, disk
  volume and Slurm behaviour at the time limit.

## 10. Checks to do by eye in the first LUMI runs

- Budget: do not submit the sweep before `gonogo` has printed its projection with the chosen `--steps`.
- Open `ranks/rank_0.log` of the first multi-rank run and read: the processor grid line, `GPU Direct
  enabled`, the kernel table (26 kernels; `opensbliblock00Kernel033` called once per step), the OPS totals
  and `Total Wall time`.
- Check in `record.json` (`ranks`) or with `rocm-smi` that 8 different GCDs per node are used.
- Compare the catalog size and the fallback list of Q2 and Q5 with the earlier 8-rank MI250X value
  (25 tuned kernels of 26; kernel 1, the initialisation loop called once, falls back to the default block).
- If `gonogo` check 7 fails, compare the kernel tables of the two named runs: call counts are integers
  and must be equal, so a difference means that the runs did not execute the same loops.
- If allocations land on different node sets, `sentinels.csv` decides whether results of different jobs
  can be compared.

## 11. Troubleshooting

| Message | What to do |
|---|---|
| `No Slurm account` | pass `--account` or export `SLURM_ACCOUNT` |
| `Required OPS file is missing or not tracked` / `does not emit the ... hooks` | wrong branch: `git checkout feature/gpu_block_size && git pull` |
| `... command not found. Load the toolchain ...` | the environment file did not load the modules; `source environment.sh` and read its output |
| `Cannot locate ROCm clang/clang++` | set `HIP_CLANG_PATH` or `TGV_CC`/`TGV_CXX` |
| `Cannot locate libmpi_gtl_hsa` / `not linked against libmpi_gtl_hsa` | load `craype-accel-amd-gfx90a`; set `TGV_GTL_DIR` or `TGV_MPI_HIP_LINK` |
| `The OPS translator needs Python >= 3.8` / `lacks the OPS translator requirements` | `module load cray-python` in the environment file, or `--translator-python` with a venv that has `ops_translator/requirements.txt` |
| `BUILD FAILED at step: OPS library (make mpi_hip)` | read the end of `build/hip/build.log`; adjust `TGV_CXXFLAGS`/`TGV_HIPFLAGS` |
| `BUILD FAILED at step: application (make opensbli_mpi_hip)` | compile errors: `TGV_HIPFLAGS`; undefined symbols: `TGV_HIP_LINK`, `TGV_HIPRAND_LINK`, `TGV_MPI_HIP_LINK`, `TGV_LINK_EXTRA` |
| `generated HIP kernels lack the block-size autotuning hooks` | the OPS checkout is not the branch with the hook-emitting translator |
| `The build is not complete` | run `build --campaign $C` |
| `INVALID [OOM]` in Q4 | the problem does not fit in a GCD: do not submit; reduce `--grid` in a new campaign |
| `INVALID [OVERPARTITION]` | a local width is below what OPS needs; that decomposition cannot run |
| `"GPU Direct enabled" not printed` | the application did not receive `-gpudirect`; check `srun.log` and the arguments in `record.json` |
| `OPS used processor grid ... but ... was forced` | the forced arguments were not honoured; do not continue, report it |
| `rank(s) ... did not load exactly the ... catalog rows` | read the named rank log; the catalog path or file is not readable from the compute nodes |
| `ranks share a GPU` | the CPU map or the task layout is wrong for the node type: `prepare --cpu-bind` |
| `the once-per-step loop opensbliblock00Kernel033 ran ... times` / `no OPS kernel table` | the run stopped early or did not get `-OPS_DIAGS=2` or the `NITER` argument; read the rank 0 log and the arguments in `record.json` |
| `AUTOTUNE_FAILED` | read the reason in `state/events.log`; the default runs of that decomposition still ran |
| `TIME GUARD` (exit 3) | normal end of an allocation: `submit --resume` |
| `The go/no-go budget was projected for ... steps` | run `gonogo --steps <N>` with the step count of the frozen plan |
| `Another process holds the lock` | another job of this campaign is running (`squeue`) |
| `Executable hash changed` / `Plan hash changed` | results of different executables or plans are never mixed: start a new campaign |

## 12. Local tests

No network, GPU, Slurm or LUMI is needed. The tests use a fake `srun`/`sbatch`/`squeue`, a mock
executable (`fixtures/mock_tgv.py`), fake compilers with the real OPS makefiles, and one real
rank-0 best-block CSV of an earlier 8-rank tgvstorenone run on LUMI (800^3, decomposition 2x2x2).

```bash
cd OPS/scripts/lumi_tgv_decomp
for f in *.sh; do bash -n "$f"; done
python3 -m py_compile *.py fixtures/*.py
python3 -m unittest
```

They take a few minutes. They do not replace the qualification stage: the mock only prints the lines
the launcher parses.
