# SENGA2 MPI-decomposition sweep on LUMI-G

This package builds SENGA2 through OPS (Fortran f2c, MPI + HIP) on LUMI and measures one
cubic problem under every valid MPI decomposition, with three ways of choosing the GPU
block size of each kernel.

| Item | Default |
|---|---|
| Application | SENGA2, repository `https://github.com/vishnu-ncl/SENGA2`, commit `a5c76d3597117a514da93ce2e8a434674cd5bf04` |
| Scale | 20 LUMI-G nodes, 160 MPI ranks, one GCD (MI250X, `gfx90a`) per rank, GPU direct |
| Problem | 1440^3 cells, 100 time steps |
| Decompositions | all 63 ordered `Px x Py x Pz` with product 160 |
| Per decomposition | 1 autotune run (A), then 3 pairs of default (D) and precomputed (P) runs |
| Extra runs | 3 unforced baseline runs, and sentinel runs that measure the noise floor |

Nothing here needs an assistant: every step is a command of `run.sh`, each prints what it
did and what to run next, and every failure names the file or log to read.
No project, account or user name is stored in the package. You give the Slurm account
with `--account` (or `SLURM_ACCOUNT`).

Contents: [1 Steps](#1-steps-in-order) · [2 Build variables](#2-build-variables-and-the-prgenv-cray-fallback) ·
[3 What a run must satisfy](#3-what-a-valid-run-is) · [4 Resume and failures](#4-allocations-resume-and-failures) ·
[5 Results](#5-results) · [6 Campaign directory](#6-campaign-directory) · [7 Exit statuses](#7-exit-statuses) ·
[8 Limitations](#8-limitations-to-carry-into-every-result) · [9 Only LUMI can verify](#9-what-only-lumi-can-verify) ·
[10 First-run checks](#10-checks-to-do-by-eye-in-the-first-lumi-runs) · [11 Troubleshooting](#11-troubleshooting) ·
[12 Local tests](#12-local-tests)

## 1. Steps, in order

All commands are run from a LUMI login node. `$C` is the campaign directory printed by
`prepare`. After `prepare`, always use the frozen copy `$C/launcher/run.sh`, so that the
code of a campaign cannot change while it runs.

```bash
git clone --branch feature/gpu_block_size https://github.com/OP-DSL/OPS.git    # or: git pull
cd OPS/scripts/lumi_senga2_decomp
```

### 1.1 Look at the plan (no Slurm, nothing is written outside the directory you name)

```bash
bash run.sh plan --output-dir ~/senga2_plan_preview
```

It writes `plan.json`, `plan.csv` (one line per run: position, stage, decomposition, mode,
repetition, pair, full argument list, seed) and the qualification list `smoke_plan.csv`.
With the defaults it reports 63 autotune, 189 default and 189 precomputed sweep runs,
3 baseline runs and 8 sentinel runs. Decompositions that cannot run (a local width that is
not an integer, or below the halo depth 5) are skipped with a `NOTICE` line; for 1440^3 on
160 ranks nothing is skipped. 16 triples are flagged `halo60_risk` (see section 8).

Options: `--nodes`, `--grid`, `--steps`, `--repetitions`, `--seed`, `--fixed-order`,
`--sentinel-every`, `--exclude PxxPyxPz`, `--exclude-halo60-risk`.

### 1.2 Create the campaign

```bash
bash run.sh prepare --account <your_project>
```

This creates `/scratch/<your_project>/ops-senga2-decomp/campaigns/SENGA2_DECOMP_<date>`
(change with `--output-root DIR` or `--campaign DIR`) and puts into it: a frozen copy of
these scripts, the environment file, a snapshot of the OPS files that are needed
(`ops/c`, `ops/fortran`, `makefiles`, `ops_translator`) and the SENGA2 sources.

SENGA2 is cloned from the pinned repository and commit, which needs network access on the
login node. Alternatives:

```bash
bash run.sh prepare --account <your_project> --senga-root /path/to/SENGA2
bash run.sh prepare --account <your_project> --senga-repo <url> --senga-commit <hash>
```

`--senga-root` copies an existing checkout (tracked files only); the checkout itself is
never modified, and files that differ from their commit are listed in `campaign.json`.
Your OPS checkout is never modified either.

Other options: `--env-file FILE` (section 2), `--hdf5 strip|keep` (section 2.3),
`--translator-python /path/to/python3`, `--partition`, `--smoke-partition`,
`--build-partition`, `--cpu-bind`, `--fingerprint-tolerance`, and the plan options of 1.1
(the campaign fixes nodes, grid, steps, repetitions and seed).

From here on:

```bash
C=/scratch/<your_project>/ops-senga2-decomp/campaigns/SENGA2_DECOMP_<date>
```

### 1.3 Build the two executables (qualification stage Q0)

```bash
bash $C/launcher/run.sh build --campaign $C --print-toolchain    # what will be used; changes nothing
bash $C/launcher/run.sh build --campaign $C --build all          # in this shell (login node)
```

or, in a Slurm job (1 node of the build partition, 3 hours by default):

```bash
bash $C/launcher/run.sh build --campaign $C --build all --submit
bash $C/launcher/run.sh build --campaign $C --build all --submit --dry-run   # show sbatch only
```

Two builds are made and kept apart, because the grid is a compile-time parameter of SENGA2:
`small` (256^3, qualification) and `full` (1440^3, the sweep). Each has its own directory
`$C/build/<name>/` with its own copy of the sources, `build.log`, `build_manifest.json`
and executable hash. For each build the launcher

1. copies the frozen sources and edits the copy: grid sizes in `src/com_senga2.F90`,
   and the HDF5 handling (section 2.3);
2. runs the OPS translator (`-t f2c_hip`, about one minute) and checks that **every**
   generated HIP kernel calls `ops_get_kernel_block_size` and
   `ops_record_kernel_performance`. If one does not, the build stops: an executable
   without the hooks would silently ignore the tuned blocks;
3. builds the OPS Fortran MPI+HIP library (`make f2c_mpi_hip`) and the three SENGA2
   helper libraries;
4. compiles the application file by file and links it. The commands are the ones of
   `makefiles/Makefile.f_app`, issued one per file so that a failure names the file;
5. checks with `ldd` that the executable links `libmpi_gtl_hsa` (GPU-aware Cray MPICH).

The translator needs Python >= 3.8 and its packages. By default a virtual environment is
created in `$C/build/translator-env` with `pip install -r ops_translator/requirements.txt`
(network access to PyPI). If that is not possible, prepare with `--translator-python`.

If a step fails, the message gives the step, the exact command and the log. Fix the cause
(usually a variable of section 2) and run the same `build` command again: finished objects
are reused; if you changed a compiler or flag, the objects are discarded automatically.
Because the environment file is frozen in the campaign, either edit
`$C/environment.sh` before the first successful build, or export the `SENGA_*` variable in
your shell for a local build. Once a build is complete it is never rebuilt.

### 1.4 Qualification (Q1 to Q5)

```bash
bash $C/launcher/run.sh smoke --campaign $C --dry-run     # list the runs, submit nothing
bash $C/launcher/run.sh smoke --campaign $C --submit      # two jobs: Q1-Q3, then Q4-Q5
```

| Stage | Build | Resources | Runs | What it proves |
|---|---|---|---|---|
| Q0 | both | build | - | toolchain, hooks, `libmpi_gtl_hsa` |
| Q1 | small 256^3 | 1 node, 1 rank | default, 100 steps | the HIP build runs on MI250X; reference fingerprint; time |
| Q2 | small | 1 rank | autotune, then replay | hooks fire, catalog not empty, replay loads exactly the catalog |
| Q3 | small | 1 node, 8 ranks | 2x2x2, 8x1x1, 1x8x1, 1x1x8, unforced, 2x2x2 without `-gpudirect` | multi-rank correctness, forced grid is the printed grid, one GCD per rank, GPU direct on/off |
| Q4 | full 1440^3 | 20 nodes, 160 ranks, 10 steps | unforced, 4x5x8, 160x1x1, 1x160x1, 1x1x160 | memory fits, no overpartitioning, identifies `d0`, thin triples |
| Q5 | full | 160 ranks, 100 steps | forced `d0`: A, then D P P D | the whole per-decomposition pipeline, measured times |

The first job (Q1 to Q3) uses one node of `--smoke-partition`; the second (Q4, Q5) uses
the 20 nodes and starts only if the first ended with status 0. Use `--stage small` or
`--stage full` to submit one of them, `--time-small` / `--time-full` for the wall times,
and `--retry-failed` to rerun failed qualification runs (old attempts are kept).
The time guard of section 1.7 also applies here, with fixed expected durations (small
build: 1500 s autotune, 600 s default/precomputed; full build: 3600 s and 1200 s);
`--no-time-guard` switches it off.
Qualification runs never enter the sweep statistics.

### 1.5 Go / no-go

```bash
bash $C/launcher/run.sh gonogo --campaign $C \
    --gpu-hours-per-node-hour <rate> --remaining-gpu-hours <hours> --disk-available-gb <GB>
```

It prints one `PASS`/`FAIL` line per check and `VERDICT: GO` or `VERDICT: NO-GO`; the exit
status is 0 for GO and 6 for NO-GO. The checks:

1. every qualification run is valid (section 3), fingerprints within tolerance;
2. in Q3 and Q4 the printed processor grid equals the forced triple; unforced grids are shown;
3. no memory failure in 160x1x1; the thin triples 1x160x1 and 1x1x160 are valid. If they
   fail, inspect them. To run the sweep without all 16 `halo60_risk` triples, repeat the
   command with `--exclude-halo60-risk`; the plan must then be frozen with the same option;
4. Q5: catalog not empty, replay complete, the two default runs within 5 % of each other;
5. budget: the cost of the sweep projected from the measured Q5 times is at most 80 % of
   what is left. The projection (runs, wall hours, node-hours) is always printed. The
   conversion to GPU-hours is **your** input, because the billing rule must be confirmed for
   your project: `--gpu-hours-per-node-hour` (check the LUMI documentation and
   `lumi-allocations`; a LUMI-G node has 4 GPU cards) and `--remaining-gpu-hours`;
6. disk: the projected log volume fits in `--disk-available-gb` (an earlier campaign died
   with `Disk quota exceeded`; the tuning log of each autotune run can exceed 100 MB).

Without the three inputs, checks 5 and 6 fail and the verdict is NO-GO. `gonogo.json`
keeps the result; `measured_times.json` keeps the measured times, which the sweep then
uses for its time guard.

### 1.6 Freeze the sweep plan

```bash
bash $C/launcher/run.sh plan --campaign $C                         # all valid decompositions
bash $C/launcher/run.sh plan --campaign $C --exclude-halo60-risk   # only if go/no-go required it
```

The plan (`$C/plan.json` and its sha256) is written once. It is never regenerated: resume
reads it, and a changed file is refused.

### 1.7 Submit, and continue over several allocations

```bash
bash $C/launcher/run.sh submit --campaign $C --dry-run                 # show sbatch and the job script
bash $C/launcher/run.sh submit --campaign $C --time 24:00:00 --jobs 2  # two chained 20-node jobs
bash $C/launcher/run.sh status --campaign $C
```

`submit` refuses to start unless the verdict is GO (`--override-gonogo "reason"` records an
explicit override). `--jobs N` chains N allocations; each one continues where the previous
stopped, and a job that finds nothing left to do ends at once. To continue later:

```bash
bash $C/launcher/run.sh submit --campaign $C --resume
bash $C/launcher/run.sh submit --campaign $C --resume --retry-failed   # also rerun failed runs
```

`--resume` refuses while an earlier sweep job is still pending or running, so that no run is
duplicated. What happens inside a job is described in section 4.

Time guard: before each run (before each D/P pair as a whole) the job compares the time
left in the allocation with the expected duration of that run times 1.2 plus 60 s, and
stops cleanly (exit status 3) instead of starting something it cannot finish. Expected
durations come from `measured_times.json` (1.5 x the Q5 elapsed time + 60 s); before that
they are 3600 s (A) and 1200 s (D, P). Override with `--est-autotune`, `--est-default`,
`--est-precomputed` (seconds), or disable with `--no-time-guard`. Each run is also killed
after 3 x its expected duration and recorded as `TIMEOUT`.

### 1.8 Collect

```bash
bash $C/launcher/run.sh collect --campaign $C
```

It can be run at any time, also on a partial campaign. See section 5.

## 2. Build variables and the PrgEnv-cray fallback

SENGA2 has never been built on LUMI. The default recipe is therefore a starting point, and
every compiler and flag can be changed without editing code.

### 2.1 Default: `environment.sh`

LUMI, `partition/G`, `PrgEnv-gnu`, `rocm`, `craype-accel-amd-gfx90a`; ROCm `clang`/`clang++`
as host compilers, `hipcc` for HIP, Cray MPICH with `libmpi_gtl_hsa`, and `ftn` (gfortran
under PrgEnv-gnu) for Fortran and for the final link. To pin module versions or change
anything, copy the file, edit it and give it to `prepare --env-file`.

`bash run.sh build --print-toolchain` (run it after `source environment.sh`) lists every
variable, its default and the value found in your environment:

| Variable | Meaning | Default |
|---|---|---|
| `SENGA_F90`, `SENGA_MPIF90` | Fortran compiler, MPI Fortran compiler and linker | `ftn` |
| `SENGA_FFLAGS` | Fortran flags, OPS library and application | `-O3 -g -ffree-form -ffree-line-length-none -fopenmp` (as `makefiles/Makefile.gnu`) |
| `SENGA_APP_FLAGS` | application Fortran flags, with its optimisation level | `-g -cpp -O3 -fallow-argument-mismatch` |
| `SENGA_APP_FLAGS_LARGE` | if set, replaces `SENGA_APP_FLAGS` for the large files | unset |
| `SENGA_LARGE_FILES` | the large generated files | the 16 files of SENGA2 `src/lumi_compile.sh` |
| `SENGA_FMOD_FLAG` | option that sets the module output directory | `-J` |
| `SENGA_CC`, `SENGA_CXX`, `SENGA_MPICXX` | host C/C++ compilers | ROCm `clang`, `clang++` (`HIP_CLANG_PATH` or `<rocm>/llvm/bin`) |
| `SENGA_HIPCC` | HIP compiler driver | `hipcc` |
| `SENGA_CXXFLAGS` | host C++ flags | `-O3 -fPIC -std=c++11` |
| `SENGA_HIPFLAGS` | HIP flags (c++11 becomes c++14 for the generated kernels) | `-O3 -std=c++11 --offload-arch=gfx90a -D__HIP_PLATFORM_AMD__` |
| `SENGA_ROCM_ROOT`, `SENGA_MPI_ROOT`, `SENGA_GTL_DIR` | installation roots | `HIP_INSTALL_PATH`/`ROCM_PATH`, `CRAY_MPICH_DIR`, `PE_MPICH_GTL_DIR_amd_gfx90a` |
| `SENGA_MPI_HIP_LINK` | MPI and GPU transport libraries | `-L<mpi>/lib -L<gtl> -lmpi -lmpi_gtl_hsa` |
| `SENGA_HIP_LINK`, `SENGA_HIPRAND_LINK` | HIP runtime, hipRAND | `-L<rocm>/lib -L<rocm>/lib64 -lamdhip64 -lrocm_smi64`, `-L<rocm>/lib -lhiprand` |
| `SENGA_LINK_EXTRA` | extra link flags | `-fopenmp -lstdc++` |
| `SENGA_HDF5_ROOT`, `SENGA_HDF5_LINK` | only with `--hdf5 keep` | `HDF5_INSTALL_PATH`/`HDF5_DIR`, `-L<hdf5>/lib -lhdf5_hl -lhdf5 -lz` |
| `SENGA_BUILD_JOBS` | parallel compile jobs | 8 |
| `SENGA_REQUIRE_GTL` | `0` skips the `libmpi_gtl_hsa` check (never for LUMI runs) | 1 |

`-fallow-argument-mismatch` needs gfortran 10 or newer. If the loaded gfortran rejects a
flag, remove it from `SENGA_APP_FLAGS`.

### 2.2 Fallback: Cray Fortran, `environment_cray.sh`

Nobody has verified which Fortran toolchain links this application on LUMI. If the
PrgEnv-gnu build fails in the Fortran compile or in the final link, create a **new**
campaign with the Cray Fortran recipe, which follows SENGA2 `src/lumi_compile.sh` and
`makefiles/Makefile.cray`:

```bash
bash run.sh prepare --account <your_project> --env-file environment_cray.sh
```

It loads `PrgEnv-cray` and sets `SENGA_FFLAGS="-g -fopenmp -M 878 -f free"`,
`SENGA_APP_FLAGS="-O3 -N 1023"` and `SENGA_APP_FLAGS_LARGE="-O1 -N 1023"`.
The last one is the Fortran optimisation option: `src/lumi_compile.sh` compiles the 16
large generated files at `-O1` with Cray Fortran. Raise it if the compiler copes, or unset
it to use `SENGA_APP_FLAGS` everywhere. The host C++ and HIP compilers stay the ROCm ones;
the file shows how to switch them to the Cray `CC` driver as `Makefile.cray` does.

Whatever toolchain is used, both executables must come from the same campaign, and results
of two campaigns must not be mixed without saying so: the compiler is part of the result.

### 2.3 HDF5: `--hdf5 strip` (default) or `keep`

`strip` reproduces what the earlier single-GPU SENGA2 campaigns did, on the campaign copy:
the three HDF5 objects are removed from the `f2c_mpi_hip` library rule; in the SENGA2
sources `use OPS_Fortran_hdf5_Declarations` and every `call ops_*hdf5*` are commented with
a marker line, and `print_dats`, `print_alldats`, `print_output` become empty routines.
The run needs no HDF5 and writes no field dumps.

`keep` leaves the sources alone and links parallel HDF5 (`module load cray-hdf5-parallel`
in the environment file, or `SENGA_HDF5_ROOT`). Be aware that SENGA2 then writes a full
HDF5 field dump at step 0 of every run, inside the measured `total_ttime`.

## 3. What a valid run is

Each run is one `srun` in a fresh directory `runs/<scope>/<run_id>/attempt_NNN/` with its
own inputs (`input/*.dat`, `TGV_species.dat`, `t_initial.dat`, empty `output/`) and its own
generated `input/cont.dat`. A run is `VALID` only if all of this holds:

- `srun` and every rank exit with status 0;
- exactly one positive `total_ttime` in the root log (`ranks/rank_0.log`);
- `GPU Direct enabled` printed by the root rank (and absent in the one run without `-gpudirect`);
- the root rank reports the expected `OPS_autotune_mode`;
- the processor grid printed by OPS equals the forced triple (unforced: it uses all ranks);
- one status record per rank, each with a different (host, GPU id);
- no `overpartitioning`, `out of memory`, `nan` or `Error`/`ERROR` text in any rank log
  (`nan` is ignored inside the OPS kernel table, whose derived columns are not physics);
- all time steps present in the step history;
- `output/tgv_stat.dat` ends with finite `tke`, `enstrophy`, `tmax` that agree with the
  reference within the tolerance (default 1e-6, relative). The reference is the first valid
  run of the same build and step count (`references/`);
- precomputed runs: every rank prints `Loaded N precomputed block sizes from <path>` with N
  equal to the catalog rows and the path of that decomposition's own catalog; rank 0 falls
  back to the default block for exactly the kernels that the autotune run executed but did
  not finish tuning; no other rank reports a catalog kernel as missing;
- default and autotune runs: no rank prints a `Loaded ... precomputed` line.

Failure categories: `OOM`, `OVERPARTITION`, `TIMEOUT`, `FINGERPRINT_MISMATCH`, `OTHER`.
The reason is in `record.json` (`reason`) and in `state/events.log`, with the directory of
the logs.

Block modes and what each rank gets:

| Mode | Arguments | Environment |
|---|---|---|
| D default | all ranks `-OPS_AUTOTUNE_MODE=0 OPS_BLOCK_SIZE_X=32 OPS_BLOCK_SIZE_Y=4 OPS_BLOCK_SIZE_Z=1` | CSV logs off on every rank |
| A autotune | rank 0 `-OPS_AUTOTUNE_MODE=1`; ranks 1..159 mode 0 with the default block | full tuning CSVs on rank 0 only |
| P precomputed | all ranks `-OPS_AUTOTUNE_MODE=2` | `OPS_BLOCKSIZE_CSV=<catalog of this decomposition>`; CSV logs off, as in D |

All runs also get `OPS_FORCE_DECOMP_X/Y/Z=` (except the unforced baseline), `-gpudirect`
and `-OPS_DIAGS=2`. These are command-line arguments of the application; OPS Fortran
truncates arguments at 63 characters and the launcher refuses longer ones. `OPS_LOGS_DIR`
always points inside the run directory.

The catalog of a decomposition is built once, from rank 0's
`ops_blocksize_best_logmod.csv` of that decomposition's own autotune run: footer lines are
skipped, one row per kernel is kept (lowest `best_time`), every block must be positive with
`bx*by*bz <= min(1024, max_threads_per_block)`. It is written as `kernel_id,bx,by,bz`,
made read-only and hashed (`catalogs/<scope>/<build>_<decomposition>/`). If the CSV is
missing, empty, has only footers, has an invalid row or two different blocks with the same
best time for one kernel, or if the autotune run is invalid, the decomposition is marked
`AUTOTUNE_FAILED`: no catalog is written, no precomputed run is launched, and no substitute
is ever fabricated. Its default runs are still executed.

## 4. Allocations, resume and failures

Order inside a sweep job:

1. the unforced baseline (3 default runs without `OPS_FORCE_DECOMP_*`). The grid OPS prints
   defines `d0`; it is not a constant in the code (`state/d0.json`);
2. one sentinel run (forced `d0`, default blocks) at the start of **every** allocation, and
   one after every 8 decompositions;
3. the decompositions in the seeded random order of the plan. For each: A first, then the
   pairs; the first mode of the pairs alternates (D P / P D / D P at even plan positions,
   P D / D P / P D at odd ones).

Rules:

- A run with a valid record is never repeated. Resume aborts if the executable hash or the
  plan hash differs from the records.
- A D/P pair is atomic within one allocation. If only one member finished, it is kept,
  marked superseded (`SUPERSEDED.json`) and the whole pair is rerun.
- A frozen catalog is reused in a later allocation only if its sha256 still matches;
  otherwise `CATALOG REJECTED` is logged and no precomputed run is launched for it.
- A failed run is never retried automatically. `--retry-failed` reruns failed runs; the
  failed attempts are kept and marked superseded.
- If a default run fails with `OOM` or `OVERPARTITION`, the decomposition is marked
  `INVALID_AT_RUNTIME` and its remaining runs are skipped; the sweep continues.
- Three consecutive failed decompositions abort the job (exit status 4) and write
  `state/ABORTED.json`. Later jobs of a chain then do nothing. After fixing the cause,
  continue with `submit --resume --retry-failed`.

## 5. Results

`collect` writes into `$C/results/`:

| File | Content |
|---|---|
| `runs.csv` | one row per run record, including failed, superseded and qualification runs (flagged; `used_in_statistics`) |
| `decompositions.csv` | one row per planned decomposition |
| `incomplete_decompositions.csv` | decompositions without a complete set, with what is missing and why |
| `sentinels.csv` | sentinel runs per allocation and overall: n, min, median, max, relative half-range |
| `collection_manifest.json` | input records with hashes, exclusions, code version, grouping key, estimator definitions |

Estimators (also in the manifest), with `T` = `total_ttime` printed by the root rank:

- `D_median_total_ttime_s`, `D_min_...`, `D_max_...` (and `P_...`): median, minimum and
  maximum over the 3 repetitions, given only when all 3 are valid. No confidence interval.
- `S_pair1..3` = `T(D, pair i) / T(P, pair i)`; `S_median_of_paired_ratios` is their median.
  `S_ratio_of_medians` is a different estimator and is tabulated beside it.
- `A_total_ttime_s`: the single autotune run; it is never used as a D or P sample.
  `search_cost_A_over_D_median` has one sample and no uncertainty.
- `catalog_rows`, `executed_kernels`, `kernel_coverage`, and `catalog_default_time_share`
  (share of the default-run kernel time covered by the catalog; flagged below 0.90).
- Covariates: `slurm_job_ids`, node lists, `halo60_risk`, `permutation_class`, `is_d0`.

A decomposition is complete with one valid autotune run and frozen catalog and all D and P
repetitions valid in same-job pairs. Incomplete ones are listed and never imputed: their
missing values stay empty. The collector does not rank decompositions and does not name a
best one.

## 6. Campaign directory

```text
$C/
  campaign.json  source_snapshot.json  environment.sh  launcher/     frozen configuration and code
  source/OPS  source/SENGA2                                          frozen sources (never edited)
  build/small  build/full     OPS/ SENGA2/ build.log build_manifest.json senga2_f2c_mpi_hip
  smoke_plan.json  plan.json  *.sha256  *.csv                        frozen plans
  runs/qualification/<run_id>/attempt_NNN/   runs/sweep/<run_id>/attempt_NNN/
      record.json        the per-run record (decomposition, mode, pair, job, nodes, arguments,
                         environment, hashes, commits, modules, times, per-rank host and GPU, verdict)
      srun.log  ranks/rank_N.log  ranks/status_N.json  execution.json
      work/              the run directory: input/cont.dat, output/tgv_stat.dat, ...
      rank_0/autotune_on/*.csv   only in autotune runs
  catalogs/<scope>/<build>_<PxxPyxPz>/blocks.csv  manifest.json
  references/            fingerprint reference per build and step count
  state/events.log  state/d0.json  state/decompositions/  state/ABORTED.json
  gonogo.json  measured_times.json  submissions.json  slurm/  results/
```

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

- **Tuning on rank 0 only, one shared catalog per decomposition.** This is acceptable here
  because the domain is periodic and 1440 is divisible by every factor of 160, so all ranks
  have the same local extents. But: (a) a kernel that rank 0 does not execute is never
  tuned and uses the default block on the ranks that do execute it; (b) each candidate
  block is timed once, in one run: the catalog is "the minimum observed once", not an
  optimum; (c) rank 0 is slower than its neighbours during the autotune run, so that run is
  not a measure of a uniformly tuned job; (d) no per-rank or per-node variation is captured.
- **No confirmation stage.** The decomposition with the lowest median among 63 three-run
  medians is optimistically biased. Any "best decomposition" figure taken from the tables
  is selection-biased until fresh runs confirm it.
- **Catalog stability is not measured.** Each decomposition is tuned once.
- **Three repetitions.** Per-decomposition statements are limited to "resolved or not"
  (ranges do not overlap and the ratio exceeds the noise floor from `sentinels.csv`).
- **One scale.** No strong scaling, other sizes, shapes, step counts or applications.
- **`halo60_risk`.** SENGA2 declares some y/z-split fields with 60-deep halos. In 16 triples
  the local y or z width is below 60. Nothing rejects them and they have never been run;
  they are flagged in the plan and in the tables, and Q4 exercises the two thinnest.
- **Output at step 0.** `cont.dat` is generated so that no dump, report or statistics output
  falls inside steps 1..100 (the three intervals are set to steps + 1; the step line is
  `<tstep> 1 100 <switch>`). SENGA2 always calls its output routine once at step 0, before
  the time loop, and that call cannot be switched off by an interval. With `--hdf5 strip`
  it writes a few numbers; it is identical in every run.
- `total_ttime` includes initialisation. Compare it with `kernel_time_s` and the halo times
  in `runs.csv`.

## 9. What only LUMI can verify

None of the following was or could be checked before the first LUMI run:

- that SENGA2 builds and links with the Cray/ROCm toolchain (which Fortran compiler,
  whether `-O1` is needed for the large files, whether HDF5 must be stripped, the
  `libmpi_gtl_hsa` link);
- that the hooks produce a non-empty best-block CSV for SENGA2 on `gfx90a`, and its size;
- that SENGA2 under OPS MPI+HIP is numerically correct with more than one rank, including
  the thin triples (it has never run on more than one rank);
- the device memory per GCD at 1440^3 on 160 ranks for the extreme decompositions;
- that GPU direct is active and that each rank has its own GCD with the CPU map used;
- that the OPS axis order matches `X,Y,Z` of the arguments, and which grid
  `MPI_Dims_create` returns for 160 ranks;
- the exact text of the log lines the validator reads (processor grid, `Loaded`, warnings,
  step history, `total_ttime`), and whether the fallback warning is printed once per kernel;
- wall times, budget consumption, disk volume and Slurm behaviour at the time limit;
- that 160 ranks can share one run directory (SENGA2 opens two small dump files on every rank).

## 10. Checks to do by eye in the first LUMI runs

- Budget: do not submit the sweep before `gonogo` has printed its projection. The only
  earlier estimate (about 23 minutes per decomposition, 24 hours of 20-node time) comes
  from another GPU on one device and is a lower bound.
- Open `ranks/rank_0.log` of the first multi-rank run and read: the processor grid line,
  `GPU Direct enabled`, the step history, `total_ttime`, and the tail of
  `work/output/tgv_stat.dat`.
- Check in `record.json` (`ranks`) or with `rocm-smi` that 8 different GCDs per node are used.
- Check that `total_ttime` is not dominated by initialisation or I/O.
- Compare the catalog size and the fallback list of Q2 and Q5 with the earlier MI300A value
  (81 of 102 kernels). A very small catalog means 100 steps do not complete the candidate
  search on this machine.
- If the time step is adaptive the final values may differ in the last digits between
  decompositions. If runs fail with `FINGERPRINT_MISMATCH` by a tiny amount, that is a
  decision for you: a new campaign with `prepare --fingerprint-tolerance <value>`. The
  tolerance is never loosened silently.
- If allocations land on different node sets, `sentinels.csv` decides whether results of
  different jobs can be compared.

## 11. Troubleshooting

| Message | What to do |
|---|---|
| `No Slurm account` | pass `--account` or export `SLURM_ACCOUNT` |
| `Cannot clone ...` | no network: clone elsewhere, check out the pinned commit, use `--senga-root` |
| `Required OPS file is missing or not tracked` / `does not emit the ... hooks` | wrong branch: `git checkout feature/gpu_block_size && git pull` |
| `... command not found. Load the toolchain ...` | the environment file did not load the modules; `source environment.sh` and read its output |
| `Cannot locate ROCm clang/clang++` | set `HIP_CLANG_PATH` or `SENGA_CC`/`SENGA_CXX` |
| `Cannot locate libmpi_gtl_hsa` / `not linked against libmpi_gtl_hsa` | load `craype-accel-amd-gfx90a`; set `SENGA_GTL_DIR` or `SENGA_MPI_HIP_LINK` |
| `The OPS translator needs Python >= 3.8` | `module load cray-python` in the environment file, or `--translator-python` |
| `BUILD FAILED ... at step: compile <file>` | read the end of `build/<name>/build.log`; adjust `SENGA_FFLAGS`/`SENGA_APP_FLAGS`, or use the Cray fallback (2.2) |
| `BUILD FAILED ... hipcc f2c_hip/f2c_hip_kernels.cpp` | read `build/<name>/hip_kernels.log`; adjust `SENGA_HIPFLAGS` |
| `BUILD FAILED ... link senga2_f2c_mpi_hip` | undefined symbols name the missing library: `SENGA_LINK_EXTRA`, `SENGA_HIP_LINK`, `SENGA_HIPRAND_LINK`, `SENGA_MPI_HIP_LINK` |
| `generated HIP kernels lack the block-size autotuning hooks` | the OPS checkout is not the branch with the hook-emitting translator |
| `The <name> build is not complete` | run `build --campaign $C --build <name>` |
| `INVALID [OOM]` in Q4 | the problem does not fit in a GCD for that triple: do not submit; reduce `--grid` in a new campaign |
| `INVALID [OVERPARTITION]` | a local width is below what OPS needs; that decomposition cannot run |
| `"GPU Direct enabled" not printed` | the application did not receive `-gpudirect`; check `srun.log` and the arguments in `record.json` |
| `OPS used processor grid ... but ... was forced` | the forced arguments were not honoured; do not continue, report it |
| `rank(s) ... did not load exactly the ... catalog rows` | read the named rank log; the catalog path or file is not readable from the compute nodes |
| `ranks share a GPU` | the CPU map or the task layout is wrong for the node type: `prepare --cpu-bind` |
| `AUTOTUNE_FAILED` | read the reason in `state/events.log`; the default runs of that decomposition still ran |
| `TIME GUARD` (exit 3) | normal end of an allocation: `submit --resume` |
| `Another process holds the lock` | another job of this campaign is running (`squeue`) |
| `Executable hash changed` / `Plan hash changed` | results of different executables or plans are never mixed: start a new campaign |

## 12. Local tests

No network, GPU, Slurm or LUMI is needed. The tests use a fake `srun`/`sbatch`/`squeue`, a
mock executable, fake compilers and a synthetic source tree; one fixture is a real
best-block CSV of an earlier single-GPU SENGA2 run.

```bash
cd OPS/scripts/lumi_senga2_decomp
for f in *.sh; do bash -n "$f"; done
python3 -m py_compile *.py fixtures/*.py
python3 -m unittest
```

They take about three minutes. They do not replace the qualification stage: the mock only
prints the lines the launcher parses.
