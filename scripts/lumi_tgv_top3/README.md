# TGV StoreNone Top 3 on LUMI

`run.sh` prepares the bundled dataset, checks selected cases, probes Slurm and submits a build followed by one GPU job per node count. It does not run autotuning. It requires Bash and Python 3.9+.

Run from any directory; OPS is located relative to the script. Override with `--ops-root`.

```bash
bash /path/to/OPS/scripts/lumi_tgv_top3/run.sh --dry-run
```

The default input is `OPS/dataset-lumi/TOP3_DATASET.tar.gz` with its `.sha256` file. The launcher verifies the checksum, extracts once under a lock, verifies every packaged file, and reuses the validated folder. An existing folder with different contents is rejected. `--dataset /path/to/TOP3_DATASET` selects an already extracted package explicitly.

## Current dataset coverage

The package has complete catalogs for **6, 8, 10, 12, 16, 20, 24, 32 and 40 nodes**. Nodes **2, 3 and 4** are missing original rank files. A dry run lists these gaps; build/submission of a selection containing them stops before any jobs are sent. No nodes are silently skipped.

```bash
bash scripts/lumi_tgv_top3/run.sh --validate \
  --nodes 6,8,10,12,16,20,24,32,40
```

## Analyze actual LUMI resources

Execute on a LUMI login node using the new user's project account:

```bash
bash scripts/lumi_tgv_top3/run.sh --analyze-resources \
  --account project_XXXXXXXXX \
  --nodes 6,8,10,12,16,20,24,32,40 \
  --output-root /scratch/project_XXXXXXXXX/user/tgv-results \
  --date-stamp campaign01
```

The analyzer saves `preflight/resource_report.json` and `submission_plan.csv`. It queries `sinfo`, `scontrol`, account associations and the user's queue, and runs **`sbatch --test-only`**, with the requested resources and walltime, for each node count. This does not submit a job. Missing commands or unavailable information are reported as unknown, not as free capacity. Node-state counts are deduplicated within each partition; nodes can belong to several partitions. Reservations and account limits can prevent immediate allocation even when nodes appear idle.

`--partition standard-g` is the default. `--partition auto` also evaluates `small-g` for 2–4 nodes and prefers the earliest available estimate, using `standard-g` as a tie breaker. It never shortens the requested walltime or uses development partitions for measurements. Estimates are snapshots, not reservations or a promise that all jobs start together.

## Environment and build

The default `environment.sh` preserves an already loaded HIP/Cray MPI environment or loads LUMI, partition/G, PrgEnv-gnu, ROCm and the gfx90a accelerator module. Site module versions can change. For a reproducible run, pass an explicit site profile:

```bash
# lumi-env.sh: use module versions available to your LUMI account.
module load LUMI
module load partition/G
module load PrgEnv-gnu
module load rocm
module load craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1
```

The profile is copied into the campaign and sourced by build and compute jobs. It must provide `hipcc`, Cray `cc`/`CC`, `CRAY_MPICH_DIR`, and the AMD GPU transport library location. The launcher configures OPS/HIP/MPI flags itself.

The launcher snapshots the relevant tracked OPS sources **including their current working-tree contents** into the result folder. This avoids concurrent builds or changing the user's checkout. The required existing local OPS changes are the TGV fourth argument for simulation steps and `OPS_DISABLE_CSV_LOGS`/`OPS_CSV_LOG_LEVEL` in the runtime. If they are absent, preparation stops. Carry these source changes when transferring OPS to another user; they are not supplied by the dataset archive.

One build job compiles OPS MPI+HIP and `TGV_StoreNone`, then freezes the executable and records its SHA256, source hashes, Git revision, compiler versions and environment. By default the build creates an isolated translator Python environment and installs `ops_translator/requirements.txt`; this requires package access. To use a prepared/offline environment, pass `--translator-python /path/to/venv/bin/python3`.

Choose an output path without whitespace: the upstream OPS Makefiles use this path during compilation.

`--build` performs the same build in the current shell environment via the profile, without submitting jobs. `--submit` instead submits a single build job in `small-g` by default, and all measurement jobs depend on its success. Customize `--build-partition`, `--build-time` and `--build-jobs` as needed.

## Submit the complete subset

```bash
bash scripts/lumi_tgv_top3/run.sh --submit \
  --account project_XXXXXXXXX \
  --nodes 6,8,10,12,16,20,24,32,40 \
  --env-file /path/to/lumi-env.sh \
  --output-root /scratch/project_XXXXXXXXX/user/tgv-results \
  --date-stamp campaign01 \
  --time 01:00:00
```

Only `--submit` sends jobs. The resource analysis and catalog validation happen before submission. The selected partition must pass its exact Slurm eligibility probe. In this example there are **one build job and nine measurement jobs**. With a complete 2–40-node dataset there would be one build job and twelve measurement jobs.

Each measurement allocation uses eight ranks and eight GPU GCDs per node, seven reserved CPU cores per rank, one OpenMP thread per process, and the LUMI CPU/GPU affinity map. Each rank selects its local GCD. GPU-aware Cray MPI and OPS `-gpudirect` are enabled; linkage and the runtime confirmation are checked.

The cases run sequentially inside the allocation:

- A: recorded MPI reference, default blocks `(32,4,1)`.
- C: recorded MPI reference, its precomputed blocks.
- D_top1/2/3: selected MPI shapes, their recorded precomputed blocks.

The reference MPI shape is forced from the dataset; it is not an observation of an automatic MPI choice. Identical precomputed cases are executed once and retain all their roles. The full plan has 57 unique measured executions; the currently complete subset has 42. Each measured case runs **once for 500 simulation steps**. Only rank 0 writes OPS diagnostic CSVs; all ranks read the correct block file and emit small status/log records.

## Recompiled executable and compatibility checks

The historical dataset was produced with other executables. This launcher performs a **new campaign transferring recorded blocks**, not a claim of exact historical-binary replay.

Before measurements, each unique precomputed decomposition runs one **separate one-step smoke test at the recorded grid and MPI shape**. These tests use precomputed mode, never autotuning, and are stored under `_compatibility/`, outside scientific results. They verify successful MPI/GPU execution, block loading on every rank, and rank-0 kernel IDs, stencil/array metadata, local geometry and thread limits against the historical tuning log. Failures stop the node job before measured cases. They consume time and must fit within `--time` along with the five measured roles.

These metadata checks **do not prove identical kernel arithmetic or numerical equivalence between source versions**. The comparison is intentionally recorded as a transfer experiment. Original manifests remain unchanged. A new campaign manifest records the new executable, source set ID and compatibility-report hash, alongside `original_manifest.json`.

## Outputs and resume

```text
<output-root>/TGV_TOP3_<stamp>/
  campaign.json
  resource_report.json
  source_snapshot.json
  source_changes.patch
  submissions.json
  build/build_manifest.json
  build/build.log
  slurm/
  OPS_LOGS_<stamp>_nNNN/
    _compatibility/
    precomputed_blocks/
    tgvstorenone_hip_<grid>_cfg1_comparison_<stamp>_nNNN/
      mode_1/decomp<shape>/iter_1/autotune_off|autotune_precomputed/
        config.txt
        run.log
        result.json
        mpi_rank_metrics.csv
        rank_0/
        ranks/
    summary.csv
    comparisons.csv
    comparisons.json
```

Main speedup is **C/D** for wall time, kernel time and halo time. Values above one favor tested MPI. A/D is also retained. With one measured repetition these are single ratios. Kernel/halo values are OPS rank means, not an additive decomposition of wall time.

```bash
bash scripts/lumi_tgv_top3/run.sh --status --campaign /path/to/TGV_TOP3_campaign01
bash scripts/lumi_tgv_top3/run.sh --collect --campaign /path/to/TGV_TOP3_campaign01
```

`--collect` writes global summaries and reports node jobs without summaries. Reinvoke the same submission command with `--resume` to retain successful cases and retry jobs that Slurm reports as failed. Running/pending/completed job IDs are not submitted again. Unknown prior job states stop resubmission to avoid duplicates. Failed execution attempts are retained separately. If a dependency was cancelled or a completed job did not produce valid outputs, inspect the campaign rather than deleting its records blindly.

## Local validation

```bash
bash -n scripts/lumi_tgv_top3/run.sh scripts/lumi_tgv_top3/environment.sh
cd scripts/lumi_tgv_top3
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_launcher
```

Tests use local fake executables and mocked Slurm; they never submit real jobs. HIP compilation, GPU Direct and actual MPI behavior require validation on LUMI.
