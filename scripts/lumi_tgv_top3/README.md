# TGV StoreNone Top 3 on LUMI

`run.sh` prepares the bundled dataset, checks selected cases, probes Slurm and submits a build followed by one GPU job per node count. It does not run autotuning. The launcher requires Bash and Python 3.6+. Compiling OPS additionally requires Python 3.8+ for its translator.

Run from any directory; OPS is located relative to the script. Override with `--ops-root`.

```bash
bash /path/to/OPS/scripts/lumi_tgv_top3/run.sh --dry-run --account project_XXXXXXXXX
```

Results and preflight reports default to `/scratch/<account>/ops-decomposition/campaigns/TGV_TOP3_<stamp>/`, following the original LUMI app's campaign location. The account comes from `--account` or `SLURM_ACCOUNT`; no project ID or username is hardcoded. For a different server/filesystem or a local dry run, pass `--output-root /path/to/results` (the campaign is created directly below that directory). Without an account or an explicit output path, actions that create reports/results stop with an explanatory error. Help and `--validate` do not require an output path. An explicit `--campaign` continues to address existing campaigns, including those created at the previous location; existing results are not moved.

The default input is `OPS/dataset-lumi/TOP3_DATASET.tar.gz` with its `.sha256` file. The launcher verifies the checksum, extracts under a lock, verifies every packaged file, and reuses the validated folder. When an updated archive explicitly supersedes the installed package, it validates both versions, preserves the old folder under `TOP3_DATASET.previous-*/TOP3_DATASET`, and activates the new one. Unknown or modified packages are rejected. `--dataset /path/to/TOP3_DATASET` selects an already extracted package explicitly. Existing campaigns remain tied to their recorded dataset identity; start a new campaign after a package upgrade.

## Current dataset coverage

Version 2 has complete runtime catalogs for **all 12 node counts: 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32 and 40**: 48 roles and 45 unique precomputed catalogs. Every catalog contains one `blocks.csv` shared by all MPI ranks, with all 25 tuned kernel IDs (2–26). Initialization kernel 1 uses the existing OPS default fallback, as in the original logs.

For 6–40 nodes the original shared catalog bytes are preserved. For 2–4 nodes, the original light exports retained only ranks 0 and 1 of per-rank catalogs. Their portable shared catalogs are now derived from the original best logs of those ranks, from the same selected tuning repetition, using the existing app's `shared_blocks` policy: most frequent block, then lowest measured best time, then block tuple. Every selected block was measured; this is not a recovery of missing per-rank measurements. Historical timings for 2–4 nodes do not measure the new shared policy. The new campaign measures it afresh. Original manifests and source logs are preserved, and `block_provenance.csv` traces all 1,125 selected blocks.

```bash
bash scripts/lumi_tgv_top3/run.sh --validate
```

## Analyze actual LUMI resources

Execute on a LUMI login node using the new user's project account:

```bash
bash scripts/lumi_tgv_top3/run.sh --analyze-resources \
  --account project_XXXXXXXXX \
  --nodes 2,3,4,6,8,10,12,16,20,24,32,40 \
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

An older system Python (3.6 or 3.7) can run help, dataset validation, dry runs and Slurm orchestration. During compilation, the launcher looks for a Python 3.8+ interpreter on `PATH` to create the translator environment. If none is available, load a newer Python in your environment profile, or supply `--translator-python` pointing to an existing environment with the translator dependencies. To choose the launcher interpreter itself, set `OPS_LAUNCHER_PYTHON=/path/to/python3`.

The launcher snapshots the relevant tracked OPS sources **including their current working-tree contents** into the result folder. This avoids concurrent builds or changing the user's checkout. The required existing local OPS changes are the TGV fourth argument for simulation steps and `OPS_DISABLE_CSV_LOGS`/`OPS_CSV_LOG_LEVEL` in the runtime. If they are absent, preparation stops. Carry these source changes when transferring OPS to another user; they are not supplied by the dataset archive.

One build job compiles OPS MPI+HIP and `TGV_StoreNone`, then freezes the executable and records its SHA256, source hashes, Git revision, compiler versions and environment. By default the build creates an isolated translator Python environment and installs `ops_translator/requirements.txt`; this requires package access. To use a prepared/offline environment, pass `--translator-python /path/to/venv/bin/python3`.

Choose an output path without whitespace: the upstream OPS Makefiles use this path during compilation.

`--build` performs the same build in the current shell environment via the profile, without submitting jobs. `--submit` instead submits a single build job in `small-g` by default, and all measurement jobs depend on its success. Customize `--build-partition`, `--build-time` and `--build-jobs` as needed.

## Submit the complete dataset

```bash
bash scripts/lumi_tgv_top3/run.sh --submit \
  --account project_XXXXXXXXX \
  --nodes 2,3,4,6,8,10,12,16,20,24,32,40 \
  --env-file /path/to/lumi-env.sh \
  --date-stamp campaign01 \
  --time 01:00:00
```

Only `--submit` sends jobs. The resource analysis and catalog validation happen before submission. The selected partition must pass its exact Slurm eligibility probe. In this example there are **one build job and twelve measurement jobs**. Choose a subset with `--nodes`, for example `--nodes 2,3`.

Each measurement allocation uses eight ranks and eight GPU GCDs per node, seven reserved CPU cores per rank, one OpenMP thread per process, and the LUMI CPU/GPU affinity map. Each rank selects its local GCD. GPU-aware Cray MPI and OPS `-gpudirect` are enabled; linkage and the runtime confirmation are checked.

The cases run sequentially inside the allocation:

- A: recorded MPI reference, default blocks `(32,4,1)`.
- C: recorded MPI reference, its precomputed blocks.
- D_top1/2/3: selected MPI shapes, their recorded precomputed blocks.

The reference MPI shape is forced from the dataset; it is not an observation of an automatic MPI choice. Identical precomputed cases are executed once and retain all their roles. The full plan has 57 unique measured executions. Each measured case runs **once for 500 simulation steps**. Only rank 0 writes OPS diagnostic CSVs; every rank reads the case's shared block file and emits a small status/log record.

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

The source snapshot requires the tracked TGV Makefile and both application headers before submission. The build generates translated sources before invoking the parallel MPI/HIP executable target, so a clean checkout needs no previously generated files.

Before compiling, the launcher locates the translator environment's bundled `clang/native/libclang.so`, exports its full filename as `LIBCLANG_PATH`, and verifies native C++ parsing. An explicit `LIBCLANG_PATH` from the environment profile takes precedence and must point to a compatible library file. The selected path is recorded in `build_manifest.json`.

After a source/build fix, update the checkout and submit a **new campaign**. Existing campaigns retain their frozen source snapshot and launcher; `--resume` does not apply code updates. Cancel any jobs left waiting on a failed build before replacing that campaign.

```bash
bash -n scripts/lumi_tgv_top3/run.sh scripts/lumi_tgv_top3/environment.sh
cd scripts/lumi_tgv_top3
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_launcher
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_dataset
```

`test_dataset` requires the extracted version-2 dataset (or `OPS_TOP3_DATASET=/path/to/TOP3_DATASET`). It audits every catalog against the original sampled-rank best logs, then exercises the real rank worker and node workflow across every selected node count and rank, all 45 smoke runs and 57 measured cases, collection and resume. Slurm/application execution and GPU linkage are simulated; no real jobs are submitted. HIP compilation, GPU Direct and actual MPI behavior require validation on LUMI.

The package includes `prepare_dataset.py --validate-only` for the full checksum/catalog/provenance audit using Python 3.6+. The same script can rebuild into a new directory with `--source-package /path/to/verified/package --output-root /path/to/new/TOP3_DATASET`.
