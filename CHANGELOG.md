# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" (https://keepachangelog.com/).


## [Unreleased]

No changes yet.


---

## [v2.0.0] - 2026-02-28

Major changes since `v1.0.0` (high-level)

- New `ops_translator` (Primary / most important)
  - A new OPS code-generation engine (`ops_translator`) replaces the legacy translator.
  - Based on Python/Clang/Fparser/Jinja2 — produces improved C/C++/Fortran backends.
  - Enables broader target support and improved codegen quality (SYCL, HIP, CUDA, OpenMP offload, etc.).
  - All apps (including CloverLeaf, CloverLeaf_3D, wave_test, halfprecision) now use the new translator.
  - Migration notes: translator runtime requires updated Python dependencies (see `ops_translator/requirements.txt`), and generated output paths/filenames may differ from the legacy translator — review build scripts when upgrading.

- New back-end & language support:
  - SYCL targets and improved SYCL integration (CPU/GPU).
  - HIP support and HIP/CUDA codegen fixes.
  - Added/expanded OpenMP offload MPI backend.
  - Fortran offload support and improved Fortran templates.
  - OpenCL and OpenACC: no longer supported in this release. Legacy codegen
    and example targets referencing OpenCL/OpenACC remain in the tree for
    historical/reference purposes but are unmaintained. Users should migrate
    to SYCL, HIP, or OpenMP offload paths where possible.

- Precision & computation features:
  - Half-precision (float16) and mixed-precision support on GPUs and in code generation.
  - Multiblock mixed-precision halo-exchange fixes and enhancements.

- I/O, tooling & utilities:
  - HDF5 improvements including SoA output mode and robustness fixes for MPI HDF5 writes.
  - Energy and power measurement hooks (RAPL/powercap and GPU power capture).

- Bug fixes:
  - Fixed SYCL device selection (`OPS_SYCL_DEVICE=gpu`) crashing on empty device list.
  - Fixed Fortran I/O deadlock when using `print`/`WRITE` inside OpenMP parallel regions with Intel Fortran runtime.

- Build, CI and testing:
  - Numerous CMake and Makefile improvements (F90/CMake additions, device selection, build flags).
  - Added `EXTRA_CLEAN_FILES` support in `Makefile.c_app` for app-specific clean targets.
  - Added `clean_all_apps.sh` utility script.
  - CI/test additions and config changes (expanded tests, GPU_NUMBER config update).
  - Added `test.sh` for multiple C and Fortran apps.

- Documentation and developer experience:
  - Major documentation refreshes across the docs, developer guide updates and a new `doc/quickstart.md`.
  - Added example/test scripts for apps and updated tutorial timings.
  - Updated AUTHORS.

Notes:
- This section lists high-level, user-facing and potentially breaking changes since `v1.0.0`.
- For a detailed per-commit log see the git history; specific fixes and smaller changes remain in commit messages since the release.

Guidance:
- Add short, one-line entries under the relevant category for each PR.
- Keep `Unreleased` updated during development; on release, copy `Unreleased`
  to a new versioned section with the release date, then clear `Unreleased`.
- Link PRs or commits where helpful (e.g. "Fix memory leak (PR #123)").
