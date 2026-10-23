# TGV StoreNone Top 3 with OPS tiling

Run from any OPS checkout; no username or project is hardcoded:

```bash
bash scripts/ops_tiled/run_tgv_top3.sh --dry-run --account project_465003260 --nodes 2,3
bash scripts/ops_tiled/run_tgv_top3.sh --submit --account project_465003260 --nodes 2,3 --time 01:00:00
```

The wrapper selects the shared launcher's tiled mode. It builds
`opensbli_mpi_hip_tiled` with `OPS_LAZY` and passes these options to every measured
run and compatibility smoke test:

```text
OPS_TILING
OPS_TILING_MAXDEPTH=6
OPS_TILESIZE_X=8000 OPS_TILESIZE_Y=8000 OPS_TILESIZE_Z=8000
-gpudirect -OPS_DIAGS=2
```

Cray MPI GPU support is enabled with `MPICH_GPU_SUPPORT_ENABLED=1` and the
`libmpi_gtl_hsa` transport library. Node counts default to
2,3,4,6,8,10,12,16,20,24,32,40; `--nodes` selects a subset. The OPS checkout defaults
to the repository containing the shared launcher; `--ops-root /path/to/OPS`
overrides the source checkout and bundled dataset location.

Each node count runs A (default MPI/default blocks), C (default MPI/precomputed),
and D_top1–3 (selected MPI/precomputed), deduplicating identical cases. All use
tiling, 500 simulation steps and one measured repetition. Only rank 0 writes
diagnostic CSVs. There is no autotuning. The same frozen TOP3_DATASET is used.

Outputs go to `/scratch/<account>/ops-decomposition/campaigns/TGV_TOP3_TILED_<stamp>`;
`--output-root` overrides the parent directory. Configs record tiling settings,
default/precomputed wall times and speedups. The original non-tiled launcher keeps
its previous defaults and campaign naming. See [shared launcher](../lumi_tgv_top3/README.md)
for resource analysis, build options, collection and resume.

This is a new experiment transferring blocks selected without tiling. The
one-step precomputed smoke tests verify kernel IDs, stencil/argument metadata,
loaded blocks and thread limits. Tiling can change halo allocation and execution
ranges: these geometry differences are recorded in the compatibility report,
instead of requiring equality with the untiled run. They do not prove numerical
equivalence or that historical Top 3 rankings remain optimal. GPU execution and
full ROCm compilation must be checked on LUMI.

During the one-step smoke only, rank 0 sets `OPS_TILING_TRACE=1` to record the
actual local ranges chosen by the tiling planner. A historical kernel may be
absent from rank 0's CSV only when every traced tile for that kernel is empty.
Unexplained missing kernels, new IDs, and changed stencil metadata still fail.
The report lists `rank0_empty_kernel_ids`; those kernels have no rank-zero
metadata or runtime block-limit check. Their catalog entries remain intact for
other ranks. Timed runs disable the trace. Changing this validation requires a
new campaign/build because campaigns freeze both launcher and OPS sources.
