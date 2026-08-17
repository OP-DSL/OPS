# Tiling: build flags, WAW / RAW dependency analysis, debugging, and validation

Cache-blocking tiling fuses a queue of `ops_par_loop`s and executes them **tile-major**: for each tile, run the whole fused chain, then the next tile. Plan construction lives in `ops/c/src/core/ops_lazy.cpp` (`ops_construct_tile_plan`). The C and Fortran MPI libraries both compile that file; after editing it, rebuild **both** if you will run CloverLeaf and SENGA2.

This page is about diagnosing “tiling is slow” or “tiled SENGA2 does not match untiled”, not about how to turn tiling on (see [Performance Tuning](perf.md) and [Developing an application](devanapp.md#optimizations---tiling)).

## Check the build flags first

Tiling turns a DRAM-bandwidth-bound code into a cache-bound one. That makes tiled runs sensitive to code-generation quality in a way untiled runs are not, so a flag that looks harmless on an untiled benchmark can erase most of the tiling speedup.

`-ffloat-store` is the one that has actually bitten us. It forces every floating-point intermediate to memory. It exists to suppress x87 excess precision, which x86-64 SSE2 does not have, so on x86-64 it buys no accuracy. It still vectorizes and still contracts to FMA, but every intermediate makes a round trip through the stack: on a four-array FP loop `g++ -O3 -march=native` emits 25 arithmetic/move instructions without it and 59 with it. Untiled CloverLeaf hides those extra stores behind DRAM stalls and loses only ~4%; a tiled run keeps its working set in L2/L3, becomes core-bound, and loses **~1.7x**.

CloverLeaf 7680², 50 steps, 32 MPI ranks (one per physical core), `OPS_TILING_MAXDEPTH=6 OPS_CACHE_SIZE=1`, gnu 12.3, dual Xeon Gold 6226R (22.5 MB L3 per 16-core socket):

| `ops_lazy.cpp` | `-ffloat-store` | untiled | tiled | speedup |
|---|---|---|---|---|
| `cb1e8277c` (before the tiling work) | on | 65.5 s | 50.4 s | 1.30x |
| `cb1e8277c` | off | 62.9 s | 29.3 s | 2.14x |
| `777d9bac6` (WAW absorb cascade) | on | 65.5 s | 69.3 s | 0.95x |
| `777d9bac6` | off | 62.9 s | 62.1 s | 1.01x |
| current | on | 65.4 s | 50.4 s | 1.30x |
| current | off | 62.9 s | 29.6 s | 2.11x |

Two things to read off that table. The dependency analysis and the build flag are independent, and each is worth about 2x on its own — so a tiled measurement is only meaningful once you know which build you have. And a plausible-looking 1.30x can be either a broken analysis or a slow build; the flag alone accounts for the whole 50.4 s → 29.6 s step.

`makefiles/Makefile.gnu` no longer passes `-ffloat-store`. Use `FLOAT_STORE=1` to put it back, and `IEEE=1` for strict FP (`-fno-fast-math -ffp-contract=off -fno-associative-math`), which is what the other backends have always done. Every app Makefile includes `Makefile.cuda`, `Makefile.hip` and `Makefile.hdf5` unconditionally, so a backend makefile that *assigns* `CXXFLAGS` instead of appending to it silently replaces the compiler's flags for every target. That is how `feature/memory_pool` ended up building without `-ffloat-store` while `develop` built with it: same source, same command line, 50.4 s vs 29.3 s. Check the real flags before comparing branches:

```bash
cd apps/c/CloverLeaf && make -p 2>/dev/null | grep -E "^CXXFLAGS"
```

## How a plan is built (short)

Analysis walks the fused loops **backwards**. For each dimension:

1. **Sweep 1 (RAW).** Execution range of a loop on a tile is the intersection of that loop’s iteration range with later **read** dependencies of the datasets it writes. A later stencil read of `A` forces an earlier write of `A` to extend by that stencil. Consecutive tiles abut: tile \(i+1\) begins where tile \(i\) ended.
2. **Sweep 2 (WAW + leftover).** If a **later write** of a dataset this loop accesses (read or write) extends further, this loop must cover that write plus this access’s stencil. Tiles with no dependency are filled to the natural tile (**leftover**).
3. **Sweep 3 (dead tiles).** A last live tile that shrank to empty while the loop still executes in the halo can be marked dead; its read deps are merged into the previous tile.
4. **Dependency update.** Per-tile `data_read_deps` / `data_write_deps` and MPI `data_read_deps_edge` are updated from the chosen execution ranges.

MPI halo **send** sizes come from `data_read_deps_edge`. **Recv** sizes come from the first tile’s `data_read_deps` begin (left) and the max last-tile `data_read_deps` end (right). Those two analyses must agree or you get `MPI_ERR_TRUNCATE` or a `Waitall` hang.

`OPS_TILING_MAXDEPTH=N` both enables tiling (the runtime looks for the substring `OPS_TILING`) and extends halo depth so several fused loops can share one exchange.

## The two failure modes

### 1. WAW absorb cascade (performance)

A later write on this tile can push `tile_end` past the next tile. The old neighbour-end **cap** then emptied only that neighbour and left further tiles live, which **splits** a producer / snapshot / mutate chain (SENGA2). The fix in `8ba84475a` instead **absorbed** every overlapped neighbour by taking that neighbour’s full current end.

On CloverLeaf a leftover tile’s end is often the **entire owned range**. Absorbing that, then the next leftover, collapsed expensive kernels (`PdV`, `advec_cell_kernel4_*`) onto **one full-domain tile**. Fusion still reported 100+ loops per plan, but cache blocking was gone: kernel counts dropped to about one invocation per timestep instead of hundreds of tiles. Measured cost is the entire tiling benefit — 62.1 s tiled against 62.9 s untiled.

**Current rule:** grow only to **this tile’s** later write of an accessed dataset, plus this access’s stencil (`write_deps_end - d_m_min`). If that extent fully covers a later tile, empty it (`begin = end = tile_end`) **without** taking the neighbour’s leftover end. If the previous tile already covers this tile’s natural chunk (`begin >= nat_end`), leftover must **not** refill it.

WAW applies to **reads as well as writes**. A snapshot that only reads `U` must still extend when a later mutate writes `U`. Restricting WAW to `acc != OPS_READ` corrupts SENGA2.

Do **not** mark WAW-emptied tiles as Sweep-3 dead tiles: that makes the previous tile look like the last live tile and expands it to the full owned range.

### 2. Split producer / snapshot / mutate (correctness)

Tile-major execution plus an in-place overwrite is the SENGA2 pattern, reproduced in `apps/c/tiling_fix/minisenga.cpp`:

```
write store7          (producer, extended range)
read  store7 with ±5  (consumer)
write store7          (mutate / overwrite)
read  store7 with ±5
```

If tile 0’s mutate writes into tile 1’s cells before tile 1’s consumer runs, tile 1 snapshots mutated values. Tile 0’s producer and consumer must cover the mutate range plus the consumer stencil, and tile 1 must start after that coverage. Wide stencils (SENGA ±5) need about one stencil of extra width past the natural tile, not a full-domain absorb.

CloverLeaf has no snapshot-before-mutate on the same field in the same way; neighbour-end capping was enough for it, and that is why the absorb change was a CloverLeaf performance regression without an obvious CloverLeaf correctness failure.

### RAW stacking (performance, required for abutting tiles)

Sweep 1 accumulates one stencil per fused loop that RAW-connects through a dat. That is **required** with abutting tiles: the next tile reruns the chain but starts at this tile’s end, so this tile’s last writers at the boundary need stacked intermediates computed **here**.

In practice the stacking is small. On the CloverLeaf hydro plans the largest live tile is within a few cells of nominal (`122x46` against a nominal `120x40`, `131x51` on the `update_halo` plans), and the overlap factor stays below 1. Do not attribute a missing 2x to RAW stacking without a skew line showing it — check the build flags first.

Capping Sweep 1 execution at `natural + max_stencil`, or clipping per-tile `read_deps` to stop that stack, has been tried and is **unsafe**:

| Attempt | Typical symptom |
|---|---|
| Clip `data_read_deps` after each loop | `MPI_ERR_TRUNCATE` (edge send still stacked, recv uses clipped first/last tile) |
| Cap Sweep 1 `tile_end` on all tiles | Coverage gaps, CloverLeaf `dt = 0`, hang after a few steps |
| Cap interior tiles only | First tile stacked, interior capped → `begin > end` gap; SENGA NaN / `IEEE_DIVIDE_BY_ZERO` |
| Treat any `read_deps_end > natural_end` as “true WAW” and empty later tiles | Fires on every RAW chain; SENGA fields fully diverge |

Do not clip halo `read_deps` / `read_deps_edge` to “fix” tile sizes. MPI depths must stay stacked because the **neighbouring rank** will not rerun this chain.

Terminal-read seeding (phantom consumer of every written dat on the owned range) also RAW-connects the whole plan. Disable with `OPS_TERMINAL_READ=0` only as a diagnostic: CloverLeaf interiors can stay valid via leftover, but halo tiles often get worse (`dead tile resurrected` / leftover refill).

## Generalized extra-growth rule

| Dependency | Extra growth | Unsafe |
|---|---|---|
| RAW stencil | Sweep 1: one stencil per RAW-connected loop on **this** tile (stacking is required for abutting tiles). A single wide stencil (SENGA ±5, CloverLeaf halo ±4) is included in that. | Clipping `read_deps` or capping Sweep 1 execution independently of halo deps |
| WAW producer / snapshot / mutate | Sweep 2: this loop’s range ∪ (later write of an accessed dat on **this** tile + this stencil). Empty later tiles only for that overlap. | Growing to the neighbour’s leftover full range; skipping WAW on `OPS_READ` |

## Debugging

### Rebuild (clock skew)

NFS / home clocks often make Make skip `ops_lazy.cpp`. **Force-delete** the objects, then rebuild the library you will actually link:

```bash
# C apps (CloverLeaf, minisenga)
rm -f ops/c/obj/$OPS_COMPILER/ops_lazy.o
cd ops/c && make mpi -j

# Fortran apps (SENGA2) compile a second copy
rm -f ops/fortran/obj/$OPS_COMPILER/ops_lazy.o
cd ops/fortran && make f_mpi -j
```

Confirm `ops_lazy.o` and the application binary timestamps moved. Relink `*_mpi_tiled` after the library update.

### Diagnostics

| Flag | What to look at |
|---|---|
| `-OPS_DIAGS=2` | `Tiling enabled`, kernel times and counts, `Total Wall time`, `Total tiled halo` |
| `-OPS_DIAGS=3` (or `>2`) | `Created tiling plan for N loops`, tile size, **tile skew** (proc 0) |
| `-OPS_DIAGS=4` | `Executing tiling plan for N loops` — the plan sequence, one line per flush |
| `-OPS_DIAGS=5` | Per-tile exec ranges after read/write deps, empty tiles, dataset deps |
| (always) | `dead tile … resurrected` — leftover/WAW filled a Sweep-3 dead tile |

Tile skew line (proc 0):

```text
Proc 0 tile skew: nominal 120x40x-1, max live 122x46x0 (loops update_halo_kernel1 / update_halo_kernel2_xvel_minus_4_a / -), live tiles 392/392, overlap factor 0.886
```

- **nominal** — `OPS_CACHE_SIZE` / `OPS_TILESIZE_*` guess, clamped to the owned range. It is computed per plan, so plans over different loop sets legitimately print different nominals; compare max live against the nominal on the *same* line.
- **max live** — largest live tile extent in each dim and which loop hit it.
- **live tiles / total** — if expensive kernels show `1/N` live tiles, WAW absorb (or dead-tile expansion) has collapsed the rank.
- **overlap factor** — sum of live cell counts / sum of nominal tile volumes. ~1 is healthy; ≫1 means skew/stacking; ≪1 on a 3D app with `TILESIZE=1000`, or when the nominal is larger than the owned block, is normal.

Kernel **invocation counts** (`-OPS_DIAGS=2` profiler): `advec_cell_kernel4_xdir` should be about `(tiles per rank) × (steps)`, not `~steps`. One count per step means one tile per rank for that kernel.

The plan sequence is worth a look before chasing tile shapes, since a plan that got split gives up fusion regardless of how good the tiles are. One rank is enough:

```bash
OMP_NUM_THREADS=1 mpirun -np 1 ./cloverleaf_mpi_tiled -OPS_DIAGS=4 \
  OPS_TILING_MAXDEPTH=6 OPS_CACHE_SIZE=1 2>&1 \
  | grep -oE "Executing tiling plan for [0-9]+"
```

CloverLeaf settles to one 156-loop plan per timestep; anything that forces an `ops_execute` flush (a reduction such as `calc_dt` or `field_summary`, or a lowdim/edge-dat access) ends a plan there.

Per-tile ranges (`-OPS_DIAGS=5`), grep one kernel:

```bash
grep -E "Proc 0, PdV_kernel_predict tile .* exec range" run.out
```

A healthy interior tile is close to the nominal chunk plus a small stencil. A range equal to the owned `biggest_range` on a non-edge tile is the cascade.

### MPI hangs and TRUNCATE

- `MPI_ERR_TRUNCATE` — send packed from `data_read_deps_edge`, recv from first/last-tile `data_read_deps`; they disagree. Do not clip one without the other.
- Hang in `MPI_Waitall` inside `ops_halo_exchanges_datlist` — same mismatch, or a coverage gap so some ranks never post the expected send. Check whether first/last tiles were capped while edge deps were not.
- `dt = 0` / `timestep : 0.000000` on CloverLeaf after a few steps — missing cells at tile boundaries (Sweep 1 cap / emptied tiles without coverage). Kill the job; it will not recover.
- Fortran `IEEE_DIVIDE_BY_ZERO` / NaN on SENGA2 — same class of gap, often on a rank in the middle of a fine-X split.

### What not to try (without a new detector)

- Absorbing `tile_end = MAX(tile_end, neighbour.te)` on any overlap.
- Capping WAW at the neighbour’s Sweep 1 end when `waw_end` is larger (SENGA split).
- `true_waw = live_read && rd_e > nat_e` (true on every RAW stack).
- Clipping `read_deps` to `natural + this_stencil` (halo send/recv diverge).
- Marking WAW-emptied tiles `dead_tiles` (previous tile expands to full owned).

## Validation

Use the **same** MPI rank count, thread count, process binding **and compiler flags** for tiled and reference runs. Tiling changes execution order; bitwise identity is not required, but large field diffs are.

### CloverLeaf (performance + “still tiled”)

Use `clover_bm7680_short.in` (7680², 50 steps, `test_problem 6`), which checks kinetic energy to 0.001% and prints `PASSED` / `FAILED`. Copy it over `clover.in`, which is what the app reads.

```bash
module load OP2-env gnu12   # or your site modules
export OPS_INSTALL_PATH=.../OPS/ops
export OPS_COMPILER=gnu

cd ops/c && make mpi -j
cd ../../apps/c/CloverLeaf
make cloverleaf_mpi_tiled -j

# Untiled
OMP_NUM_THREADS=1 mpirun -np 32 -bind-to core ./cloverleaf_mpi_tiled -OPS_DIAGS=2

# Tiled
OMP_NUM_THREADS=1 mpirun -np 32 -bind-to core ./cloverleaf_mpi_tiled \
  -OPS_DIAGS=2 OPS_TILING_MAXDEPTH=6 OPS_CACHE_SIZE=1
```

Expect ~63 s untiled and ~30 s tiled on 32 Cascade Lake cores, both `PASSED`. If tiled lands near 50 s, check `CXXFLAGS` before touching `ops_lazy.cpp`.

Tiling is insensitive to the exact tile shape over a wide range — all of these `PASSED` at 7680²/50 steps, with the largest live tile at or below nominal:

| tiling | live tiles | overlap factor | wall |
|---|---|---|---|
| `OPS_CACHE_SIZE=1` | 392/392 | 0.89 | 29.7 s |
| `OPS_CACHE_SIZE=0.5` | 216/216 | 0.97 | 30.9 s |
| `OPS_CACHE_SIZE=2` | 54/54 | 0.97 | 37.0 s |
| `OPS_TILESIZE_X=120 OPS_TILESIZE_Y=17` | 904/904 | 1.00 | 28.7 s |
| `OPS_TILESIZE_X=2000 OPS_TILESIZE_Y=64` | 30/30 | 0.48 | 54.0 s |
| `OPS_TILESIZE_X=61 OPS_TILESIZE_Y=1000` | 32/32 | 0.94 | 88.2 s |

The two slow rows are cache-footprint effects, not analysis failures: a 2000-wide tile exceeds the owned X extent, and a 61x1000 column has poor spatial locality per row. Use them as a sanity check that a bad tile shape degrades smoothly rather than producing a wrong answer.

For a quick shape check use `end_step=2` and `-OPS_DIAGS=4`: look for `Created tiling plan for 137 loops` (hydro), several hundred `live tiles`, max live not equal to the owned domain, and no `TRUNCATE`. Then run 50 steps and confirm timesteps stay O(10⁻⁴), not zero.

### SENGA2 (correctness vs untiled)

`apps/fortran/SENGA2/senga_debug.sh` builds `senga2_mpi_tiled` with `DEBUG=1` and compares HDF5 dumps. Rebuild `ops/fortran` first. Use the HDF5 `h5diff` that matches the library you linked (not a random distro copy).

Configs:

| Name | Runtime | What it tests |
|---|---|---|
| Large tiles | `OPS_TILING_MAXDEPTH=6 OPS_TILESIZE_X=Y=Z=1000` | One tile per rank; must match untiled |
| Default | `OPS_TILING_MAXDEPTH=6` | Several tiles in Y/Z, usually one in X |
| Fine X | `OPS_TILING_MAXDEPTH=6 OPS_TILESIZE_X=25 OPS_TILESIZE_Y=1000 OPS_TILESIZE_Z=1000` | Three tiles in X; WAW split shows up here |

```bash
cd ops/fortran && make f_mpi -j
cd ../../apps/fortran/SENGA2
bash senga_debug.sh
# or, if you already have an untiled dump in output_ref/:
export PATH=".../hdf5/bin:$PATH"
cd apps/fortran/SENGA2
rm -rf output; mkdir -p output
OMP_NUM_THREADS=1 mpirun -np 32 -bind-to core ./senga2_mpi_tiled \
  -OPS_DIAGS=3 OPS_TILING_MAXDEPTH=6 OPS_TILESIZE_X=25 \
  OPS_TILESIZE_Y=1000 OPS_TILESIZE_Z=1000
h5diff -p 1e-10 output/timestep00000001.h5 output_ref/timestep00000001.h5
```

Two harness traps, both of which produce “millions of diffs on every field” and neither of which is a tiling bug:

- **Stale ranks.** A 3.3 GB dump per configuration takes minutes to write. If you kill a run and immediately start the next, 32 surviving ranks keep writing into `output/` and you compare a mixture. Wait for `pgrep -f senga2_mpi_tiled` to come back empty between runs.
- **Different step counts.** `input/cont.dat` is untracked and holds the step count and dump interval, so it is easy to change it between runs. `adaptt` prints one `MPI_MAX error` line per timestep on rank 0; `grep -c "MPI_MAX error"` must match across the runs you compare.

Healthy fine-X: `live tiles 3/3`, max live in X about `25` plus a wide stencil (often ~35), not the full owned ~64 unless a specific kernel’s WAW requires it.

`h5diff -p 1e-10` uses **relative** tolerance. Fine-X can report **48 diffs in `WRUN` only** with absolute error ~10⁻¹⁷ on a field of magnitude ~10⁻⁸ (relative ~10⁻⁹). That is fused-loop rounding, not a split-chain bug. Failure looks like **millions of diffs on many fields** (e.g. `WRUN` count equal to the grid size).

`h5diff` counts differences but not their size, which is the difference between rounding and divergence. To get magnitudes, compare a field directly:

```python
import h5py, numpy as np
ref = h5py.File("output_ref/timestep00000001.h5")["SENGA_GRID/WRUN"][()]
t = h5py.File("output_tiled25/timestep00000001.h5")["SENGA_GRID/WRUN"][()]
ad = np.abs(ref - t)
print("n", np.count_nonzero(ad > 1e-10), "abs max", ad.max(),
      "rel max", np.nanmax(ad / np.maximum(np.abs(ref), 1e-30)))
```

### Mini WAW app

`apps/c/tiling_fix` (`minisenga`) is a small 3D stand-in for the store7 write / ±5 read / overwrite chain. Build `minisenga_mpi_tiled` and run with several tiles in the stenciled direction, e.g. `OPS_TILING OPS_TILESIZE_Y=33` (see the comment at the top of `minisenga.cpp`). It has no built-in checksum, so it is a plan-shape and crash canary rather than a pass/fail test; read the tile ranges out of `-OPS_DIAGS=5`.

## Code map

| Location | Role |
|---|---|
| `makefiles/Makefile.gnu` | `CXXFLAGS`; `-ffloat-store` costs ~1.7x on tiled runs |
| `ops/c/src/core/ops_lazy.cpp` — `ops_construct_tile_plan` | Sweeps 1–3, leftover, terminal reads, tile skew print |
| `ops/c/src/core/ops_lazy.cpp` — `ops_compute_mpi_dependencies` | `data_read_deps_edge` |
| `ops/c/src/mpi/ops_mpi_rt_support.cpp` — `ops_halo_exchanges_datlist` | Pack/unpack using plan depths |
| `apps/c/CloverLeaf` | Performance canary (many fused loops, small stencils) |
| `apps/fortran/SENGA2` | Correctness canary (wide stencil, snapshot/mutate) |
| `apps/c/tiling_fix` | Minimal WAW chain |
