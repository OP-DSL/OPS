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
| `-OPS_DIAGS=3` (or `>2`) | `Created tiling plan for N loops`, tile size, **tile skew**, **unblocked loops**, **biggest tiles vs nominal**, and **WAW cause** (proc 0) |
| `OPS_MDIM_SKIP_WAW=1` | Skip Sweep-2 WAW on dats with `dim>1`. Hypothesis test only; does not change SENGA2's 1179-loop plan. |
| `OPS_SWEEP3_NO_CASCADE=1` | Only the geometric last tile may become Sweep-3 dead. **Unsafe** on SENGA2 (halo-depth crash). |
| `OPS_LASTLIVE_EXPAND=1` | Restore the old Sweep-1 / leftover full-range shortcut when the next tile is Sweep-3 dead. Default **off**: only the geometric last tile may take that shortcut; Sweep 3 still merges halo `read_deps` into the previous tile. |
| `SENGA_TILING_SPLIT=N` | SENGA2 only: insert `ops_execute` after named call sites (see the split experiment below). |
| `-OPS_DIAGS=4` | `Executing tiling plan for N loops` — the plan sequence, one line per flush |
| `-OPS_DIAGS=5` | Per-tile exec ranges after read/write deps, empty tiles, dataset deps |
| (always) | `dead tile … resurrected` — leftover/WAW filled a Sweep-3 dead tile |

Tile skew line (proc 0):

```text
Proc 0 tile skew: nominal 120x40x-1, max live 122x46x0 (loops update_halo_kernel1 / update_halo_kernel2_xvel_minus_4_a / -), live tiles 392/392, overlap factor 0.886
```

- **nominal** — `OPS_CACHE_SIZE` / `OPS_TILESIZE_*` guess, clamped to the owned range. It is computed per plan, so plans over different loop sets legitimately print different nominals; compare max live against the nominal on the *same* line.
- **max live** — the largest extent in each dimension, maximised over tiles and loops **independently per dimension**. It is not necessarily one tile: `max live 74x74x138` can mean one tile is 74 wide in x, a different tile is 74 in y, and a third is 138 in z. Do not read it as a tile footprint; that is what the next line is for.
- **live tiles / total** — if expensive kernels show `1/N` live tiles, WAW absorb (or dead-tile expansion) has collapsed the rank.
- **overlap factor** — sum of live cell counts / sum of nominal tile volumes, over live tiles only. ~1 is healthy. ≫1 does *not* by itself mean redundant execution: a loop that runs its whole range on a single tile contributes its full volume to the numerator but only one nominal tile to the denominator, so a handful of unblocked loops can push the factor to 5 with no repeated work at all. ≪1 on a 3D app with `TILESIZE=1000`, or when the nominal is larger than the owned block, is normal.

When any loop's largest single tile exceeds 1.5x the nominal tile, a second line names the worst three:

```text
Proc 0 biggest tiles vs nominal: temper_kernel_eqA 35.8x (74x74x138, 1 live tiles); set_zero_kernel_MD5 35.8x (74x74x138, 1 live tiles); set_zero_kernel 35.8x (74x74x138, 1 live tiles);
```

This is the line to read when tiling does not pay off. It is a real tile footprint, so it distinguishes the two ways a plan can lose cache blocking:

- **`N live tiles` with a large ratio** — the loop is skewed/stacked and genuinely redoes work.
- **`1 live tiles` with a large ratio**, as above — the loop is not blocked at all. It executes its whole range in one go, streaming its dats through the cache once per plan and evicting everything the neighbouring tiles just built. No redundant arithmetic, but the fused chain around it is broken.

Absence of the line means every loop is within 1.5x of nominal. CloverLeaf never prints it.

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
- `OPS_SWEEP3_NO_CASCADE=1` (halo depths explode on SENGA2; MPI packing asks for a 197-deep `DRHS` halo).
- Restoring `OPS_LASTLIVE_EXPAND=1` (Sweep 1 / leftover treat a Sweep-3-dead neighbour as geometric last and fill to the owned end; that is the leftover cascade).

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

Current status on 256³, 10 steps, 32 ranks, `-O3`: all three tiled configurations agree with untiled. Large tiles and default have **no** cell above a 10⁻¹⁰ relative difference on any field; fine-X has 48 cells, all in `WRUN`, with absolute error 1.3×10⁻¹⁷ on a field of magnitude 4.9×10⁻⁸, and every other field at machine epsilon.

`h5diff` counts differences but not their size, which is the difference between rounding and divergence. To get magnitudes, compare a field directly:

```python
import h5py, numpy as np
ref = h5py.File("output_ref/timestep00000001.h5")["SENGA_GRID/WRUN"][()]
t = h5py.File("output_tiled25/timestep00000001.h5")["SENGA_GRID/WRUN"][()]
ad = np.abs(ref - t)
print("n", np.count_nonzero(ad > 1e-10), "abs max", ad.max(),
      "rel max", np.nanmax(ad / np.maximum(np.abs(ref), 1e-30)))
```

### SENGA2 performance: an open case

SENGA2 is correct under tiling but barely gains from it. Same 256³ / 10-step / 32-rank `-O3` runs, timed by SENGA’s own `total_ttime`:

| configuration | time |
|---|---|
| untiled | 458.9 s |
| large tiles (one tile per rank) | 488.6 s |
| default | 452.4 s |
| fine X (`TILESIZE_X=25`) | 554.2 s |

One tile per rank is the cost of the tiled code path with none of its benefit: 6.5% over untiled. Default tiling recovers that and 1.4% more, so the cache blocking is worth only about 8% against the same code path — far less than the 2.1x CloverLeaf gets, on a rank whose 64x64x128 block with 30-plus multi-component fields is nowhere near cache-resident.

The footprint line points at why. In the two large plans (1179 and 1180 loops):

```text
Proc 0 tile skew: nominal 69x18x17, max live 74x74x138 (loops temper_kernel_eqA / ...), live tiles 32/32, overlap factor 5.362
Proc 0 unblocked loops: 1015/1179 (1 live tile, >1.5x nominal); WAW growth via dim>1 dats: 0 loop-dims
Proc 0 biggest tiles vs nominal: temper_kernel_eqA 35.8x (74x74x138, 1 live tiles); set_zero_kernel_MD5 35.8x (74x74x138, 1 live tiles); set_zero_kernel 35.8x (74x74x138, 1 live tiles);
Proc 0 WAW cause for biggest tiles: temper_kernel_eqA (no Sweep-2 WAW recorded); ...
```

It is not three kernels. **1015 of the 1179 loops** in that plan run as one full-domain tile. `temper_kernel_eqA` is only the first name on the list because it sits at the top of `temper.F90`. There is also no Sweep-2 WAW on those worst loops: tile 0 is expanded by leftover / Sweep 1 once Sweep 3 has marked later tiles dead, so tile 0 looks like the last live tile and is filled to the owned-plus-halo range (`74x74x138`).

SENGA does **not** have streaming stores. The eviction is the unblocked loops themselves writing the whole rank on tile 0.

### Structured split experiment

`ops_execute` breaks a fused plan. `SENGA_TILING_SPLIT` (env var, `apps/fortran/SENGA2/senga_tiling_split.F90`) inserts it without a rebuild:

| value | flush after |
|---|---|
| 0 | none (baseline) |
| 3 | `rhscal` |
| 6 | `temper` (inside `rhscal`) |
| 7 | around the `YRHS-MDIM` species-copy groups |
| 9 | after each species copy into `YRHS-MDIM` |
| 8 | all of 1–7 |

`OPS_MDIM_SKIP_WAW=1` skips Sweep-2 WAW on dats with `dim>1`. That tests the hypothesis that a multi-species dat cannot express “I only write component *k*”, so a later write of any component looks like a WAW on the whole dat. `copy_kernel_sdim_to_mdim` really does write one component of a 9-component dat (`OPS_ACC_MD1(ispec,0,0,0)`); the generated code declares `OPS_WRITE` on all 9, the source declares `OPS_RW`. The API cannot name the component. That is real, and it is **not** what unblocks `temper`.

1-step, 256³, 32 ranks, `OPS_TILING_MAXDEPTH=6` (the default tile size that prints `69x18x17`, not `OPS_CACHE_SIZE=1`):

| config | big-plan unblocked | eqA blocked? | notes |
|---|---|---|---|
| split 0 (fused) | 1015/1179 | no, 35.8x | overlap 5.36; 0 Sweep-2 WAW; 0 dim>1 WAW |
| split 6 (after `temper`) | 0/35 in the temper plan | **yes** (overlap 0.58) | remaining rhscal plan 980/1144 unblocked (`maths_kernel_eqT`) |
| split 3 (after `rhscal`) | 808/988 | no | unblocking writer is still inside `rhscal` |
| split 7 / 9 (mdim copies) | 1015/1179 | no | identical to baseline |
| `OPS_MDIM_SKIP_WAW=1` | 1015/1179 | no | identical to baseline |
| `OPS_SWEEP3_NO_CASCADE=1` | (crashes) | — | halo depths 69–197 on `DRHS` vs depth 6; tiles `74x266x266` |

So:

1. **False multi-species WAW is not the skew.** Skipping it changes nothing. Isolating the copy loops changes nothing. `eqA` does not even touch `YRHS-MDIM`; it reads `DRHS/URHS/VRHS/WRHS/ERHS` and writes `TCOEFF`.
2. **`eqA` is unblocked by later loops after `temper`.** Flushing after `temper` restores a 35-loop plan with every loop blocked. The first loop of the leftover plan is `maths_kernel_eqT` (`ERHS = ERHS/DRHS` in `rhscal.F90`), which **writes `ERHS`** that `eqA` only reads — a true snapshot/mutate WAW on a scalar dat. Fuse them and `eqA` must cover `eqT`'s range; if `eqT` itself is unblocked, `eqA` follows.
3. **Most of `rhscal` is the same shape.** Split 6 only saves the 35 temper loops. 980 later loops stay unblocked, and 1-step `total_ttime` stays ~59 s for every split. An `ops_execute` after `temper` is the right way to keep `eqA` blocked; it is not enough to make tiling pay.
4. **Do not turn off the Sweep-3 dead-tile cascade.** Empty last-live tiles have to be marked dead so halo work attaches to the previous tile. Stopping the cascade (`OPS_SWEEP3_NO_CASCADE=1`) leaves those deps on empty tiles and the MPI packing asks for a 197-deep halo. Same class of failure as clipping `read_deps`.

The remaining analysis bug is: leftover fills a tile to the full owned end when the *next* tile is Sweep-3 dead, and Sweep 3 will mark the previous empty tile dead too, until tile 0 is last live. Restricting that without re-breaking halo depths is still open. Any change needs the CloverLeaf 7680² `PASSED` canary and the SENGA2 tiled-vs-untiled dump comparison.

To rerun the 1-step matrix: `apps/fortran/SENGA2/senga_tiling_split.sh` (restores `input/cont.dat`). Rebuild `ops/fortran` after editing `ops_lazy.cpp`. `SENGA_TILING_SPLIT` and `OPS_MDIM_SKIP_WAW` default off.

### Mini WAW app

`apps/c/tiling_fix` (`minisenga`) is a small 3D stand-in for the store7 write / ±5 read / overwrite chain. Build `minisenga_mpi_tiled` and run with several tiles in the stenciled direction, e.g. `OPS_TILING OPS_TILESIZE_Y=33` (see the comment at the top of `minisenga.cpp`). It has no built-in checksum, so it is a plan-shape and crash canary rather than a pass/fail test; read the tile ranges out of `-OPS_DIAGS=5`.

## Code map

| Location | Role |
|---|---|
| `makefiles/Makefile.gnu` | `CXXFLAGS`; `-ffloat-store` costs ~1.7x on tiled runs |
| `ops/c/src/core/ops_lazy.cpp` — `ops_construct_tile_plan` | Sweeps 1–3, leftover, terminal reads, tile skew / WAW-cause print, `OPS_MDIM_SKIP_WAW`, `OPS_SWEEP3_NO_CASCADE`, `OPS_LASTLIVE_EXPAND` |
| `ops/c/src/core/ops_lazy.cpp` — `ops_compute_mpi_dependencies` | `data_read_deps_edge` |
| `ops/c/src/mpi/ops_mpi_rt_support.cpp` — `ops_halo_exchanges_datlist` | Pack/unpack using plan depths |
| `apps/c/CloverLeaf` | Performance canary (many fused loops, small stencils) |
| `apps/fortran/SENGA2` | Correctness canary (wide stencil, snapshot/mutate) |
| `apps/c/tiling_fix` | Minimal WAW chain |
