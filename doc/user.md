---
author:
- Mike Giles, Istvan Reguly, Gihan Mudalige
date: May 2019
title: OPS C++ User's Manual
---





# OPS C++ API

## Initialisation declaration and termination routines

###  {#section .unnumbered}

::: list
plus 1pt minus 1pt

the usual command line arguments

an integer which defines the level of debugging diagnostics and
reporting to be performed
:::

Currently, higher `diags_level`s does the following checks\
`diags_level` $=$ 1 : no diagnostics, default to achieve best runtime
performance.\
`diags_level` $>$ 1 : print block decomposition and `ops_par_loop`
timing breakdown.\
`diags_level` $>$ 4 : print intra-block halo buffer allocation feedback
(for OPS internal development only)\
`diags_level` $>$ 5 : check if intra-block halo MPI sends depth match
MPI receives depth (for OPS internal development only)\

###  {#section-1 .unnumbered}

::: list
plus 1pt minus 1pt

dimension of the block

a name used for output diagnostics
:::

###  {#section-2 .unnumbered}

::: list
plus 1pt minus 1pt

dimension of the block

a name used for output diagnostics

hdf5 file to read and obtain the block information from
:::

Although this routine does not read in any extra information about the
block from the named HDF5 file than what is already specified in the
arguments, it is included here for error checking (e.g. check if blocks
defined in an HDF5 file is matching with the declared arguments in an
application) and completeness.\

###  {#section-3 .unnumbered}

::: list
plus 1pt minus 1pt

structured block

dimension of dataset (number of items per grid element)

size in each dimension of the block

base indices in each dimension of the block

padding from the face in the negative direction for each dimension (used
for block halo)

padding from the face in the positive direction for each dimension (used
for block halo)

input data of type `T`

the name of type used for output diagnostics (e.g. "double", "float")

a name used for output diagnostics
:::

The `size` allows to declare different sized data arrays on a given
`block`. `d_m` and `d_p` are depth of the "block halos" that are used to
indicate the offset from the edge of a block (in both the negative and
positive directions of each dimension).\
\

###  {#section-4 .unnumbered}

::: list
plus 1pt minus 1pt

structured block

dimension of dataset (number of items per grid element)

the name of type used for output diagnostics (e.g. "double", "float")

name of the dat used for output diagnostics

hdf5 file to read and obtain the data from
:::

###  {#section-5 .unnumbered}

::: list
plus 1pt minus 1pt

a name used to identify the constant

dimension of dataset (number of items per element)

the name of type used for output diagnostics (e.g. "double", "float")

pointer to input data of type `T`
:::

###  {#section-6 .unnumbered}

::: list
plus 1pt minus 1pt

a name used to identify the constant

dimension of dataset (number of items per element)

the name of type used for output diagnostics (e.g. "double", "float")

pointer to new values for constant of type `T`
:::

###  {#section-7 .unnumbered}

::: list
plus 1pt minus 1pt

origin dataset

destination dataset

defines an iteration size (number of indices to iterate over in each
direction)

indices of starting point in \"from\" dataset

indices of starting point in \"to\" dataset

direction of incrementing for \"from\" for each dimension of `iter_size`

direction of incrementing for \"to\" for each dimension of `iter_size`
:::

A from_dir \[1,2\] and a to_dir \[2,1\] means that x in the first block
goes to y in the second block, and y in first block goes to x in second
block. A negative sign indicates that the axis is flipped. (Simple
example: a transfer from (1:2,0:99,0:99) to (-1:0,0:99,0:99) would use
iter_size = \[2,100,100\], from_base = \[1,0,0\], to_base = \[-1,0,0\],
from_dir = \[0,1,2\], to_dir = \[0,1,2\]. In more complex case this
allows for transfers between blocks with different orientations.)\

###  {#section-8 .unnumbered}

::: list
plus 1pt minus 1pt

origin dataset

destination dataset

hdf5 file to read and obtain the data from
:::

###  {#section-9 .unnumbered}

::: list
plus 1pt minus 1pt

number of halos in `halos`

array of halos
:::

###  {#section-10 .unnumbered}

::: list
plus 1pt minus 1pt

size of data in bytes

the name of type used for output diagnostics (e.g. "double", "float")

name of the dat used for output diagnostics
:::

::: list
plus 1pt minus 1pt

the `ops_reduction` handle

a pointer to write the results to, memory size has to match the declared
:::

###  {#section-11 .unnumbered}

::: list
plus 1pt minus 1pt

string describing the partitioning method. Currently this string is not
used internally, but is simply a place-holder to indicate different
partitioning methods in the future.
:::

###  {#section-12 .unnumbered}

::: list
plus 1pt minus 1pt
:::

## Diagnostics and output routines

###  {#section-13 .unnumbered}

::: list
plus 1pt minus 1pt
:::

###  {#section-14 .unnumbered}

::: list
plus 1pt minus 1pt
:::

###  {#section-15 .unnumbered}

::: list
plus 1pt minus 1pt

variable to hold the CPU time at the time of invocation

variable to hold the elapsed time at the time of invocation
:::

###  {#section-16 .unnumbered}

::: list
plus 1pt minus 1pt

ops_block to be written

hdf5 file to write to
:::

###  {#section-17 .unnumbered}

::: list
plus 1pt minus 1pt

ops_stencil to be written

hdf5 file to write to
:::

###  {#section-18 .unnumbered}

::: list
plus 1pt minus 1pt

ops_dat to be written

hdf5 file to write to
:::

###  {#section-19 .unnumbered}

::: list
plus 1pt minus 1pt

ops_dat to to be written

text file to write to
:::

###  {#section-20 .unnumbered}

::: list
plus 1pt minus 1pt

output stream, use stdout to print to standard out
:::

###  {#section-21 .unnumbered}

::: list
plus 1pt minus 1pt

ops_dat to to be checked
:::

## Halo exchange

###  {#section-22 .unnumbered}

::: list
plus 1pt minus 1pt

the halo group
:::

## Parallel loop syntax

A parallel loop with N arguments has the following syntax:

###  {#section-23 .unnumbered}

::: list
plus 1pt minus 1pt

user's kernel function with N arguments

name of kernel function, used for output diagnostics

the ops_block over which this loop executes

dimension of loop iteration

iteration range array

arguments
:::

The **ops_arg** arguments in **ops_par_loop** are provided by one of the
following routines, one for global constants and reductions, and the
other for OPS datasets.

###  {#section-24 .unnumbered}

::: list
plus 1pt minus 1pt

data array

array dimension

string representing the type of data held in data

access type
:::

###  {#section-25 .unnumbered}

::: list
plus 1pt minus 1pt

an `ops_reduction` handle

array dimension (according to `type`)

string representing the type of data held in data

access type
:::

###  {#section-26 .unnumbered}

::: list
plus 1pt minus 1pt

dataset

stencil for accessing data

string representing the type of data held in dataset

access type
:::

###  {#section-27 .unnumbered}

::: list
plus 1pt minus 1pt
:::

## Stencils

The final ingredient is the stencil specification, for which we have two
versions: simple and strided.\

###  {#section-28 .unnumbered}

::: list
plus 1pt minus 1pt

dimension of loop iteration

number of points in the stencil

stencil for accessing data

string representing the name of the stencil
:::

###  {#section-29 .unnumbered}

::: list
plus 1pt minus 1pt

dimension of loop iteration

number of points in the stencil

stencil for accessing data

stride for accessing data

string representing the name of the stencil\
:::

###  {#section-30 .unnumbered}

::: list
plus 1pt minus 1pt

dimension of loop iteration

number of points in the stencil

string representing the name of the stencil

hdf5 file to write to
:::

In the strided case, the semantics for the index of data to be accessed,
for stencil point `p`, in dimension `m` are defined as:\
,\
where `loop_index[m]` is the iteration index (within the user-defined
iteration space) in the different dimensions.

If, for one or more dimensions, both `stride[m]` and `stencil[p*dims+m]`
are zero, then one of the following must be true;

-   the dataset being referenced has size 1 for these dimensions

-   these dimensions are to be omitted and so the dataset has dimension
    equal to the number of remaining dimensions.

See `OPS/apps/c/CloverLeaf/build_field.cpp` and
`OPS/apps/c/CloverLeaf/generate.cpp` for an example
`ops_decl_strided_stencil` declaration and its use in a loop,
respectively.\
These two stencil definitions probably take care of all of the cases in
the Introduction except for multiblock applications with interfaces with
different orientations -- this will need a third, even more general,
stencil specification. The strided stencil will handle both multigrid
(with a stride of 2 for example) and the boundary condition and reduced
dimension applications (with a stride of 0 for the relevant dimensions).

## Checkpointing

OPS supports the automatic checkpointing of applications. Using the API
below, the user specifies the file name for the checkpoint and an
average time interval between checkpoints, OPS will then automatically
save all necessary information periodically that is required to
fast-forward to the last checkpoint if a crash occurred. Currently, when
re-launching after a crash, the same number of MPI processes have to be
used. To enable checkpointing mode, the `OPS_CHECKPOINT` runtime
argument has to be used.\

###  {#section-31 .unnumbered}

::: list
plus 1pt minus 1pt

name of the file for checkpointing. In MPI, this will automatically be
post-fixed with the rank ID.

average time (seconds) between checkpoints

a combinations of flags, listed in `ops_checkpointing.h`:\
OPS_CHECKPOINT_INITPHASE - indicates that there are a number of parallel
loops at the very beginning of the simulations which should be excluded
from any checkpoint; mainly because they initialise datasets that do not
change during the main body of the execution. During restore mode these
loops are executed as usual. An example would be the computation of the
mesh geometry, which can be excluded from the checkpoint if it is
re-computed when recovering and restoring a checkpoint. The API call
void `ops_checkpointing_initphase_done()` indicates the end of this
initial phase.

OPS_CHECKPOINT_MANUAL_DATLIST - Indicates that the user manually
controls the location of the checkpoint, and explicitly specifies the
list of `ops_dat`s to be saved.

OPS_CHECKPOINT_FASTFW - Indicates that the user manually controls the
location of the checkpoint, and it also enables fast-forwarding, by
skipping the execution of the application (even though none of the
parallel loops would actually execute, there may be significant work
outside of those) up to the checkpoint.

OPS_CHECKPOINT_MANUAL - Indicates that when the corresponding API
function is called, the checkpoint should be created. Assumes the
presence of the above two options as well.
:::

###  {#section-32 .unnumbered}

::: list
plus 1pt minus 1pt

number of datasets to be saved

arrays of `ops_dat` handles to be saved
:::

###  {#section-33 .unnumbered}

::: list
plus 1pt minus 1pt

size of the payload in bytes

pointer to memory into which the payload is packed
:::

###  {#section-34 .unnumbered}

::: list
plus 1pt minus 1pt

number of datasets to be saved

arrays of `ops_dat` handles to be saved

size of the payload in bytes

pointer to memory into which the payload is packed
:::

###  {#section-35 .unnumbered}

::: list
plus 1pt minus 1pt

number of datasets to be saved

arrays of `ops_dat` handles to be saved

size of the payload in bytes

pointer to memory into which the payload is packed
:::

The suggested use of these **manual** functions is of course when the
optimal location for checkpointing is known - one of the ways to
determine that is to use the built-in algorithm. More details of this
will be reported in a tech-report on checkpointing, to be published
later.

## Access to OPS data

This section describes APIS that give the user access to internal data
structures in OPS and return data to user-space. These should be used
cautiously and sparsely, as they can affect performance significantly

###  {#section-36 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset
:::

###  {#section-37 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset
:::

###  {#section-38 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset

the chunk index (has to be 0)

an array populated with the displacement of the chunk within the
"global" distributed array

an array populated with the spatial extents
:::

###  {#section-39 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset

the chunk index (has to be 0)

an array populated with the displacement of the chunk within the
"global" distributed array

an array populated with the spatial extents

an array populated strides in spatial dimensions needed for column-major
indexing

an array populated with padding on the left in each dimension. Note that
these are negative values

an array populated with padding on the right in each dimension
:::

###  {#section-40 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset

the chunk index (has to be 0)

a stencil used to determine required MPI halo exchange depths

when set to OPS_HOST or OPS_DEVICE, returns a pointer to data in that
memory space, otherwise must be set to 0, and returns whether data is in
the host or on the device
:::

###  {#section-41 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset

the chunk index (has to be 0)

the kind of access that was used by the user (OPS_READ if it was read
only, OPS_WRITE if it was overwritten, OPS_RW if it was read and
written)
:::

###  {#section-42 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset

the chunk index (has to be 0)

the kind of access that was used by the user (OPS_READ if it was read
only, OPS_WRITE if it was overwritten, OPS_RW if it was read and
written)

set to OPS_HOST or OPS_DEVICE
:::

###  {#section-43 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset

the chunk index (has to be 0)

pointer to memory which should be filled by OPS
:::

###  {#section-44 .unnumbered}

::: list
plus 1pt minus 1pt

the dataset

the chunk index (has to be 0)

pointer to memory which should be copied to OPS
:::

# Tiling for Cache-blocking

OPS has a code generation (ops_gen_mpi_lazy) and build target for
tiling. Once compiled, to enable, use the `OPS_TILING` runtime parameter
- this will look at the L3 cache size of your CPU and guess the correct
tile size. If you want to alter the amount of cache to be used for the
guess, use the `OPS_CACHE_SIZE=XX` runtime parameter, where the value is
in Megabytes. To manually specify the tile sizes, use the
OPS_TILESIZE_X, OPS_TILESIZE_Y, and OPS_TILESIZE_Z runtime arguments.

When MPI is combined with OpenMP tiling can be extended to the MPI
halos. Set `OPS_TILING_MAXDEPTH` to increase the the halo depths so that
halos for multiple `ops_par_loops` can be exchanged with a single MPI
message (see [@TPDS2017] for more details)\
To test, compile CloverLeaf under `apps/c/CloverLeaf`, modify clover.in
to use a $6144^2$ mesh, then run as follows:\
For OpenMP with tiling:\
`export OMP_NUM_THREADS=xx; numactl -physnodebind=0 ./cloverleaf_tiled OPS_TILING`\
For MPI+OpenMP with tiling:\
`export OMP_NUM_THREADS=xx; mpirun -np xx ./cloverleaf_mpi_tiled OPS_TILING OPS_TILING_MAXDEPTH=6`\
To manually specify the tile sizes (in number of grid points), use the
OPS_TILESIZE_X, OPS_TILESIZE_Y, and OPS_TILESIZE_Z runtime arguments:\
`export OMP_NUM_THREADS=xx; numactl -physnodebind=0 ./cloverleaf_tiled OPS_TILING OPS_TILESIZE_X=600 OPS_TILESIZE_Y=200 `

# CUDA and OpenCL Runtime Arguments

The CUDA (and OpenCL) thread block sizes can be controlled by setting
the `OPS_BLOCK_SIZE_X, OPS_BLOCK_SIZE_Y` and `OPS_BLOCK_SIZE_Z` runtime
arguments. For example :\
`./cloverleaf_cuda OPS_BLOCK_SIZE_X=64 OPS_BLOCK_SIZE_Y=4`\
`OPS_CL_DEVICE=XX` runtime flag sets the OpenCL device to execute the
code on.\
Usually `OPS_CL_DEVICE=0` selects the CPU and `OPS_CL_DEVICE=1` selects
GPUs.

# Executing with GPUDirect

GPU direct support for MPI+CUDA, to enable (on the OPS side) add
**-gpudirect** when running the executable. You may also have to use
certain environmental flags when using different MPI distributions. For
an example of the required flags and environmental settings on the
Cambridge Wilkes2 GPU cluster see:\
<https://docs.hpc.cam.ac.uk/hpc/user-guide/performance-tips.html>

# OPS User Kernels

In OPS, the elemental operation carried out per mesh/grid point is
specified as an outlined function called a *user kernel*. An example
taken from the Cloverleaf application is given in Figure
[\[fig:example\]](#fig:example){reference-type="ref"
reference="fig:example"}.\

``` {.cpp mathescape="" linenos="" startFrom="1" numbersep="0pt" gobble="2" frame="lines" framesep="1mm"}
void accelerate_kernel( const ACC<double> &density0, const ACC<double> &volume,
                ACC<double> &stepbymass, const ACC<double> &xvel0, ACC<double> &xvel1,
                const ACC<double> &xarea, const ACC<double> &pressure,
                const ACC<double> &yvel0, ACC<double> &yvel1,
                const ACC<double> &yarea, const ACC<double> &viscosity) {

  double nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  nodal_mass = ( density0(-1,-1) * volume(-1,-1)
    + density0(0,-1) * volume(0,-1)
    + density0(0,0) * volume(0,0)
    + density0(-1,0) * volume(-1,0) ) * 0.25;

  stepbymass(0,0) = 0.5*dt/ nodal_mass;

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1(0,0) = xvel0(0,0) - stepbymass(0,0) *
            ( xarea(0,0)  * ( pressure(0,0) - pressure(-1,0) ) +
              xarea(0,-1) * ( pressure(0,-1) - pressure(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1(0,0) = yvel0(0,0) - stepbymass(0,0) *
            ( yarea(0,0)  * ( pressure(0,0) - pressure(0,-1) ) +
              yarea(-1,0) * ( pressure(-1,0) - pressure(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, 0,-1};

  xvel1(0,0) = xvel1(0,0) - stepbymass(0,0) *
            ( xarea(0,0) * ( viscosity(0,0) - viscosity(-1,0) ) +
              xarea(0,-1) * ( viscosity(0,-1) - viscosity(-1,-1) ) );

  //{0,0, -1,0, 0,-1, -1,-1};
  //{0,0, -1,0};

  yvel1(0,0) = yvel1(0,0) - stepbymass(0,0) *
            ( yarea(0,0) * ( viscosity(0,0) - viscosity(0,-1) ) +
              yarea(-1,0) * ( viscosity(-1,0) - viscosity(-1,-1) ) );


}
```

[\[fig:example\]]{#fig:example label="fig:example"}

\
\
\
\
This user kernel is then used in an `ops_par_loop` (Figure
[\[fig:parloop\]](#fig:parloop){reference-type="ref"
reference="fig:parloop"}). The key aspect to note in the user kernel in
Figure [\[fig:example\]](#fig:example){reference-type="ref"
reference="fig:example"} is the use of the ACC\<\> objects and their
parentheses operator. These specify the stencil in accessing the
elements of the respective data arrays.

``` {.cpp mathescape="" linenos="" startFrom="1" numbersep="0pt" gobble="2" frame="lines" framesep="2mm"}
    int rangexy_inner_plus1[] = {x_min,x_max+1,y_min,y_max+1};

    ops_par_loop(accelerate_kernel, "accelerate_kernel", clover_grid, 2, rangexy_inner_plus1,
     ops_arg_dat(density0, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
     ops_arg_dat(volume, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
     ops_arg_dat(work_array1, 1, S2D_00, "double", OPS_WRITE),
     ops_arg_dat(xvel0, 1, S2D_00, "double", OPS_READ),
     ops_arg_dat(xvel1, 1, S2D_00, "double", OPS_INC),
     ops_arg_dat(xarea, 1, S2D_00_0M1, "double", OPS_READ),
     ops_arg_dat(pressure, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
     ops_arg_dat(yvel0, 1, S2D_00, "double", OPS_READ),
     ops_arg_dat(yvel1, 1, S2D_00, "double", OPS_INC),
     ops_arg_dat(yarea, 1, S2D_00_M10, "double", OPS_READ),
     ops_arg_dat(viscosity, 1, S2D_00_M10_0M1_M1M1, "double", OPS_READ));
```

[\[fig:parloop\]]{#fig:parloop label="fig:parloop"}

::: thebibliography
1 OP2 for Many-Core Platforms, 2013.
<http://www.oerc.ox.ac.uk/projects/op2>

Istvan Z. Reguly, G.R. Mudalige, Mike B. Giles. Loop Tiling in
Large-Scale Stencil Codes at Run-time with OPS. (2017) IEEE Transactions
on Parallel and Distributed Systems.
<http://dx.doi.org/10.1109/TPDS.2017.2778161>
:::
