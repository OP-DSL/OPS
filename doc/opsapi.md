# OPS API

## Overview

The key characteristic of structured mesh applications is the implicit connectivity between neighboring mesh elements (such as vertices, cells). The main idea is that operations involve looping over a "rectangular" multi-dimensional set of mesh points using one or more "stencils" to access data. In multi-block meshes, we have several structured blocks.  The connectivity between the faces of different blocks can be quite complex, and in particular they may not be oriented in the same way, i.e.~an $i,j$ face of one block may correspond to the $j,k$ face of another block.  This is awkward and hard to handle simply.

## Key Concepts and Structure

The OPS API allows to declare a computation over such multi-block structured meshes. An OPS application can generally be declared in two key parts: (1) initialisation and (2) iteration over the mesh (carried out as a parallel loop). During the initialisation phase, one or more blocks (we call these `ops_block`s) are defined: these only have a dimensionality (i.e. 1D, 2D, etc.), and serve to group datasets together. Datasets are defined on a block, and have a specific size (in each dimension of the block), which may be slightly different across different datasets (e.g. staggered grids), in some directions they may be degenerate (a size of 1), or they can represent data associated with different multigrid levels (where their size if a multiple or a fraction of other datasets). Datasets can be declared with empty (NULL) pointers, then OPS will allocate the appropriate amount of memory, may be passed non-NULL pointers (currently only supported in non-MPI environments), in which case OPS will assume the memory is large enough for the data and the block halo, and there are HDF5 dataset declaration routines which allow the distributed reading of datasets from HDF5 files. The concept of blocks is necessary to group datasets together, as in a multi-block problem, in a distributed memory environment, OPS needs to be able to determine how to
decompose the problem.

The initialisation phase usually also consists of defining the stencils to be used later on (though they can be defined later as well), which describe the data access patterns used in parallel loops. Stencils are always relative to the "current" point; e.g. if at iteration $(i,j)$, we wish to access $(i{-}1,j)$ and $(i,j)$, then the stencil will have two points: $\{(-1, 0), (0, 0)\}$. To support degenerate datasets (where in one of the dimensions the dataset's size is 1), as well as for multigrid, there are special strided, restriction, and prolongation stencils: they differ from normal stencils in that as one steps through a grid in a parallel loop, the stepping is done with a non-unit stride
for these datasets. For example, in a 2D problem, if we have a degenerate dataset called xcoords, size $(N,1)$, then we will need a stencil with stride $(1,0)$ to access it in a regular 2D loop.

Finally, the initialisation phase may declare a number of global constants - these are variables in global scope that can be accessed from within elemental kernels, without having to pass them in explicitly. These may be scalars or small arrays, generally for values that do not change during execution, though they may be updated during execution
with repeated calls to `ops_decl_const`.

The initialisation phase is terminated by a call to `ops_partition`.

The bulk of the application consists of parallel loops, implemented using calls to `ops_par_loop`. These constructs work with datasets, passed through the opaque `ops_dat` handles declared during the initialisation phase. The iterations of parallel loops are semantically independent, and it is the responsibility of the user to enforce this:
the order in which iterations are executed cannot affect the result (within the limits of floating point precision). Parallel loops are defined on a block, with a prescribed iteration range that is always defined from the perspective of the dataset written/modified (the sizes of datasets, particularly in multigrid situations, may be very
different). Datasets are passed in using `ops_arg_dat`, and during execution, values at the current grid point will be passed to the user kernel. These values are passed wrapped in a templated `ACC<>` object (templated on the type of the data), whose parentheses operator is overloaded, which the user must use to specify the relative offset to
access the grid point's neighbours (which accesses have to match the the declared stencil). Datasets written may only be accessed with a one-point, zero-offset stencil (otherwise the parallel semantics may be violated).

Other than datasets, one can pass in read-only scalars or small arrays that are iteration space invariant with `ops_arg_gbl` (typically weights, $\delta t$, etc. which may be different in different loops). The current iteration index can also be passed in with `ops_arg_idx`, which will pass a globally consistent index to the user kernel (i.e.
also under MPI).

Reductions in loops are done using the `ops_arg_reduce` argument, which takes a reduction handle as an argument. The result of the reduction can then be acquired using a separate call to `ops_reduction_result`. The semantics are the following: a reduction handle after it was declared is in an "uninitialised" state. The first time it is used as an argument to a loop, its type is determined (increment/min/max), and is initialised appropriately $(0,\infty,-\infty)$, and subsequent uses of the handle in parallel loops are combined together, up until the point, where the result is acquired using `ops_reduction_result`, which then sets it back to an uninitialised state. This also implies, that different parallel loops, which all use the same reduction handle, but are otherwise independent, are independent and their partial reduction results can be combined together associatively and commutatively.

OPS takes responsibility for all data, its movement and the execution of parallel loops. With different execution hardware and optimisations, this means OPS will **re-organise** data as well as execution (potentially across different loops), and therefore **any data accesses or manipulation must only be done through the OPS API**. 

This restriction is exploited by a lazy execution mechanism in OPS. The idea is that OPS API calls that do not return a result need not be executed immediately, rather queued, and once an API call requires returning some data, operations in the queue are executed, and the result is returned. This allows OPS to analyse and optimise operations
in the queue together. This mechanism is fully automated by OPS, and is used with the various `_tiled` executables. For more information on how to use this mechanism for improving CPU performance, see Section on Tiling. Some API calls triggering the execution of queued operations include `ops_reduction_result`, and the functions in the
data access API.

To further clarify some of the important issues encountered when designing the OPS API, we note here some needs connected with a 3D application:
*   When looping over the interior with loop indices $i,j,k$, often there are 1D arrays which are referenced using just one of the indices.
*   To implement boundary conditions, we often loop over a 2D face, accessing both the 3D dataset and data from a 2D dataset.
*   To implement periodic boundary conditions using dummy "halo" points, we sometimes have to copy one plane of boundary data to another.  e.g. if the first dimension has size $I$ then we might copy the plane $i=I-2$ to plane $i=0$, and plane $i=1$ to plane $i=I-1$.
*   In multigrid, we are working with two grids with one having twice as many points as the other in each direction. To handle this we require a stencil with a non-unit stride.
*   In multi-block grids, we have several structured blocks. The connectivity between the faces of different blocks can be quite complex, and in particular they may not be oriented in the same way, i.e. an $i,j$ face of one block may correspond to the $j,k$ face of another block. 

OPS handle all of these different requirements through stencil definitions.

## OPS C++ API 
### Initialisation and termination routines
### Declaration routines
### Diagnostic and output routines
### Halo exchange
### Parallel loop syntax
### Stencils
### Checkpointing
### Access to OPS data
