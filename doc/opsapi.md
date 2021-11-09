# OPS API

## Overview

The key characteristic of structured mesh applications is the implicit connectivity between neighboring mesh elements (such as vertices, cells). The main idea is that operations involve looping over a "rectangular" multi-dimensional set of mesh points using one or more "stencils" to access data. In multi-block meshes, we have several structured blocks.  The connectivity between the faces of different blocks can be quite complex, and in particular they may not be oriented in the same way, i.e.~an $i,j$ face of one block may correspond to the $j,k$ face of another block.  This is awkward and hard to handle simply.

## Key Concepts and Structure

The OPS API allows to declare a computation over such multi-block structured meshes. An OPS application can generally be declared in two key parts: (1) initialisation and (2) iteration over the mesh (carried out as a parallel loop). During the initialisation phase, one or more blocks (we call these `ops_block`s) are defined: these only have a dimensionality (i.e. 1D, 2D, etc.), and serve to group datasets together. Datasets are defined on a block, and have a specific size (in each dimension of the block), which may be slightly different across different datasets (e.g. staggered grids), in some directions they may be degenerate (a size of 1), or they can represent data associated with different multigrid levels (where their size if a multiple or a fraction of other datasets). Datasets can be declared with empty (NULL) pointers, then OPS will allocate the appropriate amount of memory, may be passed non-NULL pointers (currently only supported in non-MPI environments), in which case OPS will assume the memory is large enough for the data and the block halo, and there are HDF5 dataset declaration routines which allow the distributed reading of datasets from HDF5 files. The concept of blocks is necessary to group datasets together, as in a multi-block problem, in a distributed memory environment, OPS needs to be able to determine how to
decompose the problem.

The initialisation phase usually also consists of defining the stencils to be used later on (though they can be defined later as well), which describe the data access patterns used in parallel loops. Stencils are always relative to the "current" point; e.g. if at iteration $(i,j)$, we wish to access $(i-1,j)$ and $(i,j)$, then the stencil will have two points: $\{(-1, 0), (0, 0)\}$. To support degenerate datasets (where in one of the dimensions the dataset's size is 1), as well as for multigrid, there are special strided, restriction, and prolongation stencils: they differ from normal stencils in that as one steps through a grid in a parallel loop, the stepping is done with a non-unit stride
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

OPS takes responsibility for all data, its movement and the execution of parallel loops. With different execution hardware and optimisations, this means OPS will **re-organise** data as well as execution (potentially across different loops), and therefore **data accesses or manipulation should only be done through the OPS API**. There is an external data access API that allows access to the data stored by OPS which in turn allows interfacing with external libraries.

This restriction is exploited by a lazy execution mechanism in OPS. The idea is that OPS API calls that do not return a result need not be executed immediately, rather queued, and once an API call requires returning some data, operations in the queue are executed, and the result is returned. This allows OPS to analyse and optimise operations
in the queue together. This mechanism is fully automated by OPS, and is used with the various `_tiled` executables. For more information on how to use this mechanism for improving CPU performance, see Section on Tiling. Some API calls triggering the execution of queued operations include `ops_reduction_result`, and the functions in the
data access API.

To further clarify some of the important issues encountered when designing the OPS API, we note here some needs connected with a 3D application:

* When looping over the interior with loop indices $i,j,k$, often there are 1D arrays which are referenced using just one of the indices.
* To implement boundary conditions, we often loop over a 2D face, accessing both the 3D dataset and data from a 2D dataset.
* To implement periodic boundary conditions using dummy "halo" points, we sometimes have to copy one plane of boundary data to another.  e.g. if the first dimension has size $I$ then we might copy the plane $i=I-2$ to plane $i=0$, and plane $i=1$ to plane $i=I-1$.
* In multigrid, we are working with two grids with one having twice as many points as the other in each direction. To handle this we require a stencil with a non-unit stride.
* In multi-block grids, we have several structured blocks. The connectivity between the faces of different blocks can be quite complex, and in particular they may not be oriented in the same way, i.e. an $i,j$ face of one block may correspond to the $j,k$ face of another block.

OPS handle all of these different requirements through stencil definitions.

## OPS C and C++ API

Both C and C++ styles API are provided for utilizing the capabilities provided by the OPS library. They are essentially the same although there are minor differences in syntax. The C++ API is mainly designed for data abstraction, which  therefore provides better data encapsulation and the support of multiple instances and threading (OpenMP currently). In the following both C style routines and C++ class and methods will be introduced according to their functionality with a notice (C) or (C++). If there is no such notice, the routine either applies to both or might not provided by the C++ API.

To enable the C++ API, a compiler directive ``OPS_CPP_API`` is required.

### Initialisation and termination routines
#### C Style
##### ops_init

__void ops_init(int argc, char** argv, int diags_level)__

This routine must be called before all other OPS routines

| Arguments      | Description |
| ----------- | ----------- |
| argc, argv      | the usual command line arguments      |
| diags_level   |  an integer which defines the level of debugging diagnostics and reporting to be performed |

Currently, higher diags_levels does the following checks

`diags_level` $=$ 1 : no diagnostics, default to achieve best runtime
performance.

`diags_level` $>$ 1 : print block decomposition and `ops_par_loop`
timing breakdown.

`diags_level` $>$ 4 : print intra-block halo buffer allocation feedback
(for OPS internal development only)

`diags_level` $>$ 5 : check if intra-block halo MPI sends depth match
MPI receives depth (for OPS internal development only)

#### ops_exit

__void ops_exit()__

This routine must be called last to cleanly terminate the OPS computation.
#### C++ style

With the C++ style APIs, all data structures (block, data and stencils etc ) are encapsulated into a class  ``OPS_instance``. Thus, we can allocate multiple instances of ``OPS_instance`` by using the class constructor, for example,

```c++
// Allocate an instance
OPS_instance *instance = new OPS_instance(argc,argv,1,ss);
```

where the meaning of arguments are same to the C API, while the extra argument (i.e., ss) is for accpeting the messages.

An explicit termination is not needed for the C++ API, although we need to "delete" the instance in if it is allocated through pointer, i.e.,
```C++
delete instance;
```

### Declaration routines

#### Block
##### ops_decl_block (C)

__ops_block ops_decl_block(int dims, char *name)__

This routine defines a structured grid block.
| Arguments      | Description |
| ----------- | ----------- |
| dims    | dimension of the block    |
| name  |  a name used for output diagnostics |

##### OPS_instance::decl_block (C++)

A method of the OPS_instance class for declaring a block, which accepts same arguments with the C style function. A OPS_instance object should be constructed before this. The method returns a pointer to a ops_block type variable, where ops_block is an alias to a pointer type of ops_block_core. An example is

```C++
ops_block grid2D = instance->decl_block(2, "grid2D");
```

##### ops_decl_block_hdf5 (C)

__ops_block ops_decl_block_hdf5(int dims, char *name, char *file)__

This routine reads the details of a structured grid block from a named HDF5 file

| Arguments      | Description |
| ----------- | ----------- |
| dims    | dimension of the block    |
| name  |  a name used for output diagnostics |
| file |hdf5 file to read and obtain the block information from|

Although this routine does not read in any extra information about the
block from the named HDF5 file than what is already specified in the
arguments, it is included here for error checking (e.g. check if blocks
defined in an HDF5 file is matching with the declared arguments in an
application) and completeness.

#### Dat (ops_cat_core)
##### ops_decl_dat (C)

__ops_dat ops_decl_dat(ops block block, int dim, int *size, int *base, int *dm, int *d p, T *data, char *type, char *name)__

This routine defines a dataset.

| Arguments      | Description |
| ----------- | ----------- |
|block   |      structured block |
|dim     |      dimension of dataset (number of items per grid element) |
|size    |  size in each dimension of the block |
|base    |  base indices in each dimension of the block |
|d_m    |  padding from the face in the negative direction for each dimension (used for block halo) |
|d_p    |  padding from the face in the positive direction for each dimension (used for block halo) |
|data    |     input data of type *T* |
|type     |     the name of type used for output diagnostics (e.g. ``double``,``float``)|
|name     |     a name used for output diagnostics|

The `size` allows to declare different sized data arrays on a given
`block`. `d_m` and `d_p` are depth of the "block halos" that are used to
indicate the offset from the edge of a block (in both the negative and
positive directions of each dimension).

##### ops_block_core::decl_dat (C++)
The method ops_block_core::decl_dat is used to define a ops_dat object, which accepts almost same arguments with the C conterpart where the block argument is not necessary, e.g.,
```C++
//declare ops_dat with dim = 2
ops_dat dat0    = grid2D->decl_dat(2, size, base, d_m, d_p, temp, "double", "dat0");
ops_dat dat1    = grid2D->decl_dat(2, size, base, d_m, d_p, temp, "double", "dat1");
```
where grid2D is a ops_block_core object which shall be defined before this.

##### ops_decl_dat_hdf5 (C)

__ops_dat ops_decl_dat_hdf5(ops_block block, int dim, char *type, char *name, char *file)__

This routine defines a dataset to be read in from a named hdf5 file

| Arguments      | Description |
| ----------- | ----------- |
|block  |   structured block|
|dim     |  dimension of dataset (number of items per grid element)|
type    |  the name of type used for output diagnostics (e.g. ``double``,``float``)|
|name   |   name of the dat used for output diagnostics|
|file   |   hdf5 file to read and obtain the data from|

#### Global constant
##### ops_decl_const (C)

__void ops_decl_const(char const * name, int dim, char const * type, T * data )__

This routine defines a global constant: a variable in global scope. Global constants need to be declared upfront
 so that they can be correctly handled for different parallelization. For e.g CUDA on GPUs. Once defined
 they remain unchanged throughout the program, unless changed by a call to ops_update_const(..). The ``name'' and``type''
 parameters **must** be string literals since they are used in the code generation step

| Arguments      | Description |
| ----------- | ----------- |
|name |         a name used to identify the constant |
|dim |           dimension of dataset (number of items per element) |
|type |          the name of type used for output diagnostics (e.g. ``double``, ``float``) |
|data |          pointer to input data of type *T* |

##### OPS_instance::decl_const (C++)

The method accepts same arguments with its C counterpart.

#### Halo definition
##### ops_decl_halo (C)

__ops_halo ops_decl_halo(ops_dat from, ops_dat to, int *iter_size, int* from_base, int *to_base, int *from_dir, int *to_dir)__

| Arguments      | Description |
| ----------- | ----------- |
|from | origin dataset |
|to|  destination dataset |
|item_size |  defines an iteration size (number of indices to iterate over in each direction) |
|from_base |  indices of starting point in \"from\" dataset|
|to_base | indices of starting point in \"to\" dataset |
|from_dir | direction of incrementing for \"from\" for each dimension of `iter_size` |
|to_dir |  direction of incrementing for \"to\" for each dimension of `iter_size`|

A from_dir \[1,2\] and a to_dir \[2,1\] means that x in the first block
goes to y in the second block, and y in first block goes to x in second
block. A negative sign indicates that the axis is flipped. (Simple
example: a transfer from (1:2,0:99,0:99) to (-1:0,0:99,0:99) would use
iter_size = \[2,100,100\], from_base = \[1,0,0\], to_base = \[-1,0,0\],
from_dir = \[0,1,2\], to_dir = \[0,1,2\]. In more complex case this
allows for transfers between blocks with different orientations.)

##### OPS_instance::decl_halo (C++)
The method accepts same arguments with its C counterpart.

##### ops_decl_halo_hdf5 (C)

__ops_halo ops_decl_halo_hdf5(ops_dat from, ops_dat to, char* file)__

This routine reads in a halo relationship between two datasets defined on two different blocks from a named HDF5 file

| Arguments      | Description |
| ----------- | ----------- |
|from|      origin dataset|
|to|        destination dataset|
|file|      hdf5 file to read and obtain the data from|

##### ops_decl_halo_group (C)

__ops_halo_group ops_decl_halo_group(int nhalos, ops_halo *halos)__

This routine defines a collection of halos. Semantically, when an exchange is triggered for all halos in a group, there is no order defined in which they are carried out.
| Arguments      | Description |
| ----------- | ----------- |
|nhalos|         number of halos in *halos* |
|halos|           array of halos|

##### OPS_instance::decl_halo_group (C++)

The method accepts same arguments with its C counterpart.

#### Reduction handle
##### ops_decl_reduction_handle (C)

__ops_reduction ops_decl_reduction_handle(int size, char *type, char *name)__
This routine defines a reduction handle to be used in a parallel loop

| Arguments      | Description |
| ----------- | ----------- |
|size|      size of data in bytes |
|type|          the name of type used for output diagnostics (e.g. ``double``,``float``) |
|name|          name of the dat used for output diagnostics|

__{void ops_reduction_result(ops_reduction handle, T *result)
{This routine returns the reduced value held by a reduction handle. When OPS uses lazy execution, this will trigger the execution of all previously queued OPS operations.}

|handle|  the *ops_reduction* handle |
|result|  a pointer to write the results to, memory size has to match the declared |

##### OPS_instance::decl_reduction_handle (C++)
The method accepts same arguments with its C counterpart.
#### Partition
##### ops_partition (C)

__ops_partition(char *method)__

Triggers a multi-block partitioning across a distributed memory set of processes. (links to a dummy function for single node parallelizations). This routine should only be called after all the ops_halo ops_decl_block
and ops_halo ops_decl_dat statements have been declared

| Arguments      | Description |
| ----------- | ----------- |
|method| string describing the partitioning method. Currently this string is not used internally, but is simply a place-holder to indicate different partitioning methods in the future. |


##### OPS_instance::partition (C++)

The method accepts same arguments with its C counterpart.
### Diagnostic and output routines

#### ops_diagnostic_output (C)

__void ops_diagnostic_output()__

This routine prints out various useful bits of diagnostic info about sets, mappings and datasets. Usually used right
after an ops_partition() call to print out the details of the decomposition

#### OPS_instance::diagnostic_output (C++)
Same to the C counterpart.
#### ops_printf

__void ops_printf(const char * format, ...)__

This routine simply prints a variable number of arguments; it is created is in place of the standard C
printf function which would print the same on each MPI process

#### ops_timers

__void ops_timers(double *cpu, double *et)__
 gettimeofday() based timer to start/end timing blocks of code

| Arguments      | Description |
| ----------- | ----------- |
|cpu|  variable to hold the CPU time at the time of invocation|
|et| variable to hold the elapsed time at the time of invocation|

#### ops_fetch_block_hdf5_file

__void ops_fetch_block_hdf5_file(ops_block block, char *file)__

Write the details of an ops_block to a named HDF5 file. Can be used over MPI (puts the data in an ops_dat into an
HDF5 file using MPI I/O)

| Arguments      | Description |
| ----------- | ----------- |
|block|  ops_block to be written|
|file|     hdf5 file to write to|

#### ops_fetch_stencil_hdf5_file

__void ops_fetch_stencil_hdf5_file(ops_stencil stencil, char *file)__

Write the details of an ops_block to a named HDF5 file. Can be used over MPI (puts the data in an ops_dat into an HDF5 file using MPI I/O)

| Arguments      | Description |
| ----------- | ----------- |
|stencil|  ops_stencil to be written
|file|     hdf5 file to write to

#### ops_fetch_dat_hdf5_file

__void ops_fetch_dat_hdf5_file(ops_dat dat, const char *file)__

Write the details of an ops_block to a named HDF5 file. Can be used over MPI (puts the data in an ops_dat into an
HDF5 file using MPI I/O)

| Arguments      | Description |
| ----------- | ----------- |
|dat|  ops_dat to be written|
|file|     hdf5 file to write to|

#### ops_print_dat_to_txtfile

__void ops_print_dat_to_txtfile(ops_dat dat, chat *file)__
Write the details of an ops_block to a named text file. When used under an MPI parallelization each MPI process
will write its own data set separately to the text file. As such it does not use MPI I/O. The data can be viewed using
a simple text editor

| Arguments      | Description |
| ----------- | ----------- |
|dat|  ops_dat to to be written|
|file|     text file to write to|

#### ops_timing_output

__void ops_timing_output(FILE *os)__

Print OPS performance performance details to output stream

| Arguments      | Description |
| ----------- | ----------- |
|os|    output stream, use stdout to print to standard out|

#### ops_NaNcheck

__void ops_NaNcheck(ops_dat dat)__

Check if any of the values held in the *dat* is a NaN. If a NaN
is found, prints an error message and exits.

| Arguments      | Description |
| ----------- | ----------- |
|dat|  ops_dat to to be checked|

### Halo exchange

#### ops_halo_transfer (C)

__void ops_halo_transfer(ops_halo_group group)__

This routine exchanges all halos in a halo group and will block execution of subsequent computations that depend on
the exchanged data.

| Arguments      | Description |
| ----------- | ----------- |
|group|         the halo group|

### Parallel loop syntax

A parallel loop with N arguments has the following syntax:

#### ops_par_loop

__void ops_par_loop(void (*kernel)(...),char *name, ops_block block, int dims, int *range, ops_arg arg1,ops_arg arg2, ..., ops_arg argN )__

| Arguments      | Description |
| ----------- | ----------- |
|kernel|     user's kernel function with N arguments|
|name|       name of kernel function, used for output diagnostics|
|block|      the ops_block over which this loop executes|
|dims|       dimension of loop iteration|
|range|      iteration range array|
|args|       arguments|

The **ps_arg** arguments in **ops_par_loop** are provided by one of the
following routines, one for global constants and reductions, and the other
for OPS datasets.

#### ops_arg_gbl

__ops_arg ops_arg_gbl(T *data, int dim, char *type, ops_access acc)__

Passes a scalar or small array that is invariant of the iteration space (not to be confused with ops_decl_const, which facilitates global scope variables).

| Arguments      | Description |
| ----------- | ----------- |
|data|       data array|
|dim|        array dimension|
|type|       string representing the type of data held in data|
|acc|        access type|

#### ops_arg_reduce

__ops_arg ops_arg_reduce(ops_reduction handle, int dim, char *type, ops_access acc)__

Passes a pointer to a variable that needs to be incremented (or swapped for min/max reduction) by the user kernel.

| Arguments      | Description |
| ----------- | ----------- |
|handle|       an  *ops_reduction* handle|
|dim|        array dimension (according to *type*)|
|type|       string representing the type of data held in data|
|acc|        access type|

#### ops_arg_dat

__ops_arg ops_arg_dat(ops_dat dat, ops_stencil stencil, char *type,ops_access acc)__

Passes a pointer wrapped in ac ACC<> object to the value(s) at the current grid point to the user kernel. The ACC object's parentheses operator has to be used for dereferencing the pointer.

| Arguments      | Description |
| ----------- | ----------- |
|dat|        dataset|
|stencil|    stencil for accessing data|
|type|       string representing the type of data held in dataset|
|acc|        access type|

#### ops_arg_idx

__ops_arg ops_arg_idx()__

Give you an array of integers (in the user kernel) that have the index of
the current grid point, i.e. idx[0] is the index in x, idx[1] is the index in y, etc. This is a globally consistent
index, so even if the block is  distributed across different MPI partitions, it gives you the same indexes. Generally
used to generate initial geometry.

### Stencils

The final ingredient is the stencil specification, for which we have two versions: simple and strided.

#### ops_decl_stencil (C)

__ops_stencil ops_decl_stencil(int dims,int points, int *stencil, char *name)__

| Arguments      | Description |
| ----------- | ----------- |
|dims|     dimension of loop iteration|
|points|   number of points in the stencil|
|stencil|  stencil for accessing data|
|name| string representing the name of the stencil|

#### OPS_instance::decl_stencil (C++)

The method accepts same arguments with its C counterpart.
#### ops_decl_strided_stencil (C)

__ops_stencil ops_decl_strided_stencil(int dims, int points, int *stencil, int *stride, char *name)__

| Arguments      | Description |
| ----------- | ----------- |
|dims|       dimension of loop iteration|
|points|     number of points in the stencil|
|stencil|    stencil for accessing data|
|stride|     stride for accessing data|
|name| string representing the name of the stencil|

#### OPS_instance::decl_strided_stencil (C++)

The method accepts same arguments with its C counterpart.

#### ops_decl_stencil_hdf5

__ops_stencil ops_decl_stencil_hdf5(int dims,int points, char *name, char* file)__

| Arguments      | Description |
| ----------- | ----------- |
|dims|     dimension of loop iteration|
|points|   number of points in the stencil|
|name|     string representing the name of the stencil|
|file|     hdf5 file to write to|

 In the strided case, the semantics for the index of data to be
accessed, for stencil point*p*, in dimension *m* are defined as

```c++
 stride[m]*loop_index[m] + stencil[p*dims+m]
```

where ``loop_index[m]`` is the iteration index (within the
user-defined iteration space) in the different dimensions.

If, for one or more dimensions, both ``stride[m]`` and
``stencil[p*dims+m]`` are zero, then one of the following must be true;

* the dataset being referenced has size 1 for these dimensions

* these dimensions are to be omitted and so the dataset has
dimension equal to the number of remaining dimensions.

See *OPS/apps/c/CloverLeaf/build_field.cpp* and *OPS/apps/c/CloverLeaf/generate.cpp* for an example *ops_decl_strided_stencil* declaration and its use in a loop,respectively.

These two stencil definitions probably take care of all of the
cases in the Introduction except for multiblock applications with interfaces
with different orientations -- this will need a third, even more general,
stencil specification. The strided stencil will handle both multigrid
(with a stride of 2 for example) and the boundary condition and reduced
dimension applications (with a stride of 0 for the relevant dimensions).

### Checkpointing

OPS supports the automatic checkpointing of applications. Using the API below, the user specifies the file name for the checkpoint and an average time interval between checkpoints, OPS will then automatically save all necessary information periodically that is required to fast-forward to the last checkpoint if a crash occurred. Currently, when re-launching after a crash, the same number of MPI processes have to be used. To enable checkpointing mode, the *OPS_CHECKPOINT* runtime argument has to be used.

#### ops_checkpointing_init

__bool ops_checkpointing_init(const char *filename, double interval, int options)__

Initialises the checkpointing system, has to be called after *ops_partition*. Returns true if the application launches in restore
mode, false otherwise.

| Arguments      | Description |
| ----------- | ----------- |
|filename| name of the file for checkpointing. In MPI, this will automatically be post-fixed with the rank ID.|
|interval| average time (seconds) between checkpoints|
|options| a combinations of flags, listed in *ops_checkpointing.h*, also see below|

* OPS_CHECKPOINT_INITPHASE - indicates that there are a number of parallel loops at the very beginning of the simulations which should be excluded from any checkpoint; mainly because they initialise datasets that do not change during the main body of the execution. During restore mode these loops are executed as usual. An example would be the computation of the mesh geometry, which can be excluded from the checkpoint if it is re-computed when recovering and restoring a checkpoint. The API call *void ops_checkpointing_initphase_done()* indicates the end of this initial phase.

* OPS_CHECKPOINT_MANUAL_DATLIST - Indicates that the user manually controls the location of the checkpoint, and explicitly specifies the list of *ops_dat*s to be saved.

* OPS_CHECKPOINT_FASTFW - Indicates that the user manually controls the location of the checkpoint, and it also enables fast-forwarding, by skipping the execution of the application (even though none of the parallel loops would actually execute, there may be significant work outside of those) up to the checkpoint

* OPS_CHECKPOINT_MANUAL - Indicates that when the corresponding API function is called, the checkpoint should be created. Assumes the presence of the above two options as well.

#### ops_checkpointing_manual_datlist

__void ops_checkpointing_manual_datlist(int ndats, ops_dat *datlist)__

A user can call this routine at a point in the code to mark the location of a checkpoint.  At this point, the list of datasets specified
will be saved. The validity of what is saved is not checked by the checkpointing algorithm assuming that the user knows
what data sets to be saved for full recovery. This routine should be called frequently (compared to check-pointing frequency) and it will trigger the creation of the checkpoint the first time it is called after the timeout occurs.

| Arguments      | Description |
| ----------- | ----------- |
|ndats| number of datasets to be saved|
|datlist| arrays of *ops_dat* handles to be saved|

#### ops_checkpointing_fastfw

__bool ops_checkpointing_fastfw(int nbytes, char *payload)__

A use can call this routine at a point in the code to mark the location of a checkpoint.  At this point, the
specified payload (e.g. iteration count, simulation time, etc.) along with the necessary datasets, as determined by the
checkpointing algorithm will be saved. This routine should be called frequently (compared to checkpointing frequency),
will trigger the creation of the checkpoint the first time it is called after the timeout occurs. In restore mode,
will restore all datasets the first time it is called, and returns true indicating that the saved payload is returned
in payload. Does not save reduction data.

| Arguments      | Description |
| ----------- | ----------- |
|nbytes| size of the payload in bytes|
|payload| pointer to memory into which the payload is packed|

#### ops_checkpointing_manual_datlist_fastfw

__bool ops_checkpointing_manual_datlist_fastfw(int ndats, op_dat *datlist, int nbytes, char *payload)__

Combines the manual datlist and fastfw calls.

| Arguments      | Description |
| ----------- | ----------- |
|ndats| number of datasets to be saved|
|datlist| arrays of *ops_dat* handles to be saved|
|nbytes| size of the payload in bytes|
|payload| pointer to memory into which the payload is packed|

#### ops_checkpointing_manual_datlist_fastfw_trigger

__bool ops_checkpointing_manual_datlist_fastfw_trigger(int ndats, opa_dat *datlist, int
nbytes, char *payload)__

With this routine it is possible to manually trigger checkpointing, instead of relying on the timeout process. as such
it combines the manual datlist and fastfw calls, and triggers the creation of a checkpoint when called.

| Arguments      | Description |
| ----------- | ----------- |
|ndats| number of datasets to be saved|
|datlist| arrays of *ops_dat* handles to be saved|
|nbytes| size of the payload in bytes|
|payload| pointer to memory into which the payload is packed|

The suggested use of these **manual** functions is of course when the optimal location for checkpointing
is known - one of the ways to determine that is to use the built-in algorithm. More details of this will be reported
in a tech-report on checkpointing, to be published later.

### Access to OPS data

This section describes APIs that give the user access to internal data structures in OPS and return data to user-space. These should be used cautiously and sparsely, as they can affect performance significantly

#### ops_dat_get_local_npartitions (C)

__int ops_dat_get_local_npartitions(ops_dat dat)__

This routine returns the number of chunks of the given dataset held by the current process.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset|

#### ops_dat_core::get_local_npartitions (C++)
The C++ version of ``ops_dat_get_local_npartitions``, which does not require input.
#### ops_dat_get_global_npartitions (C)

__int ops_dat_get_global_npartitions(ops_dat dat)__

This routine returns the number of chunks of the given dataset held by all processes.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset

#### ops_dat_core::get_global_npartitions (C++)
The C++ version of ``ops_dat_get_global_npartitions``, which does not require input.
#### ops_dat_get_extents (C)

__void ops_dat_get_extents(ops_dat dat, int part, int *disp, int *sizes)__

This routine returns the MPI displacement and size of a given chunk of the given dataset on the current process.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset|
|part|        the chunk index (has to be 0)|
|disp|        an array populated with the displacement of the chunk within the ``global'' distributed array|
|sizes|       an array populated with the spatial extents|

#### ops_dat_core::get_extents (C++)
The C++ version of ``ops_dat_get_extents`` where the arguments are the same except no need of the ops_dat arguments.

#### ops_dat_get_raw_metadata (C)

__char* ops_dat_get_raw_metadata(ops_dat dat, int part, int *disp, int *size, int *stride, int *d_m, int *d_p)__

This routine returns array shape metadata corresponding to the ops_dat. Any of the arguments that are not of interest, may be NULL.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset|
|part|        the chunk index (has to be 0)|
|disp|        an array populated with the displacement of the chunk within the ``global'' distributed array|
|size|       an array populated with the spatial extents
|stride|      an array populated strides in spatial dimensions needed for column-major indexing|
|d_m|      an array populated with padding on the left in each dimension. Note that these are negative values|
|d_p|      an array populated with padding on the right in each dimension|

#### ops_dat_core::get_raw_metadata (C++)
The C++ version of ``ops_dat_get_raw_metadata`` where the arguments are the same except no need of the ops_dat arguments.
#### ops_dat_get_raw_pointer (C)

__char* ops_dat_get_raw_pointer(ops_dat dat, int part, ops_stencil stencil, ops_memspace *memspace)__

This routine returns a pointer to the internally stored data, with MPI halo regions automatically updated as required by the supplied stencil. The strides required to index into the dataset are also given.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset|
|part|        the chunk index (has to be 0)|
|stencil|     a stencil used to determine required MPI halo exchange depths|
|memspace|       when set to OPS_HOST or OPS_DEVICE, returns a pointer to data in that memory space, otherwise must be set to 0, and returns whether data is in the host or on the device|

#### ops_dat_core::get_raw__pointer (C++)
The C++ version of ``ops_dat_get_raw_pointer`` where the arguments are the same except no need of the ops_dat arguments.
#### ops_dat_release_raw_data (C)

__void ops_dat_release_raw_data(ops_dat dat, int part, ops_access acc)__

Indicates to OPS that a dataset previously accessed with ops_dat_get_raw_pointer is released by the user, and also tells OPS how it was accessed.

A single call to ops_dat_release_raw_data() releases all pointers obtained by previous calls to ops_dat_get_raw_pointer() calls on the same dat and with the same *memspace argument, i.e. calls do not nest.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset
|part|        the chunk index (has to be 0)|
|acc|     the kind of access that was used by the user (OPS_READ if it was read only, OPS_WRITE if it was overwritten, OPS_RW if it was read and written)|

#### ops_dat_core::_release_raw_data (C++)
The C++ version of ``ops_dat_release_raw_data`` where the arguments are the same except no need of the ops_dat arguments.
#### ops_dat_fetch_data (C)

__void ops_dat_fetch_data(ops_dat dat, int part, int *data)__

This routine copies the data held by OPS to the user-specified memory location, which needs to be at least as large as indicated by the sizes parameter of ops_dat_get_extents.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset|
|part|        the chunk index (has to be 0) |
|data|        pointer to memory which should be filled by OPS|

#### ops_dat_fetch_data_memspace (C)

__void ops_dat_fetch_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace)__

This routine copies the data held by OPS to the user-specified memory location, as which needs to be at least as large as indicated by the sizes parameter of ops_dat_get_extents.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset|
|part|        the chunk index (has to be 0) |
|data|        pointer to memory which should be filled by OPS|
| memspace |the memory space where the data pointer is|
#### ops_dat_core::fetch_data (C++)
The C++ version of ``ops_dat_fetch_data_memspace`` where the arguments the same except no need of the ops_dat arguments.
#### ops_dat_set_data (C)

__void ops_dat_set_data(ops_dat dat, int part, int *data)__

This routine copies the data given  by the user to the internal data structure used by OPS. User data needs to be laid out in column-major order and strided as indicated by the sizes parameter of ops_dat_get_extents.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset|
|part|        the chunk index (has to be 0)|
|data|        pointer to memory which should be copied to OPS |


#### ops_dat_set_data_memspace (C)

__void ops_dat_set_data_memspace(ops_dat dat, int part, char *data, ops_memspace memspace)__

This routine copies the data given  by the user to the internal data structure used by OPS. User data needs to be laid out in column-major order and strided as indicated by the sizes parameter of ops_dat_get_extents.

| Arguments      | Description |
| ----------- | ----------- |
|dat|         the dataset|
|part|        the chunk index (has to be 0)|
|data|        pointer to memory which should be copied to OPS |
|memspace| the memory space where the data pointer is|

#### ops_dat_core::set_data (C++)
The C++ version of ``ops_dat_set_data_memspace`` where the arguments the same except no need of the ops_dat arguments.
### Linear algebra solvers

####  Tridiagonal solver
This section specifies APIs that allow [Tridsolver](https://github.com/OP-DSL/tridsolver) (a tridiagonal solver library) to be called from OPS. The library can be used to solve a large number of tridiagonal systems of equations stored in multidimensional datasets. Parameters that are passed to Tridsolver from OPS are stored in an `ops_tridsolver_params` object. The constructor for this class takes the `ops_block` that the datasets are defined over as an argument and optionally also a solving strategy to use (only relevant to MPI applications). The following solving strategies are available (see Tridsolver for more details about these):

- GATHER_SCATTER (not available for GPUs)
- ALLGATHER
- LATENCY_HIDING_TWO_STEP
- LATENCY_HIDING_INTERLEAVED
- JACOBI
- PCR (default)

Then parameters specific to different solving strategies can be set using setter methods. For applications using MPI, it is beneficial to reuse `ops_tridsolver_params` objects between solves as much as possible due to set up times involved with creating Tridsolver's MPI communicators.

##### ops_tridMultiDimBatch

__void ops_tridMultiDimBatch(int ndim, int solvedim, int* dims, ops_dat a, ops_dat b, ops_dat c, ops_dat d, ops_tridsolver_params *tridsolver_ctx)__

This solves multiple tridiagonal systems of equations in multidimensional datasets along the specified dimension. The matrix is stored in the `a` (bottom diagonal), `b` (central diagonal) and `c` (top diagonal) datasets. The right hand side is stored in the `d` dataset and the result is also written to this dataset.

| Arguments      | Description |
| ----------- | ----------- |
|ndim| the dimension of the datasets |
|solvedim| the dimension to solve along |
|dims| the size of each dimension (excluding any padding) |
|a| the dataset for the lower diagonal |
|b| the dataset for the central diagonal |
|c| the dataset for the upper diagonal |
|d| the dataset for the right hand side, also where the solution is written to |
|tridsolver_ctx| an object containing the parameters for the Tridsolver library |

##### ops_tridMultiDimBatch_Inc

__void ops_tridMultiDimBatch(int ndim, int solvedim, int* dims, ops_dat a, ops_dat b, ops_dat c, ops_dat d, ops_dat u, ops_tridsolver_params *tridsolver_ctx)__

This solves multiple tridiagonal systems of equations in multidimensional datasets along the specified dimension. The matrix is stored in the `a` (bottom diagonal), `b` (central diagonal) and `c` (top diagonal) datasets. The right hand side is stored in the `d` dataset and the result is added to the `u` dataset.

| Arguments      | Description |
| ----------- | ----------- |
|ndim| the dimension of the datasets |
|solvedim| the dimension to solve along |
|dims| the size of each dimension (excluding any padding) |
|a| the dataset for the lower diagonal |
|b| the dataset for the central diagonal |
|c| the dataset for the upper diagonal |
|d| the dataset for the right hand side |
|u| the dataset that the soluion is added to |
|tridsolver_ctx| an object containing the parameters for the Tridsolver library |

## Runtime Flags and Options

The following is a list of all the runtime flags and options that can be used when executing OPS generated applications.

* `OPS_DIAGS=` : set OPS diagnostics level at runtime.

  `OPS_DIAGS=1` - no diagnostics, default level to achieve the best runtime performance.

  `OPS_DIAGS>1` - print block decomposition and `ops_par_loop` timing breakdown.

  `OPS_DIAGS>4` - print intra-block halo buffer allocation feedback (for OPS internal development only).

  `OPS_DIAGS>5` - check if intra-block halo MPI sends depth match MPI receives depth (for OPS internal development only).

* `OPS_BLOCK_SIZE_X=`, `OPS_BLOCK_SIZE_Y=` and `OPS_BLOCK_SIZE_Y=` : The CUDA (and OpenCL) thread block sizes in X, Y and Z dimensions. The sizes should be an integer between 1 - 1024, and currently they should be selected such that `OPS_BLOCK_SIZE_X`*`OPS_BLOCK_SIZE_Y`*`OPS_BLOCK_SIZE_Z`< 1024

* `-gpudirect` : Enable GPU direct support when executing MPI+CUDA executables.

* `OPS_CL_DEVICE=` : Select the OpenCL device for execution. Usually `OPS_CL_DEVICE=0` selects the CPU and `OPS_CL_DEVICE=1` selects GPUs. The selected device will be reported by OPS during execution.

* `OPS_TILING` : Execute OpenMP code with cache blocking tiling. See the [Performance Tuning](https://github.com/OP-DSL/OPS/blob/MarkdownDocDev/doc/perf.md) section.
* `OPS_TILING_MAXDEPTH=` : Execute MPI+OpenMP code with cache blocking tiling and further communication avoidance. See the [Performance Tuning](https://github.com/OP-DSL/OPS/blob/MarkdownDocDev/doc/perf.md) section.

## Doxygen
Doxygen generated from OPS source can be found [here](https://op-dsl-ci.gitlab.io/ops-ci/).
