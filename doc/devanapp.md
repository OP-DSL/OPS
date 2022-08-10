# Developing an OPS Application
This page provides a tutorial in the basics of using OPS for multi-block structured mesh application development. This is taken from a [presentation](https://op-dsl.github.io/docs/OPS/tutorial.pdf) given initially in April 2018 and subsequently updated for the latest release of OPS. 

## OPS Abstraction
OPS is a Domain Specific Language embedded in C/C++ and Fortran, targeting the development of multi-block structured mesh computations. The abstraction has two distinct components:  the definition of the mesh, and operations over the mesh.
* Defining a number of 1-3D blocks, and on them a number of datasets, which have specific extents in the different dimensions.
* Describing a parallel loop over a given block, with a given iteration range, executing a given "kernel function" at each mesh point, and describing what datasets are going to be accessed and how.
* Additionally, one needs to declare stencils (access patterns) that will be used in parallel loops to access datasets, and any global constants (read-only global scope variables)

Data and computations expressed this way can be automatically managed and parallelised by the OPS library. Higher dimensions are supported in the backend, but not currently by the code generators.

## Example Application
In this tutorial we will use an example application, a simple 2D iterative Laplace equation solver. 
* Go to the `OPS/apps/c/laplace2dtutorial/original` directory
* Open the `laplace2d.cpp` file
* It uses an $imax$ x $jmax$ mesh, with an additional 1 layers of boundary cells on all sides
* There are a number of loops that set the boundary conditions along the four edges
* The bulk of the simulation is spent in a whilel oop, repeating a stencil kernel with a maximum reduction, and a copy kernel
* Compile and run the code !
Note: The following tutorial details the step-by-step approach for using OPS for Laplace (C version) application development. Similar step-by-step approach is also followed for the Laplace Fortran version and can be found at `OPS/apps/fortran/laplace2dtutorial`.

## Original - Initialisation
The original code begins with initializing the data arrays used in the calculation:
```
//Size along y
int jmax = 4094;
//Size along x
int imax = 4094;
//Size along x
int iter_max = 100;

double pi  = 2.0 * asin(1.0);
const double tol = 1.0e-6;
double error     = 1.0;

double *A;
double *Anew;
double *y0;

A    = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));
Anew = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));
y0   = (double *)malloc((imax+2) * sizeof(double));

memset(A, 0, (imax+2) * (jmax+2) * sizeof(double));
```
## Original - Boundary loops
The application sen sets boundary conditions:
```
for (int i = 0; i < imax+2; i++)
    A[(0)*(imax+2)+i]   = 0.0;

for (int i = 0; i < imax+2; i++)
    A[(jmax+1)*(imax+2)+i] = 0.0;

for (int j = 0; j < jmax+2; j++) {
    A[(j)*(imax+2)+0] = sin(pi * j / (jmax+1));
}

for (int j = 0; j < imax+2; j++) {
    A[(j)*(imax+2)+imax+1] = sin(pi * j / (jmax+1))*exp(-pi);
}
```  
Note how in the latter two loops the loop index is used.

## Original - Main iteration
The main iterative loop is a while loop iterating until the error tolarance is at a set level and the number of iterations are les than the maximum set. 
```
while ( error > tol && iter < iter_max ) {
  error = 0.0;
  for( int j = 1; j < jmax+1; j++ ) {
    for( int i = 1; i < imax+1; i++) {
      Anew[(j)*(imax+2)+i] = 0.25f * 
      ( A[(j)*(imax+2)+i+1] + A[(j)*(imax+2)+i-1]
      + A[(j-1)*(imax+2)+i] + A[(j+1)*(imax+2)+i]);
      error = fmax( error, fabs(Anew[(j)*(imax+2)+i]-A[(j)*(imax+2)+i]));
    }
  }
  for( int j = 1; j < jmax+1; j++ ) {
    for( int i = 1; i < imax+1; i++) {
      A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];    
    }
  }
  if(iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);        
  iter++;
}
```
## Build OPS
Build OPS using instructions in the [Getting Started](https://ops-dsl.readthedocs.io/en/markdowndocdev/installation.html#getting-started) page. 

## Step 1 - Preparing to use OPS
Firstly, include the appropriate header files, then initialise OPS, and at the end finalise it.
* Define that this application is 2D, include the OPS header file, and create a header file where the outlined "elemental kernels" will live.
```
#define OPS_2D
#include <ops_seq.h>
#include "laplace_kernels.h" 
```
* Initialise and finalise OPS
```  
int main(int argc, const char** argv) {
  //Initialise the OPS library, passing runtime args, and setting diagnostics level to low (1)
  ops_init(argc, argv,1);
  ...
  ...
  //Finalising the OPS library
  ops_exit();
}  
```  
By this point you need OPS set up - take a look at the Makefile in step1, and observe that the include and library paths are added, and we link against `ops_seq`.

## Step 2 - OPS declarations
Now declare a block and data on the block :
```
//The 2D block
ops_block block = ops_decl_block(2, "my_grid");

//The two datasets
int size[] = {imax, jmax};
int base[] = {0,0};
int d_m[] = {-1,-1};
int d_p[] = {1,1};
ops_dat d_A    = ops_decl_dat(block, 1, size, base,
                               d_m, d_p, A,    "double", "A");
ops_dat d_Anew = ops_decl_dat(block, 1, size, base,
                               d_m, d_p, Anew, "double", "Anew");
```
Data sets have a size (number of mesh points in each dimension). There is passing for halos or boundaries in the positive (`d_p`) and negative directions (`d_m`). Here we use a 1 thick boundary layer. Base index can be defined as it may be different from 0 (e.g. in Fortran). Item these with a 0 base index and a 1 wide halo, these datasets can be indexed from −1 tosize +1.

OPS supports gradual conversion of applications to its API, but in this case the described data sizes will need to match:  the allocated memory and its extents need to be correctly described to OPS. In this example we have two `(imax+ 2) ∗ (jmax+ 2)` size arrays, and the total size in each dimension needs to matchsize `[i] + d_p[i] − d_m[i]`.  This is only supported for the sequential and OpenMP backends. If a `NULL` pointer is passed, OPS will allocate the data internally.

We also need to declare the stencils that will be used - in this example most loops use a simple 1-point stencil, and one uses a 5-point stencil:
```
//Two stencils, a 1-point, and a 5-point
int s2d_00[] = {0,0};
ops_stencil S2D_00 = ops_decl_stencil(2,1,s2d_00,"0,0");
int s2d_5pt[] = {0,0, 1,0, -1,0, 0,1, 0,-1};
ops_stencil S2D_5pt = ops_decl_stencil(2,5,s2d_5pt,"5pt");
```  
Different names may be used for stencils in your code, but we suggest using some convention.

## Step 3 - First parallel loop
You can now convert the first loop to use OPS:
```
for (int i = 0; i < imax+2; i++)
    A[(0)*(imax+2)+i]   = 0.0;
```    
This is a loop on the ottom boundary of the domain, which is at the −1 index for our dataset, therefore our iteration range will be over the entire domain, including halos in the X direction, and the bottom boundary in the Y direction.  The iteration range is given as beginning (inclusive) and end (exclusive) indices in the x, y, etc.  directions.
```
int bottom_range[] = {-1, imax+1, -1, 0};
```
Next, we need to outline the “elemental” into `laplacekernels.h`, and place the appropriate access objects - `ACC<double> &A`, in the kernel’s formal parameter list, and `(i,j)` are the stencil offsets in the X and Y directions respectively:
```
void set_zero(ACC<double> &A) {
  A(0,0) = 0.0;
}
```
The OPS parallel loop can now be written as follows:
```
ops_par_loop(set_zero, "set_zero", block, 2, bottom_range,
      ops_arg_dat(d_A, 1, S2D_00, "double", OPS_WRITE));
```
The loop will execute `set_zero` at each mesh point defined in the iteration range, and write the dataset `d_A` with the 1-point stencil. The `ops_par_loop` implies that the order in which mesh points will be executed will not affect the end result (within machine precision).

There are three more loops which set values to zero, they can be trivially replaced with the code above, only altering the iteration range. In the main while loop, the second simpler loop simply copies data from one array to another, this time on the interior of the domain:
```
int interior_range[] = {0,imax,0,jmax};
ops_par_loop(copy, "copy", block, 2, interior_range,
    ops_arg_dat(d_A,    1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_READ));
```
And the corresponding outlined elemental kernel is as follows:
```
void copy(ACC<double> &A, const ACC<double> &Anew) {
  A(0,0) = Anew(0,0);
}
```
## Step 4 - Indexes and global constants
There are two sets of boundary loops which use the loop variable j - this is a common technique to initialise data, such as coordinates `(x = i∗dx)`. OPS has a special argument `ops_arg_idx` which gives us a globally coherent (including over MPI) iteration index - between the bounds supplied in the iteration range.
```
ops_par_loop(left_bndcon, "left_bndcon", block, 2, left_range,
      ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_idx());
```
And the corresponding outlined user kernel is as follows.  Observe the `idx` argument and the +1 offset due to the difference in indexing:
```
void left_bndcon(ACC<double> &A, const int *idx) {
  A(0,0) = sin(pi * (idx[1]+1) / (jmax+1));
}
```
This kernel also uses two variables,`jmax` and `pi` that do not depend on the iteration index - they are iteration space invariant.  OPS has two ways of supporting this:

1) Global scope constants, through `ops_decl_const`, as done in this example: we need to move the declaration of the `imax`,`jmax` and `pi` variables to global scope (outside of main), and call the OPS API:
```
//declare and define global constants
ops_decl_const("imax",1,"int",&imax);
ops_decl_const("jmax",1,"int",&jmax);
ops_decl_const("pi",1,"double",&pi);
```
These ariables do not need to be passed in to the elemental kernel, they are accessible in all elemental kernels.

2) The other option is to explicitly pass it to the elemental kernel with `ops_arg_gbl`:  this is for scalars and small arrays that should not be in global scope.


## Step 5 - Complex stencils and reductions
There is only one loop left, which uses a 5 point stencil and a reduction.  It can be outlined as usual, and for the stencil, we will use `S2Dpt5`.
```
ops_par_loop(apply_stencil, "apply_stencil", block, 2, interior_range,
        ops_arg_dat(d_A,    1, S2D_5pt, "double", OPS_READ),
        ops_arg_dat(d_Anew, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_reduce(h_err, 1, "double", OPS_MAX))
```
And the corresponding outlined elemental kernel is as follows.  Observe the stencil offsets used to access the adjacent 4 points:
```
void apply_stencil(const ACC<double> &A, ACC<double> &Anew, double *error) {
  Anew(0,0) = 0.25f * ( A(1,0) + A(-1,0)
      + A(0,-1) + A(0,1));
  *error = fmax( *error, fabs(Anew(0,0)-A(0,0)));
}
```
The loop also has a special argument for the reduction, `ops_arg_reduce`.  As the first argument, it takes a reduction handle, which has to be defined separately:
```
ops_reduction h_err = ops_decl_reduction_handle(sizeof(double), "double", "error");
``` 
Reductions may be increment (`OPS_INC`), min (`OPS_MIN`) or max (`OPS_MAX`). The user kernel will have to perform the reduction operation, reducing the passed in value as well as the computed value.

The result of the reduction can be queried from the handle as follows:
```
 ops_reduction_result(h_err, &error);
```

Multiple parallel loops may use the same handle, and their results will be combined, until the result is queried by the user.  Parallel loops that only have the reduction handle in common are semantically independent.

## Step 6 - Handing it all to OPS

We have now successfully converted all computations on the mesh to OPS parallel loops. In order for OPS to manage data and parallelisations better, we should let OPS allocate the datasets - instead of passing in the pointers to memory allocated by us, we just pass in NULL (`A` and `Anew`). Parallel I/O can be done using HDF5 - see the ops_hdf5.h header.

All data and parallelisation is now handed to OPS. We can now also compile the developer MPI version of the code - see the Makefile, and try building `laplace2d_mpi`.

## Step 7 - Code generation

Now that the developer versions of our code work, it’s time to generate code.  On the console, type: 
```
$OPSINSTALLPATH/../ops_translator/c/ops.py laplace2d.cpp
```
We have provided a Makefile which can use several different compilers (intel, cray, pgi, clang), we suggest modifying it for your own applications. Try building CUDA, OpenMP, MPI+CUDA, MPI+OpenMP, and other versions of the code. You can take a look at the generated kernels for different parallelisations under the appropriate subfolders. 

If you add the−`OPS_DIAGS=2` runtime flag, at the end of execution, OPS will report timings and achieved bandwidth for each of your kernels. For more options, see [Runtime Flags and Options](https://ops-dsl.readthedocs.io/en/markdowndocdev/devanapp.html#runtime-flags-and-options).


## Code generated versions
OPS will generate and compile a large number of different versions.
* `laplace2d_dev_seq` and `laplace2d_dev_mpi` :  these do not use code generation, they are intended for development only
* `laplace2d_seq` and `laplace2d_mpi` : baseline sequential and MPI implementations
* `laplace2d_openmp` : baseline OpenMP implementation
* `laplace2d_cuda`, `laplace2d_opencl`, `laplace2d_openacc` : implementations targeting GPUs 
* `laplace2d_mpiinline` : optimised implementation with MPI+OpenMP
* `laplace2d_tiled`: optimised implementation with OpenMP that improves spatial and temporal locality

## Optimizations - general
Try the following performance tuning options
* `laplace2d_cuda`, `laplace2d_opencl` : you can set the `OPS_BLOCK_SIZE_X` and `OPS_BLOCK_SIZE_Y` runtime arguments to control thread block or work group sizes 
* `laplace2d_mpi_cuda`, `laplace2d_mpi_openacc` : add the `-gpudirect` runtime flag to enable GPU Direct communications


## Optimizations - tiling

Tiling uses lazy execution: as parallel loops follow one another, they are not executed, but put in a queue, and only once some data needs to be returned to the user (e.g.  result of a reduction) do these loops have to be executed.

With a chain of loops queued, OPS can analyse them together and come up with a tiled execution schedule.

This works over MPI as well:  OPS extends the halo regions, and does one big halo exchange instead of several smaller ones. In the current `laplace2d` code, every stencil application loop is also doing a reduction, therefore only two loops are queued. Try modifying the code so the reduction only happens every 10 iterations ! On a Xeon E5-2650, one can get a 2.5x speedup.

The following versions can be executed with the tiling optimzations.

* `laplace2d_tiled`, `laplace2d_mpi_tiled` : add the `OPS_TILING` runtime flag, and move `-OPSDIAGS=3` to see the cache blocking tiling at work. For some applications, such as this one, the initial guess gives too large tiles, try setting `OPS_CACHE_SIZE` to a lower value (in MB, for L3 size).  Thread affinity control and using 1 process per socket isstrongly recommended.  E.g. `OMP_NUM_THREADS=20 numactl--cpunodebind=0 ./laplace2dtiled -OPSDIAGS=3 OPS_TILING OPS_CACHE_SIZE=5`. Over MPI, you will have to set `OPS_TILING_MAX_DEPTH` to extend halo regions.






