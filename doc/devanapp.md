# Developing an OPS Application
This page provides a tutorial in the basics of using OPS for multi-block structured mesh application development. This is taken from a [presentation](https://op-dsl.github.io/docs/OPS/tutorial.pdf) given initially in April 2018 and subsequently updated for the latest release of OPS. 

## OPS Abstraction
OPS is a Domain Specific Language embedded in C/C++ and Fortran, targeting the development of multi-block structured mesh computations. The abstraction has two distinct components:  the definition of the mesh, and operations over the mesh.
* Defining a number of 1-3D blocks, and on them a number of datasets, which have specific extents in the different dimensions.
* Describing a parallel loop over a given block, with a given iteration range, executing a given "kernel function" at each grid point, and describing what datasets are going to be accessed and how.
* Additionally, one needs to declare stencils (access patterns) that will be used in parallel loops to access datasets, and any global constants (read-only global scope variables)

Data and computations expressed this way can be automatically managed and parallelised by the OPS library. Higher dimensions are supported in the backend, but not currently by the code generators.

## Example Application
In this tutorial we will use an example application, a simple 2D iterative Laplace equation solver. 
* Go to the `OPS/apps/c/laplace2dtutorial/original` directory
* Open the `laplace2d.cpp` file
* It uses an $imax$ x $jmax$ grid, with an additional 1 layers of boundary cells on all sides
* There are a number of loops that set the boundary conditions along the four edges
* The bulk of the simulation is spent in a whilel oop, repeating a stencil kernel with a maximum reduction, and a copy kernel
* Compile and run the code !

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
## Step 2 - OPS declarations
## Step 3 - First parallel loop
## Step 4 - Indexes and global constants
## Step 5 - Complex stencils and reductions
## Step 6 - Handing it all to OPS
## Step 7 - Code generation
## Code generated versions
## Optimizations - general
## Optimizations - tiling

## Supported Paralleizations
## Code-generation Flags
## Runtime Flags and Options
