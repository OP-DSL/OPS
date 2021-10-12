# Developing an OPS Application
This page provides a tutorial in the basics of using OPS for multi-block structured mesh application development. This is taken from a [presentation](https://op-dsl.github.io/docs/OPS/tutorial.pdf) given initially in April 2018 and subsequently updated for the latest release of OPS. 

## OPS Abstraction
OPS is a Domain Specific Language embedded in C/C++ and Fortran, targeting the development of multi-block structured mesh computations. The abstraction has two distinct components:  the definition of the mesh, and operations over the mesh.
* Defining a number of 1-3D blocks, and on them a number of datasets, which have specific extents in the different dimensions.
* Describing a parallel loop over a given block, with a given iteration range, executing a given "kernel function" at each grid point, and describing what datasets are going to be accessed and how.
* Additionally, one needs to declare stencils (access patterns) that will be used in parallel loops to access datasets, and any global constants (read-only global scope variables)

Data and computations expressed this way can be automatically managed and parallelised by the OPS library. Higher dimensions supported in the backend, but not currently by the code generators.

## Example Application
## Original - Initialisation
## Original - Boundary loops
## Original - Main iteration
## Build OPS
## Step 1 - Preparing to use OPS
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
