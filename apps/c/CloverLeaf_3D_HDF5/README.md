Same as CloverLeaf_3D but uses HDF5 files to read in the values of the initial arrays.
Maintained here as a test application to exercise HDF5 file I/O functionality.
Compile and run ``generate_file`` (generate_file.cpp) to obtain the required
hdf5 file (for a given clover.in input deck) before running any of the cloverleaf_* applications.

I.e. do the following to run for example the mpi version of CloverLeaf_3D
```
make generate_file
./generate_file
make cloverleaf_mpi
mpirun -np 20 cloverleaf_mpi
```

See CloverLeaf_3D directory for details on compiling the CloverLeaf_3D applications
