void write_constants(const char* filename){
ops_write_const_hdf5("block0np0", 1, "int", (char*)&block0np0, filename);
ops_write_const_hdf5("Delta0block0", 1, "double", (char*)&Delta0block0, filename);
ops_write_const_hdf5("HDF5_timing", 1, "int", (char*)&HDF5_timing, filename);
ops_write_const_hdf5("c0", 1, "double", (char*)&c0, filename);
ops_write_const_hdf5("niter", 1, "int", (char*)&niter, filename);
ops_write_const_hdf5("dt", 1, "double", (char*)&dt, filename);
ops_write_const_hdf5("simulation_time", 1, "double", (char*)&simulation_time, filename);
ops_write_const_hdf5("start_iter", 1, "int", (char*)&start_iter, filename);
ops_write_const_hdf5("iter", 1, "int", (char*)&iter, filename);
}

