#include "constants.h"

void HDF5_IO_Write_0_opensbliblock00(ops_block& opensbliblock00, ops_dat& phi_B0, ops_dat& x0_B0, int HDF5_timing){
double cpu_start0, elapsed_start0;
if (HDF5_timing == 1){
ops_timers(&cpu_start0, &elapsed_start0);
}
// Writing OPS datasets
char name0[80];
sprintf(name0, "opensbli_output.h5");
ops_fetch_block_hdf5_file(opensbliblock00, name0);
ops_fetch_dat_hdf5_file(phi_B0, name0);
ops_fetch_dat_hdf5_file(x0_B0, name0);
// Writing simulation constants
write_constants(name0);
if (HDF5_timing == 1){
double cpu_end0, elapsed_end0;
ops_timers(&cpu_end0, &elapsed_end0);
ops_printf("-----------------------------------------\n");
ops_printf("Time to write HDF5 file: %s: %lf\n", name0, elapsed_end0-elapsed_start0);
ops_printf("-----------------------------------------\n");
fflush(stdout);
}
}

