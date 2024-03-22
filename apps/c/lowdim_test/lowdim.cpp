/* Code to test the functionality of HDF5 IO in OPS
 * 
 * this reads two 2D datasets from HDF5 on a 3D block and then adds a number to the
 * dataset and writes out the HDF5 file
 * 
 * @author Satya P Jammy
 * 
 * The base is generated using OpenSBLI
 */
#include <stdlib.h> 
#include <string.h> 
#include <math.h>
#define OPS_3D
#include "ops_seq_v2.h"
#include "lowdim_kernels.h"

int main(int argc, char **argv) 
{
  // Initializing OPS 
  ops_init(argc,argv,1);

  // Define and Declare OPS Block
  ops_block block = ops_decl_block(3, "block");
  int halo_p[] = {1, 1, 1};
  int halo_m[] = {-1, -1, -1};
  int size[] = {10,10,10};
  int base[] = {0, 0, 0};
  double* value = NULL;
  ops_dat dat3D = ops_decl_dat(block, 1, size, base, halo_m, halo_p, value, "double", "dat3D");

  halo_p[0] = 1; halo_p[1] = 1; halo_p[2] = 0;
  halo_m[0] = -1; halo_m[1] = -1; halo_m[2] = 0;
  size[0] = 10; size[1] = 10; size[2] = 1;
  ops_dat dat2D_XY = ops_decl_dat(block, 1, size, base, halo_m, halo_p, value, "double", "dat2D_XY");
  halo_p[0] = 0; halo_p[1] = 1; halo_p[2] = 1;
  halo_m[0] = 0; halo_m[1] = -1; halo_m[2] = -1;
  size[0] = 1; size[1] = 10; size[2] = 10;
  ops_dat dat2D_YZ = ops_decl_dat(block, 1, size, base, halo_m, halo_p, value, "double", "dat2D_YZ");
  halo_p[0] = 1; halo_p[1] = 0; halo_p[2] = 1;
  halo_m[0] = -1; halo_m[1] = 0; halo_m[2] = -1;
  size[0] = 10; size[1] = 1; size[2] = 10;
  ops_dat dat2D_XZ = ops_decl_dat(block, 1, size, base, halo_m, halo_p, value, "double", "dat2D_XZ");

  halo_p[0] = 1; halo_p[1] = 0; halo_p[2] = 0;
  halo_m[0] = -1; halo_m[1] = 0; halo_m[2] = 0;
  size[0] = 10; size[1] = 1; size[2] = 1;
  ops_dat dat1D_X = ops_decl_dat(block, 1, size, base, halo_m, halo_p, value, "double", "dat1D_X");
  halo_p[0] = 0; halo_p[1] = 1; halo_p[2] = 0;
  halo_m[0] = 0; halo_m[1] = -1; halo_m[2] = 0;
  size[0] = 1; size[1] = 10; size[2] = 1;
  ops_dat dat1D_Y = ops_decl_dat(block, 1, size, base, halo_m, halo_p, value, "double", "dat1D_Y");
  halo_p[0] = 0; halo_p[1] = 0; halo_p[2] = 1;
  halo_m[0] = 0; halo_m[1] = 0; halo_m[2] = -1;
  size[0] = 1; size[1] = 1; size[2] = 10;
  ops_dat dat1D_Z = ops_decl_dat(block, 1, size, base, halo_m, halo_p, value, "double", "dat1D_Z");

  // Define and declare stencils
  int s3D_000[] = {0, 0, 0};
  int stride3D_x[] = {1,0,0};
  int stride3D_y[] = {0,1,0};
  int stride3D_z[] = {0,0,1};
  ops_stencil S3D_000           = ops_decl_stencil(3,1,s3D_000,"S3D_000");
  ops_stencil S3D_000_STRID3D_X = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_x, "s2D_000_stride3D_x");
  ops_stencil S3D_000_STRID3D_Y = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_y, "s2D_000_stride3D_y");
  ops_stencil S3D_000_STRID3D_Z = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_z, "s2D_000_stride3D_z");
  int stride3D_xy[] = {1,1,0};
  int stride3D_yz[] = {0,1,1};
  int stride3D_xz[] = {1,0,1};
  ops_stencil S3D_000_STRID3D_XY = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_xy, "s2D_000_stride3D_xy");
  ops_stencil S3D_000_STRID3D_YZ = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_yz, "s2D_000_stride3D_yz");
  ops_stencil S3D_000_STRID3D_XZ = ops_decl_strided_stencil( 3, 1, s3D_000, stride3D_xz, "s2D_000_stride3D_xz");


  // Init OPS partition
  ops_partition("");


  double val = 0.0;
  int range_3D[] = {0, 10, 0, 10, 0, 10};
  ops_par_loop(set_val, "set_val", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  val = 1.0;
  int range_2D_XY[] = {0, 10, 0, 10, 0, 1};
  ops_par_loop(set_val, "set_val", block, 3, range_2D_XY,
      ops_arg_dat(dat2D_XY, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  val = 2.0;
  int range_2D_YZ[] = {0, 1, 0, 10, 0, 10};
  ops_par_loop(set_val, "set_val", block, 3, range_2D_YZ,
      ops_arg_dat(dat2D_YZ, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  val = 3.0;
  int range_2D_XZ[] = {0, 10, 0, 1, 0, 10};
  ops_par_loop(set_val, "set_val", block, 3, range_2D_XZ,
      ops_arg_dat(dat2D_XZ, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  val = 4.0;
  int range_1D_X[] = {0, 10, 0, 1, 0, 1};
  ops_par_loop(set_val, "set_val", block, 3, range_1D_X,
      ops_arg_dat(dat1D_X, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  val = 5.0;
  int range_1D_Y[] = {0, 1, 0, 10, 0, 1};
  ops_par_loop(set_val, "set_val", block, 3, range_1D_Y,
      ops_arg_dat(dat1D_Y, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  val = 6.0;
  int range_1D_Z[] = {0, 1, 0, 1, 0, 10};
  ops_par_loop(set_val, "set_val", block, 3, range_1D_Z,
      ops_arg_dat(dat1D_Z, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  // Now access them with strided stencils
  ops_par_loop(calc, "calc", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_READ),
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_READ),
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_READ),
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_READ),
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_READ),
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_READ));


  ops_dump_to_hdf5("output.h5");
  ops_printf("PASSED");

  ops_exit();
  return 0;
  //Main program end 
}
