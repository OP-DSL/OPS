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


void checkError(int error_count, const char* test_name) {
  if (error_count > 0)
  {
    ops_printf("%d errors found in %s\n", error_count, test_name);
    ops_printf("TEST FAILED\n");
    ops_exit();
    exit(1);
  }
}


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


  ops_reduction reduct_err = ops_decl_reduction_handle(sizeof(int), "int", "reduct_err");

  // Init OPS partition
  ops_partition("");


  
  int range_3D[] = {0, 10, 0, 10, 0, 10};
  int range_2D_XY[] = {0, 10, 0, 10, 0, 1};
  int range_2D_YZ[] = {9, 10, 0, 10, 0, 10};
  int range_2D_XZ[] = {0, 10, 5, 6, 0, 10};

  size[0] = 10; size[1] = 10; size[2] = 10;
  ops_par_loop(set3D, "set3D", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx());

 
  ops_printf("Check OPS_MAX on 2D directions...\n");
  
  double val= -INFINITY_double;
  
  ops_par_loop(set_valXY, "set_valXY", block, 3, range_2D_XY,
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valYZ, "set_valYZ", block, 3, range_2D_YZ,
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valXZ, "set_valXZ", block, 3, range_2D_XZ,
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));
      
  // Reduction to lower dimension
  ops_par_loop(reduct22D_max, "reduct22D_max", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_READ),
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_MAX),
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_MAX),
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_MAX));


  int error_count = 0;
  ops_par_loop(check2D_XY_max, "check2D_XY_max", block, 3, range_2D_XY,
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_XY_max");

  ops_par_loop(check2D_XZ_max, "check2D_XZ_max", block, 3, range_2D_XZ,
    ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_READ),
    ops_arg_gbl(size, 3, "int", OPS_READ),
    ops_arg_idx(),
    ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_XZ_max");

  
  ops_par_loop(check2D_YZ_max, "check2D_YZ_max", block, 3, range_2D_YZ,
    ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_READ),
    ops_arg_gbl(size, 3, "int", OPS_READ),
    ops_arg_idx(),
    ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_YZ_max");

  ops_printf("OPS_MAX on 2D directions OK\n");

  
  ops_printf("Check OPS_MIN on 2D directions...\n");
  
  val= INFINITY_double;
  ops_par_loop(set_valXY, "set_valXY", block, 3, range_2D_XY,
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valYZ, "set_valYZ", block, 3, range_2D_YZ,
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valXZ, "set_valXZ", block, 3, range_2D_XZ,
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));
      
  // Reduction to lower dimension
  ops_par_loop(reduct22D_min, "reduct22D_min", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_READ),
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_MIN),
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_MIN),
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_MIN));


  error_count = 0;
  ops_par_loop(check2D_XY_min, "check2D_XY_min", block, 3, range_2D_XY,
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_XY_min");

  ops_par_loop(check2D_XZ_min, "check2D_XZ_min", block, 3, range_2D_XZ,
    ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_READ),
    ops_arg_gbl(size, 3, "int", OPS_READ),
    ops_arg_idx(),
    ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_XZ_min");

  
  ops_par_loop(check2D_YZ_min, "check2D_YZ_min", block, 3, range_2D_YZ,
    ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_READ),
    ops_arg_gbl(size, 3, "int", OPS_READ),
    ops_arg_idx(),
    ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_YZ_min");

  ops_printf("OPS_MIN on 2D directions OK\n");


  
  ops_printf("Check OPS_INC on 2D directions...\n");
  
  val= 0.0;
  ops_par_loop(set_valXY, "set_valXY", block, 3, range_2D_XY,
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valYZ, "set_valYZ", block, 3, range_2D_YZ,
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valXZ, "set_valXZ", block, 3, range_2D_XZ,
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));
      
  // Reduction to lower dimension
  ops_par_loop(reduct22D_inc, "reduct22D_inc", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_READ),
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_INC),
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_INC),
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_INC));


  error_count = 0;
  ops_par_loop(check2D_XY_inc, "check2D_XY_inc", block, 3, range_2D_XY,
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_XY_inc");

  ops_par_loop(check2D_XZ_inc, "check2D_XZ_inc", block, 3, range_2D_XZ,
    ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_READ),
    ops_arg_gbl(size, 3, "int", OPS_READ),
    ops_arg_idx(),
    ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_XZ_inc");

  
  ops_par_loop(check2D_YZ_inc, "check2D_YZ_inc", block, 3, range_2D_YZ,
    ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_READ),
    ops_arg_gbl(size, 3, "int", OPS_READ),
    ops_arg_idx(),
    ops_arg_reduce(reduct_err, 1, "int", OPS_INC));


  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check2D_YZ_inc");

  ops_printf("OPS_INC on 2D directions OK\n");




  int range_1D_X[] = {0, 10, 0, 1, 5, 6};
  int range_1D_Y[] = {2, 3, 0, 10, 9, 10};
  int range_1D_Z[] = {0, 1, 9, 10, 0, 10};
 
  ops_printf("Check OPS_MAX on 1D directions...\n");
  
  val= -INFINITY_double;

  ops_par_loop(set_valX, "set_valX", block, 3, range_1D_X,
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valY, "set_valY", block, 3, range_1D_Y,
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valZ, "set_valZ", block, 3, range_1D_Z,
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));



  ops_par_loop(reduct21D_max, "reduct21D_max", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_READ),
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_MAX),
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_MAX),
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_MAX));

  ops_par_loop(check1D_X_max, "check1D_X_max", block, 3, range_1D_X,
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_X_max");
  
  
  ops_par_loop(check1D_Y_max, "check1D_Y_max", block, 3, range_1D_Y,
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_Y_max");
  

  ops_par_loop(check1D_Z_max, "check1D_Z_max", block, 3, range_1D_Z,
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_Z_max");

  ops_printf("OPS_MAX on 1D directions OK\n");


  ops_printf("Check OPS_MIN on 1D directions...\n");

  val= INFINITY_double;

  ops_par_loop(set_valX, "set_valX", block, 3, range_1D_X,
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valY, "set_valY", block, 3, range_1D_Y,
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valZ, "set_valZ", block, 3, range_1D_Z,
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));



  ops_par_loop(reduct21D_min, "reduct21D_min", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_READ),
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_MIN),
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_MIN),
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_MIN));




  ops_par_loop(check1D_X_min, "check1D_X_min", block, 3, range_1D_X,
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_X_min");
  
  
  ops_par_loop(check1D_Y_min, "check1D_Y_min", block, 3, range_1D_Y,
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_Y_min");
  

  ops_par_loop(check1D_Z_min, "check1D_Z_min", block, 3, range_1D_Z,
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_Z_min");

  ops_printf("OPS_MIN on 1D directions OK\n");


  ops_printf("Check OPS_INC on 1D directions...\n");
 
  val= 0.0;

  ops_par_loop(set_valX, "set_valX", block, 3, range_1D_X,
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valY, "set_valY", block, 3, range_1D_Y,
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));

  ops_par_loop(set_valZ, "set_valZ", block, 3, range_1D_Z,
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ));


  ops_par_loop(reduct21D_inc, "reduct21D_inc", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_READ),
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_INC),
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_INC),
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_INC));


  ops_par_loop(check1D_X_inc, "check1D_X_inc", block, 3, range_1D_X,
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_X_inc");
  
  
  ops_par_loop(check1D_Y_inc, "check1D_Y_inc", block, 3, range_1D_Y,
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_Y_inc");
  

  ops_par_loop(check1D_Z_inc, "check1D_Z_inc", block, 3, range_1D_Z,
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check1D_Z_inc");

  ops_printf("OPS_INC on 1D directions OK\n");


  ops_printf("Check reading from strided dats...\n");

  val=1.0;
  ops_par_loop(set_valXY_idx, "set_valXY_idx", block, 3, range_2D_XY,
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx());

  val=100.0;
  ops_par_loop(set_valYZ_idx, "set_valYZ_idx", block, 3, range_2D_YZ,
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx());
  
  val=10000.0;
  ops_par_loop(set_valXZ_idx, "set_valXZ_idx", block, 3, range_2D_XZ,
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx());
      
  val=0.01;
  ops_par_loop(set_valX_idx, "set_valX_idx", block, 3, range_1D_X,
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ),
      ops_arg_idx());

  val=0.0001;
  ops_par_loop(set_valY_idx, "set_valY_idx", block, 3, range_1D_Y,
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ),
      ops_arg_idx());
  
  val=0.000001;
  ops_par_loop(set_valZ_idx, "set_valZ_idx", block, 3, range_1D_Z,
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_WRITE),
      ops_arg_gbl(&val, 1, "double", OPS_READ),
      ops_arg_idx());

  // Now access them with strided stencils
  ops_par_loop(calc, "calc", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_WRITE),
      ops_arg_dat(dat2D_XY, 1, S3D_000_STRID3D_XY, "double", OPS_READ),
      ops_arg_dat(dat2D_YZ, 1, S3D_000_STRID3D_YZ, "double", OPS_READ),
      ops_arg_dat(dat2D_XZ, 1, S3D_000_STRID3D_XZ, "double", OPS_READ),
      ops_arg_dat(dat1D_X, 1, S3D_000_STRID3D_X, "double", OPS_READ),
      ops_arg_dat(dat1D_Y, 1, S3D_000_STRID3D_Y, "double", OPS_READ),
      ops_arg_dat(dat1D_Z, 1, S3D_000_STRID3D_Z, "double", OPS_READ));

  ops_par_loop(check_3D, "check_3D", block, 3, range_3D,
      ops_arg_dat(dat3D, 1, S3D_000, "double", OPS_READ),
      ops_arg_gbl(size, 3, "int", OPS_READ),
      ops_arg_idx(),
      ops_arg_reduce(reduct_err, 1, "int", OPS_INC));

  ops_reduction_result(reduct_err, &error_count);

  checkError(error_count, "check_3D");
      

  ops_printf("Calc done\n");
  char name0[80];
  sprintf(name0, "output.h5");
  ops_fetch_block_hdf5_file(block, name0);
  ops_fetch_dat_hdf5_file(dat2D_XZ, name0);
  ops_fetch_dat_hdf5_file(dat2D_XY, name0);
  ops_fetch_dat_hdf5_file(dat2D_YZ, name0);
  ops_fetch_dat_hdf5_file(dat1D_X, name0);
  ops_fetch_dat_hdf5_file(dat1D_Y, name0);
  ops_fetch_dat_hdf5_file(dat1D_Z, name0);
  ops_fetch_dat_hdf5_file(dat3D, name0);


  ops_printf("All checks PASSED\n");

  ops_exit();
  return 0;
  //Main program end 
}
