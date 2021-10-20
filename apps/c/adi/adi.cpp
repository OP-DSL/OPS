/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OPS distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/** @brief 3D heat diffusion PDE solved using ADI
 *  @author Endre Lazlo, converted to OPS by Gihan Mudalige
 *  @details PDE is solved with the ADI (Alternating Direction Implicit) method
 *  uses the Scalar tridiagonal solver for CPU and GPU written by Endre. Lazslo
**/

// standard headers
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

// OPS header file
#define OPS_3D
#include "ops_seq.h"

#ifdef OPS_MPI
#include "ops_mpi_core.h"
#endif

#include "data.h"

#include "init_kernel.h"
#include "preproc_kernel.h"

#include <mpi.h>
//#include "print_kernel.h"

extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"bx",   required_argument, 0,  0   },
  {"by",   required_argument, 0,  0   },
  {"bz",   required_argument, 0,  0   },
  {"m", required_argument, 0,  0   },
  {"iter", required_argument, 0,  0   },
  {"halo", required_argument, 0,  0   },
  {"t", no_argument, 0,  0   },
  {"help", no_argument,       0,  'h' },
  {0,      0,                 0,  0   }
};

void print_help() {
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER\n");
  exit(0);
}

typedef double APP_FP;
// writes the whole dataset to a file named as the executable_name.dat
void dump_data(APP_FP *data, const int nx, const int ny, const int nz,
               const int ldim, const char *filename) {
  // Set output filname to binary executable.dat
  char out_filename[256];
  strcpy(out_filename, filename);
  strcat(out_filename, ".dat");
  // print data to file
  FILE *fout;
  fout = fopen(out_filename, "w");
  if (fout == NULL) {
    printf(
        "ERROR: File stream could not be opened. Data will not be written to "
        "file!\n");
  } else {
    // fwrite(data,sizeof(float),(nx+STRIDE)*ny*nz,fout);
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          int ind = i + j * ldim + k * ldim * ny;
          // data[ind] = i + j*nx + k*nx*ny;
          // fprintf(fout, "%.10f ", data[ind]);
          // printf("%.10f ", data[ind]);
          fwrite(&data[ind], sizeof(APP_FP), 1, fout); // binary dump
        }
        // fprintf(fout, "\n");
      }
    }
    // fwrite(h_u,sizeof(float),nx*ny*nz,fout);
    fclose(fout);
  }
}

void dump_and_exit(APP_FP *data, const int nx, const int ny, const int nz,
                   const int ldim, const char *filename, const int iteration,
                   const int max_iteration) {
  dump_data(data, nx, ny, nz, ldim, filename);
  if (iteration == max_iteration) exit(0);
}

#ifdef OPS_MPI
void ignore_mpi_halo_rms(ops_dat dat) {
  double sum = 0.0;

  int host = OPS_HOST;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");
  const double *ptr = (double *)ops_dat_get_raw_pointer(dat, 0, S3D_000, &host);
  sub_dat *sd = OPS_sub_dat_list[dat->index];
  int pads_m[] = {-1 * (dat->d_m[0] + sd->d_im[0]), -1 * (dat->d_m[1] + sd->d_im[1]), -1 * (dat->d_m[2] + sd->d_im[2])};
  int pads_p[] = {dat->d_p[0] + sd->d_ip[0], dat->d_p[1] + sd->d_ip[1], dat->d_p[2] + sd->d_ip[2]};

  int dims[] = {dat->size[0] - pads_m[0] - pads_p[0],
                dat->size[1] - pads_m[1] - pads_p[1],
                dat->size[2] - pads_m[2] - pads_p[2]};

  for(int z = 0; z < dims[2]; z++) {
    for(int y = 0; y < dims[1]; y++) {
      for(int x = 0; x < dims[0]; x++) {
        int offset = z * dat->size[1] * dat->size[0];
        offset += y * dat->size[0];
        offset += x;
        sum += ptr[offset];
      }
    }
  }

  ops_dat_release_raw_data(dat, 0, OPS_READ);

  double global_sum = 0.0;
  MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  ops_printf("Sum: %.15g\n", global_sum);
}
#endif

#ifndef OPS_MPI
void zero_halo(ops_dat dat) {
  int host = OPS_HOST;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");
  double *ptr = (double *)ops_dat_get_raw_pointer(dat, 0, S3D_000, &host);
  int pads_m[] = {-1 * (dat->d_m[0]), -1 * (dat->d_m[1]), -1 * (dat->d_m[2])};
  int pads_p[] = {dat->d_p[0], dat->d_p[1], dat->d_p[2]};

  int dims[] = {dat->size[0] - pads_m[0] - pads_p[0],
                dat->size[1] - pads_m[1] - pads_p[1],
                dat->size[2] - pads_m[2] - pads_p[2]};

  for(int z = dims[2]; z < dims[2] + pads_p[2]; z++) {
    for(int y = dims[1]; y < dims[1] + pads_p[1]; y++) {
      for(int x = dims[0]; x < dims[0] + pads_p[0]; x++) {
        int offset = z * dat->size[1] * dat->size[0];
        offset += y * dat->size[0];
        offset += x;
        ptr[offset] = 0.0;
      }
    }
  }

  ops_dat_release_raw_data(dat, 0, OPS_WRITE);
}
#endif

#ifndef OPS_MPI
int check_halo_is_zero(ops_dat dat) {
  int host = OPS_HOST;
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");
  double *ptr = (double *)ops_dat_get_raw_pointer(dat, 0, S3D_000, &host);
  int pads_m[] = {-1 * (dat->d_m[0]), -1 * (dat->d_m[1]), -1 * (dat->d_m[2])};
  int pads_p[] = {dat->d_p[0], dat->d_p[1], dat->d_p[2]};

  int dims[] = {dat->size[0] - pads_m[0] - pads_p[0],
                dat->size[1] - pads_m[1] - pads_p[1],
                dat->size[2] - pads_m[2] - pads_p[2]};

  for(int z = dims[2]; z < dims[2] + pads_p[2]; z++) {
    for(int y = dims[1]; y < dims[1] + pads_p[1]; y++) {
      for(int x = dims[0]; x < dims[0] + pads_p[0]; x++) {
        int offset = z * dat->size[1] * dat->size[0];
        offset += y * dat->size[0];
        offset += x;
        if(ptr[offset] != 0.0) {
          ops_dat_release_raw_data(dat, 0, OPS_READ);
          return false;
        }
      }
    }
  }

  ops_dat_release_raw_data(dat, 0, OPS_READ);
  return true;
}
#endif

// declare defaults options
int nx;
int ny;
int nz;
int ldim;
int iter;
int opts[3], pads[3], synch;
int bx, by, bz;
int m;
int halo;
int t;

// declare constants
double lambda;

int main(int argc, char *argv[]) {
  // Set defaults options
  nx = 256;
  ny = 256;
  nz = 256;
  opts[0] = 0;
  opts[1] = 0;
  opts[2] = 0;
  iter = 10;
  synch = 1;
  bx = 16384;
  by = 16384;
  bz = 16384;
  m = 0;
  halo = 1;
  t = 0;

  // constants
  lambda = 1.0f;

  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"nx"  ) == 0) nx = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"ny"  ) == 0) ny = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"nz"  ) == 0) nz = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"bx"  ) == 0) bx = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"by"  ) == 0) by = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"bz"  ) == 0) bz = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"m"  ) == 0) m = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"iter") == 0) iter = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"halo") == 0) halo = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"t") == 0) t = 1;
    if(strcmp((char*)options[opt_index].name,"help") == 0) print_help();
  }

  /**--- Initialisation----**/

  // OPS initialisation
  ops_init(argc, argv, 2);

 /**--- OPS declarations----**/
  // declare block
  ops_block heat3D = ops_decl_block(3, "Heat3D");

  // declare data on blocks
  // max halo depths for the dat in the positive direction
  int d_p[3] = {halo, halo, halo};
  // max halo depths for the dat in the negative direction*/
  int d_m[3] = {-1 * halo, -1 * halo, -1 * halo};

  int size[3] = {nx, ny, nz};  // size of the dat -- should be identical to the
                               // block on which its define on
  int base[3] = {0, 0, 0};
  double *temp = NULL;

  ops_dat h_u =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_u");
  ops_dat h_du =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_du");
  ops_dat h_ax =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_ax");
  ops_dat h_bx =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_bx");
  ops_dat h_cx =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_cx");
  ops_dat h_ay =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_ay");
  ops_dat h_by =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_by");
  ops_dat h_cy =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_cy");
  ops_dat h_az =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_az");
  ops_dat h_bz =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_bz");
  ops_dat h_cz =
      ops_decl_dat(heat3D, 1, size, base, d_m, d_p, temp, "double", "h_cz");

  // declare stencils
  int s3D_000[] = {0, 0, 0};
  ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");

  int s3D_7pt[] = {0, 0, 0, -1, 0, 0, 1,  0, 0, 0, -1,
                   0, 0, 1, 0,  0, 0, -1, 0, 0, 1};
  ops_stencil S3D_7PT = ops_decl_stencil(3, 7, s3D_7pt, "3d7Point");

  // declare constants
  ops_decl_const("nx", 1, "int", &nx);
  ops_decl_const("ny", 1, "int", &ny);
  ops_decl_const("nz", 1, "int", &nz);
  ops_decl_const("lambda", 1, "double", &lambda);

  // decompose the block
  ops_partition("2D_BLOCK_DECOMPSE");

  // compute tridiagonal system sizes
  double ct0, ct1, et0, et1, ct2, et2, ct3, et3;
  double total_preproc, total_x, total_y, total_z;
  total_preproc = total_x = total_y = total_z = 0.0;

  ops_printf("\nNumber of iterations: %d\n", iter);
  ops_printf("\nGrid dimensions: %d x %d x %d\n", nx, ny, nz);
  printf("\nLocal dimensions: %d x %d x %d\n", h_u->size[0], h_u->size[1], h_u->size[2]);
  ops_diagnostic_output();

  /**-------- Initialize-------**/
  int iter_range[] = {0, nx, 0, ny, 0, nz};
  ops_par_loop(init_kernel, "init_kernel", heat3D, 3, iter_range,
               ops_arg_dat(h_u, 1, S3D_000, "double", OPS_WRITE),
               ops_arg_idx());

  ops_timers(&ct0, &et0);

  ops_tridsolver_params::SolveStrategy strat;
  switch (m) {
    case 0:
      strat = ops_tridsolver_params::GATHER_SCATTER;
      break;
    case 1:
      strat = ops_tridsolver_params::ALLGATHER;
      break;
    case 2:
      strat = ops_tridsolver_params::LATENCY_HIDING_TWO_STEP;
      break;
    case 3:
      strat = ops_tridsolver_params::LATENCY_HIDING_INTERLEAVED;
      break;
    case 4:
      strat = ops_tridsolver_params::JACOBI;
      break;
    case 5:
      strat = ops_tridsolver_params::PCR;
      break;
  }
  ops_tridsolver_params *trid_ctx_x = new ops_tridsolver_params(heat3D, strat);
  ops_tridsolver_params *trid_ctx_y = new ops_tridsolver_params(heat3D, strat);
  ops_tridsolver_params *trid_ctx_z = new ops_tridsolver_params(heat3D, strat);

  trid_ctx_x->set_batch_size(bx);
  trid_ctx_x->set_batch_size(by);
  trid_ctx_x->set_batch_size(bz);

  for (int it = 0; it < iter; it++) {  // Start main iteration loop

    /**-----calculate r.h.s. and set tri-diagonal
     * -----coefficients-----------**/
    int iter_range[] = {0, nx, 0, ny, 0, nz};

    ops_timers(&ct2, &et2);
    ops_par_loop(preproc_kernel, "preproc_kernel", heat3D, 3, iter_range,
                 ops_arg_dat(h_u, 1, S3D_7PT, "double", OPS_READ),
                 ops_arg_dat(h_du, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_ax, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_bx, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_cx, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_ay, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_by, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_cy, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_az, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_bz, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_dat(h_cz, 1, S3D_000, "double", OPS_WRITE),
                 ops_arg_idx());
    ops_timers(&ct3, &et3);
    total_preproc += et3 - et2;
    //ops_printf("Elapsed preproc (sec): %lf (s)\n", et3 - et2);

    /**---- perform tri-diagonal solves in x-direction--**/
    ops_timers(&ct2, &et2);
    ops_tridMultiDimBatch(3, 0, size, h_ax, h_bx, h_cx, h_du, h_u, trid_ctx_x);
    ops_timers(&ct3, &et3);
    total_x += et3 - et2;
    //ops_printf("Elapsed trid_x (sec): %lf (s)\n", et3 - et2);

    /**---- perform tri-diagonal solves in y-direction--**/
    ops_timers(&ct2, &et2);
    ops_tridMultiDimBatch(3, 1, size, h_ay, h_by, h_cy, h_du, h_u, trid_ctx_y);
    ops_timers(&ct3, &et3);
    total_y += et3 - et2;
    //ops_printf("Elapsed trid_y (sec): %lf (s)\n", et3 - et2);

    /**---- perform tri-diagonal solves in z-direction--**/
    ops_timers(&ct2, &et2);
    ops_tridMultiDimBatch_Inc(3, 2, size, h_az, h_bz, h_cz, h_du, h_u, trid_ctx_z);
    ops_timers(&ct3, &et3);
    total_z += et3 - et2;
    //ops_printf("Elapsed trid_z (sec): %lf (s)\n", et3 - et2);
  }  // End main iteration loop

  delete trid_ctx_x;
  delete trid_ctx_y;
  delete trid_ctx_z;

  ops_timers(&ct1, &et1);

#ifndef OPS_MPI
  if(check_halo_is_zero(h_u)) {
    ops_printf("Halo Test: PASSED\n");
  } else {
    ops_printf("Halo Test: FAILED\n");
  }
#endif

  /**---- dump solution to HDF5 file with OPS-**/
  ops_fetch_block_hdf5_file(heat3D, "adi.h5");
  ops_fetch_dat_hdf5_file(h_u, "adi.h5");

  // Print out a padded h5 file (assuming that no padding is currently being used)
  // Used for test script, should only be used on sequential solvers
  // (MPI halo will be non zero)
#ifndef OPS_MPI
  if(t) {
    int dp[3] = {1, 1, 1};
    int dm[3] = {-1, -1, -1};
    ops_dat h_u_pad =
        ops_decl_dat(heat3D, 1, size, base, dm, dp, temp, "double", "h_u");
    zero_halo(h_u_pad);
    int host = OPS_HOST;
    int s3D_000[] = {0, 0, 0};
    ops_stencil S3D_000 = ops_decl_stencil(3, 1, s3D_000, "000");
    double *ptr_pad = (double *)ops_dat_get_raw_pointer(h_u_pad, 0, S3D_000, &host);
    double *ptr = (double *)ops_dat_get_raw_pointer(h_u, 0, S3D_000, &host);
    int pads_m[] = {-1 * (h_u_pad->d_m[0]), -1 * (h_u_pad->d_m[1]), -1 * (h_u_pad->d_m[2])};
    int pads_p[] = {h_u_pad->d_p[0], h_u_pad->d_p[1], h_u_pad->d_p[2]};

    int dims[] = {h_u_pad->size[0] - pads_m[0] - pads_p[0],
                  h_u_pad->size[1] - pads_m[1] - pads_p[1],
                  h_u_pad->size[2] - pads_m[2] - pads_p[2]};

    for(int z = 0; z < dims[2]; z++) {
      for(int y = 0; y < dims[1]; y++) {
        for(int x = 0; x < dims[0]; x++) {
          int offset_pad = z * h_u_pad->size[1] * h_u_pad->size[0];
          offset_pad += y * h_u_pad->size[0];
          offset_pad += x;
          int offset = z * dims[1] * dims[0] + y * dims[0] + x;
          ptr_pad[offset_pad] = ptr[offset];
        }
      }
    }

    ops_dat_release_raw_data(h_u, 0, OPS_READ);
    ops_dat_release_raw_data(h_u_pad, 0, OPS_WRITE);
    ops_fetch_block_hdf5_file(heat3D, "adi_pad.h5");
    ops_fetch_dat_hdf5_file(h_u_pad, "adi_pad.h5");
  }
#endif

#ifdef OPS_MPI
  ignore_mpi_halo_rms(h_du);
  ignore_mpi_halo_rms(h_u);
#else
  ldim = nx; // non padded size along x
  // dump the whole raw matrix
  dump_data((double *)(h_u->data), nx, ny, nz, ldim, argv[0]);
#endif

  ops_printf("\nTotal Wall time (s): %lf\n", et1 - et0);
  ops_printf("Preproc total time (s): %lf\n", total_preproc);
  ops_printf("X Dim total time (s): %lf\n", total_x);
  ops_printf("Y Dim total time (s): %lf\n", total_y);
  ops_printf("Z Dim total time (s): %lf\n", total_z);
  ops_exit();
}
