//
// auto-generated by ops.py
//
#ifdef _OPENMP
#include <omp.h>
#endif

// user function
inline void initialise_chunk_kernel_zero(double *var) { *var = 0.0; }

// host stub function
void ops_par_loop_initialise_chunk_kernel_zero(char const *name,
                                               ops_block block, int dim,
                                               int *range, ops_arg arg0) {

  // Timing
  double t1, t2, c1, c2;

  int offs[1][2];
  ops_arg args[1] = {arg0};

#ifdef CHECKPOINTING
  if (!ops_checkpointing_before(args, 1, range, 5))
    return;
#endif

  if (OPS_diags > 1) {
    ops_timing_realloc(5, "initialise_chunk_kernel_zero");
    OPS_kernels[5].count++;
    ops_timers_core(&c1, &t1);
  }

#ifdef OPS_MPI
  sub_block_list sb = OPS_sub_block_list[block->index];
#endif

  // compute locally allocated range for the sub-block

  int start[2];
  int end[2];
  int arg_idx[2];

#ifdef OPS_MPI
  if (!sb->owned)
    return;
  for (int n = 0; n < 2; n++) {
    start[n] = sb->decomp_disp[n];
    end[n] = sb->decomp_disp[n] + sb->decomp_size[n];
    if (start[n] >= range[2 * n]) {
      start[n] = 0;
    } else {
      start[n] = range[2 * n] - start[n];
    }
    if (sb->id_m[n] == MPI_PROC_NULL && range[2 * n] < 0)
      start[n] = range[2 * n];
    if (end[n] >= range[2 * n + 1]) {
      end[n] = range[2 * n + 1] - sb->decomp_disp[n];
    } else {
      end[n] = sb->decomp_size[n];
    }
    if (sb->id_p[n] == MPI_PROC_NULL &&
        (range[2 * n + 1] > sb->decomp_disp[n] + sb->decomp_size[n]))
      end[n] += (range[2 * n + 1] - sb->decomp_disp[n] - sb->decomp_size[n]);
    if (end[n] < start[n])
      end[n] = start[n];
  }
#else
  for (int n = 0; n < 2; n++) {
    start[n] = range[2 * n];
    end[n] = range[2 * n + 1];
  }
#endif
#ifdef OPS_DEBUG
  ops_register_args(args, "initialise_chunk_kernel_zero");
#endif

  offs[0][0] = args[0].stencil->stride[0] * 1; // unit step in x dimension
  offs[0][1] =
      off2D(1, &start[0], &end[0], args[0].dat->size, args[0].stencil->stride) -
      offs[0][0];

  int off0_0 = offs[0][0];
  int off0_1 = offs[0][1];
  int dat0 = (OPS_soa ? args[0].dat->type_size : args[0].dat->elem_size);

  // Halo Exchanges
  ops_H_D_exchanges_host(args, 1);
  ops_halo_exchanges(args, 1, range);
  ops_H_D_exchanges_host(args, 1);

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  xdim0 = args[0].dat->size[0];

  if (OPS_diags > 1) {
    ops_timers_core(&c2, &t2);
    OPS_kernels[5].mpi_time += t2 - t1;
  }

#pragma omp parallel for
  for (int thr = 0; thr < nthreads; thr++) {

    int y_size = end[1] - start[1];
    char *p_a[1];

    int start_i = start[1] + ((y_size - 1) / nthreads + 1) * thr;
    int finish_i =
        start[1] + MIN(((y_size - 1) / nthreads + 1) * (thr + 1), y_size);

    // get address per thread
    int start0 = start[0];
    int start1 = start_i;

    // set up initial pointers
    int d_m[OPS_MAX_DIM];
#ifdef OPS_MPI
    for (int d = 0; d < dim; d++)
      d_m[d] =
          args[0].dat->d_m[d] + OPS_sub_dat_list[args[0].dat->index]->d_im[d];
#else
    for (int d = 0; d < dim; d++)
      d_m[d] = args[0].dat->d_m[d];
#endif
    int base0 = dat0 * 1 * (start0 * args[0].stencil->stride[0] -
                            args[0].dat->base[0] - d_m[0]);
    base0 = base0 +
            dat0 * args[0].dat->size[0] * (start1 * args[0].stencil->stride[1] -
                                           args[0].dat->base[1] - d_m[1]);
    p_a[0] = (char *)args[0].data + base0;

    for (int n_y = start_i; n_y < finish_i; n_y++) {
      for (int n_x = start[0]; n_x < start[0] + (end[0] - start[0]) / SIMD_VEC;
           n_x++) {
// call kernel function, passing in pointers to data -vectorised
#pragma simd
        for (int i = 0; i < SIMD_VEC; i++) {
          initialise_chunk_kernel_zero((double *)p_a[0] + i * 1 * 1);
        }

        // shift pointers to data x direction
        p_a[0] = p_a[0] + (dat0 * off0_0) * SIMD_VEC;
      }

      for (int n_x = start[0] + ((end[0] - start[0]) / SIMD_VEC) * SIMD_VEC;
           n_x < end[0]; n_x++) {
        // call kernel function, passing in pointers to data - remainder
        initialise_chunk_kernel_zero((double *)p_a[0]);

        // shift pointers to data x direction
        p_a[0] = p_a[0] + (dat0 * off0_0);
      }

      // shift pointers to data y direction
      p_a[0] = p_a[0] + (dat0 * off0_1);
    }
  }

  if (OPS_diags > 1) {
    ops_timers_core(&c1, &t1);
    OPS_kernels[5].time += t1 - t2;
  }

  ops_set_dirtybit_host(args, 1);

  ops_set_halo_dirtybit3(&args[0], range);

  if (OPS_diags > 1) {
    // Update kernel record
    ops_timers_core(&c2, &t2);
    OPS_kernels[5].mpi_time += t2 - t1;
    OPS_kernels[5].transfer += ops_compute_transfer(dim, start, end, &arg0);
  }
}