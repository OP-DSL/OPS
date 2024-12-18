// Auto-generated at 2024-12-18 22:21:36.432919 by ops-translator


//  ==================
//  Host stub function
//  ==================
#ifndef OPS_LAZY
void ops_par_loop_copy(
    const char *name,
    ops_block block,
    int dim,
    int *range,
    ops_arg arg0,
    ops_arg arg1
)
{ 
#else
void ops_par_loop_copy_execute(ops_kernel_descriptor *desc)
{
    ops_block block = desc->block;
    int dim = desc->dim;
    int *range = desc->range;
    ops_arg arg0 = desc->args[0];
    ops_arg arg1 = desc->args[1];
#endif

//  ======
//  Timing
//  ======
    double __t1, __t2, __c1, __c2;

    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

#if defined(CHECKPOINTING) && !defined(OPS_LAZY)
    if (!ops_checkpointing_before(args, 2, range, 5)) return;
#endif

    if (block->instance->OPS_diags > 1)
    {
        ops_timing_realloc(block->instance, 5, "copy");
        block->instance->OPS_kernels[5].count++;
        ops_timers_core(&__c1, &__t1);
    }

#ifdef OPS_DEBUG
    ops_register_args(block->instance, args, "copy");
#endif

//  =================================================
//  compute locally allocated range for the sub-block
//  =================================================
    int start_indx[2];
    int end_indx[2];
#if defined(OPS_MPI) && !defined(OPS_LAZY)
    int arg_idx[2];
#endif

#if defined(OPS_LAZY) || !defined(OPS_MPI)
    for (int n = 0; n < 2; n++) {
        start_indx[n] = range[2*n];
        end_indx[n]   = range[2*n+1];
    }
#else
    if (compute_ranges(args, 2, block, range, start_indx, end_indx, arg_idx) < 0) return;
#endif

//  ======================================================
//  Initialize global variable with the dimensions of dats
//  ======================================================
    int xdim0_copy = args[0].dat->size[0];
    int xdim1_copy = args[1].dat->size[0];

//  =======================================================
//  Set up initial pointers and exchange halos if necessary
//  =======================================================
    int base0 = args[0].dat->base_offset;
    double * __restrict__ A_p = (double *)(args[0].data + base0);

    int base1 = args[1].dat->base_offset;
    double * __restrict__ Anew_p = (double *)(args[1].data + base1);

#ifndef OPS_LAZY
//  ==============
//  Halo exchanges
//  ==============
    ops_H_D_exchanges_host(args, 2);
    ops_halo_exchanges(args, 2, range);
    ops_H_D_exchanges_host(args, 2);
#endif //OPS_LAZY

    if (block->instance->OPS_diags > 1)
    {
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[5].mpi_time += __t2 - __t1;
    }

    #pragma omp parallel for
      for (int n_y = start_indx[1]; n_y < end_indx[1]; n_y++)
      {
#ifdef __INTEL_COMPILER
        #pragma loop_count(10000)
        #pragma omp simd
#elif defined(__clang__)
        #pragma clang loop vectorize(assume_safety)
#elif defined(__GNUC__)
        #pragma GCC ivdep
#else
        #pragma simd
#endif
        for(int n_x = start_indx[0]; n_x < end_indx[0]; n_x++)
        {

             ACC<double> A(xdim0_copy, A_p + (n_x * 1) + (n_y * xdim0_copy * 1));

            const  ACC<double> Anew(xdim1_copy, Anew_p + (n_x * 1) + (n_y * xdim1_copy * 1));

  A(0,0) = Anew(0,0);

        }
    }

    if (block->instance->OPS_diags > 1)
    {
        ops_timers_core(&__c1, &__t1);
        block->instance->OPS_kernels[5].time += __t1 - __t2;
    }

#ifndef OPS_LAZY
    ops_set_dirtybit_host(args, 2);
    ops_set_halo_dirtybit3(&args[0], range);
#endif

    if (block->instance->OPS_diags > 1)
    {
//      ====================
//      Update kernel record
//      ====================
        ops_timers_core(&__c2, &__t2);
        block->instance->OPS_kernels[5].mpi_time += __t2 -__t1;
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg0);
        block->instance->OPS_kernels[5].transfer += ops_compute_transfer(dim, start_indx, end_indx, &arg1);
    }
}

#ifdef OPS_LAZY
void ops_par_loop_copy(
    const char *name,
    ops_block block,
    int dim,
    int *range,
    ops_arg arg0,
    ops_arg arg1
    )
{
    ops_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    create_kerneldesc_and_enque("copy", args, 2, 5, dim, 0, range, block, ops_par_loop_copy_execute);
}
#endif
