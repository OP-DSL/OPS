
// host stub function
void ops_par_loop_viscosity_kernel(char const *name, ops_block Block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6) {

  buildOpenCLKernels();

  ops_arg args[7] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6};

  sub_block_list sb = OPS_sub_block_list[Block->index];
  //compute localy allocated range for the sub-block
  int start_add[2];
  int end_add[2];
  for ( int n=0; n<2; n++ ){
    start_add[n] = sb->istart[n];end_add[n] = sb->iend[n]+1;
    if (start_add[n] >= range[2*n]) {
      start_add[n] = 0;
    }
    else {
      start_add[n] = range[2*n] - start_add[n];
    }
    if (end_add[n] >= range[2*n+1]) {
      end_add[n] = range[2*n+1] - sb->istart[n];
    }
    else {
      end_add[n] = sb->sizes[n];
    }
  }
  
  int x_size = end_add[0]-start_add[0];
  int y_size = end_add[1]-start_add[1];

  int xdim0 = args[0].dat->block_size[0]*args[0].dat->dim;
  int xdim1 = args[1].dat->block_size[0]*args[1].dat->dim;
  int xdim2 = args[2].dat->block_size[0]*args[2].dat->dim;
  int xdim3 = args[3].dat->block_size[0]*args[3].dat->dim;
  int xdim4 = args[4].dat->block_size[0]*args[4].dat->dim;
  int xdim5 = args[5].dat->block_size[0]*args[5].dat->dim;
  int xdim6 = args[6].dat->block_size[0]*args[6].dat->dim;
  
  //Timing
  double t1,t2,c1,c2;
  //ops_timing_realloc(34,"viscosity_kernel");
  //ops_timers_core(&c1,&t1);
 

  //dim3 grid( (x_size-1)/OPS_block_size_x+ 1, (y_size-1)/OPS_block_size_y + 1, 1);
  //dim3 block(OPS_block_size_x,OPS_block_size_y,1);

  size_t globalWorkSize[3] = {((x_size-1)/OPS_block_size_x+ 1)*OPS_block_size_x, ((y_size-1)/OPS_block_size_y + 1)*OPS_block_size_y, 1};
  size_t localWorkSize[3] =  {OPS_block_size_x,OPS_block_size_y,1};
    
  int dat0 = args[0].dat->size;
  int dat1 = args[1].dat->size;
  int dat2 = args[2].dat->size;
  int dat3 = args[3].dat->size;
  int dat4 = args[4].dat->size;
  int dat5 = args[5].dat->size;
  int dat6 = args[6].dat->size;
  
  cl_mem p_a[7];
  
  //set up initial pointers
  int base0 = dat0 * 1 * 
  (start_add[0] * args[0].stencil->stride[0] - args[0].dat->offset[0]);
  base0 = base0  + dat0 * args[0].dat->block_size[0] * 
  (start_add[1] * args[0].stencil->stride[1] - args[0].dat->offset[1]);
  base0 = base0/dat0;
  //p_a[0] = (char *)args[0].data_d + base0;

  //set up initial pointers
  int base1 = dat1 * 1 * 
  (start_add[0] * args[1].stencil->stride[0] - args[1].dat->offset[0]);
  base1 = base1  + dat1 * args[1].dat->block_size[0] * 
  (start_add[1] * args[1].stencil->stride[1] - args[1].dat->offset[1]);
  //p_a[1] = (char *)args[1].data_d + base1;
  base1 = base1/dat1;

  //set up initial pointers
  int base2 = dat2 * 1 * 
  (start_add[0] * args[2].stencil->stride[0] - args[2].dat->offset[0]);
  base2 = base2  + dat2 * args[2].dat->block_size[0] * 
  (start_add[1] * args[2].stencil->stride[1] - args[2].dat->offset[1]);
  //p_a[2] = (char *)args[2].data_d + base2;
  base2 = base2/dat2;

  //set up initial pointers
  int base3 = dat3 * 1 * 
  (start_add[0] * args[3].stencil->stride[0] - args[3].dat->offset[0]);
  base3 = base3  + dat3 * args[3].dat->block_size[0] * 
  (start_add[1] * args[3].stencil->stride[1] - args[3].dat->offset[1]);
  //p_a[3] = (char *)args[3].data_d + base3;
  base3 = base3/dat3;

  //set up initial pointers
  int base4 = dat4 * 1 * 
  (start_add[0] * args[4].stencil->stride[0] - args[4].dat->offset[0]);
  base4 = base4  + dat4 * args[4].dat->block_size[0] * 
  (start_add[1] * args[4].stencil->stride[1] - args[4].dat->offset[1]);
  //p_a[4] = (char *)args[4].data_d + base4;
  base4 = base4/dat4;

  //set up initial pointers
  int base5 = dat5 * 1 * 
  (start_add[0] * args[5].stencil->stride[0] - args[5].dat->offset[0]);
  base5 = base5  + dat5 * args[5].dat->block_size[0] * 
  (start_add[1] * args[5].stencil->stride[1] - args[5].dat->offset[1]);
  //p_a[5] = (char *)args[5].data_d + base5;
  base5 = base5/dat5;

  //set up initial pointers
  int base6 = dat6 * 1 * 
  (start_add[0] * args[6].stencil->stride[0] - args[6].dat->offset[0]);
  base6 = base6  + dat6 * args[6].dat->block_size[0] * 
  (start_add[1] * args[6].stencil->stride[1] - args[6].dat->offset[1]);
  //p_a[6] = (char *)args[6].data_d + base6;
  base6 = base6/dat6;


  ops_H_D_exchanges_cuda(args, 7);

  //if (OPS_diags>1) cutilSafeCall(cudaDeviceSynchronize());
  //ops_set_dirtybit_cuda(args, 7);

  //Update kernel record
  //ops_timers_core(&c2,&t2);
  
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 0, sizeof(cl_mem), (void*) &arg0.data_d ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 1, sizeof(cl_mem), (void*) &arg1.data_d )); 
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 2, sizeof(cl_mem), (void*) &arg2.data_d ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 3, sizeof(cl_mem), (void*) &arg3.data_d ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 4, sizeof(cl_mem), (void*) &arg4.data_d ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 5, sizeof(cl_mem), (void*) &arg5.data_d ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 6, sizeof(cl_mem), (void*) &arg6.data_d ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 7, sizeof(cl_int), (void*) &x_size ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 8, sizeof(cl_int), (void*) &y_size ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 9, sizeof(cl_int), (void*) &xdim0 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 10, sizeof(cl_int), (void*) &xdim1 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 11, sizeof(cl_int), (void*) &xdim2 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 12, sizeof(cl_int), (void*) &xdim3 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 13, sizeof(cl_int), (void*) &xdim4 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 14, sizeof(cl_int), (void*) &xdim5 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 15, sizeof(cl_int), (void*) &xdim6 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 16, sizeof(cl_int), (void*) &base0 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 17, sizeof(cl_int), (void*) &base1 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 18, sizeof(cl_int), (void*) &base2 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 19, sizeof(cl_int), (void*) &base3 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 20, sizeof(cl_int), (void*) &base4 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 21, sizeof(cl_int), (void*) &base5 ));
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0], 22, sizeof(cl_int), (void*) &base6 ));
  
  /*clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]10, sizeof(cl_mem), (void*) &Plan->offset) );
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]11, sizeof(cl_mem), (void*) &Plan->nelems) );
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]12, sizeof(cl_mem), (void*) &Plan->nthrcol) );
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]13, sizeof(cl_mem), (void*) &Plan->thrcol) );
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]14, sizeof(cl_int), (void*) &Plan->ncolblk[col]) ); // int array is on host
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]15, sizeof(cl_int), (void*) &set_size) );
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]16, nshared, NULL) );
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]17, sizeof(cl_int), (void*) &OPS_opencl_core.constant[0]) ); // xdim0_viscosity_kernel
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]18, sizeof(cl_int), (void*) &OPS_opencl_core.constant[1]) ); // xdim1_viscosity_kernel
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]19, sizeof(cl_int), (void*) &OPS_opencl_core.constant[2]) ); // xdim2_viscosity_kernel
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]20, sizeof(cl_int), (void*) &OPS_opencl_core.constant[3]) ); // xdim3_viscosity_kernel
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]21, sizeof(cl_int), (void*) &OPS_opencl_core.constant[4]) ); // xdim4_viscosity_kernel
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]22, sizeof(cl_int), (void*) &OPS_opencl_core.constant[5]) ); // xdim5_viscosity_kernel
  clSafeCall( clSetKernelArg(OPS_opencl_core.kernel[0]23, sizeof(cl_int), (void*) &OPS_opencl_core.constant[6]) ); // xdim6_viscosity_kernel
  */

  //call/enque opencl kernel wrapper function
  clSafeCall( clEnqueueNDRangeKernel(OPS_opencl_core.command_queue, OPS_opencl_core.kernel[0], 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL) );
  clSafeCall( clFinish(OPS_opencl_core.command_queue) );
  ops_set_dirtybit_cuda(args, 7);
  
  ops_H_D_exchanges(args, 7);

  
}
