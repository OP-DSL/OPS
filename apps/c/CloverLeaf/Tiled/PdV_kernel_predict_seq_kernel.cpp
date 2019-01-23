//
// auto-generated by ops.py
//
#define OPS_ACC0(x,y) (n_x*1+n_y*xdim0_PdV_kernel_predict*1+x+xdim0_PdV_kernel_predict*(y))
#define OPS_ACC1(x,y) (n_x*1+n_y*xdim1_PdV_kernel_predict*1+x+xdim1_PdV_kernel_predict*(y))
#define OPS_ACC2(x,y) (n_x*1+n_y*xdim2_PdV_kernel_predict*1+x+xdim2_PdV_kernel_predict*(y))
#define OPS_ACC3(x,y) (n_x*1+n_y*xdim3_PdV_kernel_predict*1+x+xdim3_PdV_kernel_predict*(y))
#define OPS_ACC4(x,y) (n_x*1+n_y*xdim4_PdV_kernel_predict*1+x+xdim4_PdV_kernel_predict*(y))
#define OPS_ACC5(x,y) (n_x*1+n_y*xdim5_PdV_kernel_predict*1+x+xdim5_PdV_kernel_predict*(y))
#define OPS_ACC6(x,y) (n_x*1+n_y*xdim6_PdV_kernel_predict*1+x+xdim6_PdV_kernel_predict*(y))
#define OPS_ACC7(x,y) (n_x*1+n_y*xdim7_PdV_kernel_predict*1+x+xdim7_PdV_kernel_predict*(y))
#define OPS_ACC8(x,y) (n_x*1+n_y*xdim8_PdV_kernel_predict*1+x+xdim8_PdV_kernel_predict*(y))
#define OPS_ACC9(x,y) (n_x*1+n_y*xdim9_PdV_kernel_predict*1+x+xdim9_PdV_kernel_predict*(y))
#define OPS_ACC10(x,y) (n_x*1+n_y*xdim10_PdV_kernel_predict*1+x+xdim10_PdV_kernel_predict*(y))
#define OPS_ACC11(x,y) (n_x*1+n_y*xdim11_PdV_kernel_predict*1+x+xdim11_PdV_kernel_predict*(y))


//user function

// host stub function
void ops_par_loop_PdV_kernel_predict_execute(ops_kernel_descriptor *desc) {
  ops_block block = desc->block;
  int dim = desc->dim;
  int *range = desc->range;
  ops_arg arg0 = desc->args[0];
  ops_arg arg1 = desc->args[1];
  ops_arg arg2 = desc->args[2];
  ops_arg arg3 = desc->args[3];
  ops_arg arg4 = desc->args[4];
  ops_arg arg5 = desc->args[5];
  ops_arg arg6 = desc->args[6];
  ops_arg arg7 = desc->args[7];
  ops_arg arg8 = desc->args[8];
  ops_arg arg9 = desc->args[9];
  ops_arg arg10 = desc->args[10];
  ops_arg arg11 = desc->args[11];

  //Timing
  double t1,t2,c1,c2;

  ops_arg args[12] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11};



  #ifdef CHECKPOINTING
  if (!ops_checkpointing_before(args,12,range,55)) return;
  #endif

  if (OPS_diags > 1) {
    OPS_kernels[55].count++;
    ops_timers_core(&c2,&t2);
  }

  //compute locally allocated range for the sub-block
  int start[2];
  int end[2];

  for ( int n=0; n<2; n++ ){
    start[n] = range[2*n];end[n] = range[2*n+1];
  }

  #ifdef OPS_DEBUG
  ops_register_args(args, "PdV_kernel_predict");
  #endif



  //set up initial pointers and exchange halos if necessary
  int base0 = args[0].dat->base_offset;
  const double * __restrict__ ACC<> &xarea = (double *)(args[0].data + base0);

  int base1 = args[1].dat->base_offset;
  const double * __restrict__ ACC<> &xvel0 = (double *)(args[1].data + base1);

  int base2 = args[2].dat->base_offset;
  const double * __restrict__ ACC<> &yarea = (double *)(args[2].data + base2);

  int base3 = args[3].dat->base_offset;
  const double * __restrict__ ACC<> &yvel0 = (double *)(args[3].data + base3);

  int base4 = args[4].dat->base_offset;
  double * __restrict__ ACC<> &volume_change = (double *)(args[4].data + base4);

  int base5 = args[5].dat->base_offset;
  const double * __restrict__ ACC<> &volume = (double *)(args[5].data + base5);

  int base6 = args[6].dat->base_offset;
  const double * __restrict__ ACC<> &pressure = (double *)(args[6].data + base6);

  int base7 = args[7].dat->base_offset;
  const double * __restrict__ ACC<> &density0 = (double *)(args[7].data + base7);

  int base8 = args[8].dat->base_offset;
  double * __restrict__ ACC<> &density1 = (double *)(args[8].data + base8);

  int base9 = args[9].dat->base_offset;
  const double * __restrict__ ACC<> &viscosity = (double *)(args[9].data + base9);

  int base10 = args[10].dat->base_offset;
  const double * __restrict__ ACC<> &energy0 = (double *)(args[10].data + base10);

  int base11 = args[11].dat->base_offset;
  double * __restrict__ ACC<> &energy1 = (double *)(args[11].data + base11);


  //initialize global variable with the dimension of dats
  int xdim0_PdV_kernel_predict = args[0].dat->size[0];
  int xdim1_PdV_kernel_predict = args[1].dat->size[0];
  int xdim2_PdV_kernel_predict = args[2].dat->size[0];
  int xdim3_PdV_kernel_predict = args[3].dat->size[0];
  int xdim4_PdV_kernel_predict = args[4].dat->size[0];
  int xdim5_PdV_kernel_predict = args[5].dat->size[0];
  int xdim6_PdV_kernel_predict = args[6].dat->size[0];
  int xdim7_PdV_kernel_predict = args[7].dat->size[0];
  int xdim8_PdV_kernel_predict = args[8].dat->size[0];
  int xdim9_PdV_kernel_predict = args[9].dat->size[0];
  int xdim10_PdV_kernel_predict = args[10].dat->size[0];
  int xdim11_PdV_kernel_predict = args[11].dat->size[0];

  if (OPS_diags > 1) {
    ops_timers_core(&c1,&t1);
    OPS_kernels[55].mpi_time += t1-t2;
  }

  #pragma omp parallel for
  for ( int n_y=start[1]; n_y<end[1]; n_y++ ){
    #ifdef intel
    #pragma loop_count(10000)
    #pragma omp simd aligned(ACC<> &xarea,ACC<> &xvel0,ACC<> &yarea,ACC<> &yvel0,ACC<> &volume_change,ACC<> &volume,ACC<> &pressure,ACC<> &density0,ACC<> &density1,ACC<> &viscosity,ACC<> &energy0,ACC<> &energy1)
    #else
    #pragma simd
    #endif
    for ( int n_x=start[0]; n_x<end[0]; n_x++ ){
      


  double recip_volume, energy_change;
  double right_flux, left_flux, top_flux, bottom_flux, total_flux;

  left_flux = ( xarea(0,0) * ( xvel0(0,0) + xvel0(0,1) +
                                xvel0(0,0) + xvel0(0,1) ) ) * 0.25 * dt * 0.5;
  right_flux = ( xarea(1,0) * ( xvel0(1,0) + xvel0(1,1) +
                                 xvel0(1,0) + xvel0(1,1) ) ) * 0.25 * dt * 0.5;

  bottom_flux = ( yarea(0,0) * ( yvel0(0,0) + yvel0(1,0) +
                                  yvel0(0,0) + yvel0(1,0) ) ) * 0.25* dt * 0.5;
  top_flux = ( yarea(0,1) * ( yvel0(0,1) + yvel0(1,1) +
                               yvel0(0,1) + yvel0(1,1) ) ) * 0.25 * dt * 0.5;

  total_flux = right_flux - left_flux + top_flux - bottom_flux;

  volume_change(0,0) = (volume(0,0))/(volume(0,0) + total_flux);




  recip_volume = 1.0/volume(0,0);

  energy_change = ( pressure(0,0)/density0(0,0) +
                    viscosity(0,0)/density0(0,0) ) * total_flux * recip_volume;
  energy1(0,0) = energy0(0,0) - energy_change;
  density1(0,0) = density0(0,0) * volume_change(0,0);


    }
  }
  if (OPS_diags > 1) {
    ops_timers_core(&c2,&t2);
    OPS_kernels[55].time += t2-t1;
  }

  if (OPS_diags > 1) {
    //Update kernel record
    ops_timers_core(&c1,&t1);
    OPS_kernels[55].mpi_time += t1-t2;
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg0);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg1);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg2);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg3);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg4);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg5);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg6);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg7);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg8);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg9);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg10);
    OPS_kernels[55].transfer += ops_compute_transfer(dim, start, end, &arg11);
  }
}
#undef OPS_ACC0
#undef OPS_ACC1
#undef OPS_ACC2
#undef OPS_ACC3
#undef OPS_ACC4
#undef OPS_ACC5
#undef OPS_ACC6
#undef OPS_ACC7
#undef OPS_ACC8
#undef OPS_ACC9
#undef OPS_ACC10
#undef OPS_ACC11


void ops_par_loop_PdV_kernel_predict(char const *name, ops_block block, int dim, int* range,
 ops_arg arg0, ops_arg arg1, ops_arg arg2, ops_arg arg3,
 ops_arg arg4, ops_arg arg5, ops_arg arg6, ops_arg arg7,
 ops_arg arg8, ops_arg arg9, ops_arg arg10, ops_arg arg11) {
  ops_kernel_descriptor *desc = (ops_kernel_descriptor *)malloc(sizeof(ops_kernel_descriptor));
  desc->name = name;
  desc->block = block;
  desc->dim = dim;
  desc->device = 1;
  desc->index = 55;
  desc->hash = 5381;
  desc->hash = ((desc->hash << 5) + desc->hash) + 55;
  for ( int i=0; i<4; i++ ){
    desc->range[i] = range[i];
    desc->orig_range[i] = range[i];
    desc->hash = ((desc->hash << 5) + desc->hash) + range[i];
  }
  desc->nargs = 12;
  desc->args = (ops_arg*)malloc(12*sizeof(ops_arg));
  desc->args[0] = arg0;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg0.dat->index;
  desc->args[1] = arg1;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg1.dat->index;
  desc->args[2] = arg2;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg2.dat->index;
  desc->args[3] = arg3;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg3.dat->index;
  desc->args[4] = arg4;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg4.dat->index;
  desc->args[5] = arg5;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg5.dat->index;
  desc->args[6] = arg6;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg6.dat->index;
  desc->args[7] = arg7;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg7.dat->index;
  desc->args[8] = arg8;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg8.dat->index;
  desc->args[9] = arg9;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg9.dat->index;
  desc->args[10] = arg10;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg10.dat->index;
  desc->args[11] = arg11;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg11.dat->index;
  desc->function = ops_par_loop_PdV_kernel_predict_execute;
  if (OPS_diags > 1) {
    ops_timing_realloc(55,"PdV_kernel_predict");
  }
  ops_enqueue_kernel(desc);
  }