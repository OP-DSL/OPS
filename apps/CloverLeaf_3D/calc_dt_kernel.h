#ifndef CALC_DT_KERNEL_H
#define CALC_DT_KERNEL_H

#include "data.h"
#include "definitions.h"

void calc_dt_kernel(const double *celldx, const double *celldy, const double *soundspeed,
                    const double *viscosity, const double *density0, const double *xvel0,
                    const double *xarea, const double *volume, const double *yvel0,
                    const double *yarea, double *dt_min /*dt_min is work_array1*/,
                    const double *celldz, const double *zvel0, const double *zarea) {//11,12,13

  double div, dsx, dsy, dsz, dtut, dtvt, dtct, dtwt, dtdivt, cc, dv1, dv2, jk_control;

  dsx = celldx[OPS_ACC0(0,0,0)];
  dsy = celldy[OPS_ACC1(0,0,0)];
  dsz = celldz[OPS_ACC11(0,0,0)];

  cc = soundspeed[OPS_ACC2(0,0,0)] * soundspeed[OPS_ACC2(0,0,0)];
  cc = cc + 2.0 * viscosity[OPS_ACC3(0,0,0)]/density0[OPS_ACC4(0,0,0)];
  cc = MAX(sqrt(cc),g_small);

  dtct = dtc_safe * MIN(dsx,MIN(dsy,dsz))/cc;

  div=0.0;

  //00_10_01_11

  dv1 = (xvel0[OPS_ACC5(0,0,0)] + xvel0[OPS_ACC5(0,1,0)] + xvel0[OPS_ACC5(0,0,1)] + xvel0[OPS_ACC5(0,1,1)]) * xarea[OPS_ACC6(0,0,0)];
  dv2 = (xvel0[OPS_ACC5(1,0,0)] + xvel0[OPS_ACC5(1,1,0)] + xvel0[OPS_ACC5(1,0,1)] + xvel0[OPS_ACC5(1,1,1)]) * xarea[OPS_ACC6(1,0,0)];

  div = div + dv2 - dv1;

  dtut = dtu_safe * 2.0 * volume[OPS_ACC7(0,0,0)]/MAX(MAX(fabs(dv1), fabs(dv2)), g_small * volume[OPS_ACC7(0,0,0)]);

  dv1 = (yvel0[OPS_ACC8(0,0,0)] + yvel0[OPS_ACC8(1,0,0)] + yvel0[OPS_ACC8(0,0,1)] + yvel0[OPS_ACC8(1,0,1)]) * yarea[OPS_ACC9(0,0,0)];
  dv2 = (yvel0[OPS_ACC8(0,1,0)] + yvel0[OPS_ACC8(1,1,0)] + yvel0[OPS_ACC8(0,1,1)] + yvel0[OPS_ACC8(1,1,1)]) * yarea[OPS_ACC9(0,1,0)];

  div = div + dv2 - dv1;

  dtvt = dtv_safe * 2.0 * volume[OPS_ACC7(0,0,0)]/MAX(MAX(fabs(dv1),fabs(dv2)), g_small * volume[OPS_ACC7(0,0,0)]);
  
  dv1 = (zvel0[OPS_ACC12(0,0,0)] + zvel0[OPS_ACC12(1,0,0)] + zvel0[OPS_ACC12(0,1,0)] + zvel0[OPS_ACC12(1,1,0)]) * zarea[OPS_ACC13(0,0,0)];
  dv2 = (zvel0[OPS_ACC12(0,0,1)] + zvel0[OPS_ACC12(1,0,1)] + zvel0[OPS_ACC12(0,1,1)] + zvel0[OPS_ACC12(1,1,1)]) * zarea[OPS_ACC13(0,0,1)];

  div = div + dv2 - dv1;

  dtwt = dtw_safe * 2.0 * volume[OPS_ACC7(0,0,0)]/MAX(MAX(fabs(dv1),fabs(dv2)), g_small * volume[OPS_ACC7(0,0,0)]);

  div = div/(2.0 * volume[OPS_ACC7(0,0,0)]);

  if(div < -g_small)
    dtdivt = dtdiv_safe * (-1.0/div);
  else
    dtdivt = g_big;

  //dt_min is work_array1
  dt_min[OPS_ACC10(0,0,0)] = MIN(MIN(MIN(dtct, dtut), MIN(dtvt, dtdivt)),dtwt);
  //if(dt_min[OPS_ACC10(0,0,0)]<0)
  //  printf("dt_min %lf, dtct %lf, dtut %lf, dtvt %lf, dtdivt %lf, dtwt %lf ",dt_min[OPS_ACC10(0,0,0)], dtct, dtut, dtvt,dtdivt, dtwt);
  //printf("dsx %lf, dsy %lf ",dsx,dsy);
}

void calc_dt_kernel_min(const double* dt_min /*dt_min is work_array1*/,
                    double* dt_min_val) {
  *dt_min_val = MIN(*dt_min_val, dt_min[OPS_ACC0(0,0,0)]);
  //printf("%lf ",*dt_min_val);
}

void calc_dt_kernel_get(const double* cellx, const double* celly, double* xl_pos, double* yl_pos, const double *cellz, double *zl_pos) {
  *xl_pos = cellx[OPS_ACC0(0,0,0)];
  *yl_pos = celly[OPS_ACC1(0,0,0)];
  *zl_pos = cellz[OPS_ACC4(0,0,0)];
}

void calc_dt_kernel_print(const double *xvel0, const double *yvel0, const double *zvel0,
                        const double *density0, const double *energy0,
                        const double *pressure, const double *soundspeed, double *output) {
  output[0] = xvel0[OPS_ACC0(0,0,0)];
  output[1] = yvel0[OPS_ACC1(0,0,0)];
  output[2] = zvel0[OPS_ACC2(0,0,0)];  
  output[3] = xvel0[OPS_ACC0(1,0,0)];
  output[4] = yvel0[OPS_ACC1(1,0,0)];
  output[5] = zvel0[OPS_ACC2(0,0,0)];  
  output[6] = xvel0[OPS_ACC0(1,1,0)];
  output[7] = yvel0[OPS_ACC1(1,1,0)]; 
  output[8] = zvel0[OPS_ACC2(0,0,0)];
  output[9] = xvel0[OPS_ACC0(0,1,0)];
  output[10] = yvel0[OPS_ACC1(0,1,0)]; 
  output[11] = zvel0[OPS_ACC2(0,0,0)];
  output[12] = xvel0[OPS_ACC0(0,0,1)]; 
  output[13] = yvel0[OPS_ACC1(0,0,1)]; 
  output[14] = zvel0[OPS_ACC2(0,0,1)];
  output[15] = xvel0[OPS_ACC0(1,0,1)]; 
  output[16] = yvel0[OPS_ACC1(1,0,1)]; 
  output[17] = zvel0[OPS_ACC2(0,0,1)];
  output[18] = xvel0[OPS_ACC0(1,1,1)]; 
  output[19] = yvel0[OPS_ACC1(1,1,1)]; 
  output[20] = zvel0[OPS_ACC2(0,0,1)];
  output[21] = xvel0[OPS_ACC0(0,1,1)]; 
  output[22] = yvel0[OPS_ACC1(0,1,1)]; 
  output[23] = zvel0[OPS_ACC2(0,0,1)];
  output[24] = density0[OPS_ACC3(0,0,0)];
  output[25] = energy0[OPS_ACC4(0,0,0)];
  output[26] = pressure[OPS_ACC5(0,0,0)];
  output[27] = soundspeed[OPS_ACC6(0,0,0)];  
  
}
#endif
