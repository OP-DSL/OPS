#ifndef CALC_DT_KERNEL_H
#define CALC_DT_KERNEL_H

#include "data.h"
#include "definitions.h"

void calc_dt_kernel(const ACC<double> &celldx, const ACC<double> &celldy, const ACC<double> &soundspeed,
                    const ACC<double> &viscosity, const ACC<double> &density0, const ACC<double> &xvel0,
                    const ACC<double> &xarea, const ACC<double> &volume, const ACC<double> &yvel0,
                    const ACC<double> &yarea, ACC<double> &dt_min /*dt_min is work_array1*/) {

  double div, dsx, dsy, dtut, dtvt, dtct, dtdivt, cc, dv1, dv2;//, jk_control;

  dsx = celldx(0,0);
  dsy = celldy(0,0);

  cc = soundspeed(0,0) * soundspeed(0,0);
  cc = cc + 2.0 * viscosity(0,0)/density0(0,0);
  cc = MAX(sqrt(cc),g_small);

  dtct = dtc_safe * MIN(dsx,dsy)/cc;

  div=0.0;

  //00_10_01_11

  dv1 = (xvel0(0,0) + xvel0(0,1)) * xarea(0,0);
  dv2 = (xvel0(1,0) + xvel0(1,1)) * xarea(1,0);

  div = div + dv2 - dv1;

  dtut = dtu_safe * 2.0 * volume(0,0)/MAX(MAX(fabs(dv1), fabs(dv2)), g_small * volume(0,0));

  dv1 = (yvel0(0,0) + yvel0(1,0)) * yarea(0,0);
  dv2 = (yvel0(0,1) + yvel0(1,1)) * yarea(0,1);

  div = div + dv2 - dv1;

  dtvt = dtv_safe * 2.0 * volume(0,0)/MAX(MAX(fabs(dv1),fabs(dv2)), g_small * volume(0,0));

  div = div/(2.0 * volume(0,0));

  if(div < -g_small)
    dtdivt = dtdiv_safe * (-1.0/div);
  else
    dtdivt = g_big;

  //dt_min is work_array1
  dt_min(0,0) = MIN(MIN(dtct, dtut), MIN(dtvt, dtdivt));
  //printf("dt_min %lf, dtct %lf ",dt_min(0,0), dtct);
  //printf("dsx %lf, dsy %lf ",dsx,dsy);
}

void calc_dt_kernel_min(const ACC<double> &dt_min /*dt_min is work_array1*/,
                    double* dt_min_val) {
  *dt_min_val = MIN(*dt_min_val, dt_min(0,0));
  //printf("%lf ",*dt_min_val);
}

void calc_dt_kernel_get(const ACC<double> &cellx, const ACC<double> &celly, double* xl_pos, double* yl_pos) {
  *xl_pos = cellx(0,0);
  *yl_pos = celly(0,0);
}

void calc_dt_kernel_getx(ACC<double> &cellx, double* xl_pos) {
  *xl_pos = cellx(0,0);
}

void calc_dt_kernel_gety(ACC<double> &celly,double* yl_pos) {
  *yl_pos = celly(0,0);
}

void calc_dt_kernel_print(const ACC<double> &xvel0, const ACC<double> &yvel0,
                        const ACC<double> &density0, const ACC<double> &energy0,
                        const ACC<double> &pressure, const ACC<double> &soundspeed, double *output) {
  output[0] = xvel0(1,0);
  output[1] = yvel0(1,0);
  output[2] = xvel0(-1,0);
  output[3] = yvel0(-1,0);
  output[4] = xvel0(0,1);
  output[5] = yvel0(0,1);
  output[6] = xvel0(0,-1);
  output[7] = yvel0(0,-1);
  output[8] = density0(0,0);
  output[9] = energy0(0,0);
  output[10]= pressure(0,0);
  output[11]= soundspeed(0,0);

}
#endif
