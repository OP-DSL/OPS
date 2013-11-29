#ifndef CALC_DT_KERNEL_H
#define CALC_DT_KERNEL_H

#include "data.h"
#include "definitions.h"

void calc_dt_kernel(double *celldx, double *celldy, double *soundspeed,
                    double *viscosity, double *density0, double *xvel0,
                    double *xarea, double *volume, double *yvel0,
                    double *yarea, double *dt_min /*dt_min is work_array1*/) {

  double div, dsx, dsy, dtut, dtvt, dtct, dtdivt, cc, dv1, dv2, jk_control;

  dsx = celldx[OPS_ACC0(0,0)];
  dsy = celldy[OPS_ACC2(0,0)];

  cc = soundspeed[OPS_ACC2(0,0)] * soundspeed[OPS_ACC2(0,0)];
  cc = cc + 2.0 * viscosity[OPS_ACC3(0,0)]/density0[OPS_ACC4(0,0)];
  cc = MAX(sqrt(cc),g_small);

  dtct = dtc_safe * MIN(dsx,dsy)/cc;

  div=0.0;

  //00_10_01_11

  dv1 = (xvel0[OPS_ACC5(0,0)] + xvel0[OPS_ACC5(0,1)]) * xarea[OPS_ACC6(0,0)];
  dv2 = (xvel0[OPS_ACC5(1,0)] + xvel0[OPS_ACC5(1,1)]) * xarea[OPS_ACC6(1,0)];

  div = div + dv2 - dv1;

  dtut = dtu_safe * 2.0 * volume[OPS_ACC7(0,0)]/MAX(MAX(fabs(dv1), fabs(dv2)), g_small * volume[OPS_ACC7(0,0)]);

  dv1 = (yvel0[OPS_ACC8(0,0)] + yvel0[OPS_ACC8(1,0)]) * yarea[OPS_ACC9(0,0)];
  dv2 = (yvel0[OPS_ACC8(0,1)] + yvel0[OPS_ACC8(1,1)]) * yarea[OPS_ACC9(0,1)];

  div = div + dv2 - dv1;

  dtvt = dtv_safe * 2.0 * volume[OPS_ACC7(0,0)]/MAX(MAX(fabs(dv1),fabs(dv2)), g_small * volume[OPS_ACC7(0,0)]);

  div = div/(2.0 * volume[OPS_ACC7(0,0)]);

  if(div < -g_small)
    dtdivt = dtdiv_safe * (-1.0/div);
  else
    dtdivt = g_big;

  //dt_min is work_array1
  dt_min[OPS_ACC10(0,0)] = MIN(MIN(dtct, dtut), MIN(dtvt, dtdivt));

}

void calc_dt_kernel_min(double* dt_min /*dt_min is work_array1*/,
                    double* dt_min_val) {
  *dt_min_val = MIN(*dt_min_val, dt_min[OPS_ACC0(0,0)]);
}

void calc_dt_kernel_get(double* cellx, double* celly,
                        double* xl_pos, double* yl_pos) {
  *xl_pos = cellx[OPS_ACC0(0,0)];
  *yl_pos = celly[OPS_ACC1(0,0)];
}

void calc_dt_kernel_print(double *cellx, double *celly,
                        double *xvel0, double *yvel0,
                        double *density0, double *energy0,
                        double *pressure, double *soundspeed) {
  printf("Cell velocities:\n");
  printf("%E, %E \n",xvel0[OPS_ACC2(1,0)], yvel0[OPS_ACC3(1,0)]); //xvel0(jldt  ,kldt  ),yvel0(jldt  ,kldt  )
  printf("%E, %E \n",xvel0[OPS_ACC2(-1,0)], yvel0[OPS_ACC3(-1,0)]); //xvel0(jldt+1,kldt  ),yvel0(jldt+1,kldt  )
  printf("%E, %E \n",xvel0[OPS_ACC2(0,1)], yvel0[OPS_ACC3(0,1)]); //xvel0(jldt+1,kldt+1),yvel0(jldt+1,kldt+1)
  printf("%E, %E \n",xvel0[OPS_ACC2(0,-1)], yvel0[OPS_ACC3(0,-1)]); //xvel0(jldt  ,kldt+1),yvel0(jldt  ,kldt+1)

  printf("density, energy, pressure, soundspeed = %lf, %lf, %lf, %lf \n",
    density0[OPS_ACC4(0,0)], energy0[OPS_ACC5(0,0)], pressure[OPS_ACC6(0,0)], soundspeed[OPS_ACC7(0,0)]);
}
#endif
