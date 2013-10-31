#ifndef CALC_DT_KERNEL_H
#define CALC_DT_KERNEL_H

#include "data.h"
#include "definitions.h"

void calc_dt_kernel(double** celldx, double** celldy, double **soundspeed,
                    double **viscosity, double **density0, double **xvel0,
                    double **xarea, double **volume, double **yvel0,
                    double **yarea, double** dt_min /*dt_min is work_array1*/) {

  double div, dsx, dsy, dtut, dtvt, dtct, dtdivt, cc, dv1, dv2, jk_control;

  dsx = **celldx;
  dsy = **celldy;

  cc = (**soundspeed) * (**soundspeed);
  cc = cc + 2.0 * (**viscosity)/(**density0);
  cc = MAX(sqrt(cc),g_small);

  //printf("dtc_safe %3.15e \n",dtc_safe);

  dtct = dtc_safe * MIN(dsx,dsy)/cc;



  div=0.0;

  //00_10_01_11

  dv1 = (*xvel0[0] + *xvel0[2]) * (*xarea[0]);
  dv2 = (*xvel0[1] + *xvel0[3]) * (*xarea[1]);

  div = div + dv2 - dv1;

  dtut = dtu_safe * 2.0 * (**volume)/MAX(MAX(fabs(dv1), fabs(dv2)), g_small * (**volume));

  dv1 = (*yvel0[0] + *yvel0[1]) * (*yarea[0]);
  dv2 = (*yvel0[2] + *yvel0[3]) * (*yarea[1]);

  div = div + dv2 - dv1;

  dtvt = dtv_safe * 2.0 * (**volume)/MAX(MAX(fabs(dv1),fabs(dv2)), g_small * (**volume));

  div = div/(2.0*(**volume));

  if(div < -g_small)
    dtdivt = dtdiv_safe * (-1.0/div);
  else
    dtdivt = g_big;

  //dt_min is work_array1
  **dt_min = MIN(MIN(dtct, dtut), MIN(dtvt, dtdivt));
  //printf("dt_min %3.15e \n",**dt_min);
}


void calc_dt_kernel_min(double** dt_min /*dt_min is work_array1*/,
                    double* dt_min_val) {
  *dt_min_val = MIN(*dt_min_val,**dt_min);
}

void calc_dt_kernel_get(double** cellx, double** celly,
                        double* xl_pos, double* yl_pos) {
  *xl_pos = **cellx;
  *yl_pos = **celly;
}

void calc_dt_kernel_print(double** cellx, double** celly,
                        double** xvel0, double** yvel0,
                        double** density0, double** energy0,
                        double** pressure, double** soundspeed) {
  printf("Cell velocities:\n");
  printf("%E, %E \n",*xvel0[0], *yvel0[0]); //xvel0(jldt  ,kldt  ),yvel0(jldt  ,kldt  )
  printf("%E, %E \n",*xvel0[1], *yvel0[1]); //xvel0(jldt+1,kldt  ),yvel0(jldt+1,kldt  )
  printf("%E, %E \n",*xvel0[3], *yvel0[3]); //xvel0(jldt+1,kldt+1),yvel0(jldt+1,kldt+1)
  printf("%E, %E \n",*xvel0[2], *yvel0[2]); //xvel0(jldt  ,kldt+1),yvel0(jldt  ,kldt+1)

  printf("density, energy, pressure, soundspeed = %lf, %lf, %lf, %lf \n",
    **density0, **energy0, **pressure, **soundspeed);
}

#endif
