#ifndef CALC_DT_KERNEL_H
#define CALC_DT_KERNEL_H


inline void calc_dt_kernel(const double *celldx, const double *celldy, const double *soundspeed,
                   const double *viscosity, const double *density0, const double *xvel0,
                    const double *xarea, const double *volume, const double *yvel0,
                   const double *yarea, double *dt_min /*dt_min is work_array1*/) {

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
  //printf("dt_min %3.15e \n",**dt_min);
}



inline void calc_dt_kernel_min(const double* dt_min /*dt_min is work_array1*/,
                    double* dt_min_val) {
  *dt_min_val = MIN(*dt_min_val, dt_min[OPS_ACC0(0,0)]);
}


inline void calc_dt_kernel_get(const double* cellx, const double* celly,
                        double* xl_pos, double* yl_pos) {
  *xl_pos = cellx[OPS_ACC0(0,0)];
  *yl_pos = celly[OPS_ACC1(0,0)];
  //printf("xl_pos %lf yl_pos %lf\n",*xl_pos,*yl_pos);
}


inline void calc_dt_kernel_print(const double *cellx, const double *celly,
                        const double *xvel0, const double *yvel0,
                        const double *density0, const double *energy0,
                        const double *pressure, const double *soundspeed) {
  printf("Cell velocities:\n");
  printf("%E, %E \n",xvel0[OPS_ACC2(1,0)], yvel0[OPS_ACC3(1,0)]); //xvel0(jldt  ,kldt  ),yvel0(jldt  ,kldt  )
  printf("%E, %E \n",xvel0[OPS_ACC2(-1,0)], yvel0[OPS_ACC3(-1,0)]); //xvel0(jldt+1,kldt  ),yvel0(jldt+1,kldt  )
  printf("%E, %E \n",xvel0[OPS_ACC2(0,1)], yvel0[OPS_ACC3(0,1)]); //xvel0(jldt+1,kldt+1),yvel0(jldt+1,kldt+1)
  printf("%E, %E \n",xvel0[OPS_ACC2(0,-1)], yvel0[OPS_ACC3(0,-1)]); //xvel0(jldt  ,kldt+1),yvel0(jldt  ,kldt+1)

  printf("density, energy, pressure, soundspeed = %lf, %lf, %lf, %lf \n",
    density0[OPS_ACC4(0,0)], energy0[OPS_ACC5(0,0)], pressure[OPS_ACC6(0,0)], soundspeed[OPS_ACC7(0,0)]);
}

#endif
