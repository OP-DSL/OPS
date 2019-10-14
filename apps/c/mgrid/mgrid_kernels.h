#ifndef MGRID_KERNELS_H
#define MGRID_KERNELS_H

void mgrid_populate_kernel_1(double *val, int *idx) {
  val[OPS_ACC0(0,0)] = (double)(idx[0]+6*idx[1]);
}

void mgrid_populate_kernel_2(double *val, int *idx) {
  val[OPS_ACC0(0,0)] = (double)(idx[0]+4*idx[1]);
}

void mgrid_populate_kernel_3(double *val, int *idx) {
  val[OPS_ACC0(0,0)] = (double)(idx[0]+24*idx[1]);
}

void mgrid_restrict_kernel(const double *fine, double *coarse, int *idx) {
  //coarse[OPS_ACC1(0,0)] = 1000000*fine[OPS_ACC0(-1,0)]+1000*fine[OPS_ACC0(0,0)]+fine[OPS_ACC0(1,0)];
  coarse[OPS_ACC1(0,0)] = fine[OPS_ACC0(0,0)];
}

void mgrid_prolong_kernel(const double *coarse, double *fine, int *idx) {
  fine[OPS_ACC1(0,0)] = coarse[OPS_ACC0(0,0)];//10000*coarse[OPS_ACC0(-1,0)]+100*coarse[OPS_ACC0(0,0)]+coarse[OPS_ACC0(1,0)];
}

void prolong_check(const double *val, int *idx, int *err, const int *sizex, const int *sizey) {
  int lerr = 0;
  lerr |= (val[OPS_ACC0(0,0)] != idx[0]/4 + (idx[1]/4)*(*sizex/4));
  if (lerr) printf("ERR (%d, %d): value %g, expected %d\n", idx[0], idx[1], val[OPS_ACC0(0,0)], idx[0]/4 + (idx[1]/4)*(*sizex/4));
  int xm = (idx[0]-1)<0 ? *sizex-1 : idx[0]-1;
  int xp = (idx[0]+1)>=*sizex ? 0 : idx[0]+1;
  int ym = (idx[1]-1)<0 ? *sizey-1 : idx[1]-1;
  int yp = (idx[1]+1)>=*sizey ? 0 : idx[1]+1;
  lerr |= (val[OPS_ACC0(1,0)] != xp/4 + (idx[1]/4)*(*sizex/4));
  if (lerr) printf("ERR (%d+1, %d): value %g, expected %d\n", idx[0], idx[1], val[OPS_ACC0(1,0)], xp/4 + (idx[1]/4)*(*sizex/4));
  lerr |= (val[OPS_ACC0(-1,0)] != xm/4 + (idx[1]/4)*(*sizex/4));
  if (lerr) printf("ERR (%d-1, %d): value %g, expected %d\n", idx[0], idx[1], val[OPS_ACC0(-1,0)], xm/4 + (idx[1]/4)*(*sizex/4));
  lerr |= (val[OPS_ACC0(0,1)] != idx[0]/4 + (yp/4)*(*sizex/4));
  if (lerr) printf("ERR (%d, %d+1): value %g, expected %d\n", idx[0], idx[1], val[OPS_ACC0(0,1)], idx[0]/4 + (xm/4)*(*sizex/4));
  lerr |= (val[OPS_ACC0(0,-1)] != idx[0]/4 + (ym/4)*(*sizex/4));
  if (lerr) printf("ERR (%d, %d-1): value %g, expected %d\n", idx[0], idx[1], val[OPS_ACC0(0,-1)], idx[0]/4 + (ym/4)*(*sizex/4));

  if (lerr != 0) *err = 1;
  else *err = 0;
  
}

void restrict_check(const double *val, int *idx, int *err, const int *sizex) {
  if (val[OPS_ACC0(0,0)] != idx[0]*4 + idx[1]*4**sizex) {
    printf("ERR (%d, %d): value %g expected %d\n", idx[0], idx[1], val[OPS_ACC0(0,0)], idx[0]*4 + idx[1]*4**sizex);
    *err = 1;
  } else
    *err = 0;
}

#endif //MGRID_PROLONG_KERNELS_H
