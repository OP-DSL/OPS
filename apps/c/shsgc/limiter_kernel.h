#ifndef LIMITER_KERNEL_H
#define LIMITER_KERNEL_H

#include "vars.h"


void limiter_kernel(const double* al, double *tht, double* gt) {

  double aalm, aal, all, ar, gtt;
  for (int m=0; m < 3 ;m++) {
    aalm = fabs(al[OPS_ACC_MD0(m,-1)]);
    aal = fabs(al[OPS_ACC_MD0(m,0)]);
    tht[OPS_ACC_MD1(m,0)] = fabs (aal - aalm) / (aal + aalm + del2);
    all = al[OPS_ACC_MD0(m,-1)];
    ar = al[OPS_ACC_MD0(m,0)];
    gtt = all * ( ar * ar + del2 ) + ar * (all * all + del2);
    gt[OPS_ACC_MD2(m,0)]= gtt / (ar * ar + all * all + 2.00 * del2);
  }
}

#endif