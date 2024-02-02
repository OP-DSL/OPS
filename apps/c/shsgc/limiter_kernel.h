#ifndef LIMITER_KERNEL_H
#define LIMITER_KERNEL_H

#include "vars.h"


void limiter_kernel(const ACC<double>& al, ACC<double> &tht, ACC<double>& gt) {

  double aalm, aal, all, ar, gtt;
  for (int m=0; m < 3 ;m++) {
    aalm = fabs(al(m,-1));
    aal = fabs(al(m,0));
    tht(m,0) = fabs (aal - aalm) / (aal + aalm + del2);
    all = al(m,-1);
    ar = al(m,0);
    gtt = all * ( ar * ar + del2 ) + ar * (all * all + del2);
    gt(m,0)= gtt / (ar * ar + all * all + 2.00 * del2);
  }
}
#endif
