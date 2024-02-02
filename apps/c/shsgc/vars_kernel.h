#ifndef VARS_KERNEL_H
#define VARS_KERNEL_H

#include "vars.h"


void vars_kernel(const ACC<double>& alam, const ACC<double>& al, const ACC<double> &gt, ACC<double>& cmp,  ACC<double>& cf) {

  double  anu, aaa, ga, qf, ww;
  for (int m=0; m < 3 ;m++) {
			anu = alam(m,0);
			aaa = al(m,0);
			ga = aaa * ( gt(m,1) - gt(m,0)) / (pow(aaa,2.0) + del2);
			qf = sqrt ( con + pow(anu,2.0));
			cmp(m,0) = 0.50 * qf;
			ww = anu + cmp(m,0) * ga;
			qf = sqrt(con + pow(ww,2.0));
			cf(m,0) = qf;
		}
}
#endif
