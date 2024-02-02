#ifndef SET_FIELD_KERN_H
#define SET_FIELD_KERN_H

void set_field_kernel(const double *energy0, double *energy1) {
	energy1[OPS_ACC1(0,0)] = energy0[OPS_ACC0(0,0)];
}

#endif
