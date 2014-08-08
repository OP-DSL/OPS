#ifndef MBLOCK_KERNEL_H
#define MBLOCK_KERNEL_H

void mblock_populate_kernel(double *val, int *idx) {
  val[OPS_ACC0(0,0)] = (double)(idx[0]+20*idx[1]);
}


#endif //MBLOCK_KERNEL_H
