#ifndef MBLOCK_KERNEL_H
#define MBLOCK_KERNEL_H

void mblock_populate_kernel(double *val, int *idx) {
  val[OPS_ACC0(0,0,0,0)] = (double)(idx[0] + 4*idx[1] + 16*idx[2] + 64*idx[3]);
}


#endif //MBLOCK_KERNEL_H
