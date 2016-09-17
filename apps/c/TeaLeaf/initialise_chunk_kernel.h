#ifndef INITIALISE_CHUNK_KERNEL_H
#define INITIALISE_CHUNK_KERNEL_H

#include "data.h"
#include "definitions.h"


void initialise_chunk_kernel_zero(double *var) {
  *var = 0.0;
}

void initialise_chunk_kernel_zero_x(double *var) {
  *var = 0.0;
}

void initialise_chunk_kernel_zero_y(double *var) {
  *var = 0.0;
}

void initialise_chunk_kernel_x(double *vertexx, const int *xx, double *vertexdx) {

  int x_min=field.x_min-2;
  double min_x, d_x;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  min_x=grid.xmin+d_x*field.left;

  vertexx[OPS_ACC0(0,0)] = min_x + d_x * (xx[OPS_ACC1(0,0)] - x_min);
  vertexdx[OPS_ACC2(0,0)] = (double)d_x;
}

void initialise_chunk_kernel_y(double *vertexy, const int *yy, double *vertexdy) {

  int y_min=field.y_min-2;
  double min_y, d_y;

  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;
  min_y=grid.ymin+d_y*field.bottom;

  vertexy[OPS_ACC0(0,0)] = min_y + d_y * (yy[OPS_ACC1(0,0)] - y_min);
  vertexdy[OPS_ACC2(0,0)] = (double)d_y;
}

void initialise_chunk_kernel_xx(int *xx, int *idx) {
  xx[OPS_ACC0(0,0)] = idx[0]-2;
}

void initialise_chunk_kernel_yy(int *yy, int *idx) {
  yy[OPS_ACC0(0,0)] = idx[1]-2;
}


void initialise_chunk_kernel_cellx(const double *vertexx, double* cellx, double *celldx) {

  double d_x;
  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;

  cellx[OPS_ACC1(0,0)]  = 0.5*( vertexx[OPS_ACC0(0,0)] + vertexx[OPS_ACC0(1,0)] );
  celldx[OPS_ACC2(0,0)]  = d_x;

}

void initialise_chunk_kernel_celly(const double *vertexy, double *celly, double *celldy) {

  double d_y;
  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;

  celly[OPS_ACC1(0,0)] = 0.5*( vertexy[OPS_ACC0(0,0)]+ vertexy[OPS_ACC0(0,1)] );
  celldy[OPS_ACC2(0,0)] = d_y;

  //printf("d_y %lf celldy %lf ",d_y, celldy[OPS_ACC2(0,0)]);

}

void initialise_chunk_kernel_volume(double *volume, const double *celldy, double *xarea,
                                         const double *celldx, double *yarea) {

  double d_x, d_y;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;

  volume[OPS_ACC0(0,0)] = d_x*d_y;
  xarea[OPS_ACC2(0,0)] = celldy[OPS_ACC1(0,0)];
  yarea[OPS_ACC4(0,0)] = celldx[OPS_ACC3(0,0)];
}


#endif
