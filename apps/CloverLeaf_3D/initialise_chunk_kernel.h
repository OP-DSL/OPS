#ifndef INITIALISE_CHUNK_KERNEL_H
#define INITIALISE_CHUNK_KERNEL_H

#include "data.h"
#include "definitions.h"

void initialise_chunk_kernel_x(double *vertexx, const int *xx, double *vertexdx) {
  int x_min=field.x_min-2;
  int x_max=field.x_max-2;

  double min_x, d_x;
  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  min_x=grid.xmin+d_x*field.left;

  vertexx[OPS_ACC0(0,0,0)] = min_x + d_x * (xx[OPS_ACC1(0,0,0)] - x_min);
  vertexdx[OPS_ACC2(0,0,0)] = (double)d_x;
}

void initialise_chunk_kernel_y(double *vertexy, const int *yy, double *vertexdy) {
  int y_min=field.y_min-2;
  int y_max=field.y_max-2;

  double min_y, d_y;
  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;
  min_y=grid.ymin+d_y*field.bottom;

  vertexy[OPS_ACC0(0,0,0)] = min_y + d_y * (yy[OPS_ACC1(0,0,0)] - y_min);
  vertexdy[OPS_ACC2(0,0,0)] = (double)d_y;
}

void initialise_chunk_kernel_z(double *vertexz, const int *zz, double *vertexdz) {
  int z_min=field.z_min-2;
  int z_max=field.z_max-2;

  double min_z, d_z;
  d_z = (grid.zmax - grid.zmin)/(double)grid.z_cells;
  min_z=grid.zmin+d_z*field.back;

  vertexz[OPS_ACC0(0,0,0)] = min_z + d_z * (zz[OPS_ACC1(0,0,0)] - z_min);
  vertexdz[OPS_ACC2(0,0,0)] = (double)d_z;
}

void initialise_chunk_kernel_cellx(const double *vertexx, double* cellx, double *celldx) {
  double d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  cellx[OPS_ACC1(0,0,0)]  = 0.5*( vertexx[OPS_ACC0(0,0,0)] + vertexx[OPS_ACC0(1,0,0)] );
  celldx[OPS_ACC2(0,0,0)]  = d_x;
}

void initialise_chunk_kernel_celly(const double *vertexy, double* celly, double *celldy) {
  double d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;
  celly[OPS_ACC1(0,0,0)]  = 0.5*( vertexy[OPS_ACC0(0,0,0)] + vertexy[OPS_ACC0(0,1,0)] );
  celldy[OPS_ACC2(0,0,0)]  = d_y;
}

void initialise_chunk_kernel_cellz(const double *vertexz, double* cellz, double *celldz) {
  double d_z = (grid.zmax - grid.zmin)/(double)grid.z_cells;
  cellz[OPS_ACC1(0,0,0)]  = 0.5*( vertexz[OPS_ACC0(0,0,0)] + vertexz[OPS_ACC0(0,0,1)] );
  celldz[OPS_ACC2(0,0,0)]  = d_z;
}

void initialise_chunk_kernel_volume(double *volume, const double *celldy, double *xarea,
                                         const double *celldx, double *yarea, const double *celldz, double *zarea) {

  double d_x, d_y, d_z;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;
  d_z = (grid.zmax - grid.zmin)/(double)grid.z_cells;

  volume[OPS_ACC0(0,0,0)] = d_x*d_y*d_z;
  xarea[OPS_ACC2(0,0,0)] = celldy[OPS_ACC1(0,0,0)]*celldz[OPS_ACC5(0,0,0)];
  yarea[OPS_ACC4(0,0,0)] = celldx[OPS_ACC3(0,0,0)]*celldz[OPS_ACC5(0,0,0)];
  zarea[OPS_ACC6(0,0,0)] = celldx[OPS_ACC3(0,0,0)]*celldy[OPS_ACC1(0,0,0)];
}


#endif
