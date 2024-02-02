#ifndef INITIALISE_CHUNK_KERNEL_H
#define INITIALISE_CHUNK_KERNEL_H

#include "data.h"
#include "definitions.h"

void initialise_chunk_kernel_x(ACC<double> &vertexx, const ACC<int> &xx, ACC<double> &vertexdx) {

  int x_min=field.x_min-2;
  double min_x, d_x;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  min_x=grid.xmin+d_x*field.left;

  vertexx(0,0) = min_x + d_x * (xx(0,0) - x_min);
  vertexdx(0,0) = (double)d_x;
}

void initialise_chunk_kernel_y(ACC<double> &vertexy, const ACC<int> &yy, ACC<double> &vertexdy) {

  int y_min=field.y_min-2;
  double min_y, d_y;

  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;
  min_y=grid.ymin+d_y*field.bottom;

  vertexy(0,0) = min_y + d_y * (yy(0,0) - y_min);
  vertexdy(0,0) = (double)d_y;
}

void initialise_chunk_kernel_xx(ACC<int> &xx, int *idx) {
  xx(0,0) = idx[0]-2;
}

void initialise_chunk_kernel_yy(ACC<int> &yy, int *idx) {
  yy(0,0) = idx[1]-2;
}


void initialise_chunk_kernel_cellx(const ACC<double> &vertexx, ACC<double> &cellx, ACC<double> &celldx) {

  double d_x;
  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;

  cellx(0,0)  = 0.5*( vertexx(0,0) + vertexx(1,0) );
  celldx(0,0)  = d_x;

}

void initialise_chunk_kernel_celly(const ACC<double> &vertexy, ACC<double> &celly, ACC<double> &celldy) {

  double d_y;
  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;

  celly(0,0) = 0.5*( vertexy(0,0)+ vertexy(0,1) );
  celldy(0,0) = d_y;

  //printf("d_y %lf celldy %lf ",d_y, celldy(0,0));

}

void initialise_chunk_kernel_volume(ACC<double> &volume, const ACC<double> &celldy, ACC<double> &xarea,
                                         const ACC<double> &celldx, ACC<double> &yarea) {

  double d_x, d_y;

  d_x = (grid.xmax - grid.xmin)/(double)grid.x_cells;
  d_y = (grid.ymax - grid.ymin)/(double)grid.y_cells;

  volume(0,0) = d_x*d_y;
  xarea(0,0) = celldy(0,0);
  yarea(0,0) = celldx(0,0);
}


#endif
