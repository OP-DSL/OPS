#ifndef INITIALISE_CHUNK_KERNEL_H
#define INITIALISE_CHUNK_KERNEL_H

#include "data.h"
#include "definitions.h"

inline void initialise_chunk_kernel_x(double **vertexx, int **xx, double **vertexdx) {

  int x_min=field->x_min;
  int x_max=field->x_max;
  int y_min=field->y_min;
  int y_max=field->y_max;

  double min_x, min_y, d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  min_x=grid->xmin+d_x*field->left;
  min_y=grid->ymin+d_y*field->bottom;

  **vertexx = min_x + d_x *(double)(((int)(**xx)) - x_min);
  **vertexdx = (double)d_x;
  //printf("%lf ", **vertexx);
}


inline void initialise_chunk_kernel_y(double **vertexy, int **yy, double **vertexdy) {

  int x_min=field->x_min;
  int x_max=field->x_max;
  int y_min=field->y_min;
  int y_max=field->y_max;

  double min_x, min_y, d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  min_x=grid->xmin+d_x*field->left;
  min_y=grid->ymin+d_y*field->bottom;

  **vertexy = min_y + d_y * (double)(((int)(**yy)) - y_min);
  **vertexdy = (double)d_y;
  //printf("%lf ", **vertexy);
}


inline void initialise_chunk_kernel_cellx(double **vertexx, double** cellx, double **celldx) {

  int x_min=field->x_min;
  int x_max=field->x_max;;
  int y_min=field->y_min;
  int y_max=field->y_max;

  double min_x, min_y, d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  min_x=grid->xmin+d_x;
  min_y=grid->ymin+d_y;

  **cellx = 0.5*( *vertexx[0] + *vertexx[1] );
  **celldx = d_x;
  //printf("%lf %lf\n", *vertexx[0], *vertexx[1]);

}

inline void initialise_chunk_kernel_celly(double **vertexy, double** celly, double **celldy) {

  int x_min=field->x_min;
  int x_max=field->x_max;;
  int y_min=field->y_min;
  int y_max=field->y_max;

  double min_x, min_y, d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  min_x=grid->xmin+d_x;
  min_y=grid->ymin+d_y;

  **celly = 0.5*( *vertexy[0] + *vertexy[1] );
  **celldy = d_y;
  //printf("%lf ", (double)**celldy);

}

inline void initialise_chunk_kernel_volume(double **volume, double **celldy, double **xarea,
                                         double **celldx, double **yarea) {

  double d_x, d_y;

  d_x = (grid->xmax - grid->xmin)/(double)grid->x_cells;
  d_y = (grid->ymax - grid->ymin)/(double)grid->y_cells;

  **volume = d_x*d_y;
  **xarea=**celldy;
  **yarea=**celldx;
  //printf("%lf ", (double)**xarea);
}


#endif
