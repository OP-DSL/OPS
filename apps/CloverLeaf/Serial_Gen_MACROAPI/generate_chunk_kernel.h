#ifndef GENERATE_CHUNK_KERNEL_H
#define GENERATE_CHUNK_KERNEL_H

#include "data.h"
#include "definitions.h"

inline void generate_chunk_kernel( double **vertexx, double **vertexy,
                     double **energy0, double **density0,
                     double **xvel0,  double **yvel0,
                     double **cellx, double **celly) {

  double radius, x_cent, y_cent;



  //State 1 is always the background state

  **energy0= states[0]->energy;
  **density0= states[0]->density;
  **xvel0=states[0]->xvel;
  **yvel0=states[0]->yvel;

  for(int i = 1; i<number_of_states; i++) {

    x_cent=states[i]->xmin;
    y_cent=states[i]->ymin;

    if (states[i]->geometry == g_rect) {
      if(*vertexx[1] >= states[i]->xmin  && *vertexx[0] < states[i]->xmax) {
        if(*vertexy[1] >= states[i]->ymin && *vertexy[0] < states[i]->ymax) {

          **energy0 = states[i]->energy;
          **density0 = states[i]->density;

          *xvel0[0] = states[i]->xvel;
          *xvel0[1] = states[i]->xvel;
          *xvel0[2] = states[i]->xvel;
          *xvel0[3] = states[i]->xvel;

          *yvel0[0] = states[i]->yvel;
          *yvel0[1] = states[i]->yvel;
          *yvel0[2] = states[i]->yvel;
          *yvel0[3] = states[i]->yvel;
        }
      }

    }
    else if(states[i]->geometry == g_circ) {
      radius = sqrt ((**cellx - x_cent) * (**cellx - x_cent) +
                     (**celly - y_cent) * (**celly - y_cent));
      if(radius <= states[i]->radius) {
        **energy0 = states[i]->energy;
        **density0 = states[i]->density;

        *xvel0[0] = states[i]->xvel;
        *xvel0[1] = states[i]->xvel;
        *xvel0[2] = states[i]->xvel;
        *xvel0[3] = states[i]->xvel;

        *yvel0[0] = states[i]->yvel;
        *yvel0[1] = states[i]->yvel;
        *yvel0[2] = states[i]->yvel;
        *yvel0[3] = states[i]->yvel;
      }
    }
    else if(states[i]->geometry == g_point) {
      if(*vertexx[0] == x_cent && *vertexy[0] == y_cent) {
        **energy0 = states[i]->energy;
        **density0 = states[i]->density;

        *xvel0[0] = states[i]->xvel;
        *xvel0[1] = states[i]->xvel;
        *xvel0[2] = states[i]->xvel;
        *xvel0[3] = states[i]->xvel;

        *yvel0[0] = states[i]->yvel;
        *yvel0[1] = states[i]->yvel;
        *yvel0[2] = states[i]->yvel;
        *yvel0[3] = states[i]->yvel;
      }
    }
  }
}

#endif
