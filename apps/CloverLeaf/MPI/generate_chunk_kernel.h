#ifndef GENERATE_CHUNK_KERNEL_H
#define GENERATE_CHUNK_KERNEL_H

#include "data.h"
#include "definitions.h"



void generate_chunk_kernel( double *vertexx, double *vertexy,
                     double *energy0, double *density0,
                     double *xvel0,  double *yvel0,
                     double *cellx, double *celly) {

  double radius, x_cent, y_cent;



  //State 1 is always the background state

  energy0[OPS_ACC2(0,0)]= states[0]->energy;
  density0[OPS_ACC3(0,0)]= states[0]->density;
  xvel0[OPS_ACC4(0,0)]=states[0]->xvel;
  yvel0[OPS_ACC5(0,0)]=states[0]->yvel;

  for(int i = 1; i<number_of_states; i++) {

    x_cent=states[i]->xmin;
    y_cent=states[i]->ymin;

    if (states[i]->geometry == g_rect) {
      if(vertexx[OPS_ACC0(1,0)] >= states[i]->xmin  && vertexx[OPS_ACC0(0,0)] < states[i]->xmax) {
        if(vertexy[OPS_ACC1(0,1)] >= states[i]->ymin && vertexy[OPS_ACC1(0,0)] < states[i]->ymax) {

          energy0[OPS_ACC2(0,0)] = states[i]->energy;
          density0[OPS_ACC3(0,0)] = states[i]->density;

          //S2D_00_P10_0P1_P1P1
          xvel0[OPS_ACC4(0,0)] = states[i]->xvel;
          xvel0[OPS_ACC4(1,0)] = states[i]->xvel;
          xvel0[OPS_ACC4(0,1)] = states[i]->xvel;
          xvel0[OPS_ACC4(1,1)] = states[i]->xvel;

          yvel0[OPS_ACC5(0,0)] = states[i]->yvel;
          yvel0[OPS_ACC5(1,0)] = states[i]->yvel;
          yvel0[OPS_ACC5(0,1)] = states[i]->yvel;
          yvel0[OPS_ACC5(1,1)] = states[i]->yvel;
        }
      }

    }
    else if(states[i]->geometry == g_circ) {
      radius = sqrt ((cellx[OPS_ACC6(0,0)] - x_cent) * (cellx[OPS_ACC6(0,0)] - x_cent) +
                     (celly[OPS_ACC7(0,0)] - y_cent) * (celly[OPS_ACC7(0,0)] - y_cent));
      if(radius <= states[i]->radius) {
        energy0[OPS_ACC2(0,0)] = states[i]->energy;
        density0[OPS_ACC3(0,0)] = states[i]->density;

        //S2D_00_P10_0P1_P1P1
        xvel0[OPS_ACC4(0,0)] = states[i]->xvel;
        xvel0[OPS_ACC4(1,0)] = states[i]->xvel;
        xvel0[OPS_ACC4(0,1)] = states[i]->xvel;
        xvel0[OPS_ACC4(1,1)] = states[i]->xvel;

        yvel0[OPS_ACC5(0,0)] = states[i]->yvel;
        yvel0[OPS_ACC5(1,0)] = states[i]->yvel;
        yvel0[OPS_ACC5(0,1)] = states[i]->yvel;
        yvel0[OPS_ACC5(1,1)] = states[i]->yvel;
      }
    }
    else if(states[i]->geometry == g_point) {
      if(vertexx[OPS_ACC0(0,0)] == x_cent && vertexy[OPS_ACC1(0,0)] == y_cent) {
        energy0[OPS_ACC2(0,0)] = states[i]->energy;
        density0[OPS_ACC3(0,0)] = states[i]->density;

        //S2D_00_P10_0P1_P1P1
        xvel0[OPS_ACC4(0,0)] = states[i]->xvel;
        xvel0[OPS_ACC4(1,0)] = states[i]->xvel;
        xvel0[OPS_ACC4(0,1)] = states[i]->xvel;
        xvel0[OPS_ACC4(1,1)] = states[i]->xvel;

        yvel0[OPS_ACC5(0,0)] = states[i]->yvel;
        yvel0[OPS_ACC5(1,0)] = states[i]->yvel;
        yvel0[OPS_ACC5(0,1)] = states[i]->yvel;
        yvel0[OPS_ACC5(1,1)] = states[i]->yvel;
      }
    }
  }
}


#endif
