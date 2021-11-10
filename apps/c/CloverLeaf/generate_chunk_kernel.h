#ifndef GENERATE_CHUNK_KERNEL_H
#define GENERATE_CHUNK_KERNEL_H

#include "data.h"
#include "definitions.h"



void generate_chunk_kernel( const ACC<double> &vertexx, const ACC<double> &vertexy,
                     ACC<double> &energy0, ACC<double> &density0,
                     ACC<double> &xvel0,  ACC<double> &yvel0,
                     const ACC<double> &cellx, const ACC<double> &celly) {

  double radius, x_cent, y_cent;
  int is_in = 0;
  int is_in2 = 0;
  //State 1 is always the background state

  energy0(0,0)= states[0].energy;
  density0(0,0)= states[0].density;
  xvel0(0,0)=states[0].xvel;
  yvel0(0,0)=states[0].yvel;

  for(int i = 1; i<number_of_states; i++) {

    x_cent=states[i].xmin;
    y_cent=states[i].ymin;
    is_in = 0;
    is_in2 = 0;

    if (states[i].geometry == g_rect) {
      for (int i1 = -1; i1 <= 0; i1++) {
        for (int j1 = -1; j1 <= 0; j1++) {
          if(vertexx(1+i1,0) >= states[i].xmin  && vertexx(0+i1,0) < states[i].xmax) {
            if(vertexy(0,1+j1) >= states[i].ymin && vertexy(0,0+j1) < states[i].ymax) {
              is_in = 1;
            }
          }
        }
      }
      if(vertexx(1,0) >= states[i].xmin  && vertexx(0,0) < states[i].xmax) {
        if(vertexy(0,1) >= states[i].ymin && vertexy(0,0) < states[i].ymax) {
          is_in2 = 1;
        }
      }
      if (is_in2) {
        energy0(0,0) = states[i].energy;
        density0(0,0) = states[i].density;
      }
      if (is_in) {
        xvel0(0,0) = states[i].xvel;
        yvel0(0,0) = states[i].yvel;
      }
    }
    else if(states[i].geometry == g_circ) {
      for (int i1 = -1; i1 <= 0; i1++) {
        for (int j1 = -1; j1 <= 0; j1++) {
          radius = sqrt ((cellx(i1,0) - x_cent) * (cellx(i1,0) - x_cent) +
                     (celly(0,j1) - y_cent) * (celly(0,j1) - y_cent));
          if (radius <= states[i].radius) {
            is_in = 1;
          }
        }
      }
      if (radius <= states[i].radius) is_in2 = 1;

      if (is_in2) {
        energy0(0,0) = states[i].energy;
        density0(0,0) = states[i].density;
      }

      if (is_in) {
        xvel0(0,0) = states[i].xvel;
        yvel0(0,0) = states[i].yvel;
      }
    }
    else if(states[i].geometry == g_point) {
      for (int i1 = -1; i1 <= 0; i1++) {
        for (int j1 = -1; j1 <= 0; j1++) {
          if(vertexx(i1,0) == x_cent && vertexy(0,j1) == y_cent) {
            is_in = 1;
          }
        }
      }
      if(vertexx(0,0) == x_cent && vertexy(0,0) == y_cent) 
        is_in2 = 1;

      if (is_in2) {
        energy0(0,0) = states[i].energy;
        density0(0,0) = states[i].density;
      } 

      if (is_in) {
        xvel0(0,0) = states[i].xvel;
        yvel0(0,0) = states[i].yvel;
      }
    }
  }
}


#endif
