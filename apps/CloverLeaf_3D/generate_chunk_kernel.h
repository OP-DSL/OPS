#ifndef GENERATE_CHUNK_KERNEL_H
#define GENERATE_CHUNK_KERNEL_H

#include "data.h"
#include "definitions.h"



void generate_chunk_kernel( const double *vertexx,
                     const double *vertexy, const double *vertexz,
                     double *energy0, double *density0,
                     double *xvel0,  double *yvel0, double *zvel0,
                     const double *cellx, const double *celly, const double *cellz) {

  double radius, x_cent, y_cent, z_cent;

  //State 1 is always the background state

  energy0[OPS_ACC3(0,0,0)]= states[0].energy;
  density0[OPS_ACC4(0,0,0)]= states[0].density;
  xvel0[OPS_ACC5(0,0,0)]=states[0].xvel;
  yvel0[OPS_ACC6(0,0,0)]=states[0].yvel;
  zvel0[OPS_ACC7(0,0,0)]=states[0].zvel;

  for(int i = 1; i<number_of_states; i++) {

    x_cent=states[i].xmin;
    y_cent=states[i].ymin;
    z_cent=states[i].zmin;

    if (states[i].geometry == g_cube) {
      if(vertexx[OPS_ACC0(1,0,0)] >= states[i].xmin  && vertexx[OPS_ACC0(0,0,0)] < states[i].xmax) {
        if(vertexy[OPS_ACC1(0,1,0)] >= states[i].ymin && vertexy[OPS_ACC1(0,0,0)] < states[i].ymax) {
          if(vertexz[OPS_ACC2(0,0,1)] >= states[i].zmin && vertexz[OPS_ACC2(0,0,0)] < states[i].zmax) {

            energy0[OPS_ACC3(0,0,0)] = states[i].energy;
            density0[OPS_ACC4(0,0,0)] = states[i].density;

            for (int ix=0;ix<2;ix++){
              for (int iy=0;iy<2;iy++){
                for (int iz=0;iz<2;iz++){
                  xvel0[OPS_ACC5(ix,iy,iz)] = states[i].xvel;
                  yvel0[OPS_ACC6(ix,iy,iz)] = states[i].yvel;
                  zvel0[OPS_ACC7(ix,iy,iz)] = states[i].zvel;
                }
              }
            }
          }
        }
      }
    }
    else if(states[i].geometry == g_sphe) {
      radius = sqrt ((cellx[OPS_ACC8(0,0,0)] - x_cent) * (cellx[OPS_ACC8(0,0,0)] - x_cent) +
                     (celly[OPS_ACC9(0,0,0)] - y_cent) * (celly[OPS_ACC9(0,0,0)] - y_cent) +
                     (cellz[OPS_ACC10(0,0,0)] - z_cent) * (cellz[OPS_ACC10(0,0,0)] - z_cent));
      if(radius <= states[i].radius) {
        energy0[OPS_ACC3(0,0,0)] = states[i].energy;
        density0[OPS_ACC4(0,0,0)] = states[i].density;

        for (int ix=0;ix<2;ix++){
          for (int iy=0;iy<2;iy++){
            for (int iz=0;iz<2;iz++){
              xvel0[OPS_ACC5(ix,iy,iz)] = states[i].xvel;
              yvel0[OPS_ACC6(ix,iy,iz)] = states[i].yvel;
              zvel0[OPS_ACC7(ix,iy,iz)] = states[i].zvel;
            }
          }
        }
      }
    }
    else if(states[i].geometry == g_point) {
      if(vertexx[OPS_ACC0(0,0,0)] == x_cent && vertexy[OPS_ACC1(0,0,0)] == y_cent && vertexz[OPS_ACC2(0,0,0)] == z_cent) {
        energy0[OPS_ACC3(0,0,0)] = states[i].energy;
        density0[OPS_ACC4(0,0,0)] = states[i].density;

        for (int ix=0;ix<2;ix++){
          for (int iy=0;iy<2;iy++){
            for (int iz=0;iz<2;iz++){
              xvel0[OPS_ACC5(ix,iy,iz)] = states[i].xvel;
              yvel0[OPS_ACC6(ix,iy,iz)] = states[i].yvel;
              zvel0[OPS_ACC7(ix,iy,iz)] = states[i].zvel;
            }
          }
        }
      }
    }
  }
}


#endif
