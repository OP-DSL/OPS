/* Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/** @brief Mesh chunk generation driver
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Invoked the users specified chunk generator.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq_opt.h"
#include "ops_seq_macro.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include "generate_chunk_kernel.h"


void generate_chunk_kernel_macro( double *vertexx, double *vertexy,
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


void generate()
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2};
  ops_par_loop_macro(generate_chunk_kernel_macro, "generate_chunk_kernel_macro", 2, rangexy,
    ops_arg_dat(vertexx,  s2D_00_P10_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(vertexy,  S2D_00_0P1_STRID2D_Y, "double", OPS_READ),
    ops_arg_dat(energy0,  S2D_00, "double", OPS_WRITE),
    ops_arg_dat(density0, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(xvel0,    S2D_00_P10_0P1_P1P1, "double", OPS_WRITE),
    ops_arg_dat(yvel0,    S2D_00_P10_0P1_P1P1, "double", OPS_WRITE),
    ops_arg_dat(cellx,    s2D_00_P10_STRID2D_X, "double", OPS_READ),
    ops_arg_dat(celly,    S2D_00_0P1_STRID2D_Y, "double", OPS_READ));

}
