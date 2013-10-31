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

/** @brief call the viscosity kernels
 *  @author Wayne Gaudin
 *  @details Calculates an artificial viscosity using the Wilkin's method to
 *  smooth out shock front and prevent oscillations around discontinuities.
 *  Only cells in compression will have a non-zero value.
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

#include "viscosity_kernel.h"


void viscosity_kernel_macro( double *xvel0, double *yvel0,
                       double *celldx, double *celldy,
                       double *pressure, double *density0,
                       double *viscosity) {

  double ugrad, vgrad,
         grad2,
         pgradx,pgrady,
         pgradx2,pgrady2,
         grad,
         ygrad, xgrad,
         div,
         strain2,
         limiter,
         pgrad;

  //int s2D_00_P10_0P1_P1P1[]  = {0,0, 1,0, 0,1, 1,1};

  ugrad = (xvel0[OPS_ACC0(1,0)] + xvel0[OPS_ACC0(1,1)]) - (xvel0[OPS_ACC0(0,0)] + xvel0[OPS_ACC0(0,1)]);
  vgrad = (yvel0[OPS_ACC1(0,1)] + yvel0[OPS_ACC1(1,1)]) - (yvel0[OPS_ACC1(0,0)] + yvel0[OPS_ACC1(1,0)]);

  div = (celldx[0,0])*(ugrad) + (celldy[0,0])*(vgrad);

  strain2 = 0.5*(xvel0[OPS_ACC0(0,1)] + xvel0[OPS_ACC0(1,1)] - xvel0[OPS_ACC0(0,0)] - xvel0[OPS_ACC0(1,0)])/(celldy[OPS_ACC3(0,0)]) +
            0.5*(yvel0[OPS_ACC1(1,0)] + yvel0[OPS_ACC1(1,1)] - yvel0[OPS_ACC1(0,0)] - yvel0[OPS_ACC1(0,1)])/(celldx[OPS_ACC2(0,0)]);


  //int s2D_10_M10_01_0M1[]  = {1,0, -1,0, 0,1, 0,-1};
  pgradx  = (pressure[OPS_ACC4(1,0)] - pressure[OPS_ACC4(-1,0)])/(celldx[OPS_ACC2(0,0)]+ celldx[OPS_ACC2(1,0)]);
  pgrady = (pressure[OPS_ACC4(0,1)] - pressure[OPS_ACC4(0,-1)])/(celldy[OPS_ACC3(0,0)]+ celldy[OPS_ACC3(0,1)]);

  pgradx2 = pgradx * pgradx;
  pgrady2 = pgrady * pgrady;

  limiter = ((0.5*(ugrad)/celldx[OPS_ACC2(0,0)]) * pgradx2 +
             (0.5*(vgrad)/celldy[OPS_ACC3(0,0)]) * pgrady2 +
              strain2 * pgradx * pgrady)/ MAX(pgradx2 + pgrady2 , 1.0e-16);

  if( (limiter > 0.0) || (div >= 0.0)) {
        viscosity[OPS_ACC6(0,0)] = 0.0;
  }
  else {
    pgradx = SIGN( MAX(1.0e-16, fabs(pgradx)), pgradx);
    pgrady = SIGN( MAX(1.0e-16, fabs(pgrady)), pgrady);
    pgrad = sqrt(pgradx*pgradx + pgrady*pgrady);
    xgrad = fabs(celldx[OPS_ACC2(0,0)] * pgrad/pgradx);
    ygrad = fabs(celldy[OPS_ACC3(0,0)] * pgrad/pgrady);
    grad  = MIN(xgrad,ygrad);
    grad2 = grad*grad;

    viscosity[OPS_ACC6(0,0)] = 2.0 * (density0[OPS_ACC5(0,0)]) * grad2 * limiter * limiter;
  }
}


void viscosity_func()
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy_inner[] = {x_min,x_max,y_min,y_max}; // inner range without border

  ops_par_loop_macro(viscosity_kernel_macro, "viscosity_kernel_macro", 2, rangexy_inner,
      ops_arg_dat(xvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
      ops_arg_dat(yvel0, S2D_00_P10_0P1_P1P1, "double", OPS_READ),
      ops_arg_dat(celldx, s2D_00_P10_STRID2D_X, "double", OPS_READ),
      ops_arg_dat(celldy, S2D_00_0P1_STRID2D_Y, "double", OPS_READ),
      ops_arg_dat(pressure, S2D_10_M10_01_0M1, "double", OPS_READ),
      ops_arg_dat(density0, S2D_00, "double", OPS_READ),
      ops_arg_dat(viscosity, S2D_00, "double", OPS_WRITE));
}
