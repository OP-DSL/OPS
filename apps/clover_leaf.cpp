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

/** @brief CloverLeaf top level program: Invokes the main cycle
  * @author Wayne Gaudin
  * @details CloverLeaf in a proxy-app that solves the compressible Euler
  *  Equations using an explicit finite volume method on a Cartesian grid.
  *  The grid is staggered with internal energy, density and pressure at cell
  *  centres and velocities on cell vertices.

  *  A second order predictor-corrector method is used to advance the solution
  *  in time during the Lagrangian phase. A second order advective remap is then
  *  carried out to return the mesh to an orthogonal state.
  *
  *  NOTE: that the proxy-app uses uniformly spaced mesh. The actual method will
  *  work on a mesh with varying spacing to keep it relevant to it's parent code.
  *  For this reason, optimisations should only be carried out on the software
  *  that do not change the underlying numerical method. For example, the
  *  volume, though constant for all cells, should remain array and not be
  *  converted to a scalar.
  *
  *  This version is based on C/C++ and uses the OPS prototype highlevel domain
  *  specific API for developing Structured mesh applications
  */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

// Cloverleaf constants
#include "data.h"

// Cloverleaf definitions
#include "definitions.h"

//Cloverleaf kernels
#include "test_kernel.h"

// Cloverleaf functions
void read_input();
void initialise();



/******************************************************************************
* Initialize Global constants and variables
/******************************************************************************/
float   g_version;
int     g_ibig = 640000;
double  g_small = 1.0e-16;
double  g_big  = 1.0e+21;
int     g_name_len_max = 255 ,
        g_xdir = 1,
        g_ydir = 2;

        //These two need to be kept consistent with update_halo
int     CHUNK_LEFT    = 1,
        CHUNK_RIGHT   = 2,
        CHUNK_BOTTOM  = 3,
        CHUNK_TOP     = 4,
        EXTERNAL_FACE = -1;

int     FIELD_DENSITY0   = 1,
        FIELD_DENSITY1   = 2,
        FIELD_ENERGY0    = 3,
        FIELD_ENERGY1    = 4,
        FIELD_PRESSURE   = 5,
        FIELD_VISCOSITY  = 6,
        FIELD_SOUNDSPEED = 7,
        FIELD_XVEL0      = 8,
        FIELD_XVEL1      = 9,
        FIELD_YVEL0      =10,
        FIELD_YVEL1      =11,
        FIELD_VOL_FLUX_X =12,
        FIELD_VOL_FLUX_Y =13,
        FIELD_MASS_FLUX_X=14,
        FIELD_MASS_FLUX_Y=15,
        NUM_FIELDS       =15;

FILE    *g_out, *g_in;  //Files for input and output

int     g_rect=1,
        g_circ=2,
        g_point=3;

state_type * states;



/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{
  // OPS initialisation
  ops_init(argc,argv,5);
  ops_printf("Clover version %f\n", g_version);

  //declare blocks
  int x_cells = 10;
  int y_cells = 2;
  int dims[2] = {x_cells, y_cells};  //cloverleaf 2D block dimensions
  ops_block clover_grid = ops_decl_block(2, dims, "grid");

  //declare edges of block
  ops_block clover_xedge = ops_decl_block(1, &dims[0], "xedge");
  ops_block clover_yedge = ops_decl_block(1, &dims[1], "yedge");

  //declare data on blocks
  int x_min = 1;
  int y_min = 1;
  int x_max = x_cells;
  int y_max = y_cells;
  int offset[2] = {-2,-2};
  int size[2] = {(x_max+2)-(x_min-2), (y_max+2)-(y_min-2)};
  double* temp = NULL;

  ops_dat dencity0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "density0");
  ops_dat dencity1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "density1");
  ops_dat energy0     = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "energy0");
  ops_dat energy1     = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "energy1");
  ops_dat pressure    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "pressure");
  ops_dat viscosity   = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "viscosity");
  ops_dat soundspeed  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "soundspeed");
  ops_dat volume      = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "volume");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  ops_dat xvel0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xvel0");
  ops_dat xvel1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xvel1");
  ops_dat yvel0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yvel0");
  ops_dat yvel1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yvel1");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+2)-(y_min-2);
  ops_dat vol_flux_x  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "vol_flux_x");
  ops_dat mass_flux_x = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "mass_flux_x");
  ops_dat xarea       = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xarea");


  size[0] = (x_max+2)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  ops_dat vol_flux_y  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "vol_flux_y");
  ops_dat mass_flux_y = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "mass_flux_y");
  ops_dat yarea       = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yarea");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  ops_dat work_array1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array1");
  ops_dat work_array2    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array2");
  ops_dat work_array3    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array3");
  ops_dat work_array4    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array4");
  ops_dat work_array5    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array5");
  ops_dat work_array6    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array6");
  ops_dat work_array7    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array7");


  int size2[1] = {(x_max+2)-(x_min-2)};
  int size3[1] = {(y_max+2)-(y_min-2)};
  int size4[1] = {(x_max+3)-(x_min-2)};
  int size5[1] = {(y_max+3)-(y_min-2)};
  int offset4[2] = {-2};
  ops_dat cellx    = ops_decl_dat(clover_xedge, 1, size2, offset4, temp, "double", "cellx");
  ops_dat celly    = ops_decl_dat(clover_yedge, 1, size3, offset4, temp, "double", "celly");
  ops_dat vertexx  = ops_decl_dat(clover_xedge, 1, size4, offset4, temp, "double", "vertexx");
  ops_dat vertexy  = ops_decl_dat(clover_yedge, 1, size5, offset4, temp, "double", "vertexy");
  ops_dat celldx   = ops_decl_dat(clover_xedge, 1, size2, offset4, temp, "double", "celldx");
  ops_dat celldy   = ops_decl_dat(clover_yedge, 1, size3, offset4, temp, "double", "celldy");
  ops_dat vertexdx = ops_decl_dat(clover_xedge, 1, size4, offset4, temp, "double", "vertexdx");
  ops_dat vertexdy = ops_decl_dat(clover_yedge, 1, size5, offset4, temp, "double", "vertexdy");

  ops_diagnostic_output();

  int self[] = {0};
  ops_stencil sten1 = ops_decl_stencil( 1, 1, self, "self");
  int range[] = {0, x_max};
  ops_par_loop(test_kernel, "test_kernel", 1, range,
               ops_arg_dat(cellx, sten1, OPS_WRITE));
  printf("\n\n");
  ops_par_loop(test_kernel2, "test_kernel2", 1, range,
               ops_arg_dat(cellx, sten1, OPS_WRITE));


  ops_exit();
}



