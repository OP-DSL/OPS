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
#include "initialise_chunk_kernel.h"

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

state_type states; //global variable holding state info

grid_type grid; //global variable holding global grid info

field_type field; //global variable holding info of fields


/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{
  //set up CLoverleaf problem -- need to fill in through I/O
  grid = (grid_type ) xmalloc(sizeof(grid_type_core));
  grid->x_cells = 10;
  grid->y_cells = 2;
  grid->xmin = 0;
  grid->xmax = grid->x_cells;
  grid->ymin = 0;
  grid->ymax = grid->y_cells;

  field = (field_type ) xmalloc(sizeof(field_type_core));
  field->x_min = 1;
  field->y_min = 1;
  field->x_max = grid->x_cells;
  field->y_max = grid->y_cells;

  // OPS initialisation
  ops_init(argc,argv,5);
  ops_printf("Clover version %f\n", g_version);

  //declare blocks
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = grid->xmin;
  int x_max = grid->xmax;
  int y_min = grid->ymin;
  int y_max = grid->ymax;
  int dims[2] = {x_cells, y_cells};  //cloverleaf 2D block dimensions
  ops_block clover_grid = ops_decl_block(2, dims, "grid");

  //declare edges of block
  ops_block clover_xedge = ops_decl_block(1, &dims[0], "xedge");
  ops_block clover_yedge = ops_decl_block(1, &dims[1], "yedge");

  //declare data on blocks
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

  //initialize chunk


  int self[] = {0};
  ops_stencil sten1 = ops_decl_stencil( 1, 1, self, "self");

  int range[] = {x_min-2, x_max+3};
  ops_par_loop(initialise_chunk_kernel, "initialise_chunk_kernel", 1, range,
               ops_arg_dat(vertexx, sten1, OPS_WRITE),
               ops_arg_dat(vertexdx, sten1, OPS_WRITE));

  printf("\n\n");
  ops_par_loop(test_kernel2, "test_kernel2", 1, range,
               ops_arg_dat(vertexx, sten1, OPS_READ));
  ops_par_loop(test_kernel2, "test_kernel2", 1, range,
               ops_arg_dat(vertexdx, sten1, OPS_READ));
  printf("\n\n");

  int range2[] = {x_min-2, x_max+3, y_min-2, y_max+3};
  //need to declare a 2D loop that accesses vertexx with x_range and vertexy with yrange
  //need a strided stencil
  //int stridex[] = {1,0};
  //int stridey[] = {0,1}

/*
  int self2[] = {0,0};
  ops_stencil sten2 = ops_decl_stencil( 2, 1, self2, "self");
  int range2[] = {x_min-2, x_max+2, y_min-2, y_max+2};
  ops_par_loop(test_kernel, "test_kernel", 2, range2,
               ops_arg_dat(dencity0, sten2, OPS_WRITE));
  printf("\n\n");
  ops_par_loop(test_kernel2, "test_kernel2", 2, range2,
               ops_arg_dat(dencity0, sten2, OPS_WRITE));
  printf("\n\n");
*/

  //initialize chunk


  ops_exit();
}



