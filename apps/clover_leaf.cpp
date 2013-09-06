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



// Cloverleaf functions
void initialise();
void generate();
void ideal_gas(int predict);
void update_halo(int* fields, int depth);
void field_summary();



/******************************************************************************
* Initialize Global constants and variables
/******************************************************************************/


/**----------Cloverleaf Vars/Consts--------------**/

float   g_version = 1.0;
int     g_ibig = 640000;
double  g_small = 1.0e-16;
double  g_big  = 1.0e+21;
int     g_name_len_max = 255 ,
        g_xdir = 1,
        g_ydir = 2;

int     number_of_states;

        //These two need to be kept consistent with update_halo
int     CHUNK_LEFT    = 1,
        CHUNK_RIGHT   = 2,
        CHUNK_BOTTOM  = 3,
        CHUNK_TOP     = 4,
        EXTERNAL_FACE = -1;

int     FIELD_DENSITY0   = 0,
        FIELD_DENSITY1   = 1,
        FIELD_ENERGY0    = 2,
        FIELD_ENERGY1    = 3,
        FIELD_PRESSURE   = 4,
        FIELD_VISCOSITY  = 5,
        FIELD_SOUNDSPEED = 6,
        FIELD_XVEL0      = 7,
        FIELD_XVEL1      = 8,
        FIELD_YVEL0      = 9,
        FIELD_YVEL1      =10,
        FIELD_VOL_FLUX_X =11,
        FIELD_VOL_FLUX_Y =12,
        FIELD_MASS_FLUX_X=13,
        FIELD_MASS_FLUX_Y=14,
        NUM_FIELDS       =15;

FILE    *g_out, *g_in;  //Files for input and output

int     g_rect=1,
        g_circ=2,
        g_point=3;

state_type * states; //global variable holding state info

grid_type grid; //global variable holding global grid info

field_type field; //global variable holding info of fields

int step = 0;
int advect_x; //logical
int error_condition;
int test_problem;
int complete; //logical

int fields[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


#include "cloverleaf_ops_vars.h"


/******************************************************************************
* Main program
/******************************************************************************/
int main(int argc, char **argv)
{

  /**--------------------Set up Cloverleaf default problem-------------------**/

  //need to fill these in through I/O
  grid = (grid_type ) xmalloc(sizeof(grid_type_core));
  grid->x_cells = 10;
  grid->y_cells = 2;

  grid->xmin = 0;
  grid->ymin = 0;
  grid->xmax = grid->x_cells;
  grid->ymax = grid->y_cells;

  field = (field_type ) xmalloc(sizeof(field_type_core));
  field->x_min = 0;
  field->y_min = 0;
  field->x_max = grid->x_cells;
  field->y_max = grid->y_cells;
  field->left = 0;
  field->bottom = 0;

  number_of_states = 2;
  states =  (state_type *) xmalloc(sizeof(state_type) * number_of_states);

  //state 1
  states[0] = (state_type ) xmalloc(sizeof(state_type_core));
  states[0]->density = 0.2;
  states[0]->energy = 1.0;
  states[0]->xvel = 0.0;
  states[0]->yvel = 0.0;

  //state 2
  states[1] = (state_type ) xmalloc(sizeof(state_type_core));
  states[1]->density=1.0;
  states[1]->energy=2.5;
  states[1]->xvel = 0.0;
  states[1]->yvel = 0.0;
  states[1]->geometry=g_rect;
  states[1]->xmin=0.0;
  states[1]->xmax=5.0;
  states[1]->ymin=0.0;
  states[1]->ymax=2.0;

  float dx= (grid->xmax-grid->xmin)/(float)(grid->x_cells);
  float dy= (grid->ymax-grid->ymin)/(float)(grid->y_cells);

  for(int i = 0; i < number_of_states; i++)
  {
    states[i]->xmin = states[i]->xmin + (dx/100.00);
    states[i]->ymin = states[i]->ymin + (dy/100.00);
    states[i]->xmax = states[i]->xmax - (dx/100.00);
    states[i]->ymax = states[i]->ymax - (dy/100.00);
  }

  NUM_FIELDS = 15;


  /**-------------------OPS Initialisation and Declarations------------------**/

  // OPS initialisation
  ops_init(argc,argv,5);
  ops_printf("Clover version %f\n", g_version);

  //
  //declare blocks
  //
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;
  int dims[2] = {x_cells, y_cells};  //cloverleaf 2D block dimensions
  ops_block clover_grid = ops_decl_block(2, dims, "grid");

  //declare edges of block
  dims[0] = x_cells; dims[1] = 1;
  clover_xedge = ops_decl_block(2, dims, "xedge");

  dims[0] = 1; dims[1] = y_cells;
  clover_yedge = ops_decl_block(2, dims, "yedge");

  //
  //declare data on blocks
  //
  int offset[2] = {-2,-2};
  int size[2] = {(x_max+2)-(x_min-2), (y_max+2)-(y_min-2)};
  double* temp = NULL;

  density0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "density0");
  density1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "density1");
  energy0     = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "energy0");
  energy1     = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "energy1");
  pressure    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "pressure");
  viscosity   = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "viscosity");
  soundspeed  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "soundspeed");
  volume      = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "volume");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  xvel0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xvel0");
  xvel1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xvel1");
  yvel0    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yvel0");
  yvel1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yvel1");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+2)-(y_min-2);
  vol_flux_x  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "vol_flux_x");
  mass_flux_x = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "mass_flux_x");
  xarea       = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "xarea");

  size[0] = (x_max+2)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  vol_flux_y  = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "vol_flux_y");
  mass_flux_y = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "mass_flux_y");
  yarea       = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "yarea");

  size[0] = (x_max+3)-(x_min-2); size[1] = (y_max+3)-(y_min-2);
  work_array1    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array1");
  work_array2    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array2");
  work_array3    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array3");
  work_array4    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array4");
  work_array5    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array5");
  work_array6    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array6");
  work_array7    = ops_decl_dat(clover_grid, 1, size, offset, temp, "double", "work_array7");

  int size2[2] = {(x_max+2)-(x_min-2),1};
  int size3[2] = {1, (y_max+2)-(y_min-2)};
  int size4[2] = {(x_max+3)-(x_min-2),1};
  int size5[2] = {1,(y_max+3)-(y_min-2)};
  int offsetx[2] = {-2,0};
  int offsety[2] = {0,-2};
  cellx    = ops_decl_dat(clover_xedge, 1, size2, offsetx, temp, "double", "cellx");
  celly    = ops_decl_dat(clover_yedge, 1, size3, offsety, temp, "double", "celly");
  vertexx  = ops_decl_dat(clover_xedge, 1, size4, offsetx, temp, "double", "vertexx");
  vertexy  = ops_decl_dat(clover_yedge, 1, size5, offsety, temp, "double", "vertexy");
  celldx   = ops_decl_dat(clover_xedge, 1, size2, offsetx, temp, "double", "celldx");
  celldy   = ops_decl_dat(clover_yedge, 1, size3, offsety, temp, "double", "celldy");
  vertexdx = ops_decl_dat(clover_xedge, 1, size4, offsetx, temp, "double", "vertexdx");
  vertexdy = ops_decl_dat(clover_yedge, 1, size5, offsety, temp, "double", "vertexdy");

  //contains x indicies from 0 to xmax+3 -- needed for initialization
  int* xindex = (int *)xmalloc(sizeof(int)*size4[0]);
  for(int i=x_min-2; i<x_max+3; i++) xindex[i-offsetx[0]] = i - x_min;
  xx  = ops_decl_dat(clover_xedge, 1, size4, offsetx, xindex, "int", "xx");

  //contains y indicies from 0 to ymax+3 -- needed for initialization
  int* yindex = (int *)xmalloc(sizeof(int)*size5[1]);
  for(int i=y_min-2; i<y_max+3; i++) yindex[i-offsety[1]] = i - y_min;
  yy  = ops_decl_dat(clover_yedge, 1, size5, offsety, yindex, "int", "yy");

  //
  //Declare commonly used stencils
  //
  int self2D[] = {0,0};
  int self2D_plus1x[] = {0,0, 1,0};
  int self2D_plus1y[] = {0,0, 0,1};
  int self2D_minus1x[] = {0,0, -1,0};
  int self2D_minus1y[] = {0,0, 0,-1};

  int self2D_plus2x[] = {0,0, 2,0};
  int self2D_plus2y[] = {0,0, 0,2};
  int self2D_minus2x[] = {0,0, -2,0};
  int self2D_minus2y[] = {0,0, 0,-2};

  int self2D_plus3x[] = {0,0, 3,0};
  int self2D_plus3y[] = {0,0, 0,3};
  int self2D_minus3x[] = {0,0, -3,0};
  int self2D_minus3y[] = {0,0, 0,-3};

  int self2D_plus1xy[]  = {0,0, 1,0, 0,1, 1,1};

  int stride2D_x[] = {1,0};
  int stride2D_y[] = {0,1};

  sten_self_2D = ops_decl_stencil( 2, 1, self2D, "self1D");

  sten_self2D_plus1x = ops_decl_stencil( 2, 2, self2D_plus1x, "self2D_plus1x");
  sten_self2D_plus1y = ops_decl_stencil( 2, 2, self2D_plus1y, "self2D_plus1y");
  sten_self2D_minus1x = ops_decl_stencil( 2, 2, self2D_minus1x, "self2D_minus1x");
  sten_self2D_minus1y = ops_decl_stencil( 2, 2, self2D_minus1y, "self2D_minus1y");

  sten_self2D_plus2x = ops_decl_stencil( 2, 2, self2D_plus2x, "self2D_plus2x");
  sten_self2D_plus2y = ops_decl_stencil( 2, 2, self2D_plus2y, "self2D_plus2y");
  sten_self2D_minus2x = ops_decl_stencil( 2, 2, self2D_minus2x, "self2D_minus2x");
  sten_self2D_minus2y = ops_decl_stencil( 2, 2, self2D_minus2y, "self2D_minus2y");

  sten_self2D_plus3x = ops_decl_stencil( 2, 2, self2D_plus3x, "self2D_plus3x");
  sten_self2D_plus3y = ops_decl_stencil( 2, 2, self2D_plus3y, "self2D_plus3y");
  sten_self2D_minus3x = ops_decl_stencil( 2, 2, self2D_minus3x, "self2D_minus3x");
  sten_self2D_minus3y = ops_decl_stencil( 2, 2, self2D_minus3y, "self2D_minus3y");

  sten_self2D_plus1xy = ops_decl_stencil( 2, 4, self2D_plus1xy, "self2D_plus1xy");

  sten_self_stride2D_x = ops_decl_strided_stencil( 2, 1, self2D, stride2D_x, "self_stride2D_x");
  sten_self_stride2D_y = ops_decl_strided_stencil( 2, 1, self2D, stride2D_y, "self_stride2D_y");

  sten_self_plus1_stride2D_x = ops_decl_strided_stencil( 2, 2, self2D_plus1x, stride2D_x, "self_stride2D_x");
  sten_self_plus1_stride2D_y = ops_decl_strided_stencil( 2, 2, self2D_plus1y, stride2D_y, "self_stride2D_y");

  //print ops blocks and dats details
  ops_diagnostic_output();


  /**---------------------initialize and generate chunk----------------------**/

  initialise();
  generate();

  /**------------------------------ideal_gas---------------------------------**/

  ideal_gas(FALSE);

  /**-----------------------------update_halo--------------------------------**/

  //Prime all halo data for the first step
  fields[FIELD_DENSITY0]  = 1;
  fields[FIELD_ENERGY0]   = 1;
  fields[FIELD_PRESSURE]  = 1;
  fields[FIELD_VISCOSITY] = 1;
  fields[FIELD_DENSITY1]  = 1;
  fields[FIELD_ENERGY1]   = 1;
  fields[FIELD_XVEL0]     = 1;
  fields[FIELD_YVEL0]     = 1;
  fields[FIELD_XVEL1]     = 1;
  fields[FIELD_YVEL1]     = 1;

  update_halo(fields, 2);

  /**----------------------------field_summary-------------------------------**/

  field_summary();

  ops_fprintf(g_out," Starting the calculation\n");
  fclose(g_in);


  /***************************************************************************
  **-----------------------------hydro loop---------------------------------**
  /**************************************************************************/

  step = step + 1;

  //CALL timestep() - > calls viscosity kernel and calc_dt kernel








  //ops_print_dat_to_txtfile_core(vertexx, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(vertexdx, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(vertexy, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(vertexdy, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(cellx, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(celldx, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(celly, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(celldy, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(volume, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(xarea, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(yarea, "cloverdats.dat");

  //ops_print_dat_to_txtfile_core(vertexx, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(vertexy, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(density0, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(energy0, "cloverdats.dat");

  //ops_print_dat_to_txtfile_core(xvel0, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(xvel1, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(yvel0, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(yvel1, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(vol_flux_x, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(vol_flux_y, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(mass_flux_x, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(mass_flux_y, "cloverdats.dat");

  ops_print_dat_to_txtfile_core(density0, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(energy0, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(pressure, "cloverdats.dat");
  ops_print_dat_to_txtfile_core(soundspeed, "cloverdats.dat");

  fclose(g_out);


  ops_exit();
}


/*
initialise -
initialise_chunk_kernel.f90 - strightforward
generate_chunk_kernel.f90 - initialization .. complex
ideal_gas_kernel.f90 - somewhat ok
update_halo_kernel.f90 - boundary updating
field_summary_kernel.f90 - complex


hydro -
PdV_kernel.f90
revert_kernel.f90 - strightforward
accelerate_kernel.f90 - strightforward
flux_calc_kernel.f90 - strightforward
advec_cell_kernel.f90
advec_mom_kernel.f90 - complex
reset_field_kernel.f90 - strightforward


timestep -
calc_dt_kernel.f90 - complex
viscosity_kernel.f90


pack_kernel.f90 - mpi buffer packing
*/
