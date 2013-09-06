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

/** @brief Top level initialisation routine
 *  @author Wayne Gaudin
 *  @details Checks for the user input and either invokes the input reader or
 *  switches to the internal test problem. It processes the input and strips
 *  comments before writing a final input file.
 *  It then calls the start routine.
**/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#include "ops_seq.h"

#include "data.h"
#include "definitions.h"

//Cloverleaf kernels
#include  "update_halo_kernel.h"

void update_halo(int* fields, int depth)
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy_bottom1[] = {x_min-2,x_max+2,y_min-2,y_min-1};
  int self_bottom1[] = {0,0, 0,2};
  ops_stencil sten2D_bottom1 = ops_decl_stencil( 2, 2, self_bottom1, "sten2D_bottom1");

  int rangexy_bottom2[] = {x_min-2,x_max+2,y_min-1,y_min};
  int self_bottom2[] = {0,0, 0,1};
  ops_stencil sten2D_bottom2 = ops_decl_stencil( 2, 2, self_bottom2, "sten2D_bottom2");

  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_bottom2,
      ops_arg_dat(density0, sten2D_bottom2, OPS_RW),
      ops_arg_dat(density1, sten2D_bottom2, OPS_RW),
      ops_arg_dat(energy0, sten2D_bottom2, OPS_RW),
      ops_arg_dat(energy1, sten2D_bottom2, OPS_RW),
      ops_arg_dat(pressure, sten2D_bottom2, OPS_RW),
      ops_arg_dat(viscosity, sten2D_bottom2, OPS_RW),
      ops_arg_dat(soundspeed, sten2D_bottom2, OPS_RW));

  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_bottom1,
      ops_arg_dat(density0, sten2D_bottom1, OPS_RW),
      ops_arg_dat(density1, sten2D_bottom1, OPS_RW),
      ops_arg_dat(energy0, sten2D_bottom1, OPS_RW),
      ops_arg_dat(energy1, sten2D_bottom1, OPS_RW),
      ops_arg_dat(pressure, sten2D_bottom1, OPS_RW),
      ops_arg_dat(viscosity, sten2D_bottom1, OPS_RW),
      ops_arg_dat(soundspeed, sten2D_bottom1, OPS_RW));

  int rangexy_top1[] = {x_min-2,x_max+2,y_max+1,y_max+2};
  int self_top1[] = {0,0, 0,-2};
  ops_stencil sten2D_top1 = ops_decl_stencil( 2, 2, self_top1, "sten2D_top1");

  int rangexy_top2[] = {x_min-2,x_max+2,y_max,y_max+1};
  int self_top2[] = {0,0, 0,-1};
  ops_stencil sten2D_top2 = ops_decl_stencil( 2, 2, self_top2, "sten2D_top2");

  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_top2,
      ops_arg_dat(density0, sten2D_top2, OPS_RW),
      ops_arg_dat(density1, sten2D_top2, OPS_RW),
      ops_arg_dat(energy0, sten2D_top2, OPS_RW),
      ops_arg_dat(energy1, sten2D_top2, OPS_RW),
      ops_arg_dat(pressure, sten2D_top2, OPS_RW),
      ops_arg_dat(viscosity, sten2D_top2, OPS_RW),
      ops_arg_dat(soundspeed, sten2D_top2, OPS_RW));

  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_top1,
      ops_arg_dat(density0, sten2D_top1, OPS_RW),
      ops_arg_dat(density1, sten2D_top1, OPS_RW),
      ops_arg_dat(energy0, sten2D_top1, OPS_RW),
      ops_arg_dat(energy1, sten2D_top1, OPS_RW),
      ops_arg_dat(pressure, sten2D_top1, OPS_RW),
      ops_arg_dat(viscosity, sten2D_top1, OPS_RW),
      ops_arg_dat(soundspeed, sten2D_top1, OPS_RW));

  int rangexy_left1[] = {x_min-2,x_min-1,y_min-2,y_max+2};
  int self_left1[] = {0,0, 2,0};
  ops_stencil sten2D_left1 = ops_decl_stencil( 2, 2, self_left1, "sten2D_left1");

  int rangexy_left2[] = {x_min-1,x_min,y_min-2,y_max+2};
  int self_left2[] = {0,0, 1,0};
  ops_stencil sten2D_left2 = ops_decl_stencil( 2, 2, self_left2, "sten2D_left2");

  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_left2,
      ops_arg_dat(density0, sten2D_left2, OPS_RW),
      ops_arg_dat(density1, sten2D_left2, OPS_RW),
      ops_arg_dat(energy0, sten2D_left2, OPS_RW),
      ops_arg_dat(energy1, sten2D_left2, OPS_RW),
      ops_arg_dat(pressure, sten2D_left2, OPS_RW),
      ops_arg_dat(viscosity, sten2D_left2, OPS_RW),
      ops_arg_dat(soundspeed, sten2D_left2, OPS_RW));

  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_left1,
      ops_arg_dat(density0, sten2D_left1, OPS_RW),
      ops_arg_dat(density1, sten2D_left1, OPS_RW),
      ops_arg_dat(energy0, sten2D_left1, OPS_RW),
      ops_arg_dat(energy1, sten2D_left1, OPS_RW),
      ops_arg_dat(pressure, sten2D_left1, OPS_RW),
      ops_arg_dat(viscosity, sten2D_left1, OPS_RW),
      ops_arg_dat(soundspeed, sten2D_left1, OPS_RW));

  int rangexy_right1[] = {x_max+1,x_max+2,y_min-2,y_max+2};
  int self_right1[] = {0,0, -2,0};
  ops_stencil sten2D_right1 = ops_decl_stencil( 2, 2, self_right1, "sten2D_right1");

  int rangexy_right2[] = {x_max,x_max+1,y_min-2,y_max+2};
  int self_right2[] = {0,0, -1,0};
  ops_stencil sten2D_right2 = ops_decl_stencil( 2, 2, self_right2, "sten2D_right2");

  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_right2,
      ops_arg_dat(density0, sten2D_right2, OPS_RW),
      ops_arg_dat(density1, sten2D_right2, OPS_RW),
      ops_arg_dat(energy0, sten2D_right2, OPS_RW),
      ops_arg_dat(energy1, sten2D_right2, OPS_RW),
      ops_arg_dat(pressure, sten2D_right2, OPS_RW),
      ops_arg_dat(viscosity, sten2D_right2, OPS_RW),
      ops_arg_dat(soundspeed, sten2D_right2, OPS_RW));

  ops_par_loop(update_halo_kernel, "update_halo_kernel", 2, rangexy_right1,
      ops_arg_dat(density0, sten2D_right1, OPS_RW),
      ops_arg_dat(density1, sten2D_right1, OPS_RW),
      ops_arg_dat(energy0, sten2D_right1, OPS_RW),
      ops_arg_dat(energy1, sten2D_right1, OPS_RW),
      ops_arg_dat(pressure, sten2D_right1, OPS_RW),
      ops_arg_dat(viscosity, sten2D_right1, OPS_RW),
      ops_arg_dat(soundspeed, sten2D_right1, OPS_RW));

  //ops_print_dat_to_txtfile_core(density0, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(density1, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(energy0, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(energy1, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(pressure, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(viscosity, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(soundspeed, "cloverdats.dat");

  //int rangexy_bottom1[] = {x_min-2,x_max+2,y_min-2,y_min-1};
  int bottom1[] = {0,0, 0,3};
  ops_stencil s2D_bottom1 = ops_decl_stencil( 2, 2, bottom1, "s2D_bottom1");
  //int rangexy_bottom2[] = {x_min-2,x_max+2,y_min-1,y_min};
  int bottom2[] = {0,0, 0,2};
  ops_stencil s2D_bottom2 = ops_decl_stencil( 2, 2, bottom2, "s2D_bottom2");

  ops_par_loop(update_halo_kernel2, "update_halo_kernel2", 2, rangexy_bottom2,
      ops_arg_dat(xvel0, s2D_bottom2, OPS_RW),
      ops_arg_dat(xvel1, s2D_bottom2, OPS_RW),
      ops_arg_dat(yvel0, s2D_bottom2, OPS_RW),
      ops_arg_dat(yvel1, s2D_bottom2, OPS_RW),
      ops_arg_dat(vol_flux_x,  s2D_bottom2, OPS_RW),
      ops_arg_dat(mass_flux_x, s2D_bottom2, OPS_RW),
      ops_arg_dat(vol_flux_y,  s2D_bottom2, OPS_RW),
      ops_arg_dat(mass_flux_y, s2D_bottom2, OPS_RW));

  ops_par_loop(update_halo_kernel2, "update_halo_kernel2", 2, rangexy_bottom1,
      ops_arg_dat(xvel0, s2D_bottom1, OPS_RW),
      ops_arg_dat(xvel1, s2D_bottom1, OPS_RW),
      ops_arg_dat(yvel0, s2D_bottom1, OPS_RW),
      ops_arg_dat(yvel1, s2D_bottom1, OPS_RW),
      ops_arg_dat(vol_flux_x, s2D_bottom1, OPS_RW),
      ops_arg_dat(mass_flux_x, s2D_bottom1, OPS_RW),
      ops_arg_dat(vol_flux_y, s2D_bottom1, OPS_RW),
      ops_arg_dat(mass_flux_y, s2D_bottom1, OPS_RW));


  //int rangexy_top1[] = {x_min-2,x_max+2,y_max+1,y_max+2};
  int top1[] = {0,0, 0,-3};
  ops_stencil s2D_top1 = ops_decl_stencil( 2, 2, top1, "s2D_top1");
  //int rangexy_top2[] = {x_min-2,x_max+2,y_max,y_max+1};
  int top2[] = {0,0, 0,-2};
  ops_stencil s2D_top2 = ops_decl_stencil( 2, 2, top2, "s2D_top2");

  ops_par_loop(update_halo_kernel2, "update_halo_kernel2", 2, rangexy_top1,
      ops_arg_dat(xvel0, s2D_top1, OPS_RW),
      ops_arg_dat(xvel1, s2D_top1, OPS_RW),
      ops_arg_dat(yvel0, s2D_top1, OPS_RW),
      ops_arg_dat(yvel1, s2D_top1, OPS_RW),
      ops_arg_dat(vol_flux_x,  s2D_top1, OPS_RW),
      ops_arg_dat(mass_flux_x, s2D_top1, OPS_RW),
      ops_arg_dat(vol_flux_y,  s2D_top1, OPS_RW),
      ops_arg_dat(mass_flux_y, s2D_top1, OPS_RW));

  ops_par_loop(update_halo_kernel2, "update_halo_kernel2", 2, rangexy_top2,
      ops_arg_dat(xvel0, s2D_top2, OPS_RW),
      ops_arg_dat(xvel1, s2D_top2, OPS_RW),
      ops_arg_dat(yvel0, s2D_top2, OPS_RW),
      ops_arg_dat(yvel1, s2D_top2, OPS_RW),
      ops_arg_dat(vol_flux_x, s2D_top2, OPS_RW),
      ops_arg_dat(mass_flux_x, s2D_top2, OPS_RW),
      ops_arg_dat(vol_flux_y, s2D_top2, OPS_RW),
      ops_arg_dat(mass_flux_y, s2D_top2, OPS_RW));


  //int rangexy_left1[] = {x_min-2,x_min-1,y_min-2,y_max+2};
  int left1[] = {0,0, 3,0};
  ops_stencil s2D_left1 = ops_decl_stencil( 2, 2, left1, "s2D_left1");

  //int rangexy_left2[] = {x_min-1,x_min,y_min-2,y_max+2};
  int left2[] = {0,0, 2,0};
  ops_stencil s2D_left2 = ops_decl_stencil( 2, 2, left2, "s2D_left2");

  ops_par_loop(update_halo_kernel2, "update_halo_kernel2", 2, rangexy_left2,
      ops_arg_dat(xvel0, s2D_left2, OPS_RW),
      ops_arg_dat(xvel1, s2D_left2, OPS_RW),
      ops_arg_dat(yvel0, s2D_left2, OPS_RW),
      ops_arg_dat(yvel1, s2D_left2, OPS_RW),
      ops_arg_dat(vol_flux_x, s2D_left2, OPS_RW),
      ops_arg_dat(mass_flux_x, s2D_left2, OPS_RW),
      ops_arg_dat(vol_flux_y, s2D_left2, OPS_RW),
      ops_arg_dat(mass_flux_y, s2D_left2, OPS_RW));

  ops_par_loop(update_halo_kernel2, "update_halo_kernel2", 2, rangexy_left1,
      ops_arg_dat(xvel0, s2D_left1, OPS_RW),
      ops_arg_dat(xvel1, s2D_left1, OPS_RW),
      ops_arg_dat(yvel0, s2D_left1, OPS_RW),
      ops_arg_dat(yvel1, s2D_left1, OPS_RW),
      ops_arg_dat(vol_flux_x, s2D_left1, OPS_RW),
      ops_arg_dat(mass_flux_x, s2D_left1, OPS_RW),
      ops_arg_dat(vol_flux_y, s2D_left1, OPS_RW),
      ops_arg_dat(mass_flux_y, s2D_left1, OPS_RW));

  //int rangexy_right1[] = {x_max+1,x_max+2,y_min-2,y_max+2};
  int right1[] = {0,0, -3,0};
  ops_stencil s2D_right1 = ops_decl_stencil( 2, 2, right1, "s2D_right1");

  //int rangexy_right2[] = {x_max,x_max+1,y_min-2,y_max+2};
  int right2[] = {0,0, -2,0};
  ops_stencil s2D_right2 = ops_decl_stencil( 2, 2, right2, "s2D_right2");

  ops_par_loop(update_halo_kernel2, "update_halo_kernel2", 2, rangexy_right2,
      ops_arg_dat(xvel0, s2D_right2, OPS_RW),
      ops_arg_dat(xvel1, s2D_right2, OPS_RW),
      ops_arg_dat(yvel0, s2D_right2, OPS_RW),
      ops_arg_dat(yvel1, s2D_right2, OPS_RW),
      ops_arg_dat(vol_flux_x, s2D_right2, OPS_RW),
      ops_arg_dat(mass_flux_x, s2D_right2, OPS_RW),
      ops_arg_dat(vol_flux_y, s2D_right2, OPS_RW),
      ops_arg_dat(mass_flux_y, s2D_right2, OPS_RW));

  ops_par_loop(update_halo_kernel2, "update_halo_kernel2", 2, rangexy_right1,
      ops_arg_dat(xvel0, s2D_right1, OPS_RW),
      ops_arg_dat(xvel1, s2D_right1, OPS_RW),
      ops_arg_dat(yvel0, s2D_right1, OPS_RW),
      ops_arg_dat(yvel1, s2D_right1, OPS_RW),
      ops_arg_dat(vol_flux_x, s2D_right1, OPS_RW),
      ops_arg_dat(mass_flux_x, s2D_right1, OPS_RW),
      ops_arg_dat(vol_flux_y, s2D_right1, OPS_RW),
      ops_arg_dat(mass_flux_y, s2D_right1, OPS_RW));


  //ops_print_dat_to_txtfile_core(xvel0, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(xvel1, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(yvel0, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(yvel1, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(vol_flux_x, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(vol_flux_y, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(mass_flux_x, "cloverdats.dat");
  //ops_print_dat_to_txtfile_core(mass_flux_y, "cloverdats.dat");
}
