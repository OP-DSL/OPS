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

/** @brief momentum advection
 *  @author Wayne Gaudin, converted to OPS by Gihan Mudalige
 *  @details Performs a second order advective remap on the vertex momentum
 *  using van-Leer limiting and directional splitting.
 *  Note that although pre_vol is only set and not used in the update, please
 *  leave it in the method.
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
#include "advec_mom_kernel.h"


void advec_mom_kernel_x1_macro( double *pre_vol, double *post_vol,
                          double *volume,
                          double *vol_flux_x, double *vol_flux_y) {

  post_vol[OPS_ACC1(0,0)] = volume[OPS_ACC2(0,0)] + vol_flux_y[OPS_ACC4(0,1)] -  vol_flux_y[OPS_ACC3(0,0)];
  pre_vol[OPS_ACC0(0,0)] = post_vol[OPS_ACC1(0,0)] + vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)];

}

void advec_mom_kernel_y1_macro( double *pre_vol, double *post_vol,
                          double *volume,
                          double *vol_flux_x, double *vol_flux_y) {

  post_vol[OPS_ACC1(0,0)] = volume[OPS_ACC2(0,0)] + vol_flux_x[OPS_ACC3(1,0)] -  vol_flux_x[OPS_ACC3(0,0)];
  pre_vol[OPS_ACC0(0,0)] = post_vol[OPS_ACC1(0,0)] + vol_flux_y[OPS_ACC4(0,1)] - vol_flux_y[OPS_ACC4(0,0)];

}

void advec_mom_kernel_x2_macro( double *pre_vol, double *post_vol,
                          double *volume,
                          double *vol_flux_y) {

  post_vol[OPS_ACC1(0,0)]  = volume[OPS_ACC2(0,0)] ;
  pre_vol[OPS_ACC0(0,0)]   = post_vol[OPS_ACC1(0,0)]  + vol_flux_y[OPS_ACC3(0,1)] - vol_flux_y[OPS_ACC3(0,0)];

}

void advec_mom_kernel_y2_macro( double *pre_vol, double *post_vol,
                          double *volume,
                          double *vol_flux_x) {

  post_vol[OPS_ACC1(0,0)]  = volume[OPS_ACC2(0,0)] ;
  pre_vol[OPS_ACC0(0,0)]   = post_vol[OPS_ACC1(0,0)]  + vol_flux_x[OPS_ACC3(1,0)] - vol_flux_x[OPS_ACC3(0,0)];

}


void advec_mom_kernel_mass_flux_x_macro( double *node_flux, double *mass_flux_x) {

  //mass_flux_x accessed with: {0,0, 1,0, 0,-1, 1,-1}

  node_flux[OPS_ACC0(0,0)] = 0.25 * ( mass_flux_x[OPS_ACC1(0,-1)] + mass_flux_x[OPS_ACC1(0,0)] +
    mass_flux_x[OPS_ACC1(1,-1)] + mass_flux_x[OPS_ACC1(1,0)] ); // Mass Flux in x
}


void advec_mom_kernel_mass_flux_y_macro( double *node_flux, double *mass_flux_y) {

  //mass_flux_y accessed with: {0,0, 0,1, -1,0, -1,1}

  node_flux[OPS_ACC0(0,0)] = 0.25 * ( mass_flux_y[OPS_ACC1(-1,0)] + mass_flux_y[OPS_ACC1(0,0)] +
      mass_flux_y[OPS_ACC1(-1,1)] + mass_flux_y[OPS_ACC1(0,1)] ); // Mass Flux in y
}


void advec_mom_kernel_post_advec_x_macro( double *node_mass_post, double *post_vol,
                                  double *density1) {

  //post_vol accessed with: {0,0, -1,0, 0,-1, -1,-1}
  //density1 accessed with: {0,0, -1,0, 0,-1, -1,-1}

  node_mass_post[OPS_ACC0(0,0)] = 0.25 * ( density1[OPS_ACC2(0,-1)] * post_vol[OPS_ACC0(0,-1)] +
                              density1[OPS_ACC2(0,0)]   * post_vol[OPS_ACC1(0,0)]   +
                              density1[OPS_ACC2(-1,-1)] * post_vol[OPS_ACC1(-1,-1)] +
                              density1[OPS_ACC2(-1,0)]  * post_vol[OPS_ACC1(-1,0)]  );

}

//this is the same as advec_mom_kernel_post_advec_x ... just repeated here for debugging
void advec_mom_kernel_post_advec_y_macro( double *node_mass_post, double *post_vol,
                                  double *density1) {

  //post_vol accessed with: {0,0, -1,0, 0,-1, -1,-1}
  //density1 accessed with: {0,0, -1,0, 0,-1, -1,-1}

  node_mass_post[OPS_ACC0(0,0)] = 0.25 * ( density1[OPS_ACC2(0,-1)] * post_vol[OPS_ACC0(0,-1)] +
                              density1[OPS_ACC2(0,0)]   * post_vol[OPS_ACC1(0,0)]   +
                              density1[OPS_ACC2(-1,-1)] * post_vol[OPS_ACC1(-1,-1)] +
                              density1[OPS_ACC2(-1,0)]  * post_vol[OPS_ACC1(-1,0)]  );

}


void advec_mom_kernel1_x_macro( double *node_flux, double *node_mass_pre,
                        double *advec_vel, double *mom_flux,
                        double *celldx, double *vel1) {

  //node_flux accessed with: {0,0}
  //node_mass_pre accessed with: {0,0, 1,0}
  //celldx is accessed with {0,0, 1,0, -1,0, -2,0} striding in x
  //vel1 is accessed with {0,0, 1,0, 2,0, -1,0}

  double sigma, wind, width;
  double sigma2, wind2;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  double vdiffuw2, vdiffdw2, auw2, limiter2;

  sigma = fabs(node_flux[OPS_ACC0(0,0)])/node_mass_pre[OPS_ACC1(1,0)];
  sigma2 = fabs(node_flux[OPS_ACC0(0,0)])/node_mass_pre[OPS_ACC1(0,0)];

  width = celldx[OPS_ACC4(0,0)];
  vdiffuw = vel1[OPS_ACC5(1,0)] - vel1[OPS_ACC5(2,0)];
  vdiffdw = vel1[OPS_ACC5(0,0)] - vel1[OPS_ACC5(1,0)];
  vdiffuw2 = vel1[OPS_ACC5(0,0)] - vel1[OPS_ACC5(-1,0)];
  vdiffdw2 = -vdiffdw;

  auw = fabs(vdiffuw);
  adw = fabs(vdiffdw);
  auw2 = fabs(vdiffuw2);
  wind = 1.0;
  wind2 = 1.0;

  if(vdiffdw <= 0.0) wind = -1.0;
  if(vdiffdw2 <= 0.0) wind2 = -1.0;

  limiter = wind * MIN( width * ( (2.0 - sigma) * adw/width + (1.0 + sigma) *
                        auw/celldx[OPS_ACC4(1,0)] )/6.0 , MIN(auw, adw) );
  limiter2= wind2* MIN( width * ( (2.0 - sigma2) * adw/width + (1.0 + sigma2) *
                        auw2/celldx[OPS_ACC4(-1,0)] )/6.0, MIN(auw2,adw) );

  if((vdiffuw * vdiffdw) <= 0.0) limiter = 0.0;
  if((vdiffuw2 * vdiffdw2) <= 0.0) limiter2 = 0.0;

  if( (node_flux[OPS_ACC0(0,0)]) < 0.0) {
    advec_vel[OPS_ACC2(0,0)] = vel1[OPS_ACC5(1,0)] + (1.0 - sigma) * limiter;
  }
  else {
    advec_vel[OPS_ACC2(0,0)] = vel1[OPS_ACC5(0,0)] + (1.0 - sigma2) * limiter2;
  }

  mom_flux[OPS_ACC3(0,0)] = advec_vel[OPS_ACC2(0,0)] * node_flux[OPS_ACC0(0,0)];

}


void advec_mom_kernel1_y_macro( double *node_flux, double *node_mass_pre,
                        double *advec_vel, double *mom_flux,
                        double *celldy, double *vel1) {

  //node_flux accessed with: {0,0}
  //node_mass_pre accessed with: {0,0, 0,1}
  //celldy is accessed with {0,0, 0,1, 0,-1, 0,-2} striding in y
  //vel1 is accessed with {0,0, 0,1, 0,2, 0,-1}

  double sigma, wind, width;
  double sigma2, wind2;
  double vdiffuw, vdiffdw, auw, adw, limiter;
  double vdiffuw2, vdiffdw2, auw2, limiter2;

  sigma = fabs(node_flux[OPS_ACC0(0,0)])/node_mass_pre[OPS_ACC1(0,1)];
  sigma2 = fabs(node_flux[OPS_ACC0(0,0)])/node_mass_pre[OPS_ACC1(0,0)];

  width = celldy[OPS_ACC4(0,0)];
  vdiffuw = vel1[OPS_ACC5(0,1)] - vel1[OPS_ACC5(0,2)];
  vdiffdw = vel1[OPS_ACC5(0,0)] - vel1[OPS_ACC5(0,1)];
  vdiffuw2 = vel1[OPS_ACC5(0,0)] - vel1[OPS_ACC5(0,-1)];
  vdiffdw2 = -vdiffdw;

  auw = fabs(vdiffuw);
  adw = fabs(vdiffdw);
  auw2 = fabs(vdiffuw2);
  wind = 1.0;
  wind2 = 1.0;

  if(vdiffdw <= 0.0) wind = -1.0;
  if(vdiffdw2 <= 0.0) wind2 = -1.0;

  limiter = wind * MIN( width * ( (2.0 - sigma) * adw/width + (1.0 + sigma) *
                        auw/celldy[OPS_ACC4(0,1)] )/6.0 , MIN(auw, adw) );
  limiter2= wind2* MIN( width * ( (2.0 - sigma2) * adw/width + (1.0 + sigma2) *
                        auw2/celldy[OPS_ACC4(0,-1)] )/6.0, MIN(auw2,adw) );

  if((vdiffuw * vdiffdw) <= 0.0) limiter = 0.0;
  if((vdiffuw2 * vdiffdw2) <= 0.0) limiter2 = 0.0;

  if( (node_flux[OPS_ACC0(0,0)]) < 0.0) {
    advec_vel[OPS_ACC2(0,0)] = vel1[OPS_ACC5(0,1)] + (1.0 - sigma) * limiter;
  }
  else {
    advec_vel[OPS_ACC2(0,0)] = vel1[OPS_ACC5(0,0)] + (1.0 - sigma2) * limiter2;
  }

  mom_flux[OPS_ACC3(0,0)] = advec_vel[OPS_ACC2(0,0)] * node_flux[OPS_ACC0(0,0)];
}


void advec_mom_kernel_pre_advec_x_macro( double *node_mass_pre, double *node_mass_post,
                                  double *node_flux) {

  //node_flux accessed with: {0,0, -1,0}
  node_mass_pre[OPS_ACC0(0,0)] = node_mass_post[OPS_ACC1(0,0)] - node_flux[OPS_ACC2(-1,0)] + node_flux[OPS_ACC2(0,0)];

}
void advec_mom_kernel_pre_advec_y_macro( double *node_mass_pre, double *node_mass_post,
                                  double *node_flux) {

  //node_flux accessed with: {0,0, 0,-1}
  node_mass_pre[OPS_ACC0(0,0)] = node_mass_post[OPS_ACC1(0,0)] - node_flux[OPS_ACC2(0,-1)] + node_flux[OPS_ACC2(0,0)];

}


void advec_mom_kernel2_x_macro( double *vel1, double *node_mass_post,
                        double *node_mass_pre, double *mom_flux) {

  //mom_flux accessed with: {0,0, -1,0}
  vel1[OPS_ACC0(0,0)] = ( vel1[OPS_ACC0(0,0)] * node_mass_pre[OPS_ACC2(0,0)]  +
    mom_flux[OPS_ACC3(-1,0)] - mom_flux[OPS_ACC3(0,0)] ) / node_mass_post[OPS_ACC1(0,0)];

}


void advec_mom_kernel2_y_macro( double *vel1, double *node_mass_post,
                        double *node_mass_pre, double *mom_flux) {

  //mom_flux accessed with: {0,0, 0,-1}
  vel1[OPS_ACC0(0,0)] = ( vel1[OPS_ACC0(0,0)] * node_mass_pre[OPS_ACC2(0,0)]  +
    mom_flux[OPS_ACC3(0,-1)] - mom_flux[OPS_ACC3(0,0)] ) / node_mass_post[OPS_ACC1(0,0)];
}



void advec_mom(int which_vel, int sweep_number, int dir)
{
  //initialize sizes using global values
  int x_cells = grid->x_cells;
  int y_cells = grid->y_cells;
  int x_min = field->x_min;
  int x_max = field->x_max;
  int y_min = field->y_min;
  int y_max = field->y_max;

  int rangexy[] = {x_min-2,x_max+2,y_min-2,y_max+2}; // full range over grid

  int mom_sweep;
  ops_dat vel1;

  int vector = TRUE; //currently always use vector loops .. need to set this in input

  if( which_vel == 1) {
    vel1 = xvel1;
  }
  else {
    vel1 = yvel1;
  }

  mom_sweep = dir + 2*(sweep_number-1);
  //printf("vector %d, direction: %d sweep_number: %d mom_sweep %d \n",vector, dir, sweep_number, mom_sweep);

  if(mom_sweep == 1) { // x 1
    ops_par_loop_macro(advec_mom_kernel_x1_macro, "advec_mom_kernel_x1_macro", 2, rangexy,
        ops_arg_dat(work_array6, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00, "double", OPS_RW), //this may not be OPS_RW ... see kernel
        ops_arg_dat(volume, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, "double", OPS_READ));
  }
  else if(mom_sweep == 2) { // y 1
    ops_par_loop_macro(advec_mom_kernel_y1_macro, "advec_mom_kernel_y1_macro", 2, rangexy,
        ops_arg_dat(work_array6, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00, "double", OPS_RW), //this may not be OPS_RW ... see kernel
        ops_arg_dat(volume, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, "double", OPS_READ));
  }
  else if (mom_sweep == 3) { // x 2
    ops_par_loop_macro(advec_mom_kernel_x2_macro, "advec_mom_kernel_x2_macro", 2, rangexy,
        ops_arg_dat(work_array6, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00, "double", OPS_RW), //this may not be OPS_RW ... see kernel
        ops_arg_dat(volume, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_y, S2D_00_0P1, "double", OPS_READ));
  }
  else if (mom_sweep == 4) { // y 2
    ops_par_loop_macro(advec_mom_kernel_y2_macro, "advec_mom_kernel_y2_macro", 2, rangexy,
        ops_arg_dat(work_array6, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00, "double", OPS_RW), //this may not be OPS_RW ... see kernel
        ops_arg_dat(volume, S2D_00, "double", OPS_READ),
        ops_arg_dat(vol_flux_x, S2D_00_P10, "double", OPS_READ));
  }

  int range_fullx_party_1[] = {x_min-2,x_max+2,y_min,y_max+1}; // full x range partial y range
  int range_partx_party_1[] = {x_min-1,x_max+2,y_min,y_max+1}; // partial x range partial y range

  int range_fully_party_1[] = {x_min,x_max+1,y_min-2,y_max+2}; // full y range partial x range
  int range_partx_party_2[] = {x_min,x_max+1,y_min-1,y_max+2}; // partial x range partial y range

  if (dir == 1) {

    //Find staggered mesh mass fluxes, nodal masses and volumes.
    ops_par_loop_macro(advec_mom_kernel_mass_flux_x_macro, "advec_mom_kernel_mass_flux_x_macro", 2, range_fullx_party_1,
        ops_arg_dat(work_array1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(mass_flux_x, S2D_00_P10_0M1_P1M1, "double", OPS_READ));

    //Staggered cell mass post advection
    ops_par_loop_macro(advec_mom_kernel_post_advec_x_macro, "advec_mom_kernel_post_advec_x_macro", 2, range_partx_party_1,
        ops_arg_dat(work_array2, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
        ops_arg_dat(density1, S2D_00_M10_0M1_M1M1, "double", OPS_READ));


    //Stagered cell mass pre advection
    ops_par_loop_macro(advec_mom_kernel_pre_advec_x_macro, "advec_mom_kernel_pre_advec_x_macro", 2, range_partx_party_1,
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array2/*node_mass_post*/, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array1/*node_flux*/, S2D_00_M10, "double", OPS_READ));


    int range_plus1xy_minus1x[] = {x_min-1,x_max+1,y_min,y_max+1}; // partial x range partial y range
    if(vector == 1) {

      ops_par_loop_macro(advec_mom_kernel1_x_macro, "advec_mom_kernel1_x_macro", 2, range_plus1xy_minus1x,
        ops_arg_dat(work_array1/*node_flux*/, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00_P10, "double", OPS_READ),
        ops_arg_dat(work_array4/*advec_vel*/, S2D_00, "double", OPS_RW),
        ops_arg_dat(work_array5/*mom_flux*/, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(celldx, S2D_00_P10_M10_M20_STRID2D_X, "double", OPS_READ),
        ops_arg_dat(vel1, S2D_00_P10_P20_M10, "double", OPS_READ));


    }
    else {
      //currently ignor this section
    }

    int range_partx_party_2[] = {x_min,x_max+1,y_min,y_max+1}; // full x range partial y range

    ops_par_loop_macro(advec_mom_kernel2_x_macro, "advec_mom_kernel2_x_macro", 2, range_partx_party_2,
        ops_arg_dat(vel1, S2D_00, "double", OPS_INC),
        ops_arg_dat(work_array2/*node_mass_post*/, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, S2D_00_M10, "double", OPS_READ)
        );
  }
  else if (dir == 2) {

    //Find staggered mesh mass fluxes, nodal masses and volumes.

    ops_par_loop_macro(advec_mom_kernel_mass_flux_y_macro, "advec_mom_kernel_mass_flux_y_macro", 2, range_fully_party_1,
        ops_arg_dat(work_array1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(mass_flux_y, S2D_00_0P1_M10_M1P1, "double", OPS_READ));


    //Staggered cell mass post advection
    ops_par_loop_macro(advec_mom_kernel_post_advec_y_macro, "advec_mom_kernel_post_advec_y_macro", 2, range_partx_party_2,
        ops_arg_dat(work_array2, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array7, S2D_00_M10_0M1_M1M1, "double", OPS_READ),
        ops_arg_dat(density1, S2D_00_M10_0M1_M1M1, "double", OPS_READ));

    //Stagered cell mass pre advection
    ops_par_loop_macro(advec_mom_kernel_pre_advec_y_macro, "advec_mom_kernel_pre_advec_y_macro", 2, range_partx_party_2,
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(work_array2/*node_mass_post*/, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array1/*node_flux*/, S2D_00_0M1, "double", OPS_READ));

    int range_plus1xy_minus1y[] = {x_min,x_max+1,y_min-1,y_max+1}; // partial x range partial y range
    if(vector == 1) {

        ops_par_loop_macro(advec_mom_kernel1_y_macro, "advec_mom_kernel1_y_macro", 2, range_plus1xy_minus1y,
        ops_arg_dat(work_array1/*node_flux*/, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00_0P1, "double", OPS_READ),
        ops_arg_dat(work_array4/*advec_vel*/, S2D_00, "double", OPS_RW),
        ops_arg_dat(work_array5/*mom_flux*/, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(celldy, S2D_00_0P1_0M1_0M2_STRID2D_Y, "double", OPS_READ),
        ops_arg_dat(vel1, S2D_00_0P1_0P2_0M1, "double", OPS_READ));

    }
    else {
      //currently ignor this section
    }

    int range_partx_party_2[] = {x_min,x_max+1,y_min,y_max+1}; // full x range partial y range

    ops_par_loop_macro(advec_mom_kernel2_y_macro, "advec_mom_kernel2_y_macro", 2, range_partx_party_2,
        ops_arg_dat(vel1, S2D_00, "double", OPS_INC),
        ops_arg_dat(work_array2/*node_mass_post*/, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array3/*node_mass_pre*/, S2D_00, "double", OPS_READ),
        ops_arg_dat(work_array5/*mom_flux*/, S2D_00_0M1, "double", OPS_READ)
        );
  }

}
