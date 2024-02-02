/*Crown Copyright 2014 AWE.

 This file is part of TeaLeaf.

 TeaLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 TeaLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 TeaLeaf. If not, see http://www.gnu.org/licenses/. */

// @brief Controls the main diffusion cycle.
// @author Istvan Reguly, David Beckingsale, Wayne Gaudin
// @details Implicitly calculates the change in temperature using a Jacobi iteration


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OPS header file
#define OPS_2D
#include <ops_seq.h>

#include "tea_leaf.h"
 
#include "data.h"
#include "definitions.h"

#include "tea_leaf_ppcg_kernels.h"

void tea_leaf_init_zero2_kernel (double * p, double * z);
void tea_leaf_init_zero_kernel (double * p);
void tea_leaf_yeqx_kernel (double * p, const double * x);
void tea_leaf_yeqax_kernel (double * p, const double * x, const double * a);
void tea_leaf_dot_kernel (const double * r, const double * p, double *rro);
void tea_leaf_axpy_kernel(double * u, const double * p, const double * alpha);
void tea_leaf_axpby_kernel(double * u, const double * p, const double * alpha, const double * beta);
void tea_leaf_zeqxty_kernel(double * z, const double * x, const double * y);
void tea_leaf_recip_kernel(double * u, const double * p);
void tea_leaf_recip2_kernel(double *z, const double *x, const double *y);
void tea_leaf_norm2_kernel(const double *x, double * norm);


void tea_leaf_ppcg_init_sd(
  ops_dat r,
  ops_dat rtemp,
	ops_dat kx,
	ops_dat ky,
  ops_dat sd,
  ops_dat z,
  ops_dat utemp,
	ops_dat cp,
	ops_dat bfp,
	ops_dat Mi,
  double rx, double ry,
  double theta, int preconditioner_type)
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};

  double theta_r = 1.0/theta;
  if (preconditioner_type != TL_PREC_NONE) {
    ops_par_loop(tea_leaf_ppcg_init1_kernel, "tea_leaf_ppcg_init1_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(sd, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(rtemp, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(utemp, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(z, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&theta_r, 1, "double", OPS_READ));
  } else {
    ops_par_loop(tea_leaf_ppcg_init2_kernel, "tea_leaf_ppcg_init2_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(sd, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(rtemp, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(utemp, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&theta_r, 1, "double", OPS_READ));
  }
}

void tea_leaf_ppcg_init(
  ops_dat p,
  ops_dat r,
  ops_dat Mi,
  ops_dat z,
  ops_dat Kx,
  ops_dat Ky,
  ops_dat cp,
  ops_dat bfp,
  double rx, double ry,
  double *rro, int preconditioner_type,
  int ppcg_inner_iters, 
  double *ch_alphas, double *ch_betas, 
  double theta, double solve_time, int step)
{
    //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};

  // We divide the algorithm up into steps
  // step = 1, is a CG step
  // step = 2 or 3, is a partial step of the PPCG algorithm
  
 
  *rro = 0.0;
  
  if (step == 1) {
    ops_par_loop(tea_leaf_init_zero2_kernel, "tea_leaf_init_zero2_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(z, 1, S2D_00, "double", OPS_WRITE));
  } else if (step == 3) {
    ops_par_loop(tea_leaf_init_zero_kernel, "tea_leaf_init_zero_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE));
  }

  if (preconditioner_type != TL_PREC_NONE || (tl_ppcg_active && step ==3)) {
    //We don't apply the preconditioner on the final application of the polynomial acceleration   
    if (step == 1 || step == 2) {
      if (preconditioner_type == TL_PREC_JAC_BLOCK)
        tea_block_solve(r, z, cp, bfp, Kx, Ky, rx, ry);
      else if (preconditioner_type == TL_PREC_JAC_DIAG)
        tea_diag_solve(r, z, Mi, Kx, Ky, rx, ry);
    }

    if (step == 1 || step == 3) {
      ops_par_loop(tea_leaf_yeqx_kernel, "tea_leaf_yeqx_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(z, 1, S2D_00, "double", OPS_READ));
    }
  } else {
    if (step == 1) {
      ops_par_loop(tea_leaf_yeqx_kernel, "tea_leaf_yeqx_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(r, 1, S2D_00, "double", OPS_READ)); 
    }
  }

  if (step == 1 || step == 3) {
    ops_par_loop(tea_leaf_dot_kernel, "tea_leaf_dot_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(p, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
        ops_arg_reduce(red_temp, 1, "double", OPS_INC));
		ops_reduction_result(red_temp, rro);
  }
}

void tea_leaf_kernel_ppcg_inner(
  double *alpha, double *beta,
  double rx, double ry,
  int inner_step,
  ops_dat u,
  ops_dat r,
  ops_dat rtemp,
  ops_dat Kx,
  ops_dat Ky,
  ops_dat sd,
  ops_dat z,
  ops_dat utemp,
  ops_dat cp,
  ops_dat bfp,
  ops_dat Mi,
  int preconditioner_type)
{

  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};

  ops_par_loop(tea_leaf_ppcg_inner1_kernel, "tea_leaf_ppcg_inner1_kernel", tea_grid, 2, rangexy,
    ops_arg_dat(rtemp, 1, S2D_00, "double", OPS_INC),
    ops_arg_dat(Kx, 1, S2D_00_P10, "double", OPS_READ),
    ops_arg_dat(Ky, 1, S2D_00_0P1, "double", OPS_READ),
    ops_arg_dat(sd, 1, S2D_00_0M1_M10_P10_0P1, "double", OPS_READ),
    ops_arg_gbl(&rx, 1, "double", OPS_READ),
    ops_arg_gbl(&ry, 1, "double", OPS_READ));

  if (preconditioner_type != TL_PREC_NONE) {
    if (preconditioner_type == TL_PREC_JAC_BLOCK)
      tea_block_solve(r, z, cp, bfp, Kx, Ky, rx, ry);
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
      tea_diag_solve(r, z, Mi, Kx, Ky, rx, ry);

    ops_par_loop(tea_leaf_ppcg_inner2_kernel, "tea_leaf_ppcg_inner2_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(sd, 1, S2D_00, "double", OPS_RW),
      ops_arg_dat(utemp, 1, S2D_00, "double", OPS_INC),
      ops_arg_dat(z, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&alpha[inner_step], 1, "double", OPS_READ),
      ops_arg_gbl(&beta[inner_step], 1, "double", OPS_READ));
  } else {
    ops_par_loop(tea_leaf_ppcg_inner2_kernel, "tea_leaf_ppcg_inner2_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(sd, 1, S2D_00, "double", OPS_RW),
      ops_arg_dat(utemp, 1, S2D_00, "double", OPS_INC),
      ops_arg_dat(rtemp, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&alpha[inner_step], 1, "double", OPS_READ),
      ops_arg_gbl(&beta[inner_step], 1, "double", OPS_READ));
  }
}

void tea_leaf_ppcg_calc_zrnorm(
  ops_dat z,
  ops_dat r,
  int preconditioner_type, double *norm)
{
      //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};
  *norm = 0.0;
  if (preconditioner_type != TL_PREC_NONE || tl_ppcg_active) {
    ops_par_loop(tea_leaf_dot_kernel, "tea_leaf_dot_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(z, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
      ops_arg_reduce(red_temp, 1, "double", OPS_INC));
  } else {
    ops_par_loop(tea_leaf_norm2_kernel, "tea_leaf_norm2_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
      ops_arg_reduce(red_temp, 1, "double", OPS_INC));
  }
  ops_reduction_result(red_temp,norm);
}

void tea_leaf_ppcg_update_z(
  ops_dat z,
  ops_dat utemp)
{
      //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};
  ops_par_loop(tea_leaf_yeqx_kernel, "tea_leaf_yeqx_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(z, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(utemp, 1, S2D_00, "double", OPS_READ));  
}

void tea_leaf_ppcg_store_r(
  ops_dat r,
  ops_dat rstore)
{
      //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};
  ops_par_loop(tea_leaf_yeqx_kernel, "tea_leaf_yeqx_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(rstore, 1, S2D_00, "double", OPS_WRITE),
        ops_arg_dat(r, 1, S2D_00, "double", OPS_READ));  
}


void tea_leaf_ppcg_calc_rrn(
  ops_dat r,
  ops_dat rstore,
  ops_dat z,
  double *rrn)
{
      //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};

  *rrn = 0.0;

  ops_par_loop(tea_leaf_ppcg_reduce_kernel, "tea_leaf_ppcg_reduce_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(rstore, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
        ops_arg_dat(z, 1, S2D_00, "double", OPS_READ),
        ops_arg_reduce(red_temp, 1, "double", OPS_INC));  

  ops_reduction_result(red_temp,rrn);
}

void tea_leaf_run_ppcg_inner_steps(
  double *ch_alphas, double *ch_betas, double *theta,
  int tl_ppcg_inner_steps, double *solve_time, double rx, double ry) {

  int fields[NUM_FIELDS];
  int ppcg_cur_step;
  int t, inner_step, bounds_extra;
  int x_min_bound, x_max_bound, y_min_bound, y_max_bound;


  fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
  fields[FIELD_U] = 1;

  //IF (profiler_on) halo_time=timer()
  update_halo(fields,1);
  // IF (profiler_on) solve_time = solve_time + (timer() - halo_time)

  tea_leaf_ppcg_init_sd(vector_r, vector_rtemp, vector_Kx, vector_Ky, vector_sd, vector_z, vector_utemp, tri_cp, tri_bfp, vector_Mi, rx, ry, *theta, tl_preconditioner_type);

  // inner steps
  for( ppcg_cur_step=1;ppcg_cur_step<=tl_ppcg_inner_steps;ppcg_cur_step++) {

    fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
    fields[FIELD_SD] = 1;
    fields[FIELD_R] = 1;

    // IF (profiler_on) halo_time = timer()
    update_halo(fields,1); //halo_exchange_depth
    // IF (profiler_on) solve_time = solve_time + (timer()-halo_time)

    inner_step = ppcg_cur_step;

    fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
    fields[FIELD_SD] = 1;

      tea_leaf_kernel_ppcg_inner(                              
          ch_alphas, ch_betas,                              
          rx,
          ry,
          inner_step,
          u,                                
          vector_r,                         
          vector_rtemp,                         
          vector_Kx,                        
          vector_Ky,                        
          vector_sd,                        
          vector_z,                          
          vector_utemp,                          
          tri_cp,                          
          tri_bfp,                          
          vector_Mi,                          
          tl_preconditioner_type);

		if (ppcg_cur_step%tiling_frequency == 0) ops_execute(vector_r->block->instance);
  }

  fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
  fields[FIELD_P] = 1;
  
  // Then we update z = utemp
  tea_leaf_ppcg_update_z(vector_z, vector_utemp);
}
