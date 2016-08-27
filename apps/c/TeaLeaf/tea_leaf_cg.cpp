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
// @details Implicitly calculates the change in temperature using CG method


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

#include "tea_leaf_cg_kernels.h"

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

void tea_leaf_cg_init(
	ops_dat p,
	ops_dat r,
	ops_dat Mi,
	ops_dat z,
	ops_dat Kx,
	ops_dat Ky,
	ops_dat cp,
	ops_dat bfp,
	double rx, double ry,
	double *rro, int preconditioner_type)
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};

  *rro = 0.0;

  ops_par_loop(tea_leaf_init_zero2_kernel, "tea_leaf_init_zero2_kernel", tea_grid, 2, rangexy,
    ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE),
    ops_arg_dat(z, 1, S2D_00, "double", OPS_WRITE));


  if (preconditioner_type != TL_PREC_NONE) {

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
      tea_block_solve(r, z, cp, bfp, Kx, Ky, rx, ry);
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
      tea_diag_solve(r, z, Mi, Kx, Ky, rx, ry);
    ops_par_loop(tea_leaf_yeqx_kernel, "tea_leaf_yeqx_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(z, 1, S2D_00, "double", OPS_READ));
  } else {
    ops_par_loop(tea_leaf_yeqx_kernel, "tea_leaf_yeqx_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ));
  }

  ops_par_loop(tea_leaf_dot_kernel, "tea_leaf_dot_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(p, 1, S2D_00, "double", OPS_READ),
      ops_arg_reduce(red_temp, 1, "double", OPS_INC));

  ops_reduction_result(red_temp,rro);
}

void tea_leaf_cg_calc_w(
  ops_dat p,
  ops_dat w,
  ops_dat Kx,
  ops_dat Ky,
  double rx, double ry, double *pw)
{
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int rangexy[] = {x_min,x_max,y_min,y_max};

  *pw = 0.0;
  ops_par_loop(tea_leaf_cg_calc_w_reduce_kernel, "tea_leaf_cg_calc_w_reduce_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(w, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(Kx, 1, S2D_00_P10, "double", OPS_READ),
      ops_arg_dat(Ky, 1, S2D_00_0P1, "double", OPS_READ),
      ops_arg_dat(p, 1, S2D_00_0M1_M10_P10_0P1, "double", OPS_READ),
      ops_arg_gbl(&rx, 1, "double", OPS_READ),
      ops_arg_gbl(&ry, 1, "double", OPS_READ),
      ops_arg_reduce(red_temp, 1, "double", OPS_INC));

  ops_reduction_result(red_temp,pw);
}

void tea_leaf_cg_calc_ur(
  ops_dat u,
  ops_dat p,
  ops_dat r,
  ops_dat Mi,
  ops_dat w,
  ops_dat z,
  ops_dat cp,
  ops_dat bfp,
  ops_dat Kx,
  ops_dat Ky,
  double rx, double ry, double alpha, double *rnn, int preconditioner_type)
{
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int rangexy[] = {x_min,x_max,y_min,y_max};

  *rnn = 0.0;
  if (preconditioner_type != TL_PREC_NONE) {
    ops_par_loop(tea_leaf_axpy_kernel, "tea_leaf_axpy_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(u, 1, S2D_00, "double", OPS_INC),
        ops_arg_dat(p, 1, S2D_00, "double", OPS_READ),
        ops_arg_gbl(&alpha, 1, "double", OPS_READ));

    double malpha = -1.0 * alpha;
    ops_par_loop(tea_leaf_axpy_kernel, "tea_leaf_axpy_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(r, 1, S2D_00, "double", OPS_INC),
        ops_arg_dat(w, 1, S2D_00, "double", OPS_READ),
        ops_arg_gbl(&malpha, 1, "double", OPS_READ));


    if (preconditioner_type == TL_PREC_JAC_BLOCK)
      tea_block_solve(r, z, cp, bfp, Kx, Ky, rx, ry);
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
      tea_diag_solve(r, z, Mi, Kx, Ky, rx, ry);

    ops_par_loop(tea_leaf_dot_kernel, "tea_leaf_dot_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
      ops_arg_dat(z, 1, S2D_00, "double", OPS_READ),
      ops_arg_reduce(red_temp, 1, "double", OPS_INC));

  } else {
    ops_par_loop(tea_leaf_axpy_kernel, "tea_leaf_axpy_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(u, 1, S2D_00, "double", OPS_INC),
        ops_arg_dat(p, 1, S2D_00, "double", OPS_READ),
        ops_arg_gbl(&alpha, 1, "double", OPS_READ));

    ops_par_loop(tea_leaf_cg_calc_ur_r_reduce_kernel, "tea_leaf_cg_calc_ur_r_reduce_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(r, 1, S2D_00, "double", OPS_INC),
        ops_arg_dat(w, 1, S2D_00, "double", OPS_READ),
        ops_arg_gbl(&alpha, 1, "double", OPS_READ),
        ops_arg_reduce(red_temp, 1, "double", OPS_INC));
  }
  ops_reduction_result(red_temp,rnn);
}


void tea_leaf_cg_calc_p(
  ops_dat p,
  ops_dat r,
  ops_dat z,
  double beta, int preconditioner_type)
{
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int rangexy[] = {x_min,x_max,y_min,y_max};

  if (preconditioner_type != TL_PREC_NONE || tl_ppcg_active) {
    ops_par_loop(tea_leaf_axpy_kernel, "tea_leaf_axpy_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(p, 1, S2D_00, "double", OPS_INC),
        ops_arg_dat(z, 1, S2D_00, "double", OPS_READ),
        ops_arg_gbl(&beta, 1, "double", OPS_READ));
  } else {
    ops_par_loop(tea_leaf_axpy_kernel, "tea_leaf_axpy_kernel", tea_grid, 2, rangexy,
        ops_arg_dat(p, 1, S2D_00, "double", OPS_INC),
        ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
        ops_arg_gbl(&beta, 1, "double", OPS_READ));
  }
}



