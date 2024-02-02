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

#include "tea_leaf_cheby_kernels.h"

void tea_leaf_init_zero2_kernel (double * p, double * z);
void tea_leaf_init_zero_kernel (double * p);
void tea_leaf_yeqx_kernel (double * p, const double * x);
void tea_leaf_yeqax_kernel (double * p, const double * x, const double * a);
void tea_leaf_dot_kernel (const double * r, const double * p, double *rro);
void tea_leaf_xpy_kernel(double * u, const double * p);
void tea_leaf_xpy_kernel(double * u, const double * p);
void tea_leaf_axpy_kernel(double * u, const double * p, const double * alpha);
void tea_leaf_axpby_kernel(double * u, const double * p, const double * alpha, const double * beta);
void tea_leaf_zeqxty_kernel(double * z, const double * x, const double * y);
void tea_leaf_recip_kernel(double * u, const double * p);
void tea_leaf_recip2_kernel(double *z, const double *x, const double *y);
void tea_leaf_recip3_kernel(double *z, const double *x, const double *theta);
void tea_leaf_norm2_kernel(const double *x, double * norm);

void tea_leaf_cheby_init(
  ops_dat u,
	ops_dat u0,
  ops_dat p,
	ops_dat r,
	ops_dat Mi,
  ops_dat w,
	ops_dat z,
	ops_dat Kx,
	ops_dat Ky,
	ops_dat cp,
	ops_dat bfp,
	double rx, double ry,
	double theta, int preconditioner_type)
{
  //initialize sizes using global values
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;

  int rangexy[] = {x_min,x_max,y_min,y_max};

  ops_par_loop(tea_leaf_cheby_init_kernel, "tea_leaf_cheby_init_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(w, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(r, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(Kx, 1, S2D_00_P10, "double", OPS_READ),
      ops_arg_dat(Ky, 1, S2D_00_0P1, "double", OPS_READ),
      ops_arg_dat(u, 1, S2D_00_0M1_M10_P10_0P1, "double", OPS_READ),
      ops_arg_dat(u0, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&rx, 1, "double", OPS_READ),
      ops_arg_gbl(&ry, 1, "double", OPS_READ));

  if (preconditioner_type != TL_PREC_NONE) {

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
      tea_block_solve(r, z, cp, bfp, Kx, Ky, rx, ry);
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
      tea_diag_solve(r, z, Mi, Kx, Ky, rx, ry);

    ops_par_loop(tea_leaf_recip3_kernel, "tea_leaf_recip3_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(z, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&theta, 1, "double", OPS_READ));
  } else {
    ops_par_loop(tea_leaf_recip3_kernel, "tea_leaf_recip3_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(p, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&theta, 1, "double", OPS_READ));
  }

  double one = 1.0;
  ops_par_loop(tea_leaf_xpy_kernel, "tea_leaf_xpy_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(u, 1, S2D_00, "double", OPS_INC),
      ops_arg_dat(p, 1, S2D_00, "double", OPS_READ));

}

void tea_leaf_cheby_iterate(
  ops_dat u,
  ops_dat u0,
  ops_dat p,
  ops_dat r,
  ops_dat Mi,
  ops_dat w,
  ops_dat z,
  ops_dat Kx,
  ops_dat Ky,
  ops_dat cp,
  ops_dat bfp,
  double *ch_alphas,
  double *ch_betas,
  double rx, double ry, int step, int preconditioner_type)
{
  int x_min = field.x_min;
  int x_max = field.x_max;
  int y_min = field.y_min;
  int y_max = field.y_max;
  int rangexy[] = {x_min,x_max,y_min,y_max};

  ops_par_loop(tea_leaf_cheby_init_kernel, "tea_leaf_cheby_init_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(w, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(r, 1, S2D_00, "double", OPS_WRITE),
      ops_arg_dat(Kx, 1, S2D_00_P10, "double", OPS_READ),
      ops_arg_dat(Ky, 1, S2D_00_0P1, "double", OPS_READ),
      ops_arg_dat(u, 1, S2D_00_0M1_M10_P10_0P1, "double", OPS_READ),
      ops_arg_dat(u0, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&rx, 1, "double", OPS_READ),
      ops_arg_gbl(&ry, 1, "double", OPS_READ));

   if (preconditioner_type != TL_PREC_NONE) {

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
      tea_block_solve(r, z, cp, bfp, Kx, Ky, rx, ry);
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
      tea_diag_solve(r, z, Mi, Kx, Ky, rx, ry);

    ops_par_loop(tea_leaf_axpby_kernel, "tea_leaf_axpby_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(p, 1, S2D_00, "double", OPS_RW),
      ops_arg_dat(z, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&ch_alphas[step], 1, "double", OPS_READ),
      ops_arg_gbl(&ch_betas[step], 1, "double", OPS_READ));

  } else {
    ops_par_loop(tea_leaf_axpby_kernel, "tea_leaf_axpby_kernel", tea_grid, 2, rangexy,
      ops_arg_dat(p, 1, S2D_00, "double", OPS_RW),
      ops_arg_dat(r, 1, S2D_00, "double", OPS_READ),
      ops_arg_gbl(&ch_alphas[step], 1, "double", OPS_READ),
      ops_arg_gbl(&ch_betas[step], 1, "double", OPS_READ));
  }

  ops_par_loop(tea_leaf_xpy_kernel, "tea_leaf_xpy_kernel", tea_grid, 2, rangexy,
    ops_arg_dat(u, 1, S2D_00, "double", OPS_INC),
    ops_arg_dat(p, 1, S2D_00, "double", OPS_READ));
}

void tqli(double *d, double *e, int n, int *info) {
  int i,iter,l,m,cont;
  double b,c,dd,f,g,p,r,s;
	for (i = 1; i < n; i++) {
    e[i] = e[i+1];
  }
	e[n] = 0.0;
  *info = 0;
  for (l = 1; l <=n; l++) {
    iter=0;
    while(true) { //iterate
      for (m = l; m <= n-1;m++) {
        dd=fabs(d[m])+fabs(d[m+1]);
        if (fabs(e[m])+dd == dd) break;
      }
      if (m == l) break;
      if (iter == 30) {
        *info=1;
        return;
      }
      iter=iter+1;
      g=(d[l+1]-d[l])/(2.0*e[l]);
      r=sqrt(g*g+1.0*1.0);
      g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
      s=1.0;
      c=1.0;
      p=0.0;
      for (i = m-1; i>= l; i--) {
        f=s*e[i];
        b=c*e[i];
        r=sqrt(f*f+g*g);
        e[i+1]=r;
        if (r == 0.0) {
          d[i+1]=d[i+1]-p;
          e[m]=0.0;
          cont = 1;
          break;
        } else cont = 0;
        s=f/r;
        c=g/r;
        g=d[i+1]-p;
        r=(d[i]-g)*s+2.0*c*b;
        p=s*r;
        d[i+1]=g+p;
        g=c*r-b;
      }
      if (cont) continue;
      d[l]=d[l]-p;
      e[l]=g;
      e[m]=0.0;
    } //iterate
  }
}
void tea_calc_eigenvalues(double *cg_alphas, double *cg_betas,double *eigmin, double *eigmax, int max_iters, int tl_ch_cg_presteps, int *info) {

	int swapped = 0;
  double diag[max_iters+1];
  double offdiag[max_iters+1];
	for (int i = 0; i < max_iters+1; i++ ) {
		diag[i] = 0.0;
		offdiag[i] = 0.0;
	}
  for (int n=1;n <= tl_ch_cg_presteps;n++) {
    diag[n] = 1.0/cg_alphas[n];
    if (n > 1) diag[n] = diag[n] + cg_betas[n-1]/cg_alphas[n-1];
    if (n < tl_ch_cg_presteps) offdiag[n+1] = sqrt(cg_betas[n])/cg_alphas[n];
  }

  tqli(diag, offdiag, tl_ch_cg_presteps, info);

  // ! could just call this instead
  // !offdiag(:)=eoshift(offdiag(:),1)
  // !CALL dsterf(tl_ch_cg_presteps, diag, offdiag, info)

  if (*info != 0) return;

  // bubble sort eigenvalues
  while(true) {
    for (int n = 1; n <= tl_ch_cg_presteps-1; n++) {
      if (diag[n] >= diag[n+1]) {
        double tmp = diag[n];
        diag[n] = diag[n+1];
        diag[n+1] = tmp;
        swapped = 1;
      }
    }
    if (!swapped) break;
    swapped = 0;
  }

  *eigmin = diag[1];
  *eigmax = diag[tl_ch_cg_presteps];

  if (*eigmin < 0.0 || *eigmax < 0.0) *info = 1;
}

void tea_calc_ch_coefs(double *ch_alphas, double *ch_betas,double eigmin, double eigmax, double *theta, int max_cheby_iters) {
  
  *theta = (eigmax + eigmin)/2.0;
  double delta = (eigmax - eigmin)/2.0;
  double sigma = *theta/delta;

  double rho_old = 1.0/sigma;

  for (int n = 1; n <= max_cheby_iters; n++) { //TODO: indexing
    double rho_new = 1.0/(2.0*sigma - rho_old);

    double cur_alpha = rho_new*rho_old;
    double cur_beta = 2.0*rho_new/delta;

    ch_alphas[n] = cur_alpha;
    ch_betas[n] = cur_beta;

    rho_old = rho_new;
  }

}

void tea_leaf_cheby_first_step(double *ch_alphas, double *ch_betas, int *fields,
    double *error, double *theta, double cn, int max_cheby_iters, int *est_itc, double solve_time, double rx, double ry) {


  double bb = 0;
  // calculate 2 norm of u0
  tea_leaf_calc_2norm(0, &bb);

  // initialise 'p' array
  tea_leaf_cheby_init(u,u0,vector_p,vector_r,vector_Mi,vector_w,vector_z,vector_Kx,vector_Ky,tri_cp,tri_bfp,rx,ry,*theta,tl_preconditioner_type);

  // if (profiler_on) halo_time = timer()
  update_halo(fields,1);
  // if (profiler_on) solve_time = solve_time + (timer()-halo_time)

  tea_leaf_cheby_iterate(u,u0,vector_p,vector_r,vector_Mi,vector_w,vector_z,vector_Kx,vector_Ky,tri_cp,tri_bfp,ch_alphas, ch_betas, rx,ry,1,tl_preconditioner_type);

  tea_leaf_calc_2norm(1, error);

  double it_alpha = eps/2.0*sqrt(bb/(*error));//eps*bb/(4.0*(*error));
  double gamm = (sqrt(cn) - 1.0)/(sqrt(cn) + 1.0);
  *est_itc = round(log(it_alpha)/(log(gamm)));

  ops_fprintf(g_out,"    est itc\n%11d\n",*est_itc);
  ops_printf("    est itc\n%11d\n",*est_itc);

}

