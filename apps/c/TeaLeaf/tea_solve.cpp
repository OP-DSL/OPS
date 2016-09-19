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
 TeaLeaf. if not, see http://www.gnu.org/licenses/. */

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

 double eigmin, eigmax, theta, cn;
 int first=1;

void tea_leaf() {
  int n;
  double old_error,error,exact_error,initial_residual,c;
  int fields[NUM_FIELDS];

  double timer,halo_time,solve_time,init_time,reset_time,dot_product_time;

  // For CG solver
  double rro, pw, rrn, alpha, beta;

  // For chebyshev solver and PPCG solver
  double  cg_alphas[max_iters+1], cg_betas[max_iters+1];
  double  *ch_alphas=NULL, *ch_betas=NULL;

  int est_itc, cheby_calc_steps, max_cheby_iters, info, ppcg_inner_iters;
  int ch_switch_check;

  int cg_calc_steps;

  double cg_time, ch_time, total_solve_time, ch_per_it, cg_per_it, iteration_time;

	int halo_exchange_depth = 1;
	int zero_boundary[4] = {EXTERNAL_FACE,EXTERNAL_FACE,EXTERNAL_FACE,EXTERNAL_FACE};
	double rx, ry;

  cg_time = 0.0;
  ch_time = 0.0;
  cg_calc_steps = 0;
  ppcg_inner_iters = 0;
  ch_switch_check = 0;

  total_solve_time = 0.0;
  init_time = 0.0;
  halo_time = 0.0;
  solve_time = 0.0;

  if (coefficient != RECIP_CONDUCTIVITY && coefficient != CONDUCTIVITY) {
    ops_printf("Unknown coefficient option\n");
    exit(-1);
  }


  tl_ppcg_active = 0;  // Set to false until we have the eigenvalue estimates
  cheby_calc_steps = 0;
  cg_calc_steps = 0;

  ops_timers(&c, &total_solve_time);

  if (profiler_on) ops_timers(&c, &init_time);

  fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
  fields[FIELD_ENERGY1] = 1;
  fields[FIELD_DENSITY] = 1;

  update_halo(fields,halo_exchange_depth);

  tea_leaf_common_init(halo_exchange_depth, zero_boundary, reflective_boundary, density, energy1, u, u0, vector_r, vector_w, vector_Kx, vector_Ky, tri_cp, tri_bfp, vector_Mi, &rx, &ry, tl_preconditioner_type, coefficient);

  fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
  fields[FIELD_U] = 1;

  update_halo(fields,1);

  tea_leaf_calc_residual(u,u0,vector_r, vector_Kx, vector_Ky, rx, ry);
  tea_leaf_calc_2norm(1, &initial_residual);
  
  old_error = initial_residual;

  initial_residual=sqrt(fabs(initial_residual));

  if (verbose_on) {
      ops_fprintf(g_out,"Initial residual \n",initial_residual);
  }

  if (tl_use_cg || tl_use_chebyshev || tl_use_ppcg) {
    // All 3 of these solvers use the CG kernels to initialise
    tea_leaf_cg_init(vector_p, vector_r, vector_Mi, vector_z, vector_Kx, vector_Ky, tri_cp, tri_bfp, rx, ry, &rro, tl_preconditioner_type);
      
    // We need to update p when using CG due to matrix/vector multiplication
    fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
    fields[FIELD_U] = 1;
    fields[FIELD_P] = 1;

    //if (profiler_on) halo_time=timer()
    update_halo(fields,1);
    //if (profiler_on) init_time=init_time+(timer()-halo_time)

    fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
    fields[FIELD_P] = 1;
     
  } else if (tl_use_jacobi) {
    fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
    fields[FIELD_U] = 1;
  }

  //if (profiler_on) profiler%tea_init = profiler%tea_init + (timer() - init_time)

  if (profiler_on) ops_timers_core(&c,&solve_time);

  for (n = 1; n <= max_iters; n++) {

    ops_timers_core(&c,&iteration_time);

    if (ch_switch_check == 0) {
      if ((cheby_calc_steps > 0)) {
        // already started or already have good guesses for eigenvalues
        ch_switch_check = 1;
      } else if ((first == 0) && tl_use_ppcg && n > 1) {
        // if using PPCG, it can start almost immediately
        ch_switch_check = 1;
      } else if ((fabs(old_error) <= tl_ch_cg_epslim) && (n >= tl_ch_cg_presteps)) {
        // Error is less than set limit, and enough steps have passed to get a good eigenvalue guess
        ch_switch_check = 1;
      } else {
        // keep doing CG (or jacobi)
        ch_switch_check = 0;
      }
    }

    if ((tl_use_chebyshev || tl_use_ppcg) && ch_switch_check) {
      // on the first chebyshev steps, find the eigenvalues, coefficients,
      // and expected number of iterations
      if (cheby_calc_steps == 0) {
        // maximum number of iterations in chebyshev solver
        max_cheby_iters = max_iters - n + 2;

        if (first) {
          // calculate eigenvalues
          tea_calc_eigenvalues(cg_alphas, cg_betas, &eigmin, &eigmax, 
              max_iters, n-1, &info);
          first=0;
          if (info != 0) {
            ops_printf("Error in calculating eigenvalues\n");
            exit(-1);
          }
          eigmin = eigmin * 0.95;
          eigmax = eigmax * 1.05;
        }

        if (tl_use_chebyshev) {
          // calculate chebyshev coefficients
          // preallocate space for the coefficients
          ch_alphas = (double*)malloc((max_iters+1)*sizeof(double));
          ch_betas  = (double*)malloc((max_iters+1)*sizeof(double));
          
          tea_calc_ch_coefs(ch_alphas, ch_betas, eigmin, eigmax,
              &theta, max_cheby_iters);

          // don't need to update p any more
          fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
          fields[FIELD_U] = 1;
          
        } else if (tl_use_ppcg) {
          // currently also calculate chebyshev coefficients
          // preallocate space for the coefficients
          ch_alphas = (double*)malloc((max_iters+1)*sizeof(double));
          ch_betas  = (double*)malloc((max_iters+1)*sizeof(double));
          
          //ls_coefs becomes ch_coefs
          tea_calc_ch_coefs(ch_alphas, ch_betas, eigmin, eigmax,
              &theta, tl_ppcg_inner_steps);
          // We have the eigenvalue estimates so turn on ppcg
          tl_ppcg_active = 1;
        }

        cn = eigmax/eigmin;

        //TODO: formatting
        ops_fprintf(g_out, "Switching after %3d CG its, error %10.7E\n", n, rro);
        ops_fprintf(g_out, "Eigen min %10.6E Eigen max %10.6E Condition number %1.6lf Error %10.7E\n", eigmin, eigmax, cn, old_error);
        ops_printf("Switching after %3d CG its, error %10.7E\n", n, rro);
        ops_printf("Eigen min %10.6E Eigen max %10.6E Condition number %1.6lf Error %10.7E\n", eigmin, eigmax, cn, old_error);
                

      // Reinitialise CG with PPCG applied
      
      // Step 1: we reinitialise z if we precondition or else we just use r     
      tea_leaf_ppcg_init(vector_p,vector_r,vector_Mi,vector_z, vector_Kx,vector_Ky, tri_cp, tri_bfp, rx,ry, &rro,tl_preconditioner_type,ppcg_inner_iters,ch_alphas,ch_betas,theta,solve_time,2);
  
      // Step 2: compute a new z from r or previous z    
      tea_leaf_run_ppcg_inner_steps(ch_alphas, ch_betas, &theta,tl_ppcg_inner_steps, &solve_time, rx, ry);
      ppcg_inner_iters = ppcg_inner_iters + tl_ppcg_inner_steps;

      // Step 3: Recompute p after polynomial acceleration
      tea_leaf_ppcg_init(vector_p,vector_r,vector_Mi,vector_z, vector_Kx,vector_Ky, tri_cp, tri_bfp, rx,ry, &rro,tl_preconditioner_type,ppcg_inner_iters,ch_alphas,ch_betas,theta,solve_time,3);

      // need to update p when using CG due to matrix/vector multiplication
      fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
      fields[FIELD_U] = 1;
      fields[FIELD_P] = 1;

      //if (profiler_on) halo_time=timer()
      update_halo(fields,1);
      //if (profiler_on) init_time=init_time+(timer()-halo_time)

      fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
      fields[FIELD_P] = 1;
      }

      if (tl_use_chebyshev) {
        if (cheby_calc_steps == 0) {
          tea_leaf_cheby_first_step(ch_alphas, ch_betas, fields,
              &old_error, &theta, cn, max_cheby_iters, &est_itc, solve_time,rx,ry);

          cheby_calc_steps = 1;
        } else {
          tea_leaf_cheby_iterate(u,u0,vector_p, vector_r, vector_Mi, vector_w, vector_z, vector_Kx, vector_Ky, tri_cp, tri_bfp,ch_alphas, ch_betas, rx,ry,cheby_calc_steps,tl_preconditioner_type);

          // after an estimated number of iterations has passed, calc resid.
          // Leaving 10 iterations between each global reduction won't affect
          // total time spent much if at all (number of steps spent in
          // chebyshev is typically O(300+)) but will greatly reduce global
          // synchronisations needed
          if ((n >= est_itc) && (n%10 == 0)) {
            tea_leaf_calc_2norm(1, &error);
          }
        }
        
      // PPCG iteration, first needs an estimate of the eigenvalues  
      // We are essentially doing PPFCG(1) here where the F denotes that CG is flexible.
      // This accoutns for rounding error arising from the polynomial preconditioning
      } else if (tl_use_ppcg) {

        // w = Ap
        // pw = p.w

        tea_leaf_cg_calc_w(vector_p, vector_w, vector_Kx,vector_Ky,rx,ry,&pw);

        // Now need to store r
        tea_leaf_ppcg_store_r(vector_r,vector_rstore);
     
        // Now compute r.z for alpha
        tea_leaf_ppcg_calc_zrnorm(vector_z, vector_r, tl_preconditioner_type, &rro);
  
        // alpha = z.r / (pw)
        alpha = rro/pw;

        tea_leaf_cg_calc_ur(u,vector_p, vector_r, vector_Mi, vector_w, vector_z, tri_cp, tri_bfp, vector_Kx, vector_Ky, rx,ry,alpha, &rrn,tl_preconditioner_type);

        // not using rrn, so don't do a tea_allsum

        tea_leaf_run_ppcg_inner_steps(ch_alphas, ch_betas, &theta,
            tl_ppcg_inner_steps, &solve_time, rx, ry);

        ppcg_inner_iters = ppcg_inner_iters + tl_ppcg_inner_steps ;
 
        // We use flexible CG, FCG(1)
        tea_leaf_ppcg_calc_rrn(vector_r, vector_rstore, vector_z, &rrn);

        // beta = (r.z)_{k+1} / (r.z)_{k}
        beta = rrn/rro;

        tea_leaf_cg_calc_p(vector_p,vector_r,vector_z,beta,tl_preconditioner_type);

        error = rrn;
        rro = rrn;

      }

      cheby_calc_steps = cheby_calc_steps + 1;
      
    // Either: -  CG iteration
    //          - Or if we choose Chebyshev or PPCG we need
    //    a few CG iterations to get an estimate of the eigenvalues
    } else if (tl_use_cg || tl_use_chebyshev || tl_use_ppcg) {

      fields[FIELD_P] = 1;
      cg_calc_steps = cg_calc_steps + 1;

      // w = Ap
      // pw = p.w
      tea_leaf_cg_calc_w(vector_p, vector_w, vector_Kx,vector_Ky,rx,ry,&pw);

      alpha = rro/pw;
      cg_alphas[n] = alpha;

      // u = u + a*p
      // r = r - a*w
      tea_leaf_cg_calc_ur(u,vector_p, vector_r, vector_Mi, vector_w, vector_z, tri_cp, tri_bfp, vector_Kx, vector_Ky, rx,ry,alpha, &rrn,tl_preconditioner_type);

      beta = rrn/rro;
      cg_betas[n] = beta;

      // p = r + b*p
      tea_leaf_cg_calc_p(vector_p,vector_r,vector_z,beta,tl_preconditioner_type);

      error = rrn;
      rro = rrn;
    
    // Jacobi iteration
    } else if (tl_use_jacobi) {
      tea_leaf_jacobi_solve(rx,ry,vector_Kx, vector_Ky, &error, u0, u, vector_r);
    }

    // updates u and possibly p
    // if (profiler_on) halo_time = timer()
    update_halo(fields,1);
    // if (profiler_on) solve_time = solve_time + (timer()-halo_time)

    if (profiler_on) {
      double t;
      ops_timers_core(&c,&t);
      if (tl_use_chebyshev && ch_switch_check) {
        ch_time=ch_time+(t-iteration_time);
      } else {
        cg_time=cg_time+(t-iteration_time);
      }
    }

    error=sqrt(fabs(error));

    if (verbose_on) {
      ops_fprintf(g_out,"Residual %g\n",error);
    }
    
		if (fabs(error) < eps*initial_residual) break;

    old_error = error;

  }

  if (tl_check_result) {
    fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
    fields[FIELD_U] = 1;

    //if (profiler_on) halo_time = timer()
    update_halo(fields,1);
    //if (profiler_on) solve_time = solve_time + (timer()-halo_time)

    tea_leaf_calc_residual(u,u0,vector_r, vector_Kx, vector_Ky, rx, ry);
    tea_leaf_calc_2norm(1, &exact_error);

    exact_error = sqrt(exact_error);
  }

  // if (profiler_on) profiler%tea_solve = profiler%tea_solve + (timer() - solve_time)


  ops_fprintf(g_out,"Conduction error %-10.7E\n", error/initial_residual);
  ops_printf("Conduction error %-10.7E\n", error/initial_residual);

  if (tl_check_result) {
    ops_fprintf(g_out,"EXACT error calculated as %g\n", exact_error/initial_residual);
    ops_printf("EXACT error calculated as %g\n", exact_error/initial_residual);
  }

  ops_fprintf(g_out,"Iteration count %8d\n", n-1);
  ops_printf("Iteration count %8d\n", n-1);

  if (tl_use_ppcg) {
    ops_fprintf(g_out,"PPCG Iteration count %d (Total %d)\n", ppcg_inner_iters, ppcg_inner_iters + n - 1);
    ops_printf("PPCG Iteration count %d (Total %d)\n", ppcg_inner_iters, ppcg_inner_iters + n - 1);
  }
      

  // RESET
//  if (profiler_on) reset_time=timer()
  
  if (ch_alphas != NULL) {free(ch_alphas); free(ch_betas);}
  
  tea_leaf_finalise(energy1, density,u);

  fields[0]=0;fields[1]=0;fields[2]=0;fields[3]=0;fields[4]=0;fields[5]=0;fields[6]=0;
  fields[FIELD_ENERGY1] = 1;

  // if (profiler_on) halo_time=timer()
  update_halo(fields,1);
  // if (profiler_on) reset_time = reset_time + (timer()-halo_time)

  // if (profiler_on) profiler%tea_reset = profiler%tea_reset + (timer() - reset_time)

  if (profiler_on) {
    double t;
    ops_timers(&c, &t);
    total_solve_time = (t - total_solve_time);
    ops_fprintf(g_out, "Solve Time %g Its %d Time Per It %g\n", total_solve_time, n, total_solve_time/n);
    ops_printf(        "Solve Time %g Its %d Time Per It %g\n", total_solve_time, n, total_solve_time/n);
  }

  if (profiler_on && tl_use_chebyshev) {
    //tea_sum(ch_time)
    //tea_sum(cg_time)
    //TODO why are we summing these?

      // cg_per_it = MERGE((cg_time/cg_calc_steps)/parallel%max_task, 0.0_8, cg_calc_steps > 0)
      // ch_per_it = MERGE((ch_time/cheby_calc_steps)/parallel%max_task, 0.0_8, cheby_calc_steps > 0)

      // WRITE(0, "(a3, a16, a7, a16, a7)") "", "Time", "Its", "Per it", "Ratio"
      // WRITE(0, "(a3, f16.10, i7, f16.10, f7.2)") &
      //     "CG", cg_time + 0.0_8, cg_calc_steps, cg_per_it, 1.0_8
      // WRITE(0, "(a3, f16.10, i7, f16.10, f7.2)") "CH", ch_time + 0.0_8, cheby_calc_steps, &
      //     ch_per_it, MERGE(ch_per_it/cg_per_it, 0.0_8, cheby_calc_steps > 0)
      // WRITE(0, "('Chebyshev actually took ', i6, ' (' i6, ' off guess)')") &
      //     cheby_calc_steps, cheby_calc_steps-est_itc

      // WRITE(g_out, "(a3, a16, a7, a16, a7)") "", "Time", "Its", "Per it", "Ratio"
      // WRITE(g_out, "(a3, f16.10, i7, f16.10, f7.2)") &
      //     "CG", cg_time + 0.0_8, cg_calc_steps, cg_per_it, 1.0_8
      // WRITE(g_out, "(a3, f16.10, i7, f16.10, f7.2)") "CH", ch_time + 0.0_8, cheby_calc_steps, &
      //     ch_per_it, MERGE(ch_per_it/cg_per_it, 0.0_8, cheby_calc_steps > 0)
      // WRITE(g_out, "('Chebyshev actually took ', i6, ' (' i6, ' off guess)')") &
      //     cheby_calc_steps, cheby_calc_steps-est_itc
  }
}
