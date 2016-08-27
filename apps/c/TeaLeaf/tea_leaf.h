#ifndef TEA_FUN_HEADER_H
#define TEA_FUN_HEADER_H


void build_field();
void diffuse();
void field_summary();
void generate();
void initialise();
void initialise_chunk();

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
	double *rro, int preconditioner_type);

void tea_leaf_cg_calc_w(
  ops_dat p,
  ops_dat w,
  ops_dat Kx,
  ops_dat Ky,
  double rx, double ry, double *pw);

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
  double rx, double ry, double alpha, double *rnn, int preconditioner_type);

void tea_leaf_cg_calc_p(
  ops_dat p,
  ops_dat r,
  ops_dat z,
  double beta, int preconditioner_type);


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
	double theta, int preconditioner_type);

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
  double rx, double ry, int step, int preconditioner_type);

void tea_calc_eigenvalues(double *ch_alphas, double *ch_betas,double *eighmin, double *eigmax, int max_iters, int tl_ch_cg_presteps, int *info);
void tea_calc_ch_coefs(double *ch_alphas, double *ch_betas,double eighmin, double eigmax, double *theta, int max_cheby_iters);
void tea_leaf_cheby_first_step(double *ch_alphas, double *ch_betas, int *fields,
    double *error, double *theta, double cn, int max_cheby_iters, int *est_itc, double solve_time,double rx, double ry);

void tea_leaf_common_init(
  int halo_depth,
  int* zero_boundary,
  int reflective_boundary,
  ops_dat density,
  ops_dat energy,
  ops_dat u,
	ops_dat u0,
	ops_dat r,
  ops_dat w,
	ops_dat Kx,
	ops_dat Ky,
	ops_dat cp,
	ops_dat bfp,
  ops_dat Mi,
	double *rx, double *ry,
	int preconditioner_type, int coef);
void tea_leaf_finalise(
  ops_dat energy,
  ops_dat density,
  ops_dat u);

void tea_leaf_calc_residual(
  ops_dat u,
  ops_dat u0,
  ops_dat r,
  ops_dat Kx,
  ops_dat Ky,
  double rx, double ry);

void tea_leaf_calc_2norm_kernel(
  ops_dat arr,
  double *norm);
void tea_leaf_calc_2norm(int norm_array, double *norm);
void tea_diag_init(
	int halo_depth,
  ops_dat Mi,
  ops_dat Kx,
  ops_dat Ky,
  double rx, double ry);
void tea_diag_solve(
  ops_dat r,
  ops_dat z,
  ops_dat Mi,
  ops_dat Kx,
  ops_dat Ky,
  double rx, double ry);
void tea_block_init(
  ops_dat cp,
  ops_dat bfp,
  ops_dat Kx,
  ops_dat Ky,
  double rx, double ry);
void tea_block_solve(
  ops_dat r,
  ops_dat z,
  ops_dat cp,
  ops_dat bfp,
  ops_dat Kx,
  ops_dat Ky,
  double rx, double ry);

void tea_leaf_jacobi_solve(
  double rx, double ry,
	ops_dat Kx,
	ops_dat Ky,
  double *error,
	ops_dat u0,
	ops_dat u1,
	ops_dat un);

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
  double theta, int preconditioner_type);

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
  double theta, double solve_time, int step);

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
  int preconditioner_type);

void tea_leaf_ppcg_calc_zrnorm(
  ops_dat z,
  ops_dat r,
  int preconditioner_type, double *norm);


void tea_leaf_ppcg_update_z(
  ops_dat z,
  ops_dat utemp,
  int preconditioner_type, double *norm);

void tea_leaf_ppcg_store_r(
  ops_dat r,
  ops_dat rstore);

void tea_leaf_ppcg_calc_rrn(
  ops_dat r,
  ops_dat rstore,
  ops_dat z,
  double *rrn);

void tea_leaf_run_ppcg_inner_steps(
  double *ch_alphas, double *ch_betas, double *theta,
  int tl_ppcg_inner_steps, double *solve_time, double rx, double ry);

void tea_leaf();

void timestep();

void update_halo(int *fields, int depth);

#endif
