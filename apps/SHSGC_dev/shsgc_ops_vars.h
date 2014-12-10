ops_block shsgc_grid;

//ops dats
ops_dat x;
ops_dat rho_old, rho_new, rho_res;
ops_dat rhou_old, rhou_new, rhou_res;
// ops_dat rhov_old, rhov_new;
ops_dat rhoE_old, rhoE_new, rhoE_res;
ops_dat rhoin;
ops_dat fn, dfn;
ops_dat r, al, alam, gt, tht, ep2, cmp, cf, eff, s;
ops_dat readvar;

ops_reduction rms;
ops_dat u, v, w, T, viscu_res, viscE_res;
ops_dat u_x, u_xx, T_x, T_xx, mu, mu_x;
//
//Declare commonly used stencils
//
ops_stencil S1D_0, S1D_01, S1D_0M1;
ops_stencil S1D_0M1M2P1P2;

