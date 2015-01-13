#ifndef __OPS_DATA_H
#define __OPS_DATA_H


extern ops_block shsgc_grid;

//ops dats
extern ops_dat x;
extern ops_dat rho_old, rho_new, rho_res;
extern ops_dat rhou_old, rhou_new, rhou_res;
// ops_dat rhov_old, rhov_new;
extern ops_dat rhoE_old, rhoE_new, rhoE_res;
extern ops_dat rhoin;
extern ops_dat fn, dfn;
extern ops_dat r, al, alam, gt, tht, ep2, cmp, cf, eff, s;
extern ops_dat readvar;

extern ops_reduction rms;
extern ops_dat u, v, w, T, viscu_res, viscE_res;
extern ops_dat u_x, u_xx, T_x, T_xx, mu, mu_x;

//
//
//Declare commonly used stencils
//
extern ops_stencil S1D_0, S1D_01, S1D_0M1;
extern ops_stencil S1D_0M1M2P1P2;

#endif