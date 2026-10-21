ops_dat p_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
p_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "p_B0");
}
ops_dat Residual0_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual0_B0");
}
ops_dat Residual1_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual1_B0");
}
ops_dat Residual2_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual2_B0");
}
ops_dat Residual3_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual3_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual3_B0");
}
ops_dat Residual4_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
Residual4_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual4_B0");
}
ops_dat rho_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rho_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rho_B0");
}
ops_dat rho_RKold_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rho_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rho_RKold_B0");
}
ops_dat rhoE_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhoE_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhoE_B0");
}
ops_dat rhoE_RKold_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhoE_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhoE_RKold_B0");
}
ops_dat rhou0_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou0_B0");
}
ops_dat rhou0_RKold_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou0_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou0_RKold_B0");
}
ops_dat rhou1_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou1_B0");
}
ops_dat rhou1_RKold_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou1_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou1_RKold_B0");
}
ops_dat rhou2_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou2_B0");
}
ops_dat rhou2_RKold_B0;
{
int halo_p[] = {0, 0, 0};
int halo_m[] = {0, 0, 0};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
rhou2_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "rhou2_RKold_B0");
}
ops_dat T_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
T_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "T_B0");
}
ops_dat u0_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u0_B0");
}
ops_dat u1_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u1_B0");
}
ops_dat u2_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
u2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "u2_B0");
}
ops_dat wk0_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk0_B0");
}
ops_dat wk1_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk1_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk1_B0");
}
ops_dat wk2_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk2_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk2_B0");
}
ops_dat wk3_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk3_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk3_B0");
}
ops_dat wk4_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk4_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk4_B0");
}
ops_dat wk5_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk5_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk5_B0");
}
ops_dat wk6_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk6_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk6_B0");
}
ops_dat wk7_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk7_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk7_B0");
}
ops_dat wk8_B0;
{
int halo_p[] = {5, 5, 5};
int halo_m[] = {-5, -5, -5};
int size[] = {block0np0, block0np1, block0np2};
int base[] = {0, 0, 0};
double* value = NULL;
wk8_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk8_B0");
}