ops_dat phi_B0;
{
if (restart == 1){
phi_B0 = ops_decl_dat_hdf5(opensbliblock00, 1, "double", "phi_B0", "restart.h5");
}
else {
int halo_p[] = {5};
int halo_m[] = {-5};
int size[] = {block0np0};
int base[] = {0};
double* value = NULL;
phi_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "phi_B0");
}
}
ops_dat phi_RKold_B0;
{
int halo_p[] = {5};
int halo_m[] = {-5};
int size[] = {block0np0};
int base[] = {0};
double* value = NULL;
phi_RKold_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "phi_RKold_B0");
}
ops_dat Residual0_B0;
{
int halo_p[] = {5};
int halo_m[] = {-5};
int size[] = {block0np0};
int base[] = {0};
double* value = NULL;
Residual0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "Residual0_B0");
}
ops_dat wk0_B0;
{
int halo_p[] = {5};
int halo_m[] = {-5};
int size[] = {block0np0};
int base[] = {0};
double* value = NULL;
wk0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "wk0_B0");
}
ops_dat x0_B0;
{
int halo_p[] = {5};
int halo_m[] = {-5};
int size[] = {block0np0};
int base[] = {0};
double* value = NULL;
x0_B0 = ops_decl_dat(opensbliblock00, 1, size, base, halo_m, halo_p, value, "double", "x0_B0");
}
