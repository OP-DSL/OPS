// Boundary condition exchange code on opensbliblock00 direction 0 left
ops_halo_group periodicBC_direction0_side0_5_block0 ;
{
int halo_iter[] = {2};
int from_base[] = {0};
int to_base[] = {block0np0};
int from_dir[] = {1};
int to_dir[] = {1};
ops_halo halo0 = ops_decl_halo(phi_B0, phi_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo grp[] = {halo0};
periodicBC_direction0_side0_5_block0 = ops_decl_halo_group(1,grp);
}
// Boundary condition exchange code on opensbliblock00 direction 0 right
ops_halo_group periodicBC_direction0_side1_6_block0 ;
{
int halo_iter[] = {2};
int from_base[] = {block0np0 - 2};
int to_base[] = {-2};
int from_dir[] = {1};
int to_dir[] = {1};
ops_halo halo0 = ops_decl_halo(phi_B0, phi_B0, halo_iter, from_base, to_base, from_dir, to_dir);
ops_halo grp[] = {halo0};
periodicBC_direction0_side1_6_block0 = ops_decl_halo_group(1,grp);
}
