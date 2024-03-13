MODULE OPS_CONSTANTS
!----------shsgc Vars/Consts--------------
#ifdef OPS_WITH_CUDAFOR
use cudafor
integer, constant :: nxp_opsconstant
integer, constant :: nyp_opsconstant
integer, constant :: xhalo_opsconstant
integer, constant :: yhalo_opsconstant
real(8), constant :: xmin_opsconstant
real(8), constant :: ymin_opsconstant
real(8), constant :: xmax_opsconstant
real(8), constant :: ymax_opsconstant
real(8), constant :: dx_opsconstant
real(8), constant :: dy_opsconstant
real(8), constant :: pl_opsconstant
real(8), constant :: pr_opsconstant
real(8), constant :: rhol_opsconstant
real(8), constant :: rhor_opsconstant
real(8), constant :: ul_opsconstant
real(8), constant :: ur_opsconstant
real(8), constant :: gam_opsconstant
real(8), constant :: gam1_opsconstant
real(8), constant :: eps_opsconstant
real(8), constant :: lambda_opsconstant
!real(8), constant :: a1(3)
!real(8), constant :: a2(3)
real(8), constant :: dt_opsconstant
real(8), constant :: del2_opsconstant
real(8), constant :: akap2_opsconstant
real(8), constant :: tvdsmu_opsconstant
real(8), constant :: con_opsconstant
integer :: nxp
integer :: nyp
integer :: xhalo
integer :: yhalo
real(8) :: xmin
real(8) :: ymin
real(8) :: xmax
real(8) :: ymax
real(8) :: dx
real(8) :: dy
real(8) :: pl
real(8) :: pr
real(8) :: rhol
real(8) :: rhor
real(8) :: ul
real(8) :: ur
real(8) :: gam
real(8) :: gam1
real(8) :: eps
real(8) :: lambda
!real(8) :: a1(3)
!real(8) :: a2(3)
real(8) :: dt
real(8) :: del2
real(8) :: akap2
real(8) :: tvdsmu
real(8) :: con
#else
integer :: nxp
integer :: nyp
integer :: xhalo
integer :: yhalo
real(8) :: xmin
real(8) :: ymin
real(8) :: xmax
real(8) :: ymax
real(8) :: dx
real(8) :: dy
real(8) :: pl
real(8) :: pr
real(8) :: rhol
real(8) :: rhor
real(8) :: ul
real(8) :: ur
real(8) :: gam
real(8) :: gam1
real(8) :: eps
real(8) :: lambda
!real(8) :: a1(3)
!real(8) :: a2(3)
real(8) :: dt
real(8) :: del2
real(8) :: akap2
real(8) :: tvdsmu
real(8) :: con
#endif
END MODULE OPS_CONSTANTS
