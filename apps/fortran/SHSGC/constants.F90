MODULE OPS_CONSTANTS
!----------shsgc Vars/Consts--------------
#ifdef OPS_WITH_CUDAFOR_2 !ifdef OPS_WITH_CUDAFOR
use cudafor
integer :: nxp
integer :: nyp
integer, constant :: xhalo
integer, constant :: yhalo
real(8), constant :: xmin
real(8), constant :: ymin
real(8), constant :: xmax
real(8), constant :: ymax
real(8), constant :: dx
real(8), constant :: dy
real(8), constant :: pl
real(8), constant :: pr
real(8), constant :: rhol
real(8), constant :: rhor
real(8), constant :: ul
real(8), constant :: ur
real(8), constant :: gam
real(8), constant :: gam1
real(8), constant :: eps
real(8), constant :: lambda
!real(8), constant :: a1(3)
!real(8), constant :: a2(3)
real(8), constant :: dt
real(8), constant :: del2
real(8), constant :: akap2
real(8), constant :: tvdsmu
real(8), constant :: con
#else
integer nxp
integer nyp
integer xhalo
integer yhalo
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