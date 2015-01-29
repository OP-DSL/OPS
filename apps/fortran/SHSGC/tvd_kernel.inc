subroutine tvd_kernel(tht, ep2)

  real (kind=8), INTENT(in), DIMENSION(3) :: tht
  real (kind=8), DIMENSION(3) :: ep2

  real(8) :: maxim
  integer m

  DO m=1, 3
    IF (tht(OPS_ACC_MD1(m,0)) .gt. tht(OPS_ACC_MD1(m,1))) then
      maxim = tht(OPS_ACC_MD1(m,0))
    ELSE
      maxim = tht(OPS_ACC_MD1(m,1))
    END IF
    ep2(OPS_ACC_MD2(m,0)) = akap2 * maxim;
  END DO

end subroutine