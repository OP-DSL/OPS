subroutine test_kernel(rho_new, rms)

  real (kind=8), INTENT(in), DIMENSION(1) :: rho_new
  real (kind=8), DIMENSION(1) :: rms

  rms(1) = rms(1) + rho_new(OPS_ACC1(0))**2.0_8

end subroutine