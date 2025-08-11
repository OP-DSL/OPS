SUBROUTINE print_kernel(dat0)
    REAL(kind=8), DIMENSION(2) :: dat0

    REAL(kind=8) :: val1, val2

    val1 = dat0(OPS_ACC_MD1(1,0,0))
    val2 = dat0(OPS_ACC_MD1(2,0,0))

!    print *,"(",val1," ",val2,")"

#if !defined(OPS_WITH_CUDAFOR) && !defined(OPS_WITH_OMPOFFLOADFOR)
    print '("(",F0.6,1X,F0.6,")")', val1, val2
#endif

END SUBROUTINE print_kernel
