module math_functions
    use, intrinsic :: ISO_C_BINDING
    implicit none

    contains

!   ==============================================================================

!   SIN function
!   ------------
    function fsin_float ( var ) BIND(C,name="fsin_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = SIN(var)
    end function fsin_float

    function fsin_dble ( var ) BIND(C,name="fsin_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res
        !print *, "sin real*8 called"
        res = SIN(var)
    end function fsin_dble

!   ==============================================================================

!   SINH function
!   -------------
    function fsinh_float ( var ) BIND(C,name="fsinh_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = SINH(var)
    end function fsinh_float

    function fsinh_dble ( var ) BIND(C,name="fsinh_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = SINH(var)
    end function fsinh_dble

!   ==============================================================================

!   COS function
!   ------------
    function fcos_float ( var ) BIND(C,name="fcos_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = COS(var)
    end function fcos_float

    function fcos_dble ( var ) BIND(C,name="fcos_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = COS(var)
    end function fcos_dble

!   ==============================================================================

!   COSH function
!   -------------
    function fcosh_float ( var ) BIND(C,name="fcosh_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = COSH(var)
    end function fcosh_float

    function fcosh_dble ( var ) BIND(C,name="fcosh_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = COSH(var)
    end function fcosh_dble

!   ==============================================================================

!   TAN function
!   ------------
    function ftan_float ( var ) BIND(C,name="ftan_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = TAN(var)
    end function ftan_float

    function ftan_dble ( var ) BIND(C,name="ftan_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = TAN(var)
    end function ftan_dble

!   ==============================================================================

!   TANH function
!   -------------
    function ftanh_float ( var ) BIND(C,name="ftanh_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = TANH(var)
    end function ftanh_float

    function ftanh_dble ( var ) BIND(C,name="ftanh_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = TANH(var)
    end function ftanh_dble

!   ==============================================================================

!   ASIN function
!   -------------
    function fasin_float ( var ) BIND(C,name="fasin_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = ASIN(var)
    end function fasin_float

    function fasin_dble ( var ) BIND(C,name="fasin_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = ASIN(var)
    end function fasin_dble

!   ==============================================================================

!   ACOS function
!   -------------
    function facos_float ( var ) BIND(C,name="facos_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = ACOS(var)
    end function facos_float

    function facos_dble ( var ) BIND(C,name="facos_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = ACOS(var)
    end function facos_dble

!   ==============================================================================

!   ATAN function
!   -------------
    function fatan_float ( var ) BIND(C,name="fatan_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = ATAN(var)
    end function fatan_float

    function fatan_dble ( var ) BIND(C,name="fatan_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = ATAN(var)
    end function fatan_dble

!   ==============================================================================

!   ATAN2 function
!   -------------
    function fatan2_float ( xvar, yvar ) BIND(C,name="fatan2_float") result( res )
        real(kind=4), intent(in) :: xvar, yvar
        real(kind=4) :: res

        res = ATAN2(xvar, yvar)
    end function fatan2_float

    function fatan2_dble ( xvar, yvar ) BIND(C,name="fatan2_dble") result( res )
        real(kind=8), intent(in) :: xvar, yvar
        real(kind=8) :: res

        res = ATAN2(xvar, yvar)
    end function fatan2_dble

!   ==============================================================================

!   SQRT function
!   -------------
    function fsqrt_float ( var ) BIND(C,name="fsqrt_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = SQRT(var)
    end function fsqrt_float

    function fsqrt_dble ( var ) BIND(C,name="fsqrt_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = SQRT(var)
    end function fsqrt_dble

!   ==============================================================================

!   EXP function
!   ------------
    function fexp_float ( var ) BIND(C,name="fexp_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = EXP(var)
    end function fexp_float

    function fexp_dble ( var ) BIND(C,name="fexp_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res
        !print *, "exp real*8 called"
        res = EXP(var)
    end function fexp_dble

!   ==============================================================================

!   LOG function
!   -------------
    function flog_float ( var ) BIND(C,name="flog_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = LOG(var)
    end function flog_float

    function flog_dble ( var ) BIND(C,name="flog_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = LOG(var)
    end function flog_dble

!   ==============================================================================

!   LOG10 function
!   --------------
    function flog10_float ( var ) BIND(C,name="flog10_float") result( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = LOG10(var)
    end function flog10_float

    function flog10_dble ( var ) BIND(C,name="flog10_dble") result( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = LOG10(var)
    end function flog10_dble

!   ==============================================================================

!   POW function
!   -------------
    function fpow_float ( base, expn ) BIND(C,name="fpow_float") result( res )
        real(kind=4), intent(in) :: base,expn
        real(kind=4) :: res

        res = base**expn
    end function fpow_float

    function fpow_dble ( base, expn ) BIND(C,name="fpow_dble") result( res )
        real(kind=8), intent(in) :: base,expn
        real(kind=8) :: res

        res = base**expn
    end function fpow_dble

    function fpow_int ( base, expn ) BIND(C,name="fpow_int") result( res )
        integer(kind=4), intent(in) :: base,expn
        integer(kind=4) :: res

        res = base**expn
    end function fpow_int

end module 
