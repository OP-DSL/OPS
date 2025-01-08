MODULE math_FUNCTIONs
    use, intrinsic :: ISO_C_BINDING
    implicit none

    contains

!   ==============================================================================

!   SIN function
!   ------------
    FUNCTION ftn_sin_float ( var ) BIND(C,name="ftn_sin_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = SIN(var)
    END FUNCTION ftn_sin_float

    FUNCTION ftn_sin_double ( var ) BIND(C,name="ftn_sin_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = SIN(var)
    END FUNCTION ftn_sin_double

!   ==============================================================================

!   SINH function
!   -------------
    FUNCTION ftn_sinh_float ( var ) BIND(C,name="ftn_sinh_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = SINH(var)
    END FUNCTION ftn_sinh_float

    FUNCTION ftn_sinh_double ( var ) BIND(C,name="ftn_sinh_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = SINH(var)
    END FUNCTION ftn_sinh_double

!   ==============================================================================

!   COS function
!   ------------
    FUNCTION ftn_cos_float ( var ) BIND(C,name="ftn_cos_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = COS(var)
    END FUNCTION ftn_cos_float

    FUNCTION ftn_cos_double ( var ) BIND(C,name="ftn_cos_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = COS(var)
    END FUNCTION ftn_cos_double

!   ==============================================================================

!   COSH function
!   -------------
    FUNCTION ftn_cosh_float ( var ) BIND(C,name="ftn_cosh_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = COSH(var)
    END FUNCTION ftn_cosh_float

    FUNCTION ftn_cosh_double ( var ) BIND(C,name="ftn_cosh_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = COSH(var)
    END FUNCTION ftn_cosh_double

!   ==============================================================================

!   TAN function
!   ------------
    FUNCTION ftn_tan_float ( var ) BIND(C,name="ftn_tan_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = TAN(var)
    END FUNCTION ftn_tan_float

    FUNCTION ftn_tan_double ( var ) BIND(C,name="ftn_tan_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = TAN(var)
    END FUNCTION ftn_tan_double

!   ==============================================================================

!   TANH function
!   -------------
    FUNCTION ftn_tanh_float ( var ) BIND(C,name="ftn_tanh_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = TANH(var)
    END FUNCTION ftn_tanh_float

    FUNCTION ftn_tanh_double ( var ) BIND(C,name="ftn_tanh_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = TANH(var)
    END FUNCTION ftn_tanh_double

!   ==============================================================================

!   ASIN function
!   -------------
    FUNCTION ftn_asin_float ( var ) BIND(C,name="ftn_asin_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = ASIN(var)
    END FUNCTION ftn_asin_float

    FUNCTION ftn_asin_double ( var ) BIND(C,name="ftn_asin_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = ASIN(var)
    END FUNCTION ftn_asin_double

!   ==============================================================================

!   ACOS function
!   -------------
    FUNCTION ftn_acos_float ( var ) BIND(C,name="ftn_acos_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = ACOS(var)
    END FUNCTION ftn_acos_float

    FUNCTION ftn_acos_double ( var ) BIND(C,name="ftn_acos_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = ACOS(var)
    END FUNCTION ftn_acos_double

!   ==============================================================================

!   ATAN function
!   -------------
    FUNCTION ftn_atan_float ( var ) BIND(C,name="ftn_atan_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = ATAN(var)
    END FUNCTION ftn_atan_float

    FUNCTION ftn_atan_double ( var ) BIND(C,name="ftn_atan_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = ATAN(var)
    END FUNCTION ftn_atan_double

!   ==============================================================================

!   ATAN2 function
!   -------------
    FUNCTION ftn_atan2_float ( xvar, yvar ) BIND(C,name="ftn_atan2_float") RESULT( res )
        real(kind=4), intent(in) :: xvar, yvar
        real(kind=4) :: res

        res = ATAN2(xvar, yvar)
    END FUNCTION ftn_atan2_float

    FUNCTION ftn_atan2_double ( xvar, yvar ) BIND(C,name="ftn_atan2_double") RESULT( res )
        real(kind=8), intent(in) :: xvar, yvar
        real(kind=8) :: res

        res = ATAN2(xvar, yvar)
    END FUNCTION ftn_atan2_double

!   ==============================================================================

!   SQRT function
!   -------------
    FUNCTION ftn_sqrt_float ( var ) BIND(C,name="ftn_sqrt_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = SQRT(var)
    END FUNCTION ftn_sqrt_float

    FUNCTION ftn_sqrt_double ( var ) BIND(C,name="ftn_sqrt_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = SQRT(var)
    END FUNCTION ftn_sqrt_double

!   ==============================================================================

!   EXP function
!   ------------
    FUNCTION ftn_exp_float ( var ) BIND(C,name="ftn_exp_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = EXP(var)
    END FUNCTION ftn_exp_float

    FUNCTION ftn_exp_double ( var ) BIND(C,name="ftn_exp_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = EXP(var)
    END FUNCTION ftn_exp_double

!   ==============================================================================

!   LOG function
!   -------------
    FUNCTION ftn_log_float ( var ) BIND(C,name="ftn_log_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = LOG(var)
    END FUNCTION ftn_log_float

    FUNCTION ftn_log_double ( var ) BIND(C,name="ftn_log_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = LOG(var)
    END FUNCTION ftn_log_double

!   ==============================================================================

!   LOG10 function
!   --------------
    FUNCTION ftn_log10_float ( var ) BIND(C,name="ftn_log10_float") RESULT( res )
        real(kind=4), intent(in) :: var
        real(kind=4) :: res

        res = LOG10(var)
    END FUNCTION ftn_log10_float

    FUNCTION ftn_log10_double ( var ) BIND(C,name="ftn_log10_double") RESULT( res )
        real(kind=8), intent(in) :: var
        real(kind=8) :: res

        res = LOG10(var)
    END FUNCTION ftn_log10_double

!   ==============================================================================

!   POW function
!   -------------
    FUNCTION ftn_pow_float ( base, expn ) BIND(C,name="ftn_pow_float") RESULT( res )
        real(kind=4), intent(in) :: base,expn
        real(kind=4) :: res

        res = base**expn
    END FUNCTION ftn_pow_float

    FUNCTION ftn_pow_double ( base, expn ) BIND(C,name="ftn_pow_double") RESULT( res )
        real(kind=8), intent(in) :: base,expn
        real(kind=8) :: res

        res = base**expn
    END FUNCTION ftn_pow_double

    FUNCTION ftn_pow_int ( base, expn ) BIND(C,name="ftn_pow_int") RESULT( res )
        integer(kind=4), intent(in) :: base,expn
        integer(kind=4) :: res

        res = base**expn
    END FUNCTION ftn_pow_int

END MODULE 
