//WRITE,READ
SUBROUTINE user_kernel_eqA(A, B)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B
END SUBROUTINE

//WRITE
SUBROUTINE user_kernel_eqB(A)
real(kind=8), dimension(1) :: A
END SUBROUTINE

//RW,READ
SUBROUTINE user_kernel_eqC(A, B)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B
END SUBROUTINE

//INC,READ,READ
SUBROUTINE user_kernel_eqD(A, B, C)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B,C
END SUBROUTINE

//WRITE,READ,READ
SUBROUTINE user_kernel_eqE(A, B, C)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B,C
END SUBROUTINE

//READ,WRITE - DFBYDX
SUBROUTINE user_kernel_eqF(A, B)
real(kind=8), dimension(1), intent(in) :: A
real(kind=8), dimension(1) :: B
END SUBROUTINE

//INC,READ,READ,READ
SUBROUTINE user_kernel_eqG(A, B, C, D)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B,C,D
END SUBROUTINE

//INC,READ,READ,READ,READ,READ,READ
SUBROUTINE user_kernel_eqH(A, B, C, D, E, F, G)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B,C,D,E,F,G
END SUBROUTINE

//RW,INC,READ,READ,READ
SUBROUTINE user_kernel_eqI(A, B, C, D, E)
real(kind=8), dimension(1) :: A,B
real(kind=8), dimension(1), intent(in) :: C,E
integer(kind=4), dimension(1), intent(in) :: D
END SUBROUTINE

//INC,INC,READ,READ,READ,READ,READ,READ
SUBROUTINE user_kernel_eqJ(A, B, C, D, E, F, G, H)
real(kind=8), dimension(1) :: A,B
real(kind=8), dimension(1), intent(in) :: C,D,E,F,G,H
END SUBROUTINE

//INC,READ,READ,READ,READ,READ,READ,READ
SUBROUTINE user_kernel_eqK(A, B, C, D, E, F, G, H)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B,C,D,E,F,G,H
END SUBROUTINE

//INC,INC,READ,READ,READ,READ
SUBROUTINE user_kernel_eqL(A, B, C, D, E, F)
real(kind=8), dimension(1) :: A,B
real(kind=8), dimension(1), intent(in) :: C,D,E,F
END SUBROUTINE

//INC,READ,READ,READ,READ,READ
SUBROUTINE user_kernel_eqM(A, B, C, D, E, F)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B,C,D,E,F
END SUBROUTINE

//INC,READ,READ,READ,READ,READ,READ,READ,READ
SUBROUTINE user_kernel_eqN(A, B, C, D, E, F, G, H, I)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B,C,D,E,F,G,H,I
END SUBROUTINE

//RW,READ,READ,READ,READ,READ,READ,READ,READ
SUBROUTINE user_kernel_eqO(A, B, C, D, E, F, G, H, I)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B,C,D,E,F,G,H,I
END SUBROUTINE

//INC,READ
SUBROUTINE user_kernel_eqP(A, B)
real(kind=8), dimension(1) :: A
real(kind=8), dimension(1), intent(in) :: B
END SUBROUTINE


