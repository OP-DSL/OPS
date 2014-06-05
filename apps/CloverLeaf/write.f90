subroutine fortfunc(time,step,vol,mass,press,ie,ke,buf)
  use, intrinsic :: ISO_C_BINDING

  integer step
  double precision  time,vol,mass,press,ie,ke
  CHARACTER(LEN=350):: buf
  CHARACTER(LEN=39) :: temp0
  CHARACTER(LEN=168):: temp1
  CHARACTER(LEN=127):: temp2
  WRITE(temp0,*) 'Time ',time
  temp0 = ADJUSTL(temp0)
  WRITE(temp1,'(a20,7a16)')C_NEW_LINE//'             ','Volume','Mass','Density','Pressure','Internal Energy','Kinetic Energy','Total Energy'
  WRITE(temp2,'(a7,i7,7e16.4,a1)')C_NEW_LINE//' step:',step,vol,mass,mass/vol,press/vol,ie,ke,ie+ke,''//C_NEW_LINE
  WRITE(buf,*)temp0//temp1//temp2//CHAR(0)

  return
  end


subroutine fortfunc2(time,step,dtl_control,l,dt,jdt, kdt,  x_pos,y_pos,buf)
  use, intrinsic :: ISO_C_BINDING

  integer step,jdt, kdt,l
  double precision  time,dt,x_pos,y_pos
  CHARACTER(LEN=8):: dtl_control
  CHARACTER(LEN=350):: buf
  CHARACTER(LEN=340) :: temp0
  dtl_control =TRIM(ADJUSTL(dtl_control))
  WRITE(temp0,"(' step ', i7,' time ', f11.7,' control ',a8,'    timestep  ',1pe9.2,i8,',',i8,' x ',1pe9.2,' y ',1pe9.2)") &
                      step,time,dtl_control(1:l),dt,jdt,kdt,x_pos,y_pos
  temp0 = ADJUSTL(temp0)
  WRITE(buf,*)temp0//CHAR(0)
  return
  end
