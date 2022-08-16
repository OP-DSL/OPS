SUBROUTINE initialise()
    use OPS_Fortran_Reference
    use DATA_MODULE
    
    use, intrinsic :: ISO_C_BINDING
    
    character(len=80) :: line

    g_out = 6
    g_in = 7
    
    OPEN(FILE='clover.out',ACTION='WRITE',UNIT=g_out,IOSTAT=ios)

    IF (ios.NE.0) THEN
        call ops_printf("Error opening clover.out file.")
        call EXIT(-1)
    ENDIF

    call ops_printf("Output file clover.out opened. All output will go there\n")
    
    write(g_out, "(a)") "\n"
    write(g_out,"(a,f8.2)") " Clover version ", g_version
    write(g_out,"(a)") " Clover will run from the following input:-\n"
        
    g_in = 8
    OPEN(FILE='clover.in',ACTION='READ',STATUS='OLD',UNIT=g_in,IOSTAT=ios)
    IF (ios.NE.0) THEN
      g_in = 9
      OPEN(FILE='clover.in',UNIT=g_in,STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      write(g_in,'(A)') '*clover'
      write(g_in,'(A)') ' state 1 density=0.2 energy=1.0'
      write(g_in,'(A)') ' state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0'
      write(g_in,'(A)') ' x_cells=10'
      write(g_in,'(A)') ' y_cells=2'
      write(g_in,'(A)') ' xmin=0.0'
      write(g_in,'(A)') ' ymin=0.0'
      write(g_in,'(A)') ' xmax=10.0'
      write(g_in,'(A)') ' ymax=2.0'
      write(g_in,'(A)') ' initial_timestep=0.04'
      write(g_in,'(A)') ' timestep_rise=1.5'
      write(g_in,'(A)') ' max_timestep=0.04'
      write(g_in,'(A)') ' end_time=3.0'
      write(g_in,'(A)') ' test_problem 1'
      write(g_in,'(A)') '*endclover'
      CLOSE(g_in)
      g_in = 10
      OPEN(FILE='clover.in',ACTION='READ',STATUS='OLD',UNIT=g_in,IOSTAT=ios)
    ENDIF

    DO
        READ(g_in,"(a)",IOSTAT=ios) line
        IF (ios.NE.0) EXIT
        write(g_out, "(a)") line
    ENDDO
    
    write(g_out, "(a)") "\n"
    write(g_out, "(a)") " Initialising and generating"
    write(g_out, "(a)") "\n"

    !call read_input()

    write(g_out, "(a)") " Starting the calculation\n"

    CLOSE(g_in)

END SUBROUTINE initialise



