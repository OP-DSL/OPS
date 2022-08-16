SUBROUTINE start()

    USE DATA_MODULE

    time = 0.0
    step = 0
    dtold = dtinit
    dt    = dtinit

    CALL build_field()

    CALL initialise_chunk()

END SUBROUTINE start
