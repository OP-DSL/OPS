FC       = ifort
FFLAGS   = -module $(OPS_INSTALL_PATH)/fortran/mod/$(OPS_COMPILER)
FMODS    = -I$(OPS_INSTALL_PATH)/fortran/mod/$(OPS_COMPILER)
FLINK    = -L$(OPS_INSTALL_PATH)/fortran/lib/$(OPS_COMPILER)
OMPFLAGS = -lstdc++ -Ofast

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_step4.F90 Makefile
	$(FC) $(FFLAGS) $(FMODS) $(FLINK) $(OMPFLAGS) -o $@ constants.F90 *_seq_kernel.F90 laplace2d_step4.F90 -lops_for_seq

clean:
	$(RM) $(BIN)
