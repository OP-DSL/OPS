include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hdf5

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_original.F90 Makefile
	$(FC) $(FFLAGS) -o $@ laplace2d_original.F90 -lm

clean:
	$(RM) $(BIN)
