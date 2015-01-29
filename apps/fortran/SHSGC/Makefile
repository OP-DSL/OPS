OPS = $(OPS_INSTALL_PATH)
F_OPS = $(OPS)/fortran
F_OPS_LIB = $(F_OPS)/lib
F_OPS_MOD = $(F_OPS)/mod/$(OPS_COMPILER)
F_OPS_INC = $(F_OPS)/include

ifeq ($(OPS_COMPILER),intel)
  CC 		:= icc -std=c99
ifdef DEBUG
  #OPT	 	:= -O3 -g -xAVX -vec-report2 #-xAVX
  OPT	 	:= -O0 -g
else
  OPT	 	:= -O3 -xAVX -vec-report2 #-xAVX
endif
  CCFLAGS       := $(OPT) -Wall -Werror -wd981
  CXX 		:= icpc
  CXXFLAGS 	:= $(CCFLAGS)
  CXXLINK	:= -lstdc++
  FC		:= ifort
  FFLAGS	:= $(OPT)
  MPICC 	:= $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX 	:= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS 	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  TARGETS	:= clean shsgc_seq
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif

FLINK = -L$(F_OPS_LIB) $(CXXLINK)
FMODS = -I$(F_OPS_MOD) -I$(F_OPS_INC)

all: $(TARGETS)

shsgc_seq: shsgc_ops.F90 constants.F90
	$(FC) $(OPT) $(FMODS) -c constants.F90 ./MPI/initialize_kernel_seq_kernel.F90 \
	./MPI/drhouupdx_kernel_seq_kernel.F90 ./MPI/zerores_kernel_seq_kernel.F90 \
	./MPI/drhoudx_kernel_seq_kernel.F90 ./MPI/save_kernel_seq_kernel.F90 \
        ./MPI/drhoEpudx_kernel_seq_kernel.F90 ./MPI/updateRK3_kernel_seq_kernel.F90 \
	./MPI/Riemann_kernel_seq_kernel.F90 ./MPI/limiter_kernel_seq_kernel.F90 \
	./MPI/tvd_kernel_seq_kernel.F90 ./MPI/vars_kernel_seq_kernel.F90 \
	./MPI/calupwindeff_kernel_seq_kernel.F90 ./MPI/fact_kernel_seq_kernel.F90 \
	./MPI/update_kernel_seq_kernel.F90 ./MPI/test_kernel_seq_kernel.F90 shsgc_ops.F90
	$(FC) $(OPT) $(FLINK) shsgc_ops.o constants.o initialize_kernel_seq_kernel.o \
	drhouupdx_kernel_seq_kernel.o zerores_kernel_seq_kernel.o save_kernel_seq_kernel.o \
	drhoudx_kernel_seq_kernel.o drhoEpudx_kernel_seq_kernel.o updateRK3_kernel_seq_kernel.o \
	Riemann_kernel_seq_kernel.o limiter_kernel_seq_kernel.o tvd_kernel_seq_kernel.o \
	vars_kernel_seq_kernel.o calupwindeff_kernel_seq_kernel.o \
	fact_kernel_seq_kernel.o update_kernel_seq_kernel.o test_kernel_seq_kernel.o \
	-o shsgc_seq -lops_for_seq

clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f shsgc_seq
