OPS = $(OPS_INSTALL_PATH)
F_OPS = $(OPS)/fortran
F_OPS_LIB = $(F_OPS)/lib
F_OPS_MOD = $(F_OPS)/mod/$(OPS_COMPILER)
F_OPS_INC = $(F_OPS)/include

ifeq ($(OPS_COMPILER),intel)
  CC 		:= icc -std=c99
ifdef DEBUG
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
  TARGETS	:= clean multiDim_seq
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif

FLINK = -L$(F_OPS_LIB)
FMODS = -I$(F_OPS_MOD) -I$(F_OPS_INC)

all: $(TARGETS)

multiDim_seq: multiDim_ops.F90
	$(FC) $(OPT) $(FMODS) -c multiDim_print_kernel_seq_kernel.F90 \
	multiDim_copy_kernel_seq_kernel.F90 multiDim_kernel_seq_kernel.F90 multiDim_ops.F90
	$(FC) $(OPT) $(FLINK) multiDim_ops.o multiDim_kernel_seq_kernel.o multiDim_copy_kernel_seq_kernel.o \
	multiDim_print_kernel_seq_kernel.o -o multiDim_seq -lops_for_seq

clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f multiDim_seq
