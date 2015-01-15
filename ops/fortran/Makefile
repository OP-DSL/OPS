#
# The following environment variables are required:
#
# CUDA_INSTALL_PATH - CUDA base directory
# OPS_COMPILER      - Compiler toolchain: gnu (gcc, gfortran) or pgi (pgc++, pgfortran)
# OPS_INSTALL_PATH  - OPS base directory
#
OPS = $(OPS_INSTALL_PATH)
C_OPS = $(OPS)/c
C_OPS_INC = $(C_OPS)/include
C_OPS_SRC = $(C_OPS)/src

F_OPS = $(OPS)/fortran
F_OPS_INC = $(F_OPS)/include
F_OPS_SRC = $(F_OPS)/src
F_OPS_OBJ = $(F_OPS)/obj
F_OPS_LIB = $(F_OPS)/lib
F_OPS_MOD = $(F_OPS)/mod

ifeq ($(OPS_COMPILER),intel)
  CC 		:= icc -std=c99
ifdef DEBUG
  OPT	 	:= -O0 -g
else
  OPT	 	:= -O3 -xAVX #vec-report -xAVX
endif
  CCFLAGS       := $(OPT) -Wall -Werror -wd981
  CXX 		:= icpc
  CXXFLAGS 	:= $(CCFLAGS)
  F_INC_MOD	:= $(F_OPS_MOD)/intel
  FC		:= ifort
  FFLAGS	:= $(OPT) -module $(F_INC_MOD)
  MPICC 	:= $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX 	:= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS 	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  TARGETS	:= clean core f_seq
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif

ifdef HDF5_INSTALL_PATH
  HDF5_INC 	:= -DCHECKPOINTING -I$(HDF5_INSTALL_PATH)/include
  CHKCC		:= $(MPICXX)
else
  CHKCC		:= $(CC)
endif

.PHONY: clean mkdir

all:	$(TARGETS)

mkdir:
	@mkdir -p $(F_OPS_LIB) $(F_OPS_OBJ) $(F_OPS_MOD) $(F_INC_MOD)

core:	mkdir $(C_OPS_INC)/ops_lib_core.h $(C_OPS_SRC)/core/ops_lib_core.c $(C_OPS_SRC)/externlib/ops_util.c
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lazy.cpp -o $(F_OPS_OBJ)/ops_lazy.o
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/externlib/ops_util.c -o $(F_OPS_OBJ)/ops_util.o
	$(CHKCC) $(CXXFLAGS) -I$(C_OPS_INC) $(HDF5_INC) -c $(C_OPS_SRC)/externlib/ops_checkpointing.c -o $(F_OPS_OBJ)/ops_checkpointing.o

f_seq: 	mkdir core $(F_OPS_OBJ)/ops_lib_core.o
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/sequential/ops_seq.c -o $(F_OPS_OBJ)/ops_seq.o
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.c -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(FFLAGS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	ar -r $(F_OPS_LIB)/libops_for_seq.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_seq.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o

clean:
	-rm -if $(F_OPS_OBJ)/*.o
	-rm -if $(F_OPS_LIB)/*.a
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_SRC)/*~
#-rm -if $(F_OPS_SRC)/ops_for_reference.c
#-rm -if $(F_OPS_SRC)/ops_cpp_reference.F90
	-rm -if $(F_OPS_MOD)/*.mod
	-rm -if $(F_INC_MOD)/*.mod
	-rm -if $(F_OPS)/*~
