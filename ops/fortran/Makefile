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

CUDA_INC = $(CUDA_INSTALL_PATH)/include

# flags for nvcc
# set NV_ARCH to select the correct one
ifndef NV_ARCH
  MESSAGE=select an NVIDA device to compile in CUDA, e.g. make NV_ARCH=KEPLER
  NV_ARCH=Kepler
endif
ifeq ($(NV_ARCH),Fermi)
  CODE_GEN_CUDA=-gencode arch=compute_20,code=sm_21
  PGI_CUDA_FORT_FLAGS=-Mcuda=cuda5.5,cc20
else
ifeq ($(NV_ARCH),Kepler)
  CODE_GEN_CUDA=-gencode arch=compute_35,code=sm_35
  PGI_CUDA_FORT_FLAGS=-Mcuda=cuda6.0,cc35
endif
endif

MPI_INC = $(MPI_INSTALL_PATH)/include
MPI_LIB = $(MPI_INSTALL_PATH)/lib

ifeq ($(OPS_COMPILER),intel)
  CC 		= icc -std=c99
ifdef DEBUG
  OPT	 	= -O0 -g
else
  OPT	 	= -O3 -xAVX #vec-report -xAVX
endif
  CCFLAGS       = $(OPT) -Wall -Werror -wd981
  CXX 		= icpc
  CXXFLAGS 	= $(CCFLAGS)
  F_INC_MOD	= $(F_OPS_MOD)/intel
  FC		= ifort
  FFLAGS	= $(OPT) -module $(F_INC_MOD)
  MPICC 	= $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX 	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS 	= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  TARGETS	= clean core f_seq f_mpi
else
ifeq ($(OPS_COMPILER),pgi)
  CC            = pgcc
ifdef DEBUG
  OPT           = -O0 -g
else
  OPT           = -O3
endif
  CCFLAGS       = $(OPT)
  NVCC		= nvcc $(CODE_GEN_CUDA)
  NVCCFLAGS	= -Xcompiler="-std=c99 -Wall -fPIC -pedantic -pipe"
  CUDA_ALIGNE_FLAG = -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CXX           = pgc++
  CXXFLAGS      = $(CCFLAGS)
  F_INC_MOD     = $(F_OPS_MOD)/pgi
  FC            = pgfortran
  FFLAGS        = $(OPT) -module $(F_INC_MOD)
  OPENACCFLAGS	= $(OPT) -module $(F_INC_MOD)/openacc #$(PGI_CUDA_FORT_FLAGS) -Minform=inform -Mvect –Mvect=cachesize:12288 -DOPS_WITH_CUDAFOR
  MPICC         = $(MPI_INSTALL_PATH)/bin/mpicc
  MPICXX        = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS      = $(CXXFLAGS)
  TARGETS       = clean core f_seq f_mpi f_cuda f_mpi_cuda f_openacc
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif

ifdef HDF5_INSTALL_PATH
  HDF5_INC 	  	:= -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB		:= -L$(HDF5_INSTALL_PATH)/lib
  CHKCC			:= $(MPICXX)
ifdef CHECKPOINTING
  HDF5_INC 	  	:= $(HDF5_INC) -DCHECKPOINTING
endif
else
  CHKCC			:= $(CC)
endif

.PHONY: clean mkdir

all:	$(TARGETS)

mkdir:
	@mkdir -p $(F_OPS_LIB) $(F_OPS_OBJ) $(F_OPS_MOD) $(F_INC_MOD) $(F_INC_MOD)/openacc

core:	mkdir $(C_OPS_INC)/ops_lib_core.h $(C_OPS_SRC)/core/ops_lib_core.c $(C_OPS_SRC)/externlib/ops_util.c
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lazy.cpp -o $(F_OPS_OBJ)/ops_lazy.o
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/externlib/ops_util.c -o $(F_OPS_OBJ)/ops_util.o
	$(FC) $(FFLAGS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o

hdf5:	mkdir core $(F_OPS_OBJ)/ops_lib_core.o
	$(CHKCC) $(CXXFLAGS) -I$(C_OPS_INC) $(HDF5_INC) -c $(C_OPS_SRC)/externlib/ops_checkpointing.c -o $(F_OPS_OBJ)/ops_checkpointing.o
	$(CC) $(CFLAGS) $(HDF5_INC) -I$(MPI_INC) -I$(C_OPS_INC) -c $(C_OPS_SRC)/externlib/ops_hdf5.c -o $(F_OPS_OBJ)/ops_hdf5.o
	$(FC) $(FFLAGS) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_hdf5.a $(F_OPS_OBJ)/ops_hdf5.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o


f_seq: 	mkdir core hdf5 $(F_OPS_OBJ)/ops_lib_core.o
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/sequential/ops_seq.c -o $(F_OPS_OBJ)/ops_seq.o
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.c -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(FFLAGS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_seq.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_seq.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

f_cuda: mkdir core
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.c -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_decl.c -o $(F_OPS_OBJ)/ops_cuda_decl.o
	$(NVCC) $(CCFLAGS) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cu -o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.c -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(FFLAGS) -DOPS_WITH_CUDAFOR -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) -DOPS_WITH_CUDAFOR -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_decl.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
	$(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

f_mpi: 	mkdir core $(F_OPS_OBJ)/ops_lib_core.o
	$(MPICC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_decl.c -o $(F_OPS_OBJ)/ops_mpi_decl.o
	$(MPICC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_core.c -o $(F_OPS_OBJ)/ops_mpi_core.o
	$(MPICC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_partition.c -o $(F_OPS_OBJ)/ops_mpi_partition.o
	$(MPICC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support.c -o $(F_OPS_OBJ)/ops_mpi_rt_support.o
	$(MPICC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support_host.c -o $(F_OPS_OBJ)/ops_mpi_rt_support_host.o
	$(MPICC) $(CXXFLAGS) -I$(MPI_INC) $(HDF5_INC) -I$(C_OPS_INC)  -I$(CUDA_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_hdf5.c -o $(F_OPS_OBJ)/ops_mpi_hdf5.o
	$(FC) $(FFLAGS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_mpi.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl.o $(F_OPS_OBJ)/ops_mpi_core.o $(F_OPS_OBJ)/ops_mpi_rt_support.o \
	$(F_OPS_OBJ)/ops_mpi_partition.o $(F_OPS_OBJ)/ops_mpi_rt_support_host.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o $(F_OPS_OBJ)/ops_mpi_hdf5.o

f_mpi_cuda: mkdir core
	$(MPICC) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INC) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_decl_cuda.c -o $(F_OPS_OBJ)/ops_mpi_decl_cuda.o
	$(MPICC) $(MPIFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_partition.c -o $(F_OPS_OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_core.c -o $(F_OPS_OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support.c -o $(F_OPS_OBJ)/ops_mpi_rt_support.o
	$(MPICC) $(CXXFLAGS) -I$(MPI_INC) $(HDF5_INC) -I$(C_OPS_INC)  -I$(CUDA_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_hdf5.c -o $(F_OPS_OBJ)/ops_mpi_hdf5.o
	$(NVCC)  $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support_cuda.cu -o $(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o -I$(MPI_INSTALL_PATH)/include
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.c -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(FC) $(FFLAGS) -DOPS_WITH_CUDAFOR -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) -DOPS_WITH_CUDAFOR -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_mpi_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_core.o $(F_OPS_OBJ)/ops_mpi_partition.o $(F_OPS_OBJ)/ops_mpi_decl_cuda.o \
	$(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(F_OPS_OBJ)/ops_mpi_rt_support.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_mpi_hdf5.o

f_openacc: mkdir core
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.c -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCCFLAGS) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_decl.c -o $(F_OPS_OBJ)/ops_cuda_decl.o
	$(NVCC) $(CCFLAGS) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cu -o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o
	$(CC) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.c -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(OPENACCFLAGS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(OPENACCFLAGS) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_openacc.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
        $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_decl.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
        $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o


clean:
	-rm -if $(F_OPS_OBJ)/*.o
	-rm -if $(F_OPS_LIB)/*.a
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_MOD)/*.mod
	-rm -if $(F_INC_MOD)/*.mod
	-rm -if $(F_OPS)/*~
