#
# The following environment variables should be predefined:
#
# OPS_COMPILER (gnu,intel,etc)
#

#
# set paths for various files
#

SRC = src
INC = include
LIB = lib
OBJ = obj

CUDA_INC = -I$(CUDA_INSTALL_PATH)/include
OPENCL_INC = -I$(OPENCL_INSTALL_PATH)/include

#
# compiler settings
#

ifeq ($(OPS_COMPILER),gnu)
  CC 	  	:= gcc
  CCFLAGS 	:= -std=c99 -fPIC -DUNIX -Wall -ffloat-store#-g
  CXX 	  	:= g++
  CXXFLAGS	:= -std=c99 -fPIC -DUNIX -Wall -ffloat-store#-g -Wextra
  MPICXX  	:= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPICC   	:= $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
else
ifeq ($(OPS_COMPILER),intel)
  CC 		:= icc -std=c99
ifdef DEBUG
  CCFLAGS 	:= -O0 -g  #-vec-report -xSSE4.2
else
  CCFLAGS 	:= -O3 -vec-report -xAVX
endif
  CXX 		:= icpc
  CXXFLAGS 	:= $(CCFLAGS)
  MPICC 	:= $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX 	:= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS 	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
else
ifeq ($(OPS_COMPILER),mic)
  CC    := icc -std=c99
ifdef DEBUG
  CCFLAGS   := -O0 -g  -mmic -#-vec-report -xSSE4.2
else
  CCFLAGS   := -O3 -vec-report -mmic -mcmodel=medium
endif
  CXX     := icpc
  CXXFLAGS  := $(CCFLAGS)
  MPICC   := $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX  := $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS  := $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
else
ifeq ($(OPS_COMPILER),pgi)
  CC 		:= pgcc -c99
  CUDA_ALIGNE_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CCFLAGS 	:= -O3 -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  CXX 		:= pgCC
  CXXFLAGS 	:= $(CCFLAGS)
  MPICXX 	:= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPICC 	:= $(MPI_INSTALL_PATH)/bin/mpicc -c99
  MPIFLAGS 	:= $(CXXFLAGS)
else
ifeq ($(OPS_COMPILER),cray)
  CC        := cc
  CUDA_ALIGNE_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CCFLAGS   := -O3 -hgnu
  CXX       := CC
  CXXFLAGS  := $(CCFLAGS)
  MPICXX    := CC
  MPICC     := cc
  MPIFLAGS  := $(CXXFLAGS)
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif
endif
endif

NVCC  := nvcc
ifndef DEBUG
    NVCCFLAGS := -O3 -arch=sm_35 -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
else
    NVCCFLAGS := -O0 -g -G -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
endif

.PHONY: clean mklib

all: clean core seq mpi cuda mpi_cuda opencl

mklib:
	@mkdir -p $(LIB) $(OBJ)

core: mklib $(INC)/ops_lib_core.h $(SRC)/core/ops_lib_core.c $(SRC)/externlib/ops_util.c
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lib_core.c -o $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lazy.cpp -o $(OBJ)/ops_lazy.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/externlib/ops_util.c -o $(OBJ)/ops_util.o

seq: mklib core $(OBJ)/ops_lib_core.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/sequential/ops_seq.c -o $(OBJ)/ops_seq.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_seq.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_util.o $(OBJ)/ops_seq.o $(OBJ)/ops_dummy_singlenode.o

cuda: mklib core $(SRC)/cuda/ops_cuda_decl.c $(OBJ)/ops_lib_core.o
	$(CC) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) $(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_decl.c -o $(OBJ)/ops_cuda_decl.o
	$(CC) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) $(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_rt_support.c -o $(OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/cuda/ops_cuda_rt_support.cu -o $(OBJ)/ops_cuda_rt_support_cuda.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_util.o $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_cuda_rt_support_cuda.o $(OBJ)/ops_cuda_decl.o $(OBJ)/ops_dummy_singlenode.o

mpi: mklib core $(OBJ)/ops_lib_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_decl.c -o $(OBJ)/ops_mpi_decl.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_host.c -o $(OBJ)/ops_mpi_rt_support_host.o
	ar -r $(LIB)/libops_mpi.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl.o $(OBJ)/ops_mpi_rt_support.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_host.o $(OBJ)/ops_mpi_partition.o

mpi_cuda: mklib cuda $(OBJ)/ops_cuda_rt_support.o
	$(MPICC) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) -I$(INC) $(CUDA_INC) -c $(SRC)/mpi/ops_mpi_decl_cuda.c -o $(OBJ)/ops_mpi_decl_cuda.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_cuda.cu -o $(OBJ)/ops_mpi_rt_support_cuda.o -I$(MPI_INSTALL_PATH)/include
	ar -r $(LIB)/libops_mpi_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda.o $(OBJ)/ops_mpi_rt_support.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition.o $(OBJ)/ops_cuda_rt_support.o

opencl: mklib core $(SRC)/opencl/ops_opencl_decl.c $(OBJ)/ops_lib_core.o
	$(CXX) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_decl.c -o $(OBJ)/ops_opencl_decl.o
	$(CXX) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support.c -o $(OBJ)/ops_opencl_rt_support.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_util.o $(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_opencl_decl.o $(OBJ)/ops_dummy_singlenode.o


clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(INC)/*~
	-rm -if *~
