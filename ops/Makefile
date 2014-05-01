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

#
# compiler settings
#

ifeq ($(OPS_COMPILER),gnu)
  CC 	  	:= gcc
  CCFLAGS 	:= -std=c99 -fPIC -DUNIX -Wall -ffloat-store#-g
  CXX 	  	:= g++
  CXXFLAGS	:= -std=c99 -fPIC -DUNIX -Wall -ffloat-store#-g -Wextra
  MPICXX  	:= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPIFLAGS	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
else
ifeq ($(OPS_COMPILER),intel)
  CC 		:= icc
ifdef DEBUG
  CCFLAGS 	:= -O0 -g  #-vec-report -xSSE4.2
else
  CCFLAGS 	:= -O3 -vec-report -xAVX 
endif
  CXX 		:= icpc
  CXXFLAGS 	:= $(CCFLAGS)
  MPICXX 	:= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPIFLAGS 	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
	NVCC  := nvcc
ifdef DEBUG
	NVCCFLAGS := -O3 -arch=sm_35
else
	NVCCFLAGS := -O0 -g -G
endif
else
ifeq ($(OPS_COMPILER),pgi)
  CC 		:= pgcc
  CUDA_ALIGNE_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CCFLAGS 	:= -O3 -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  CXX 		:= pgCC
  CXXFLAGS 	:= $(CCFLAGS)
  MPICXX 	:= $(MPI_INSTALL_PATH)/bin/mpiCC
	NVCC  := nvcc
  MPIFLAGS 	:= $(CXXFLAGS)
ifdef DEBUG
	NVCCFLAGS := -O3 -arch=sm_35
else
	NVCCFLAGS := -O0 -g -G
endif
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif

.PHONY: clean mklib

all: clean core seq openmp mpi cuda mpi_cuda
mklib:
	@mkdir -p $(LIB) $(OBJ)

core: mklib $(INC)/ops_lib_core.h $(SRC)/core/ops_lib_core.c $(SRC)/externlib/ops_util.c
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lib_core.c -o $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/externlib/ops_util.c -o $(OBJ)/ops_util.o

seq: mklib core $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/sequential/ops_seq.c -o $(OBJ)/ops_seq.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_seq.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_seq.o $(OBJ)/ops_dummy_singlenode.o

openmp: mklib core $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/openmp/ops_openmp_decl.c -o $(OBJ)/ops_openmp_decl.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_openmp.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_openmp_decl.o $(OBJ)/ops_dummy_singlenode.o

cuda: mklib core $(SRC)/cuda/ops_cuda_decl.c $(OBJ)/ops_lib_core.o
	$(CXX) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) $(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_decl.c -o $(OBJ)/ops_cuda_decl.o
	$(CXX) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) $(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_rt_support.c -o $(OBJ)/ops_cuda_rt_support.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_cuda_decl.o $(OBJ)/ops_dummy_singlenode.o

mpi: mklib core $(OBJ)/ops_lib_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_decl.c -o $(OBJ)/ops_mpi_decl.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_host.c -o $(OBJ)/ops_mpi_rt_support_host.o
	ar -r $(LIB)/libops_mpi.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl.o $(OBJ)/ops_mpi_rt_support.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_host.o $(OBJ)/ops_mpi_partition.o

mpi_cuda: mklib cuda $(OBJ)/ops_cuda_rt_support.o
	$(MPICXX) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) -I$(INC) $(CUDA_INC) -c $(SRC)/mpi/ops_mpi_decl_cuda.c -o $(OBJ)/ops_mpi_decl_cuda.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_cuda.cu -o $(OBJ)/ops_mpi_rt_support_cuda.o
	ar -r $(LIB)/libops_mpi_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda.o $(OBJ)/ops_mpi_rt_support.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition.o $(OBJ)/ops_cuda_rt_support.o

clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(INC)/*~
	-rm -if *~
