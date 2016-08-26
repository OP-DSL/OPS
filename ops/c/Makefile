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
ifeq ($(OPS_COMPILER),clang)
  CC 	  	:= clang
ifdef DEBUG
  CCFLAGS 	:= -O0 -std=c99 -fPIC -DUNIX -Wall -g #-ffloat-store
else
  CCFLAGS 	:= -O3 -std=c99 -fPIC -DUNIX -Wall #-g -ffloat-store
endif
  CXX 	  	:= clang++
  CXXFLAGS	:= -fPIC -DUNIX -Wall # -ffloat-store -std=c99
  MPICXX  	:= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPICC   	:= $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
else
ifeq ($(OPS_COMPILER),gnu)
  CC 	  	:= gcc
ifdef DEBUG
  CCFLAGS 	:= -O0 -std=c99 -fPIC -DUNIX -Wall -ffloat-store -g
else
  CCFLAGS 	:= -O3 -std=c99 -fPIC -DUNIX -Wall -ffloat-store#-g
endif
  CXX 	  	:= g++
  CXXFLAGS	:= -std=c99 -fPIC -DUNIX -Wall -ffloat-store
  MPICXX  	:= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPICC   	:= $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
else
ifeq ($(OPS_COMPILER),intel)
  CC 		:= icc -std=c99
ifdef DEBUG
  CCFLAGS 	:= -O0 -g
else
  CCFLAGS 	:= -O3 -xCORE_AVX2 -qopenmp #vec-report -xAVX
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
  CCFLAGS   := -O0 -g  -mmic
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
ifdef DEBUG
  CCFLAGS 	:= -O0 -g
else
#  CCFLAGS 	:= -O3
  CCFLAGS 	:= -O2 -fastsse -gopt -Mipa=fast #-Mlist #-mp=nonuma -Kieee
endif
  CUDA_ALIGNE_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CXX 		:= pgc++
  CXXFLAGS 	:= $(CCFLAGS)
  MPICXX 	:= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPICC 	:= $(MPI_INSTALL_PATH)/bin/mpicc -c99
  MPIFLAGS 	:= $(CXXFLAGS)
else
ifeq ($(OPS_COMPILER),cray) #Archer and Titan
  CC        := cc
  CUDA_ALIGNE_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CCFLAGS   := -O3 -hgnu
  CXX       := CC
  CXXFLAGS  := $(CCFLAGS)
  MPICXX    := CC
  MPICC     := cc
  MPIFLAGS  := $(CXXFLAGS)
else
ifeq ($(OPS_COMPILER),xl) #for IBM BlueGene Q
  CC        := mpixlc
  CUDA_ALIGNE_FLAG :=
  CCFLAGS   := -O5 -qnoeh -qipa=partition=large -g -qfullpath -Q -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036 -qsrcmsg -qfloat=nomaf
  CXX       := mpixlcxx
  CXXFLAGS  := $(CCFLAGS)
  MPICXX    := mpixlcxx
  MPICC     := mpixlc
  MPIFLAGS  := $(CXXFLAGS)
else
ifeq ($(OPS_COMPILER),xlp8) #for IBM Power8
  CC        := mpicc
  CUDA_ALIGNE_FLAG :=
  CCFLAGS   := -qarch=pwr8 -qtune=pwr8 -O5 -qaltivec=le 
  CXX       := mpicxx
  CXXFLAGS  := $(CCFLAGS)
  MPICXX    := mpicxx
  MPICC     := mpicc
  MPIFLAGS  := $(CXXFLAGS)
else

print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif
endif
endif
endif
endif
endif

NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc
ifndef DEBUG
    NVCCFLAGS := -O3 -arch=sm_35 -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
    #NVCCFLAGS := -O3 -arch=sm_21 -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
else
    NVCCFLAGS := -O0 -g -G -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
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

.PHONY: clean mklib

all: clean core hdf5_seq seq mpi cuda mpi_cuda opencl mpi_opencl

mklib:
	@mkdir -p $(LIB) $(OBJ)

core: mklib $(INC)/ops_lib_core.h $(SRC)/core/ops_lib_core.c $(SRC)/externlib/ops_util.c
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lib_core.c -o $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lazy.cpp -o $(OBJ)/ops_lazy.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/externlib/ops_util.c -o $(OBJ)/ops_util.o
	$(CHKCC) $(CXXFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/ops_checkpointing.c -o $(OBJ)/ops_checkpointing.o
	$(CHKCC) $(CXXFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/ops_checkpointing_strategy.c -o $(OBJ)/ops_checkpointing_strategy.o

hdf5_seq: mklib core $(OBJ)/ops_lib_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/externlib/ops_hdf5.c -o $(OBJ)/ops_hdf5.o

seq: mklib core hdf5_seq $(OBJ)/ops_lib_core.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/sequential/ops_seq.c -o $(OBJ)/ops_seq.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_seq.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_seq.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_hdf5.o $(OBJ)/ops_checkpointing_strategy.o

cuda: mklib core hdf5_seq $(SRC)/cuda/ops_cuda_decl.c $(OBJ)/ops_lib_core.o
	$(CC) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) $(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_decl.c -o $(OBJ)/ops_cuda_decl.o
	$(CC) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) $(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_rt_support.c -o $(OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/cuda/ops_cuda_rt_support.cu -o $(OBJ)/ops_cuda_rt_support_cuda.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_cuda_rt_support_cuda.o \
	$(OBJ)/ops_cuda_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_hdf5.o $(OBJ)/ops_checkpointing_strategy.o

mpi: mklib core $(OBJ)/ops_lib_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_decl.c -o $(OBJ)/ops_mpi_decl.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_host.c -o $(OBJ)/ops_mpi_rt_support_host.o
	$(MPICC) $(MPIFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/mpi/ops_mpi_hdf5.c -o $(OBJ)/ops_mpi_hdf5.o
	ar -r $(LIB)/libops_mpi.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl.o $(OBJ)/ops_mpi_rt_support.o $(OBJ)/ops_checkpointing_strategy.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_host.o $(OBJ)/ops_mpi_partition.o $(OBJ)/ops_mpi_hdf5.o

mpi_cuda: mklib cuda $(OBJ)/ops_cuda_rt_support.o
	$(MPICC) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) \
	-I$(INC) $(CUDA_INC) -c $(SRC)/mpi/ops_mpi_decl_cuda.c -o $(OBJ)/ops_mpi_decl_cuda.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_cuda.cu -o \
	$(OBJ)/ops_mpi_rt_support_cuda.o -I$(MPI_INSTALL_PATH)/include
	$(MPICC) $(MPIFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/mpi/ops_mpi_hdf5.c -o $(OBJ)/ops_mpi_hdf5.o
	ar -r $(LIB)/libops_mpi_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda.o $(OBJ)/ops_mpi_rt_support.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition.o \
	$(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_mpi_hdf5.o $(OBJ)/ops_checkpointing_strategy.o

opencl: mklib core hdf5_seq $(SRC)/opencl/ops_opencl_decl.c $(OBJ)/ops_lib_core.o
	$(CXX) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_decl.c -o $(OBJ)/ops_opencl_decl.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	$(CXX) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support.c -o $(OBJ)/ops_opencl_rt_support.o
	$(CXX) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support.cpp -o $(OBJ)/ops_opencl_rt_support_cpp.o
	$(CC) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_opencl_rt_support_cpp.o \
	$(OBJ)/ops_opencl_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_hdf5.o $(OBJ)/ops_checkpointing_strategy.o

mpi_opencl: mklib opencl $(OBJ)/ops_opencl_rt_support.o
	$(MPICC) $(MPIFLAGS) -I$(INC) $(OPENCL_INC) -c $(SRC)/mpi/ops_mpi_decl_opencl.c -o $(OBJ)/ops_mpi_decl_opencl.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(CXX) $(CCFLAGS) $(OPENCL_INC) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_opencl.cpp -o \
	$(OBJ)/ops_mpi_rt_support_opencl.o -I$(MPI_INSTALL_PATH)/include
	$(MPICC) $(MPIFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/mpi/ops_mpi_hdf5.c -o $(OBJ)/ops_mpi_hdf5.o
	ar -r $(LIB)/libops_mpi_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_mpi_decl_opencl.o $(OBJ)/ops_mpi_rt_support.o \
        $(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_opencl.o $(OBJ)/ops_mpi_partition.o \
	$(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_mpi_hdf5.o $(OBJ)/ops_checkpointing_strategy.o


clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(INC)/*~
	-rm -if *~
	-rm -if *.lst
