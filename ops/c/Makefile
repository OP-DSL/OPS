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

OPENCL_INC = -I$(OPENCL_INSTALL_PATH)/include

#
# compiler settings
#

include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda

ifdef HDF5_INSTALL_PATH
  HDF5_INC 	  	:= -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB		:= -L$(HDF5_INSTALL_PATH)/lib
  CHKCC			:= $(MPICPP)
ifdef CHECKPOINTING
  HDF5_INC 	  	:= $(HDF5_INC) -DCHECKPOINTING
endif
else
  CHKCC			:= $(CC)
endif

#TRID_INSTALL_PATH := /home/mgiles/mudalige/TriDiagonal/laszloendre-tridsolver-93670b204e2c/scalar/build
TRID_INC := -I$(TRID_INSTALL_PATH)/include
TRID_LIB := -L$(TRID_INSTALL_PATH)/lib

.PHONY: clean mklib

TARGETS = clean core seq mpi 

ifdef HDF5_INSTALL_PATH
	TARGETS += hdf5_seq hdf5_mpi
endif

ifdef CUDA_INSTALL_PATH
	TARGETS += cuda mpi_cuda
endif

ifdef OPENCL_INSTALL_PATH
	TARGETS += opencl mpi_opencl
endif

ifdef TRID_INSTALL_PATH
	TARGETS += trid
endif

all: $(TARGETS)

mklib:
	@mkdir -p $(LIB) $(OBJ)

core: mklib $(INC)/ops_lib_core.h $(SRC)/core/ops_lib_core.c $(SRC)/externlib/ops_util.c
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/core/ops_lib_core.c -o $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lazy.cpp -o $(OBJ)/ops_lazy.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/externlib/ops_util.c -o $(OBJ)/ops_util.o
	$(CHKCC) $(CXXFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/ops_checkpointing.c -o $(OBJ)/ops_checkpointing.o
	$(CHKCC) $(CXXFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/ops_checkpointing_strategy.c -o $(OBJ)/ops_checkpointing_strategy.o

hdf5_seq: mklib core $(OBJ)/ops_lib_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/externlib/ops_hdf5.c -o $(OBJ)/ops_hdf5.o
	ar -r $(LIB)/libops_hdf5_seq.a $(OBJ)/ops_hdf5.o

trid: mklib core $(OBJ)/ops_lib_core.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag.c -o $(OBJ)/ops_tridiag.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(CUDA_INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag_cuda.c -o $(OBJ)/ops_tridiag_cuda.o
	$(MPICXX) $(MPIFLAGS) -I$(INC) $(CUDA_INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag_mpi.c -o $(OBJ)/ops_tridiag_mpi.o
#	ar -r $(LIB)/libops_trid.a $(OBJ)/ops_tridiag.o

seq: mklib core trid $(OBJ)/ops_lib_core.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/sequential/ops_seq.c -o $(OBJ)/ops_seq.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_seq.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_seq.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o $(OBJ)/ops_tridiag.o

cuda: mklib core trid $(SRC)/cuda/ops_cuda_decl.c $(OBJ)/ops_lib_core.o
	$(CC) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_decl.c -o $(OBJ)/ops_cuda_decl.o
	$(CC) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_rt_support.c -o $(OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/cuda/ops_cuda_rt_support.cu -o $(OBJ)/ops_cuda_rt_support_cuda.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_cuda_rt_support_cuda.o \
	$(OBJ)/ops_cuda_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o \
	$(OBJ)/ops_tridiag_cuda.o

mpi: mklib core $(OBJ)/ops_lib_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_decl.c -o $(OBJ)/ops_mpi_decl.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_host.c -o $(OBJ)/ops_mpi_rt_support_host.o
	ar -r $(LIB)/libops_mpi.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl.o $(OBJ)/ops_mpi_rt_support.o $(OBJ)/ops_checkpointing_strategy.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_host.o $(OBJ)/ops_mpi_partition.o \
	$(OBJ)/ops_tridiag_mpi.o

hdf5_mpi: mklib core $(OBJ)/ops_lib_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/mpi/ops_mpi_hdf5.c -o $(OBJ)/ops_mpi_hdf5.o
	ar -r $(LIB)/libops_hdf5_mpi.a $(OBJ)/ops_mpi_hdf5.o

mpi_cuda: mklib cuda $(OBJ)/ops_cuda_rt_support.o
	$(MPICC) $(MPIFLAGS) $(CUDA_ALIGNE_FLAG) \
	-I$(INC) -I$(CUDA_INC) -c $(SRC)/mpi/ops_mpi_decl_cuda.c -o $(OBJ)/ops_mpi_decl_cuda.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_cuda.cu -o \
	$(OBJ)/ops_mpi_rt_support_cuda.o -I$(MPI_INSTALL_PATH)/include
	ar -r $(LIB)/libops_mpi_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda.o $(OBJ)/ops_mpi_rt_support.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition.o \
	$(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_checkpointing_strategy.o

opencl: mklib core $(SRC)/opencl/ops_opencl_decl.c $(OBJ)/ops_lib_core.o
	$(CC) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_decl.c -o $(OBJ)/ops_opencl_decl.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	$(CC) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support.c -o $(OBJ)/ops_opencl_rt_support.o
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support.cpp -o $(OBJ)/ops_opencl_rt_support_cpp.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_opencl_rt_support_cpp.o \
	$(OBJ)/ops_opencl_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o

mpi_opencl: mklib opencl $(OBJ)/ops_opencl_rt_support.o
	$(MPICC) $(MPIFLAGS) -I$(INC) $(OPENCL_INC) -c $(SRC)/mpi/ops_mpi_decl_opencl.c -o $(OBJ)/ops_mpi_decl_opencl.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c -o $(OBJ)/ops_mpi_partition.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c -o $(OBJ)/ops_mpi_core.o
	$(MPICC) $(MPIFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c -o $(OBJ)/ops_mpi_rt_support.o
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_opencl.cpp -o \
	$(OBJ)/ops_mpi_rt_support_opencl.o -I$(MPI_INSTALL_PATH)/include
	ar -r $(LIB)/libops_mpi_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_mpi_decl_opencl.o $(OBJ)/ops_mpi_rt_support.o \
        $(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_opencl.o $(OBJ)/ops_mpi_partition.o \
	$(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_checkpointing_strategy.o


clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(INC)/*~
	-rm -if *~
	-rm -if *.lst
