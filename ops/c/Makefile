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
  CHKCC			:= $(MPICC)
ifdef CHECKPOINTING
  HDF5_INC 	  	:= $(HDF5_INC) -DCHECKPOINTING
endif
else
  CHKCC			:= $(CC)
endif

TRID_INSTALL_PATH ?= /home/mudalige/TriDiagonal/Tridiag-Git/tridsolver/scalar/build
# This refers to the location of the submodule in the "master" pdetk repo
#TRID_INSTALL_PATH ?= ../../../tridsolver/scalar

TRID_INC := -I$(TRID_INSTALL_PATH)/include
TRID_LIB := -L$(TRID_INSTALL_PATH)/lib

MPI_INC = $(foreach dir,$(shell $(MPICC) --showme:incdirs),-I$(dir))

.PHONY: clean mklib

TARGETS = clean core seq cuda

ifdef HDF5_INSTALL_PATH
	TARGETS += hdf5_seq hdf5_mpi
endif

ifdef CUDA_INSTALL_PATH
	TARGETS += cuda mpi_cuda
endif

ifdef OPENCL_INSTALL_PATH
	TARGETS += opencl mpi_opencl
endif

ifeq ($(OPS_COMPILER),intel)
ifdef TRID_INSTALL_PATH
		TRID = 1
		TARGETS += trid
endif
endif

all: $(TARGETS)

mklib:
	@mkdir -p $(LIB) $(OBJ)

core: mklib $(INC)/ops_lib_core.h $(SRC)/core/ops_lib_core.c $(SRC)/externlib/ops_util.c
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/core/ops_lib_core.c $(OUT)$(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lazy.cpp $(OUT)$(OBJ)/ops_lazy.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/externlib/ops_util.c $(OUT)$(OBJ)/ops_util.o
	$(CHKCC) $(CCFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/ops_checkpointing.c $(OUT)$(OBJ)/ops_checkpointing.o
	$(CHKCC) $(CCFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/ops_checkpointing_strategy.c $(OUT)$(OBJ)/ops_checkpointing_strategy.o

hdf5_seq: mklib core
	$(MPICC) $(CCFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/externlib/ops_hdf5.c $(OUT)$(OBJ)/ops_hdf5.o
	$(AR)$(LIB)/libops_hdf5_seq.a $(OBJ)/ops_hdf5.o

trid: mklib core
	$(CXX) $(CXXFLAGS) -I$(INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag.c $(OUT)$(OBJ)/ops_tridiag.o
	$(AR)$(LIB)/libops_trid_seq.a $(OBJ)/ops_tridiag.o
ifdef CUDA_INSTALL_PATH
	$(CXX) $(CXXFLAGS) -I$(INC) -I$(CUDA_INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag_cuda.c $(OUT)$(OBJ)/ops_tridiag_cuda.o
	$(AR)$(LIB)/libops_trid_cuda.a $(OBJ)/ops_tridiag_cuda.o
endif
	$(MPICXX) $(CXXFLAGS) -I$(INC) $(CUDA_INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag_mpi.c $(OUT)$(OBJ)/ops_tridiag_mpi.o
	$(AR)$(LIB)/libops_trid_mpi.a $(OBJ)/ops_tridiag_mpi.o

seq: mklib core
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/sequential/ops_seq.c $(OUT)$(OBJ)/ops_seq.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c $(OUT)$(OBJ)/ops_dummy_singlenode.o
	$(AR)$(LIB)/libops_seq.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_seq.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o 

cuda: mklib core seq $(SRC)/cuda/ops_cuda_decl.c
	$(CC) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_decl.c $(OUT)$(OBJ)/ops_cuda_decl.o
	$(CC) -DSET_CUDA_CACHE_CONFIG $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_rt_support.c $(OUT)$(OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/cuda/ops_cuda_rt_support.cu $(OUT)$(OBJ)/ops_cuda_rt_support_cuda.o
	$(AR)$(LIB)/libops_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_cuda_rt_support_cuda.o \
	$(OBJ)/ops_cuda_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o

mpi: mklib core
	$(MPICC) $(CCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_decl.c $(OUT)$(OBJ)/ops_mpi_decl.o
	$(MPICC) $(CCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.c $(OUT)$(OBJ)/ops_mpi_partition.o
	$(MPICC) $(CCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.c $(OUT)$(OBJ)/ops_mpi_core.o
	$(MPICC) $(CCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.c $(OUT)$(OBJ)/ops_mpi_rt_support.o
	$(MPICC) $(CCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_host.c $(OUT)$(OBJ)/ops_mpi_rt_support_host.o
	$(AR)$(LIB)/libops_mpi.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl.o $(OBJ)/ops_mpi_rt_support.o $(OBJ)/ops_checkpointing_strategy.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_host.o $(OBJ)/ops_mpi_partition.o

hdf5_mpi: mklib core
	$(MPICC) $(CCFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/mpi/ops_mpi_hdf5.c $(OUT)$(OBJ)/ops_mpi_hdf5.o
	$(AR)$(LIB)/libops_hdf5_mpi.a $(OBJ)/ops_mpi_hdf5.o

mpi_cuda: mklib cuda mpi
	$(MPICC) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) \
	-I$(INC) -I$(CUDA_INC) -c $(SRC)/mpi/ops_mpi_decl_cuda.c $(OUT)$(OBJ)/ops_mpi_decl_cuda.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_cuda.cu $(OUT)\
    $(OBJ)/ops_mpi_rt_support_cuda.o -I$(MPI_INSTALL_PATH)/include
	$(AR)$(LIB)/libops_mpi_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda.o $(OBJ)/ops_mpi_rt_support.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition.o \
	$(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_checkpointing_strategy.o

opencl: mklib core seq $(SRC)/opencl/ops_opencl_decl.c
	$(CC) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_decl.c $(OUT)$(OBJ)/ops_opencl_decl.o
	$(CC) $(CCFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.c $(OUT)$(OBJ)/ops_dummy_singlenode.o
	$(CC) $(CCFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support.c $(OUT)$(OBJ)/ops_opencl_rt_support.o
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support.cpp $(OUT)$(OBJ)/ops_opencl_rt_support_cpp.o
	$(AR)$(LIB)/libops_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_opencl_rt_support_cpp.o \
	$(OBJ)/ops_opencl_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o

mpi_opencl: mklib opencl mpi
	$(MPICC) $(CCFLAGS) -I$(INC) $(OPENCL_INC) -c $(SRC)/mpi/ops_mpi_decl_opencl.c $(OUT)$(OBJ)/ops_mpi_decl_opencl.o
	$(MPICPP) $(CXXFLAGS) $(OPENCL_INC) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_opencl.cpp $(OUT)\
	$(OBJ)/ops_mpi_rt_support_opencl.o -I$(MPI_INSTALL_PATH)/include
	$(AR)$(LIB)/libops_mpi_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_mpi_decl_opencl.o $(OBJ)/ops_mpi_rt_support.o \
        $(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_opencl.o $(OBJ)/ops_mpi_partition.o \
	$(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_checkpointing_strategy.o


clean:
	-rm -if $(OBJ)/*.optrpt #remove optimization output files from intel compiler
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(INC)/*~
	-rm -if *~
	-rm -if *.lst
