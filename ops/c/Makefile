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

OPS_INSTALL_PATH = $(subst /ops/c,/ops,$(shell pwd))
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda

ifdef HDF5_INSTALL_PATH
  HDF5_INC 	  	:= -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB		:= -L$(HDF5_INSTALL_PATH)/lib
  CHKCC			:= $(MPICC)
  CHKCXX			:= $(MPICXX)
ifdef CHECKPOINTING
  HDF5_INC 	  	:= $(HDF5_INC) -DCHECKPOINTING
endif
else
  CHKCC			:= $(CC)
  CHKCXX			:= $(CXX)
endif

# This refers to the location of the submodule in the "master" pdetk repo
#TRID_INSTALL_PATH ?= ../../../tridsolver/scalar

TRID_INC := -I$(TRID_INSTALL_PATH)/include
TRID_LIB := -L$(TRID_INSTALL_PATH)/lib

MPI_INC = $(foreach dir,$(shell $(MPICC) --showme:incdirs),-I$(dir))

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

core: mklib $(INC)/ops_lib_core.h $(SRC)/core/ops_lib_core.cpp $(SRC)/externlib/ops_util.cpp
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lib_core.cpp -o $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lazy.cpp -o $(OBJ)/ops_lazy.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/externlib/ops_util.cpp -o $(OBJ)/ops_util.o
	$(CHKCXX) $(CXXFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/ops_checkpointing.cpp -o $(OBJ)/ops_checkpointing.o
	$(CHKCXX) $(CXXFLAGS) -I$(INC) $(HDF5_INC) -c $(SRC)/externlib/ops_checkpointing_strategy.cpp -o $(OBJ)/ops_checkpointing_strategy.o

hdf5_seq: mklib core $(OBJ)/ops_lib_core.o
	$(MPICXX) $(CXXFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/externlib/ops_hdf5.cpp -o $(OBJ)/ops_hdf5.o
	ar -r $(LIB)/libops_hdf5_seq.a $(OBJ)/ops_hdf5.o

trid: mklib core $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag.cpp -o $(OBJ)/ops_tridiag.o
ifdef CUDA_INSTALL_PATH
	$(CXX) $(CXXFLAGS) -I$(INC) -I$(CUDA_INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag_cuda.cpp -o $(OBJ)/ops_tridiag_cuda.o
endif
	$(MPICXX) $(CXXFLAGS) -I$(INC) $(CUDA_INC) $(TRID_INC) $(TRID_LIB) -c $(SRC)/tridiag/ops_tridiag_mpi.cpp -o $(OBJ)/ops_tridiag_mpi.o
	ar -r $(LIB)/libops_trid_seq.a $(OBJ)/ops_tridiag.o
ifdef CUDA_INSTALL_PATH
	ar -r $(LIB)/libops_trid_cuda.a $(OBJ)/ops_tridiag_cuda.o
endif
	ar -r $(LIB)/libops_trid_mpi.a $(OBJ)/ops_tridiag_mpi.o

seq: mklib core $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/sequential/ops_seq.cpp -o $(OBJ)/ops_seq.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.cpp -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_seq.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_seq.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o 

cuda: mklib core $(SRC)/cuda/ops_cuda_decl.cpp $(OBJ)/ops_lib_core.o
	$(CXX) -DSET_CUDA_CACHE_CONFIG $(CXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_decl.cpp -o $(OBJ)/ops_cuda_decl.o
	$(CXX) -DSET_CUDA_CACHE_CONFIG $(CXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INC) -I$(INC) \
        -c $(SRC)/cuda/ops_cuda_rt_support.cpp -o $(OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/cuda/ops_cuda_rt_support.cu -o $(OBJ)/ops_cuda_rt_support_cuda.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.cpp -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_cuda_rt_support_cuda.o \
	$(OBJ)/ops_cuda_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o

mpi: mklib core $(OBJ)/ops_lib_core.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_decl.cpp -o $(OBJ)/ops_mpi_decl.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.cpp -o $(OBJ)/ops_mpi_partition.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.cpp -o $(OBJ)/ops_mpi_core.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.cpp -o $(OBJ)/ops_mpi_rt_support.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_host.cpp -o $(OBJ)/ops_mpi_rt_support_host.o
	ar -r $(LIB)/libops_mpi.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl.o $(OBJ)/ops_mpi_rt_support.o $(OBJ)/ops_checkpointing_strategy.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_host.o $(OBJ)/ops_mpi_partition.o

hdf5_mpi: mklib core $(OBJ)/ops_lib_core.o
	$(MPICXX) $(CXXFLAGS) -I$(INC)  $(HDF5_INC) $(HDF5_LIB) -c $(SRC)/mpi/ops_mpi_hdf5.cpp -o $(OBJ)/ops_mpi_hdf5.o
	ar -r $(LIB)/libops_hdf5_mpi.a $(OBJ)/ops_mpi_hdf5.o

mpi_cuda: mklib cuda $(OBJ)/ops_cuda_rt_support.o
	$(MPICXX) $(CXXFLAGS) $(CUDA_ALIGNE_FLAG) \
	-I$(INC) -I$(CUDA_INC) -c $(SRC)/mpi/ops_mpi_decl_cuda.cpp -o $(OBJ)/ops_mpi_decl_cuda.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.cpp -o $(OBJ)/ops_mpi_partition.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.cpp -o $(OBJ)/ops_mpi_core.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.cpp -o $(OBJ)/ops_mpi_rt_support.o
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_cuda.cu -o \
	$(OBJ)/ops_mpi_rt_support_cuda.o 
	ar -r $(LIB)/libops_mpi_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda.o $(OBJ)/ops_mpi_rt_support.o \
	$(OBJ)/ops_mpi_core.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition.o \
	$(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_checkpointing_strategy.o

opencl: mklib core $(SRC)/opencl/ops_opencl_decl.cpp $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_decl.cpp -o $(OBJ)/ops_opencl_decl.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.cpp -o $(OBJ)/ops_dummy_singlenode.o
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support.cpp -o $(OBJ)/ops_opencl_rt_support.o
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -I$(INC) \
        -c $(SRC)/opencl/ops_opencl_rt_support_kernels.cpp -o $(OBJ)/ops_opencl_rt_support_kernels.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_dummy_singlenode.cpp -o $(OBJ)/ops_dummy_singlenode.o
	ar -r $(LIB)/libops_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_opencl_rt_support_kernels.o \
	$(OBJ)/ops_opencl_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o

mpi_opencl: mklib opencl $(OBJ)/ops_opencl_rt_support.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) $(OPENCL_INC) -c $(SRC)/mpi/ops_mpi_decl_opencl.cpp -o $(OBJ)/ops_mpi_decl_opencl.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_partition.cpp -o $(OBJ)/ops_mpi_partition.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_core.cpp -o $(OBJ)/ops_mpi_core.o
	$(MPICXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support.cpp -o $(OBJ)/ops_mpi_rt_support.o
	$(MPICPP) $(CXXFLAGS) $(OPENCL_INC) -I$(INC) -c $(SRC)/mpi/ops_mpi_rt_support_opencl.cpp -o \
	$(OBJ)/ops_mpi_rt_support_opencl.o -I$(MPI_INSTALL_PATH)/include
	ar -r $(LIB)/libops_mpi_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o $(OBJ)/ops_lazy.o \
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
