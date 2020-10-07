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
LIB = lib/$(OPS_COMPILER)
OBJ = obj/$(OPS_COMPILER)

OPENCL_INC = -I$(OPENCL_INSTALL_PATH)/include

#
# compiler settings
#

OPS_INSTALL_PATH ?= $(subst /ops/c,/ops,$(shell pwd))
include ../../makefiles/Makefile.common
include ../../makefiles/Makefile.mpi
include ../../makefiles/Makefile.cuda
include ../../makefiles/Makefile.hip

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
TRID_INSTALL_PATH = $(TDMA_INSTALL_PATH)
#../../../tridsolver/scalar

TRID_INC := -I$(TRID_INSTALL_PATH)/include
TRID_LIB := -L$(TRID_INSTALL_PATH)/lib

MPI_INC = $(foreach dir,$(shell $(MPICC) --showme:incdirs),-I$(dir))

.PHONY: mklib

TARGETS = seq mpi 

ifdef HDF5_INSTALL_PATH
	TARGETS += hdf5_seq hdf5_mpi
	INC += $(HDF5_INC)
endif

ifdef CUDA_INSTALL_PATH
	TARGETS += cuda mpi_cuda
	INC += $(CUDA_INC)
endif

ifdef HIP_INSTALL_PATH
	TARGETS += hip mpi_hip
	INC += $(HIP_INC)
endif

ifdef OPENCL_INSTALL_PATH
	TARGETS += opencl mpi_opencl
	INC += $(OPENCL_INC)
endif

ifdef TRID_INSTALL_PATH
	TARGETS += trid_seq trid_mpi
ifdef CUDA_INSTALL_PATH
	TARGETS += trid_cuda trid_mpi_cuda
endif
endif

ifdef SYCL_INSTALL_PATH
	TARGETS += sycl
endif

all: $(TARGETS)

mklib:
	@mkdir -p $(LIB) $(OBJ)

.SUFFIXES: .cpp .cu .o

$(OBJ)/%.o : src/core/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/ops_hdf5.o : src/externlib/ops_hdf5.cpp
	$(CHKCXX) $(CXXFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%.o : src/externlib/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%.o : src/sequential/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%.o : src/cuda/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%.o : src/opencl/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%_mpi.o : src/externlib/%.cpp
	$(MPICXX) $(CXXFLAGS) -I$(INC)  $(HDF5_INC) -c $< -o $@ 

$(OBJ)/%_hip_mpi.o : src/mpi/%_hip.cpp
	$(HIPCC) $(HIPFLAGS) -I$(INC) $(MPI_INC) -c $< -o $@ 

$(OBJ)/%_mpi.o : src/mpi/%.cpp
	$(MPICXX) $(CXXFLAGS) -I$(INC)  $(HDF5_INC) -c $< -o $@ 

$(OBJ)/%_cuda.o : src/cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%_mpi.o : src/tridiag/%_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -I$(INC) $(TRID_INC) -c $< -o $@

$(OBJ)/%_mpi_cuda.o : src/tridiag/%_mpi_cuda.cpp
	$(MPICXX) $(CXXFLAGS) -I$(INC) $(TRID_INC) -c $< -o $@

$(OBJ)/%.o : src/tridiag/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC) $(TRID_INC) -c $< -o $@

$(OBJ)/%_cuda.o : src/tridiag/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC) $(TRID_INC) -c $< -o $@

$(OBJ)/%.o : src/cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%_hip.o : src/hip/%.cpp
	$(HIPCC) $(HIPFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%.o : src/hip/%.cpp
	$(HIPCC) $(HIPFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%.o : src/mpi/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $< -o $@

$(OBJ)/%.o : src/sycl/%.cpp
	$(SYCLCXX) $(CXXFLAGS) $(SYCL_FLAGS) -I$(INC) -c $< -o $@

hdf5_seq: mklib $(OBJ)/ops_hdf5.o 
	ar -r $(LIB)/libops_hdf5_seq.a $(OBJ)/ops_hdf5.o

trid_seq: mklib $(OBJ)/ops_tridiag.o
	ar -r $(LIB)/libops_trid_seq.a $(OBJ)/ops_tridiag.o
trid_cuda: mklib $(OBJ)/ops_tridiag_cuda.o
	ar -r $(LIB)/libops_trid_cuda.a $(OBJ)/ops_tridiag_cuda.o
trid_mpi: mklib $(OBJ)/ops_tridiag_mpi.o
	ar -r $(LIB)/libops_trid_mpi.a $(OBJ)/ops_tridiag_mpi.o
trid_mpi_cuda: mklib $(OBJ)/ops_tridiag_mpi_cuda.o
	ar -r $(LIB)/libops_trid_mpi_cuda.a $(OBJ)/ops_tridiag_mpi_cuda.o

seq: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
  $(OBJ)/ops_util.o $(OBJ)/ops_seq.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o 
	ar -r $(LIB)/libops_seq.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_seq.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o 

cuda: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
  $(OBJ)/ops_util.o $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_cuda_rt_support_cuda.o \
  $(OBJ)/ops_cuda_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o
	ar -r $(LIB)/libops_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_cuda_rt_support_cuda.o \
	$(OBJ)/ops_cuda_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o

hip: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
  $(OBJ)/ops_util.o $(OBJ)/ops_hip_rt_support.o $(OBJ)/ops_hip_rt_support_kernels.o \
  $(OBJ)/ops_hip_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o
	ar -r $(LIB)/libops_hip.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_hip_rt_support.o $(OBJ)/ops_hip_rt_support_kernels.o \
	$(OBJ)/ops_hip_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o

mpi: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
  $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o $(OBJ)/ops_checkpointing_strategy_mpi.o \
  $(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_host_mpi.o $(OBJ)/ops_mpi_partition_mpi.o 
	ar -r $(LIB)/libops_mpi.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o $(OBJ)/ops_checkpointing_strategy_mpi.o \
	$(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_host_mpi.o $(OBJ)/ops_mpi_partition_mpi.o

hdf5_mpi: mklib $(OBJ)/ops_mpi_hdf5_mpi.o 
	ar -r $(LIB)/libops_hdf5_mpi.a $(OBJ)/ops_mpi_hdf5_mpi.o

mpi_cuda: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o \
  $(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o \
  $(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition_mpi.o \
  $(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_checkpointing_strategy.o
	ar -r $(LIB)/libops_mpi_cuda.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_cuda_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o \
	$(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_cuda.o $(OBJ)/ops_mpi_partition_mpi.o \
	$(OBJ)/ops_cuda_rt_support.o $(OBJ)/ops_checkpointing_strategy.o

mpi_hip: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o \
  $(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_hip_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o \
  $(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_hip_mpi.o $(OBJ)/ops_mpi_partition_mpi.o \
  $(OBJ)/ops_hip_rt_support.o $(OBJ)/ops_checkpointing_strategy.o
	ar -r $(LIB)/libops_mpi_hip.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_util.o $(OBJ)/ops_mpi_decl_hip_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o \
	$(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_hip_mpi.o $(OBJ)/ops_mpi_partition_mpi.o \
	$(OBJ)/ops_hip_rt_support.o $(OBJ)/ops_checkpointing_strategy.o

opencl: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
  $(OBJ)/ops_util.o $(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_opencl_rt_support_kernels.o \
  $(OBJ)/ops_opencl_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o
	ar -r $(LIB)/libops_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
	$(OBJ)/ops_util.o $(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_opencl_rt_support_kernels.o \
	$(OBJ)/ops_opencl_decl.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o

mpi_opencl: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_util.o $(OBJ)/ops_lazy.o \
  $(OBJ)/ops_checkpointing.o $(OBJ)/ops_mpi_decl_opencl_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o \
        $(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_opencl_mpi.o $(OBJ)/ops_mpi_partition_mpi.o \
  $(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_checkpointing_strategy.o 
	ar -r $(LIB)/libops_mpi_opencl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_util.o $(OBJ)/ops_lazy.o \
	$(OBJ)/ops_checkpointing.o $(OBJ)/ops_mpi_decl_opencl_mpi.o $(OBJ)/ops_mpi_rt_support_mpi.o \
        $(OBJ)/ops_mpi_core_mpi.o $(OBJ)/ops_mpi_rt_support_opencl_mpi.o $(OBJ)/ops_mpi_partition_mpi.o \
	$(OBJ)/ops_opencl_rt_support.o $(OBJ)/ops_checkpointing_strategy.o

sycl: mklib $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
  $(OBJ)/ops_util.o $(OBJ)/ops_dummy_singlenode.o $(OBJ)/ops_checkpointing_strategy.o \
  $(OBJ)/ops_sycl_decl.o $(OBJ)/ops_sycl_rt_support.o $(OBJ)/ops_sycl_rt_support_kernels.o 
	ar -r $(LIB)/libops_sycl.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_instance.o $(OBJ)/ops_lazy.o $(OBJ)/ops_checkpointing.o \
		$(OBJ)/ops_util.o $(OBJ)/ops_sycl_rt_support.o $(OBJ)/ops_sycl_rt_support_kernels.o $(OBJ)/ops_sycl_decl.o $(OBJ)/ops_dummy_singlenode.o \
		$(OBJ)/ops_checkpointing_strategy.o


clean:
	-rm -if $(OBJ)/*.optrpt #remove optimization output files from intel compiler
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if include/*~
	-rm -if *~
	-rm -if *.lst
