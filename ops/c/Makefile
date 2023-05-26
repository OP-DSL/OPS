#
# The following environment variables should be predefined:
#
# OPS_COMPILER (gnu,intel,etc)
#
#
# set paths for various files
#

OPENCL_INC = -I$(OPENCL_INSTALL_PATH)/include

#
# compiler settings
#

OPS_INSTALL_PATH ?= $(subst /ops/c,/ops,$(shell pwd))
include ../../makefiles/Makefile.common
include ../../makefiles/Makefile.mpi
include ../../makefiles/Makefile.cuda
include ../../makefiles/Makefile.hip
include ../../makefiles/Makefile.tridsolver

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

MPI_INC = $(foreach dir,$(shell $(MPICC) --showme:incdirs),-I$(dir))

.PHONY: mklib

TARGETS = seq mpi 

ifdef HDF5_INSTALL_PATH
	TARGETS += hdf5_seq hdf5_mpi
	C_OPS_INC += $(HDF5_INC)
endif

ifdef CUDA_INSTALL_PATH
	TARGETS += cuda mpi_cuda
	C_OPS_INC += $(CUDA_INC)
endif

ifdef HIP_INSTALL_PATH
	TARGETS += hip mpi_hip
	C_OPS_INC += $(HIP_INC)
endif

ifdef OPENCL_INSTALL_PATH
	TARGETS += opencl mpi_opencl
	C_OPS_INC += $(OPENCL_INC)
endif

ifdef TRID_INSTALL_PATH
	TARGETS += trid_seq trid_mpi
ifdef CUDA_INSTALL_PATH
	TARGETS += trid_cuda trid_mpi_cuda
endif
endif

ifdef SYCL_INSTALL_PATH
	TARGETS += sycl mpi_sycl
endif

all: $(TARGETS)

mklib:
	@mkdir -p $(C_OPS_LIB) $(C_OPS_OBJ)

.SUFFIXES: .cpp .cu .o

$(C_OPS_OBJ)/%.o : src/core/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/sequential/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/ops_hdf5.o : src/externlib/ops_hdf5.cpp
	$(CHKCXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/ops_hdf5_common.o : src/externlib/ops_hdf5_common.cpp
	$(CHKCXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/externlib/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%_mpi.o : src/externlib/%.cpp
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC)  $(HDF5_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/ompoffload/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPOFFLOADFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%_mpi.o : src/mpi/%.cpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC)  $(HDF5_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/mpi/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%_sycl_mpi.o : src/mpi/%_sycl.cpp
	$(SYCLMPICXX) $(CXXFLAGS) $(SYCL_FLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%_hip_mpi.o : src/mpi/%_hip.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -I$(C_OPS_INC) $(MPI_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/cuda/%.cpp
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%_cuda.o : src/cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/opencl/%.cpp
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%_hip.o : src/hip/%.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/hip/%.cpp
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/sycl/%.cpp
	$(SYCLCXX) $(CXXFLAGS) $(SYCL_FLAGS) -I$(C_OPS_INC) -c $< -o $@

$(C_OPS_OBJ)/%_mpi.o : src/tridiag/%_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) $(TRID_INC) -c $< -o $@

$(C_OPS_OBJ)/%_mpi_cuda.o : src/tridiag/%_mpi_cuda.cpp
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) $(TRID_INC) -c $< -o $@

$(C_OPS_OBJ)/%.o : src/tridiag/%.cpp
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) $(TRID_INC) -c $< -o $@

$(C_OPS_OBJ)/%_cuda.o : src/tridiag/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) $(TRID_INC) -c $< -o $@

hdf5_seq: mklib $(C_OPS_OBJ)/ops_hdf5.o $(C_OPS_OBJ)/ops_hdf5_common.o
	ar -r $(C_OPS_LIB)/libops_hdf5_seq.a $(C_OPS_OBJ)/ops_hdf5.o $(C_OPS_OBJ)/ops_hdf5_common.o

trid_seq: mklib $(C_OPS_OBJ)/ops_tridiag.o
	ar -r $(C_OPS_LIB)/libops_trid_seq.a $(C_OPS_OBJ)/ops_tridiag.o
trid_cuda: mklib $(C_OPS_OBJ)/ops_tridiag_cuda.o
	ar -r $(C_OPS_LIB)/libops_trid_cuda.a $(C_OPS_OBJ)/ops_tridiag_cuda.o
trid_mpi: mklib $(C_OPS_OBJ)/ops_tridiag_mpi.o
	ar -r $(C_OPS_LIB)/libops_trid_mpi.a $(C_OPS_OBJ)/ops_tridiag_mpi.o
trid_mpi_cuda: mklib $(C_OPS_OBJ)/ops_tridiag_mpi_cuda.o
	ar -r $(C_OPS_LIB)/libops_trid_mpi_cuda.a $(C_OPS_OBJ)/ops_tridiag_mpi_cuda.o

seq: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_host_common.o  $(C_OPS_OBJ)/ops_host_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o
	ar -r $(C_OPS_LIB)/libops_seq.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
	$(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_host_common.o  $(C_OPS_OBJ)/ops_host_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o

cuda: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_cuda_common_cuda.o $(C_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
  $(C_OPS_OBJ)/ops_cuda_singlenode_cuda.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
  $(C_OPS_OBJ)/ops_device_singlenode_common.o
	ar -r $(C_OPS_LIB)/libops_cuda.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
	$(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_cuda_common_cuda.o $(C_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
	$(C_OPS_OBJ)/ops_cuda_singlenode_cuda.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(C_OPS_OBJ)/ops_device_singlenode_common.o

ompoffload: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_ompoffload_common.o $(C_OPS_OBJ)/ops_ompoffload_rt_support_kernels.o \
  $(C_OPS_OBJ)/ops_ompoffload_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
  $(C_OPS_OBJ)/ops_device_singlenode_common.o
	ar -r $(C_OPS_LIB)/libops_ompoffload.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_ompoffload_common.o $(C_OPS_OBJ)/ops_ompoffload_rt_support_kernels.o \
  $(C_OPS_OBJ)/ops_ompoffload_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
  $(C_OPS_OBJ)/ops_device_singlenode_common.o

hip: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_hip_common.o $(C_OPS_OBJ)/ops_hip_rt_support_kernels.o \
  $(C_OPS_OBJ)/ops_hip_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
  $(C_OPS_OBJ)/ops_device_singlenode_common.o
	ar -r $(C_OPS_LIB)/libops_hip.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
	$(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_hip_common.o $(C_OPS_OBJ)/ops_hip_rt_support_kernels.o \
	$(C_OPS_OBJ)/ops_hip_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(C_OPS_OBJ)/ops_device_singlenode_common.o

sycl: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_sycl_common.o $(C_OPS_OBJ)/ops_sycl_rt_support_kernels.o \
  $(C_OPS_OBJ)/ops_sycl_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
  $(C_OPS_OBJ)/ops_device_singlenode_common.o
	ar -r $(C_OPS_LIB)/libops_sycl.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_sycl_common.o $(C_OPS_OBJ)/ops_sycl_rt_support_kernels.o \
  $(C_OPS_OBJ)/ops_sycl_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
  $(C_OPS_OBJ)/ops_device_singlenode_common.o

mpi: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o $(C_OPS_OBJ)/ops_checkpointing_strategy_mpi.o \
  $(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_host_mpi.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o $(C_OPS_OBJ)/ops_host_common.o
	ar -r $(C_OPS_LIB)/libops_mpi.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
	$(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o $(C_OPS_OBJ)/ops_checkpointing_strategy_mpi.o \
	$(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_host_mpi.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o $(C_OPS_OBJ)/ops_host_common.o

hdf5_mpi: mklib $(C_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(C_OPS_OBJ)/ops_hdf5_common.o
	ar -r $(C_OPS_LIB)/libops_hdf5_mpi.a $(C_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(C_OPS_OBJ)/ops_hdf5_common.o

mpi_cuda: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o \
  $(C_OPS_OBJ)/ops_checkpointing.o $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
  $(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o \
  $(C_OPS_OBJ)/ops_cuda_rt_support.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o $(C_OPS_OBJ)/ops_cuda_common.o
	ar -r $(C_OPS_LIB)/libops_mpi_cuda.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o \
	$(C_OPS_OBJ)/ops_checkpointing.o $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(C_OPS_OBJ)/ops_cuda_rt_support.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o $(C_OPS_OBJ)/ops_cuda_common.o

mpi_hip: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o \
  $(C_OPS_OBJ)/ops_checkpointing.o $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
  $(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_hip_mpi.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o \
  $(C_OPS_OBJ)/ops_checkpointing_strategy.o $(C_OPS_OBJ)/ops_hip_common.o
	ar -r $(C_OPS_LIB)/libops_mpi_hip.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o \
	$(C_OPS_OBJ)/ops_checkpointing.o $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_hip_mpi.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(C_OPS_OBJ)/ops_checkpointing_strategy.o $(C_OPS_OBJ)/ops_hip_common.o

mpi_sycl: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o \
  $(C_OPS_OBJ)/ops_checkpointing.o $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
  $(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_sycl_mpi.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o \
  $(C_OPS_OBJ)/ops_checkpointing_strategy.o $(C_OPS_OBJ)/ops_sycl_common.o
	ar -r $(C_OPS_LIB)/libops_mpi_sycl.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o \
  $(C_OPS_OBJ)/ops_checkpointing.o $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
  $(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_sycl_mpi.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o \
  $(C_OPS_OBJ)/ops_checkpointing_strategy.o $(C_OPS_OBJ)/ops_sycl_common.o

opencl: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
  $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_opencl_common.o $(C_OPS_OBJ)/ops_opencl_rt_support_kernels.o \
  $(C_OPS_OBJ)/ops_opencl_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
  $(C_OPS_OBJ)/ops_device_singlenode_common.o
	ar -r $(C_OPS_LIB)/libops_opencl.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_lazy.o $(C_OPS_OBJ)/ops_checkpointing.o \
	$(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_opencl_common.o $(C_OPS_OBJ)/ops_opencl_rt_support_kernels.o \
	$(C_OPS_OBJ)/ops_opencl_singlenode.o $(C_OPS_OBJ)/ops_dummy_singlenode.o $(C_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(C_OPS_OBJ)/ops_device_singlenode_common.o

mpi_opencl: mklib $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_lazy.o \
  $(C_OPS_OBJ)/ops_checkpointing.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
        $(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_opencl_mpi.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o \
  $(C_OPS_OBJ)/ops_checkpointing_strategy.o $(C_OPS_OBJ)/ops_opencl_common.o
	ar -r $(C_OPS_LIB)/libops_mpi_opencl.a $(C_OPS_OBJ)/ops_lib_core.o $(C_OPS_OBJ)/ops_instance.o $(C_OPS_OBJ)/ops_util.o $(C_OPS_OBJ)/ops_lazy.o \
	$(C_OPS_OBJ)/ops_checkpointing.o $(C_OPS_OBJ)/ops_mpi_decl_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
        $(C_OPS_OBJ)/ops_mpi_core_mpi.o $(C_OPS_OBJ)/ops_mpi_rt_support_opencl_mpi.o $(C_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(C_OPS_OBJ)/ops_checkpointing_strategy.o $(C_OPS_OBJ)/ops_opencl_common.o


clean:
	-rm -if $(C_OPS_OBJ)/*.optrpt #remove optimization output files from intel compiler
	-rm -if $(C_OPS_OBJ)/*.o
	-rm -if $(C_OPS_OBJ)/*~
	-rm -if $(C_OPS_LIB)/*.a
	-rm -if $(C_OPS_SRC)/*~
	-rm -if include/*~
	-rm -if *~
	-rm -if *.lst
