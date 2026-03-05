#
# The following environment variables are required:
#
# CUDA_INSTALL_PATH - CUDA base directory
# OPS_COMPILER      - Compiler toolchain: gnu (gcc, gfortran) or pgi (pgc++, pgfortran)
# OPS_INSTALL_PATH  - OPS base directory
#
OPS_INSTALL_PATH ?= $(subst /ops/fortran,/ops,$(shell pwd))
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.hip

ifdef HDF5_INSTALL_PATH
  HDF5_INC 	  	:= -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB		:= -L$(HDF5_INSTALL_PATH)/lib
  CHKCC			:= $(MPICC)
  CHKCXX		:= $(MPICXX)
ifdef CHECKPOINTING
  HDF5_INC 	  	:= $(HDF5_INC) -DCHECKPOINTING
endif
else
  CHKCC			:= $(CC)
  CHKCXX      	:= $(CXX)
endif

CXXFLAGS += -DOPS_FTN
NVCCFLAGS += -DOPS_FTN

ifneq ($(strip $(MPI_INSTALL_PATH)),)
ifneq (,$(wildcard $(MPI_INC)))
I_MPI_INC = -I$(MPI_INC)
endif
ifneq (,$(wildcard $(MPI_LIB)))
L_MPI_LIB = -L$(MPI_LIB)
endif
else
I_MPI_INC = $(foreach dir,$(shell $(MPICC) --showme:incdirs),-I$(dir))
endif

.PHONY: mkdir

TARGETS	= f_seq f2c_seq f_mpi f2c_mpi f_ompoffload f_mpi_ompoffload

ifdef HDF5_INSTALL_PATH
	TARGETS += hdf5_seq
	C_OPS_INC += $(HDF5_INC)
endif

ifeq ($(OPS_COMPILER),pgi)
	TARGETS += f_cuda f_mpi_cuda
endif

ifdef CUDA_INSTALL_PATH
	TARGETS += f2c_cuda f2c_mpi_cuda
	C_OPS_INC += -I$(CUDA_INC)
endif

ifdef HIP_INSTALL_PATH
	TARGETS += f2c_hip f2c_mpi_hip
	C_OPS_INC += -I$(HIP_INC)
endif

ifdef SYCL_INSTALL_PATH
	TARGETS += f2c_sycl f2c_mpi_sycl
endif

GPU_VENDOR := $(shell \
    if command -v nvidia-smi >/dev/null 2>&1; then \
        echo NVIDIA; \
    elif command -v rocm-smi >/dev/null 2>&1; then \
        echo AMD; \
    elif command -v intel-smi >/dev/null 2>&1; then \
        echo INTEL; \
    elif command -v lspci >/dev/null 2>&1; then \
        if lspci | grep -i 'vga\|3d\|display' | grep -qi 'nvidia'; then echo NVIDIA; \
        elif lspci | grep -i 'vga\|3d\|display' | grep -qi 'amd'; then echo AMD; \
        elif lspci | grep -i 'vga\|3d\|display' | grep -qi 'intel'; then echo INTEL; \
        else echo UNKNOWN; fi; \
    else \
        echo UNKNOWN; \
    fi)

ifneq (,$(filter f_ompoffload f_mpi_ompoffload,$(TARGETS)))
ifeq ($(OPS_COMPILER),gnu)
TARGETS:=$(filter-out f_ompoffload f_mpi_ompoffload,$(TARGETS))
endif
ifeq ($(OPS_COMPILER),intel)
TARGETS:=$(filter-out f_ompoffload f_mpi_ompoffload,$(TARGETS))
endif
ifeq ($(OPS_COMPILER),icx)
ifeq ($(GPU_VENDOR), NVIDIA)
TARGETS:=$(filter-out f_ompoffload f_mpi_ompoffload,$(TARGETS))
endif
ifeq ($(GPU_VENDOR), AMD)
TARGETS:=$(filter-out f_ompoffload f_mpi_ompoffload,$(TARGETS))
endif
endif
endif

ifdef USE_OPENMP4
    OMP_OFFLOAD_COMMON=ops_ompoffload_common_omp4.o
else
    OMP_OFFLOAD_COMMON=ops_ompoffload_common_omp5.o
endif

CXXFLAGS_NOOMP = $(CXXFLAGS)
CXXFLAGS_NOOMP := $(filter-out -fopenmp, $(CXXFLAGS_NOOMP))
CXXFLAGS_NOOMP := $(filter-out -qopenmp, $(CXXFLAGS_NOOMP))

FFLAGS_NOOMP = $(FFLAGS)
FFLAGS_NOOMP := $(filter-out -fopenmp, $(FFLAGS_NOOMP))
FFLAGS_NOOMP := $(filter-out -qopenmp, $(FFLAGS_NOOMP))

CHKCXX += $(I_MPI_INC)

all: $(TARGETS)

mkdir:
	@mkdir -p $(F_OPS_LIB) $(F_OPS_OBJ) $(F_OPS_MOD) $(F_INC_MOD) $(F_INC_MOD)/cuda $(F_INC_MOD)/f2c_cuda $(F_INC_MOD)/ompoffload $(F_INC_MOD)/f2c_hip $(F_INC_MOD)/f2c_sycl

.SUFFIXES: .cpp .cu .F90 .o

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/core/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%_hip.o : $(C_OPS_SRC)/core/%.cpp
	$(CXX) $(CXXFLAGS_NOOMP) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/sequential/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/ops_hdf5.o : $(C_OPS_SRC)/externlib/ops_hdf5.cpp
	$(CHKCXX) $(CXXFLAGS_NOOMP) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/ops_hdf5_common.o : $(C_OPS_SRC)/externlib/ops_hdf5_common.cpp
	$(CHKCXX) $(CXXFLAGS_NOOMP) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o : $(C_OPS_SRC)/mpi/ops_mpi_hdf5.cpp
	$(MPICXX) $(CXXFLAGS_NOOMP) -I$(C_OPS_INC) $(I_MPI_INC) $(HDF5_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/externlib/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%_hip.o : $(C_OPS_SRC)/externlib/%.cpp
	$(CXX) $(CXXFLAGS_NOOMP) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%_mpi.o : $(C_OPS_SRC)/externlib/%.cpp
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) $(I_MPI_INC) $(HDF5_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/ompoffload/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPOFFLOADFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%_ompoffload_mpi.o : $(C_OPS_SRC)/mpi/%_ompoffload.cpp
	$(MPICXX) $(CXXFLAGS) $(OMPOFFLOADFLAGS) -I$(C_OPS_INC) $(I_MPI_INC) -c $< -o $@

$(F_OPS_OBJ)/%_mpi.o : $(C_OPS_SRC)/mpi/%.cpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) $(I_MPI_INC) $(HDF5_INC) -c $< -o $@

$(F_OPS_OBJ)/%_mpi_hip.o : $(C_OPS_SRC)/mpi/%.cpp
	$(MPICXX) $(CXXFLAGS_NOOMP) -I$(C_OPS_INC) $(I_MPI_INC) $(HDF5_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/mpi/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%_sycl_mpi.o : $(C_OPS_SRC)/mpi/%_sycl.cpp
	$(SYCLMPICXX) $(CXXFLAGS) $(SYCL_FLAGS) -I$(C_OPS_INC) $(I_MPI_INC) -c $< -o $@

$(F_OPS_OBJ)/%_hip_mpi_hip.o : $(C_OPS_SRC)/mpi/%_hip.cpp
	$(HIPCC) $(CXXFLAGS_NOOMP) $(HIPFLAGS) -I$(C_OPS_INC) $(I_MPI_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/cuda/%.cpp
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%_cuda.o : $(C_OPS_SRC)/cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/hip/%.cpp
	$(HIPCC) $(CXXFLAGS_NOOMP) $(HIPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/sycl/%.cpp
	$(SYCLCXX) $(CXXFLAGS) $(SYCL_FLAGS) -I$(C_OPS_INC) -c $< -o $@

#	******************************	Sequential	************************************
$(F_OPS_OBJ)/ops_for_declarations.o: $(F_OPS_SRC)/ops_for_declarations.F90
	$(FC) $(FFLAGS) $(FMODS) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(F_OPS_SRC)/%.F90 $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS) -c $< -o $@

#	******************************	F2C Sequential	*********************************
$(F_OPS_OBJ)/ops_for_declarations_f2c.o: $(F_OPS_SRC)/ops_for_declarations.F90
	$(FC) $(FFLAGS) -DOPS_F2C_INTEROP $(FMODS) -c $< -o $@

$(F_OPS_OBJ)/%_f2c.o : $(F_OPS_SRC)/%.F90 $(F_OPS_OBJ)/ops_for_declarations_f2c.o
	$(FC) $(FFLAGS) $(FMODS) -c $< -o $@

#	******************************	Cuda	*****************************************
$(F_OPS_OBJ)/ops_for_declarations_cuda.o: $(F_OPS_SRC)/ops_for_declarations.F90
	$(FC) $(FFLAGS) $(FMODS_CUDA) -c $< -o $@

$(F_OPS_OBJ)/%_cuda.o : $(F_OPS_SRC)/%.F90 $(F_OPS_OBJ)/ops_for_declarations_cuda.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) -c $< -o $@

$(F_OPS_OBJ)/%_cuda.o : $(F_OPS_SRC)/%.CUF
	$(FC) $(FFLAGS) $(CUDAFOR) $(FMODS_CUDA) -c $< -o $@

#	******************************	F2C CUDA	*************************************
$(F_OPS_OBJ)/ops_for_declarations_f2c_cuda.o: $(F_OPS_SRC)/ops_for_declarations.F90
	$(FC) $(FFLAGS) -DOPS_F2C_INTEROP $(FMODS_F2C_CUDA) -c $< -o $@

$(F_OPS_OBJ)/%_f2c_cuda.o : $(F_OPS_SRC)/%.F90 $(F_OPS_OBJ)/ops_for_declarations_f2c_cuda.o
	$(FC) $(FFLAGS) $(FMODS_F2C_CUDA) -c $< -o $@

#   ******************************  F2C HIP    *************************************
$(F_OPS_OBJ)/ops_for_declarations_f2c_hip.o: $(F_OPS_SRC)/ops_for_declarations.F90
	$(FC) $(FFLAGS_NOOMP) -DOPS_F2C_INTEROP $(FMODS_F2C_HIP) -c $< -o $@

$(F_OPS_OBJ)/%_f2c_hip.o : $(F_OPS_SRC)/%.F90 $(F_OPS_OBJ)/ops_for_declarations_f2c_hip.o
	$(FC) $(FFLAGS_NOOMP) $(FMODS_F2C_HIP) -c $< -o $@

#   ******************************  F2C SYCL    *************************************
$(F_OPS_OBJ)/ops_for_declarations_f2c_sycl.o: $(F_OPS_SRC)/ops_for_declarations.F90
	$(FC) $(FFLAGS) -DOPS_F2C_INTEROP $(FMODS_F2C_SYCL) -c $< -o $@

$(F_OPS_OBJ)/%_f2c_sycl.o : $(F_OPS_SRC)/%.F90 $(F_OPS_OBJ)/ops_for_declarations_f2c_sycl.o
	$(FC) $(FFLAGS) $(FMODS_F2C_SYCL) -c $< -o $@

#	******************************	OpenMP Offload	*********************************
$(F_OPS_OBJ)/%_ompoffload.o : $(F_OPS_SRC)/%.F90
	$(FC) $(FFLAGS) $(FMODS_OMPOFFLOAD) -c $< -o $@

#	****************************************************************************************************************************************************************************

hdf5_seq: mkdir $(F_OPS_OBJ)/ops_hdf5.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_hdf5.a $(F_OPS_OBJ)/ops_hdf5.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o

f_seq: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_host_common.o $(F_OPS_OBJ)/ops_host_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_seq.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_host_common.o $(F_OPS_OBJ)/ops_host_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

f2c_seq: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_host_common.o $(F_OPS_OBJ)/ops_host_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_for_declarations_f2c.o $(F_OPS_OBJ)/ops_for_rt_support_f2c.o $(F_OPS_OBJ)/ops_for_math_support_f2c.o
	ar -r $(F_OPS_LIB)/libops_f2c_seq.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_host_common.o $(F_OPS_OBJ)/ops_host_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_for_declarations_f2c.o $(F_OPS_OBJ)/ops_for_rt_support_f2c.o $(F_OPS_OBJ)/ops_for_math_support_f2c.o

f_cuda: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_common_cuda.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
	$(F_OPS_OBJ)/ops_cuda_singlenode_cuda.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common.o \
	$(F_OPS_OBJ)/ops_for_declarations_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_cuda.o
	ar -r $(F_OPS_LIB)/libops_for_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_common_cuda.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o $(F_OPS_OBJ)/ops_cuda_singlenode_cuda.o \
	$(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_device_singlenode_common.o \
	$(F_OPS_OBJ)/ops_for_declarations_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_cuda.o

f2c_cuda: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_common_cuda.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
	$(F_OPS_OBJ)/ops_cuda_singlenode_cuda.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common.o \
	$(F_OPS_OBJ)/ops_for_declarations_f2c_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_cuda.o $(F_OPS_OBJ)/ops_for_math_support_f2c_cuda.o
	ar -r $(F_OPS_LIB)/libops_f2c_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_common_cuda.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o $(F_OPS_OBJ)/ops_cuda_singlenode_cuda.o \
	$(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_device_singlenode_common.o \
	$(F_OPS_OBJ)/ops_for_declarations_f2c_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_cuda.o $(F_OPS_OBJ)/ops_for_math_support_f2c_cuda.o

f_ompoffload: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/$(OMP_OFFLOAD_COMMON) $(F_OPS_OBJ)/ops_ompoffload_rt_support_kernels.o \
	$(F_OPS_OBJ)/ops_ompoffload_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common.o $(F_OPS_OBJ)/ops_for_declarations_ompoffload.o $(F_OPS_OBJ)/ops_for_rt_support_ompoffload.o
	ar -r $(F_OPS_LIB)/libops_for_ompoffload.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/$(OMP_OFFLOAD_COMMON) $(F_OPS_OBJ)/ops_ompoffload_rt_support_kernels.o \
	$(F_OPS_OBJ)/ops_ompoffload_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common.o $(F_OPS_OBJ)/ops_for_declarations_ompoffload.o $(F_OPS_OBJ)/ops_for_rt_support_ompoffload.o

f2c_hip: mkdir $(F_OPS_OBJ)/ops_lib_core_hip.o $(F_OPS_OBJ)/ops_instance_hip.o $(F_OPS_OBJ)/ops_lazy_hip.o $(F_OPS_OBJ)/ops_checkpointing_hip.o \
	$(F_OPS_OBJ)/ops_util_hip.o $(F_OPS_OBJ)/ops_hip_common.o $(F_OPS_OBJ)/ops_hip_rt_support_kernels.o \
	$(F_OPS_OBJ)/ops_hip_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode_hip.o $(F_OPS_OBJ)/ops_checkpointing_strategy_hip.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common_hip.o \
	$(F_OPS_OBJ)/ops_for_declarations_f2c_hip.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_hip.o $(F_OPS_OBJ)/ops_for_math_support_f2c_hip.o
	ar -r $(F_OPS_LIB)/libops_f2c_hip.a $(F_OPS_OBJ)/ops_lib_core_hip.o $(F_OPS_OBJ)/ops_instance_hip.o $(F_OPS_OBJ)/ops_lazy_hip.o $(F_OPS_OBJ)/ops_checkpointing_hip.o \
	$(F_OPS_OBJ)/ops_util_hip.o $(F_OPS_OBJ)/ops_hip_common.o $(F_OPS_OBJ)/ops_hip_rt_support_kernels.o \
	$(F_OPS_OBJ)/ops_hip_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode_hip.o $(F_OPS_OBJ)/ops_checkpointing_strategy_hip.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common_hip.o \
	$(F_OPS_OBJ)/ops_for_declarations_f2c_hip.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_hip.o $(F_OPS_OBJ)/ops_for_math_support_f2c_hip.o

f2c_sycl: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_sycl_common.o $(F_OPS_OBJ)/ops_sycl_rt_support_kernels.o \
	$(F_OPS_OBJ)/ops_sycl_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common.o \
	$(F_OPS_OBJ)/ops_for_declarations_f2c_sycl.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_sycl.o $(F_OPS_OBJ)/ops_for_math_support_f2c_sycl.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_sycl.o
	ar -r $(F_OPS_LIB)/libops_f2c_sycl.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_sycl_common.o $(F_OPS_OBJ)/ops_sycl_rt_support_kernels.o \
	$(F_OPS_OBJ)/ops_sycl_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common.o \
	$(F_OPS_OBJ)/ops_for_declarations_f2c_sycl.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_sycl.o $(F_OPS_OBJ)/ops_for_math_support_f2c_sycl.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_sycl.o

f_mpi: 	mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o $(F_OPS_OBJ)/ops_checkpointing_strategy_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_host_mpi.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o $(F_OPS_OBJ)/ops_host_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_for_mpi.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o $(F_OPS_OBJ)/ops_checkpointing_strategy_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_host_mpi.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o $(F_OPS_OBJ)/ops_host_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations.o

f2c_mpi:  mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o $(F_OPS_OBJ)/ops_checkpointing_strategy_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_host_mpi.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o $(F_OPS_OBJ)/ops_host_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_f2c.o $(F_OPS_OBJ)/ops_for_rt_support_f2c.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c.o $(F_OPS_OBJ)/ops_for_math_support_f2c.o
	ar -r $(F_OPS_LIB)/libops_f2c_mpi.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o $(F_OPS_OBJ)/ops_checkpointing_strategy_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_host_mpi.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o $(F_OPS_OBJ)/ops_host_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_f2c.o $(F_OPS_OBJ)/ops_for_rt_support_f2c.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c.o $(F_OPS_OBJ)/ops_for_math_support_f2c.o

f_mpi_cuda: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o \
	$(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_cuda_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_cuda.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_cuda.o
	ar -r $(F_OPS_LIB)/libops_for_mpi_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o \
	$(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_cuda_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_cuda.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_cuda.o

f2c_mpi_cuda: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o \
	$(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_cuda_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_f2c_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_cuda.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_cuda.o $(F_OPS_OBJ)/ops_for_math_support_f2c_cuda.o
	ar -r $(F_OPS_LIB)/libops_f2c_mpi_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o \
	$(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_cuda_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_f2c_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_cuda.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_cuda.o $(F_OPS_OBJ)/ops_for_math_support_f2c_cuda.o

f_mpi_ompoffload: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o \
	$(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_ompoffload_mpi.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/$(OMP_OFFLOAD_COMMON) \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_ompoffload.o \
	$(F_OPS_OBJ)/ops_for_rt_support_ompoffload.o $(F_OPS_OBJ)/ops_for_hdf5_declarations_ompoffload.o
	ar -r $(F_OPS_LIB)/libops_for_mpi_ompoffload.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o \
	$(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_ompoffload_mpi.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/$(OMP_OFFLOAD_COMMON) \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_ompoffload.o \
	$(F_OPS_OBJ)/ops_for_rt_support_ompoffload.o $(F_OPS_OBJ)/ops_for_hdf5_declarations_ompoffload.o

f2c_mpi_hip: mkdir $(F_OPS_OBJ)/ops_lib_core_hip.o $(F_OPS_OBJ)/ops_instance_hip.o $(F_OPS_OBJ)/ops_lazy_hip.o \
	$(F_OPS_OBJ)/ops_checkpointing_hip.o $(F_OPS_OBJ)/ops_util_hip.o $(F_OPS_OBJ)/ops_mpi_decl_mpi_hip.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi_hip.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi_hip.o $(F_OPS_OBJ)/ops_mpi_rt_support_hip_mpi_hip.o $(F_OPS_OBJ)/ops_mpi_partition_mpi_hip.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy_hip.o $(F_OPS_OBJ)/ops_hip_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_f2c_hip.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_hip.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_hip.o $(F_OPS_OBJ)/ops_for_math_support_f2c_hip.o
	ar -r $(F_OPS_LIB)/libops_f2c_mpi_hip.a $(F_OPS_OBJ)/ops_lib_core_hip.o $(F_OPS_OBJ)/ops_instance_hip.o $(F_OPS_OBJ)/ops_lazy_hip.o \
	$(F_OPS_OBJ)/ops_checkpointing_hip.o $(F_OPS_OBJ)/ops_util_hip.o $(F_OPS_OBJ)/ops_mpi_decl_mpi_hip.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi_hip.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi_hip.o $(F_OPS_OBJ)/ops_mpi_rt_support_hip_mpi_hip.o $(F_OPS_OBJ)/ops_mpi_partition_mpi_hip.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy_hip.o $(F_OPS_OBJ)/ops_hip_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_f2c_hip.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_hip.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_hip.o $(F_OPS_OBJ)/ops_for_math_support_f2c_hip.o

f2c_mpi_sycl: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o \
	$(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_sycl_mpi.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_sycl_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_f2c_sycl.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_sycl.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_sycl.o $(F_OPS_OBJ)/ops_for_math_support_f2c_sycl.o
	ar -r $(F_OPS_LIB)/libops_f2c_mpi_sycl.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o \
	$(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_mpi.o \
	$(F_OPS_OBJ)/ops_mpi_core_mpi.o $(F_OPS_OBJ)/ops_mpi_rt_support_sycl_mpi.o $(F_OPS_OBJ)/ops_mpi_partition_mpi.o \
	$(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_sycl_common.o \
	$(F_OPS_OBJ)/ops_mpi_hdf5_mpi.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_declarations_f2c_sycl.o $(F_OPS_OBJ)/ops_for_rt_support_f2c_sycl.o \
	$(F_OPS_OBJ)/ops_for_hdf5_declarations_f2c_sycl.o $(F_OPS_OBJ)/ops_for_math_support_f2c_sycl.o

clean:
	-rm -if $(F_OPS_OBJ)/*.o $(F_OPS_OBJ)/*.optrpt
	-rm -if $(F_OPS_LIB)/*.a
	-rm -if $(C_OPS_SRC)/*~
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_MOD)/*.mod
	-rm -if $(F_INC_MOD)/*.mod
	-rm -if $(F_INC_MOD)/cuda/*.mod
	-rm -if $(F_INC_MOD)/f2c_cuda/*.mod
	-rm -if $(F_INC_MOD)/ompoffload/*.mod
	-rm -if $(F_INC_MOD)/f2c_hip/*.mod
	-rm -if $(F_INC_MOD)/f2c_sycl/*.mod
	-rm -if $(F_OPS)/*~
	-rm -if *.lst cudafe
