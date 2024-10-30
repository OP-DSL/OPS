#
# The following environment variables are required:
#
# CUDA_INSTALL_PATH - CUDA base directory
# OPS_COMPILER      - Compiler toolchain: gnu (gcc, gfortran) or pgi (pgc++, pgfortran)
# OPS_INSTALL_PATH  - OPS base directory
#
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda

ifdef HDF5_INSTALL_PATH
  HDF5_INC 	  	:= -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB		:= -L$(HDF5_INSTALL_PATH)/lib
  CHKCC			:= $(MPICPP)
  CHKCXX            := $(MPICXX)
ifdef CHECKPOINTING
  HDF5_INC 	  	:= $(HDF5_INC) -DCHECKPOINTING
endif
else
  CHKCC			:= $(CC)
  CHKCXX            := $(CXX)
endif

CXXFLAGS += -DOPS_FTN
NVCCFLAGS += -DOPS_FTN

.PHONY: mkdir

TARGETS	= f_seq f_mpi

ifdef HDF5_INSTALL_PATH
	TARGETS += hdf5_seq
	C_OPS_INC += $(HDF5_INC)
endif

ifeq ($(OPS_COMPILER),pgi)
	TARGETS += f_cuda f_mpi_cuda
	C_OPS_INC += -I$(CUDA_INC)
endif

all: $(TARGETS)

mkdir:
	@mkdir -p $(F_OPS_LIB) $(F_OPS_OBJ) $(F_OPS_MOD) $(F_INC_MOD) $(F_INC_MOD)/openacc $(F_INC_MOD)/cuda

.SUFFIXES: .cpp .cu .F90 .o

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/core/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/sequential/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/ops_hdf5.o : $(C_OPS_SRC)/externlib/ops_hdf5.cpp
	$(CHKCXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/ops_hdf5_common.o : $(C_OPS_SRC)/externlib/ops_hdf5_common.cpp
	$(CHKCXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/externlib/%.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%_mpi.o : $(C_OPS_SRC)/externlib/%.cpp
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC)  $(HDF5_INC) -c $< -o $@

$(F_OPS_OBJ)/%_mpi.o : $(C_OPS_SRC)/mpi/%.cpp
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -I$(C_OPS_INC)  $(HDF5_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/mpi/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/cuda/%.cpp
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%_cuda.o : $(C_OPS_SRC)/cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(C_OPS_SRC)/cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $< -o $@

$(F_OPS_OBJ)/ops_for_declarations.o: $(F_OPS_SRC)/ops_for_declarations.F90
	$(FC) $(FFLAGS) $(FMODS) -c $< -o $@

$(F_OPS_OBJ)/%.o : $(F_OPS_SRC)/%.F90 $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS) -c $< -o $@

$(F_OPS_OBJ)/%_cuda.o : $(F_OPS_SRC)/%.F90
	$(FC) $(FFLAGS) $(FMODS_CUDA) -c $< -o $@

$(F_OPS_OBJ)/%_cuda.o : $(F_OPS_SRC)/%.CUF
	$(FC) $(FFLAGS) $(CUDAFOR) $(FMODS_CUDA) -c $< -o $@


hdf5_seq: mkdir $(F_OPS_OBJ)/ops_hdf5.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_hdf5.a $(F_OPS_OBJ)/ops_hdf5.o $(F_OPS_OBJ)/ops_hdf5_common.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o

f_seq: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_host_common.o $(F_OPS_OBJ)/ops_host_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_seq.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_host_common.o $(F_OPS_OBJ)/ops_host_singlenode.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

f_cuda: mkdir $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_common_cuda.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
	$(F_OPS_OBJ)/ops_cuda_singlenode_cuda.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o \
	$(F_OPS_OBJ)/ops_device_singlenode_common.o \
	$(F_OPS_OBJ)/ops_for_declarations_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_cuda.o
	ar -r $(F_OPS_LIB)/libops_for_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_instance.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_common_cuda.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o $(F_OPS_OBJ)/ops_cuda_singlenode_cuda.o \
	$(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_checkpointing_strategy.o $(F_OPS_OBJ)/ops_device_singlenode_common.o \
	$(F_OPS_OBJ)/ops_for_declarations_cuda.o $(F_OPS_OBJ)/ops_for_rt_support_cuda.o

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

clean:
	-rm -if $(F_OPS_OBJ)/*.o $(F_OPS_OBJ)/*.optrpt
	-rm -if $(F_OPS_LIB)/*.a
	-rm -if $(C_OPS_SRC)/*~
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_MOD)/*.mod
	-rm -if $(F_INC_MOD)/*.mod
	-rm -if $(F_OPS)/*~
	-rm -if *.lst cudafe
