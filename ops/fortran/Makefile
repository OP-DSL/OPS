#
# The following environment variables are required:
#
# CUDA_INSTALL_PATH - CUDA base directory
# OPS_COMPILER      - Compiler toolchain: gnu (gcc, gfortran) or pgi (pgc++, pgfortran)
# OPS_INSTALL_PATH  - OPS base directory
#
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda

TARGETS	= clean core f_seq f_mpi

ifeq ($(OPS_COMPILER),pgi)
  TARGETS       += f_cuda f_mpi_cuda f_openacc
endif

ifeq ($(OPS_COMPILER),xlp8)
  TARGETS       += f_cuda f_mpi_cuda
endif


include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi

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

.PHONY: clean mkdir

all:	$(TARGETS)

mkdir:
	@mkdir -p $(F_OPS_LIB) $(F_OPS_OBJ) $(F_OPS_MOD) $(F_INC_MOD) $(F_INC_MOD)/openacc $(F_INC_MOD)/cuda

core:	mkdir $(C_OPS_INC)/ops_lib_core.h $(C_OPS_SRC)/core/ops_lib_core.cpp $(C_OPS_SRC)/externlib/ops_util.cpp
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.cpp -o $(F_OPS_OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_instance.cpp -o $(F_OPS_OBJ)/ops_instance.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lazy.cpp -o $(F_OPS_OBJ)/ops_lazy.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/externlib/ops_util.cpp -o $(F_OPS_OBJ)/ops_util.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o

hdf5:	mkdir core $(F_OPS_OBJ)/ops_lib_core.o
	$(CHKCC) $(CXXFLAGS) -I$(C_OPS_INC) $(HDF5_INC) -c $(C_OPS_SRC)/externlib/ops_checkpointing.cpp -o $(F_OPS_OBJ)/ops_checkpointing.o
	$(CXX) $(CXXFLAGS) $(HDF5_INC) -I$(MPI_INC) -I$(C_OPS_INC) -c $(C_OPS_SRC)/externlib/ops_hdf5.cpp -o $(F_OPS_OBJ)/ops_hdf5.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_hdf5.a $(F_OPS_OBJ)/ops_hdf5.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o


f_seq: 	mkdir core hdf5 $(F_OPS_OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/sequential/ops_seq.cpp -o $(F_OPS_OBJ)/ops_seq.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.cpp -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_seq.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_instance.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_seq.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

f_cuda: mkdir core
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.cpp -o $(F_OPS_OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) $(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cpp -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(CXX) $(CXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) $(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_decl.cpp -o $(F_OPS_OBJ)/ops_cuda_decl.o
	$(NVCC) $(NVCXXFLAGS) -I$(C_OPS_INC) $(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cu -o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.cpp -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) $(CUDAFOR) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) $(CUDAFOR) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_decl.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o $(F_OPS_OBJ)/ops_instance.o \
	$(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

f_mpi: 	mkdir core $(F_OPS_OBJ)/ops_lib_core.o
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_decl.cpp -o $(F_OPS_OBJ)/ops_mpi_decl.o
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_core.cpp -o $(F_OPS_OBJ)/ops_mpi_core.o
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_partition.cpp -o $(F_OPS_OBJ)/ops_mpi_partition.o
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support.cpp -o $(F_OPS_OBJ)/ops_mpi_rt_support.o
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support_host.cpp -o $(F_OPS_OBJ)/ops_mpi_rt_support_host.o
	$(MPICXX) $(CXXFLAGS) -I$(MPI_INC) $(HDF5_INC) -I$(C_OPS_INC)  $(CUDA_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_hdf5.cpp -o $(F_OPS_OBJ)/ops_mpi_hdf5.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_for_mpi.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl.o $(F_OPS_OBJ)/ops_mpi_core.o $(F_OPS_OBJ)/ops_mpi_rt_support.o $(F_OPS_OBJ)/ops_instance.o \
	$(F_OPS_OBJ)/ops_mpi_partition.o $(F_OPS_OBJ)/ops_mpi_rt_support_host.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o $(F_OPS_OBJ)/ops_mpi_hdf5.o  $(F_OPS_OBJ)/ops_for_hdf5_declarations.o

f_mpi_cuda: mkdir core
	$(MPICXX) $(CXXFLAGS) $(CUDA_ALIGNE_FLAG) $(CUDA_INC) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_decl_cuda.cpp -o $(F_OPS_OBJ)/ops_mpi_decl_cuda.o
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_partition.cpp -o $(F_OPS_OBJ)/ops_mpi_partition.o
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_core.cpp -o $(F_OPS_OBJ)/ops_mpi_core.o
	$(MPICXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support.cpp -o $(F_OPS_OBJ)/ops_mpi_rt_support.o
	$(MPICXX) $(CXXFLAGS) -I$(MPI_INC) $(HDF5_INC) -I$(C_OPS_INC)  $(CUDA_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_hdf5.cpp -o $(F_OPS_OBJ)/ops_mpi_hdf5.o
	$(NVCC)  $(NVCXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support_cuda.cu -o $(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o -I$(MPI_INSTALL_PATH)/include
	$(NVCC) $(NVCXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.cpp -o $(F_OPS_OBJ)/ops_lib_core.o
	$(NVCC) $(NVCXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) $(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cpp -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) $(CUDAFOR) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) $(CUDAFOR) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_for_mpi_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_core.o $(F_OPS_OBJ)/ops_mpi_partition.o $(F_OPS_OBJ)/ops_mpi_decl_cuda.o $(F_OPS_OBJ)/ops_instance.o \
	$(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(F_OPS_OBJ)/ops_mpi_rt_support.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_mpi_hdf5.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o

f_openacc: mkdir core
	$(NVCC) $(NVCXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.cpp -o $(F_OPS_OBJ)/ops_lib_core.o
	$(NVCC) $(NVCXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) $(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cpp -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCXXFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) $(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_decl.cpp -o $(F_OPS_OBJ)/ops_cuda_decl.o
	$(NVCC) $(NVCXXFLAGS) -I$(C_OPS_INC) $(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cu -o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.cpp -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(OPENACXXFLAGS) $(FMODS_OPENACC) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(OPENACXXFLAGS) $(FMODS_OPENACC) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	$(FC) $(OPENACXXFLAGS) $(FMODS_OPENACC) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_for_openacc.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_instance.o \
        $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_decl.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
        $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

clean:
	-rm -if $(F_OPS_OBJ)/*.o $(F_OPS_OBJ)/*.optrpt
	-rm -if $(F_OPS_LIB)/*.a
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_MOD)/*.mod
	-rm -if $(F_INC_MOD)/*.mod
	-rm -if $(F_OPS)/*~
