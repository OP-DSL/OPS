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

core:	mkdir $(C_OPS_INC)/ops_lib_core.h $(C_OPS_SRC)/core/ops_lib_core.c $(C_OPS_SRC)/externlib/ops_util.c
	$(CC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lazy.cpp -o $(F_OPS_OBJ)/ops_lazy.o
	$(CXX) $(CXXFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_amr_lazy.cpp -o $(F_OPS_OBJ)/ops_amr_lazy.o
	$(CC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/externlib/ops_util.c -o $(F_OPS_OBJ)/ops_util.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o

hdf5:	mkdir core $(F_OPS_OBJ)/ops_lib_core.o
	$(CHKCC) $(CXXFLAGS) -I$(C_OPS_INC) $(HDF5_INC) -c $(C_OPS_SRC)/externlib/ops_checkpointing.c -o $(F_OPS_OBJ)/ops_checkpointing.o
	$(CC) $(CCFLAGS) $(HDF5_INC) -I$(MPI_INC) -I$(C_OPS_INC) -c $(C_OPS_SRC)/externlib/ops_hdf5.c -o $(F_OPS_OBJ)/ops_hdf5.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_hdf5.a $(F_OPS_OBJ)/ops_hdf5.o $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o


f_seq: 	mkdir core hdf5 $(F_OPS_OBJ)/ops_lib_core.o
	$(CC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/sequential/ops_seq.c -o $(F_OPS_OBJ)/ops_seq.o
	$(CC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.c -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_seq.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_amr_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_seq.o $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

f_cuda: mkdir core
	$(CC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(CC) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.c -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(CC) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_decl.c -o $(F_OPS_OBJ)/ops_cuda_decl.o
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cu -o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o
	$(CC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.c -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) $(CUDAFOR) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) $(CUDAFOR) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	ar -r $(F_OPS_LIB)/libops_for_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_amr_lazy.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_decl.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
	$(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

f_mpi: 	mkdir core $(F_OPS_OBJ)/ops_lib_core.o
	$(MPICC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_decl.c -o $(F_OPS_OBJ)/ops_mpi_decl.o
	$(MPICC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_core.c -o $(F_OPS_OBJ)/ops_mpi_core.o
	$(MPICC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_partition.c -o $(F_OPS_OBJ)/ops_mpi_partition.o
	$(MPICC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support.c -o $(F_OPS_OBJ)/ops_mpi_rt_support.o
	$(MPICC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support_host.c -o $(F_OPS_OBJ)/ops_mpi_rt_support_host.o
	$(MPICC) $(CCFLAGS) -I$(MPI_INC) $(HDF5_INC) -I$(C_OPS_INC)  -I$(CUDA_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_hdf5.c -o $(F_OPS_OBJ)/ops_mpi_hdf5.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	$(FC) $(FFLAGS) $(FMODS) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_for_mpi.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_amr_lazy.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_decl.o $(F_OPS_OBJ)/ops_mpi_core.o $(F_OPS_OBJ)/ops_mpi_rt_support.o \
	$(F_OPS_OBJ)/ops_mpi_partition.o $(F_OPS_OBJ)/ops_mpi_rt_support_host.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o $(F_OPS_OBJ)/ops_mpi_hdf5.o  $(F_OPS_OBJ)/ops_for_hdf5_declarations.o

f_mpi_cuda: mkdir core
	$(MPICC) $(CCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(CUDA_INC) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_decl_cuda.c -o $(F_OPS_OBJ)/ops_mpi_decl_cuda.o
	$(MPICC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_partition.c -o $(F_OPS_OBJ)/ops_mpi_partition.o
	$(MPICC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_core.c -o $(F_OPS_OBJ)/ops_mpi_core.o
	$(MPICC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support.c -o $(F_OPS_OBJ)/ops_mpi_rt_support.o
	$(MPICC) $(CXXFLAGS) -I$(MPI_INC) $(HDF5_INC) -I$(C_OPS_INC)  -I$(CUDA_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_hdf5.c -o $(F_OPS_OBJ)/ops_mpi_hdf5.o
	$(NVCC)  $(NVCCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/mpi/ops_mpi_rt_support_cuda.cu -o $(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o -I$(MPI_INSTALL_PATH)/include
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.c -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) $(CUDAFOR) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) $(CUDAFOR) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	$(FC) $(FFLAGS) $(FMODS_CUDA) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_for_mpi_cuda.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o $(F_OPS_OBJ)/ops_amr_lazy.o \
	$(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_mpi_core.o $(F_OPS_OBJ)/ops_mpi_partition.o $(F_OPS_OBJ)/ops_mpi_decl_cuda.o \
	$(F_OPS_OBJ)/ops_mpi_rt_support_cuda.o $(F_OPS_OBJ)/ops_mpi_rt_support.o \
	$(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_mpi_hdf5.o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o

f_openacc: mkdir core
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_lib_core.c -o $(F_OPS_OBJ)/ops_lib_core.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.c -o $(F_OPS_OBJ)/ops_cuda_rt_support.o
	$(NVCC) $(NVCCFLAGS) $(CUDA_ALIGNE_FLAG) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_decl.c -o $(F_OPS_OBJ)/ops_cuda_decl.o
	$(NVCC) $(NVCCFLAGS) -I$(C_OPS_INC) -I$(CUDA_INC) -c $(C_OPS_SRC)/cuda/ops_cuda_rt_support.cu -o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o
	$(CC) $(CCFLAGS) -I$(C_OPS_INC) -c $(C_OPS_SRC)/core/ops_dummy_singlenode.c -o $(F_OPS_OBJ)/ops_dummy_singlenode.o
	$(FC) $(OPENACCFLAGS) -c $(F_OPS_SRC)/ops_for_declarations.F90 -o $(F_OPS_OBJ)/ops_for_declarations.o
	$(FC) $(OPENACCFLAGS) -c $(F_OPS_SRC)/ops_for_rt_support.F90 -o $(F_OPS_OBJ)/ops_for_rt_support.o
	$(FC) $(OPENACCFLAGS) -c $(F_OPS_SRC)/ops_for_hdf5_declarations.F90 -o $(F_OPS_OBJ)/ops_for_hdf5_declarations.o
	ar -r $(F_OPS_LIB)/libops_for_openacc.a $(F_OPS_OBJ)/ops_lib_core.o $(F_OPS_OBJ)/ops_lazy.o $(F_OPS_OBJ)/ops_amr_lazy.o $(F_OPS_OBJ)/ops_checkpointing.o \
        $(F_OPS_OBJ)/ops_util.o $(F_OPS_OBJ)/ops_cuda_decl.o $(F_OPS_OBJ)/ops_cuda_rt_support.o $(F_OPS_OBJ)/ops_cuda_rt_support_cuda.o \
        $(F_OPS_OBJ)/ops_dummy_singlenode.o $(F_OPS_OBJ)/ops_for_declarations.o $(F_OPS_OBJ)/ops_for_rt_support.o

clean:
	-rm -if $(F_OPS_OBJ)/*.o
	-rm -if $(F_OPS_LIB)/*.a
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_SRC)/*~
	-rm -if $(F_OPS_MOD)/*.mod
	-rm -if $(F_INC_MOD)/*.mod
	-rm -if $(F_OPS)/*~
