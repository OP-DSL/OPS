include $(OPS_INSTALL_PATH)/../makefiles/Makefile.common
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.mpi
include $(OPS_INSTALL_PATH)/../makefiles/Makefile.cuda

ifdef HDF5_INSTALL_PATH
ifdef CHECKPOINTING
  HDF5_INC              := -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB              := -L$(HDF5_INSTALL_PATH)/lib -lhdf5_hl -lhdf5 -lz
  HDF5_INC              := $(HDF5_INC) -DCHECKPOINTING
endif
endif


TARGETS	= clean mblock_seq mblock_openmp mblock_mpi mblock_mpi_openmp
ifeq ($(OPS_COMPILER),pgi)
	TARGETS       += mblock_cuda mblock_mpi_cuda
endif
ifeq ($(OPS_COMPILER),xlp8)
	TARGETS       += mblock_cuda poisson_mpi_cuda
endif

all: $(TARGETS)

mblock_seq: mblock_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(FMODS) -c constants.F90 ./MPI/mblock_populate_kernel_seq_kernel.F90 \
	mblock_ops.F90
	$(FC) $(FFLAGS) $(FLINK) mblock_ops.o constants.o mblock_populate_kernel_seq_kernel.o \
	-o mblock_seq -lops_for_seq

mblock_openmp: mblock_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(FMODS) $(OMPFLAGS) -c constants.F90 ./MPI_OpenMP/mblock_populate_kernel_omp_kernel.F90 \
        mblock_ops.F90
	$(FC) $(FFLAGS) $(FLINK) $(OMPFLAGS) mblock_ops.o constants.o mblock_populate_kernel_omp_kernel.o \
        -o mblock_openmp -lops_for_seq

mblock_cuda: mblock_ops.F90 constants.F90
ifeq ($(OPS_COMPILER),pgi)
	$(FC) $(FFLAGS) $(FMODS) -DOPS_WITH_CUDAFOR -c constants.F90 ./CUDA/mblock_populate_kernel_cuda_kernel.CUF \
        mblock_ops.F90
	$(FC) $(FFLAGS) $(FLINK) -DOPS_WITH_CUDAFOR mblock_ops.o constants.o mblock_populate_kernel_cuda_kernel.o \
        -o mblock_cuda -lops_for_cuda
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif

mblock_mpi: mblock_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) -c -DOPS_MPI constants.F90 ./MPI/mblock_populate_kernel_seq_kernel.F90 \
        mblock_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) -DOPS_MPI mblock_ops.o constants.o mblock_populate_kernel_seq_kernel.o \
        -o mblock_mpi -lops_for_mpi $(HDF5_LIB)

mblock_mpi_openmp: mblock_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) $(OMPFLAGS) -c -DOPS_MPI constants.F90 ./MPI_OpenMP/mblock_populate_kernel_omp_kernel.F90 \
        mblock_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(OMPFLAGS) -DOPS_MPI mblock_ops.o constants.o mblock_populate_kernel_omp_kernel.o \
        -o mblock_mpi_openmp -lops_for_mpi $(HDF5_LIB)

mblock_mpi_cuda: mblock_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) $(CUDAFOR) $(OPS_MPI) -c constants.F90 ./CUDA/mblock_populate_kernel_cuda_kernel.CUF \
        mblock_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(CUDAFOR) $(OPS_MPI) mblock_ops.o constants.o mblock_populate_kernel_cuda_kernel.o \
        -o mblock_mpi_cuda -lops_for_mpi_cuda $(HDF5_LIB)

clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f mblock_seq
	rm -f mblock_openmp
	rm -f mblock_cuda
	rm -f mblock_mpi
	rm -f mblock_mpi_openmp
	rm -f mblock_mpi_cuda
