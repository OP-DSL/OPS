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


TARGETS	= clean multidim_seq multidim_openmp multidim_mpi multidim_mpi_openmp
ifeq ($(OPS_COMPILER),pgi)
  TARGETS       += multidim_cuda multidim_mpi_cuda
endif

all: $(TARGETS)

multidim_seq: multidim_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) constants.F90 -c ./MPI/multidim_print_kernel_seq_kernel.F90 \
	./MPI/multidim_copy_kernel_seq_kernel.F90 ./MPI/multidim_kernel_seq_kernel.F90 ./MPI/multidim_reduce_kernel_seq_kernel.F90 multidim_ops.F90
	$(FC) $(FFLAGS) $(FLINK) multidim_ops.o constants.o multidim_kernel_seq_kernel.o multidim_copy_kernel_seq_kernel.o \
	multidim_print_kernel_seq_kernel.o multidim_reduce_kernel_seq_kernel.o -o multidim_seq -lops_for_seq

multidim_openmp: multidim_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(FMODS) $(OMPFLAGS) constants.F90 -c ./MPI/multidim_print_kernel_seq_kernel.F90 \
        ./MPI/multidim_copy_kernel_seq_kernel.F90 ./MPI/multidim_kernel_seq_kernel.F90 ./MPI/multidim_reduce_kernel_seq_kernel.F90 multidim_ops.F90
	$(FC) $(FFLAGS) $(FLINK) $(OMPFLAGS) multidim_ops.o constants.o multidim_kernel_seq_kernel.o multidim_copy_kernel_seq_kernel.o \
        multidim_print_kernel_seq_kernel.o multidim_reduce_kernel_seq_kernel.o -o multidim_openmp -lops_for_seq

multidim_cuda: multidim_ops.F90 constants.F90
ifeq ($(OPS_COMPILER),pgi)
	$(FC) $(FFLAGS) $(FMODS) $(CUDA_ALIGN_FLAG) $(CUDAFOR) constants.F90 -c ./CUDA/multidim_print_kernel_cuda_kernel.CUF \
	./CUDA/multidim_copy_kernel_cuda_kernel.CUF ./CUDA/multidim_kernel_cuda_kernel.CUF ./CUDA/multidim_reduce_kernel_cuda_kernel.CUF multidim_ops.F90
	$(FC) $(FFLAGS) $(FLINK) $(CUDA_ALIGN_FLAG) $(CUDAFOR) multidim_ops.o constants.o multidim_kernel_cuda_kernel.o \
	multidim_copy_kernel_cuda_kernel.o multidim_print_kernel_cuda_kernel.o \
	multidim_reduce_kernel_cuda_kernel.o -o multidim_cuda -lops_for_cuda
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif

multidim_mpi: multidim_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) $(OPS_MPI) -c constants.F90 ./MPI/multidim_print_kernel_seq_kernel.F90 \
        ./MPI/multidim_copy_kernel_seq_kernel.F90 ./MPI/multidim_kernel_seq_kernel.F90 ./MPI/multidim_reduce_kernel_seq_kernel.F90 multidim_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(OPS_MPI) multidim_ops.o constants.o multidim_kernel_seq_kernel.o multidim_copy_kernel_seq_kernel.o \
        multidim_print_kernel_seq_kernel.o multidim_reduce_kernel_seq_kernel.o -o multidim_mpi -lops_for_mpi $(HDF5_LIB)

multidim_mpi_openmp: multidim_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) $(OMPFLAGS) $(OPS_MPI) -c constants.F90 ./MPI/multidim_print_kernel_seq_kernel.F90 \
        ./MPI/multidim_copy_kernel_seq_kernel.F90 ./MPI/multidim_kernel_seq_kernel.F90 ./MPI/multidim_reduce_kernel_seq_kernel.F90 multidim_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(OMPFLAGS) $(OPS_MPI) multidim_ops.o constants.o multidim_kernel_seq_kernel.o multidim_copy_kernel_seq_kernel.o \
        multidim_print_kernel_seq_kernel.o multidim_reduce_kernel_seq_kernel.o -o multidim_mpi_openmp -lops_for_mpi $(HDF5_LIB)


multidim_mpi_cuda: multidim_ops.F90 constants.F90
ifeq ($(OPS_COMPILER),pgi)
	$(MPIF90) $(FFLAGS) $(FMODS) $(CUDA_ALIGN_FLAG) $(CUDAFOR) $(OPS_MPI) -c constants.F90 ./CUDA/multidim_print_kernel_cuda_kernel.CUF \
	./CUDA/multidim_copy_kernel_cuda_kernel.CUF ./CUDA/multidim_kernel_cuda_kernel.CUF ./CUDA/multidim_reduce_kernel_cuda_kernel.CUF multidim_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(CUDA_ALIGN_FLAG) $(CUDAFOR) $(OPS_MPI) multidim_ops.o constants.o multidim_kernel_cuda_kernel.o \
        multidim_copy_kernel_cuda_kernel.o multidim_print_kernel_cuda_kernel.o \
        multidim_reduce_kernel_cuda_kernel.o -o multidim_mpi_cuda -lops_for_mpi_cuda $(HDF5_LIB)
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif


clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f multidim_seq multidim_cuda multidim_openmp multidim_mpi multidim_mpi_openmp multidim_mpi_cuda
