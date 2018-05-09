
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


TARGETS       = clean poisson_seq poisson_openmp poisson_mpi poisson_mpi_openmp
ifeq ($(OPS_COMPILER),pgi)
  TARGETS       += poisson_cuda poisson_mpi_cuda poisson_openacc
endif
ifeq ($(OPS_COMPILER),xlp8)
  TARGETS       += poisson_cuda poisson_mpi_cuda
endif


all: $(TARGETS)

poisson_seq: poisson_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(FMODS) -c constants.F90 \
	./MPI/poisson_initialguess_kernel_seq_kernel.F90 \
	./MPI/poisson_populate_kernel_seq_kernel.F90 \
	./MPI/poisson_stencil_kernel_seq_kernel.F90 \
	./MPI/poisson_update_kernel_seq_kernel.F90 \
	./MPI/poisson_error_kernel_seq_kernel.F90 \
	poisson_ops.F90
	$(FC) $(FFLAGS) $(FLINK) poisson_ops.o constants.o \
	poisson_populate_kernel_seq_kernel.o \
	poisson_initialguess_kernel_seq_kernel.o \
	poisson_stencil_kernel_seq_kernel.o \
	poisson_update_kernel_seq_kernel.o \
	poisson_error_kernel_seq_kernel.o \
	-o poisson_seq -lops_for_seq

poisson_openmp: poisson_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(OMPFLAGS) $(FMODS) -c constants.F90 \
	./MPI_OpenMP/poisson_initialguess_kernel_omp_kernel.F90 \
	./MPI_OpenMP/poisson_populate_kernel_omp_kernel.F90 \
	./MPI_OpenMP/poisson_stencil_kernel_omp_kernel.F90 \
	./MPI_OpenMP/poisson_update_kernel_omp_kernel.F90 \
	./MPI_OpenMP/poisson_error_kernel_omp_kernel.F90 \
	poisson_ops.F90
	$(FC) $(FFLAGS) $(OMPFLAGS) $(FLINK) poisson_ops.o constants.o \
	poisson_initialguess_kernel_omp_kernel.o \
	poisson_populate_kernel_omp_kernel.o \
	poisson_stencil_kernel_omp_kernel.o \
	poisson_update_kernel_omp_kernel.o \
	poisson_error_kernel_omp_kernel.o \
        -o poisson_openmp -lops_for_seq

poisson_cuda: poisson_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(FMODS) $(CUDAFOR) -c constants.F90 \
        ./CUDA/poisson_initialguess_kernel_cuda_kernel.CUF \
        ./CUDA/poisson_populate_kernel_cuda_kernel.CUF \
        ./CUDA/poisson_stencil_kernel_cuda_kernel.CUF \
        ./CUDA/poisson_update_kernel_cuda_kernel.CUF \
        ./CUDA/poisson_error_kernel_cuda_kernel.CUF \
        poisson_ops.F90
	$(FC) $(FFLAGS) $(FLINK) $(CUDAFOR) poisson_ops.o constants.o \
        poisson_initialguess_kernel_cuda_kernel.o \
        poisson_populate_kernel_cuda_kernel.o \
        poisson_stencil_kernel_cuda_kernel.o \
        poisson_update_kernel_cuda_kernel.o \
        poisson_error_kernel_cuda_kernel.o \
        -o poisson_cuda -lops_for_cuda

poisson_mpi: poisson_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) $(OPS_MPI) -c constants.F90 \
        ./MPI/poisson_initialguess_kernel_seq_kernel.F90 \
        ./MPI/poisson_populate_kernel_seq_kernel.F90 \
        ./MPI/poisson_stencil_kernel_seq_kernel.F90 \
        ./MPI/poisson_update_kernel_seq_kernel.F90 \
        ./MPI/poisson_error_kernel_seq_kernel.F90 \
        poisson_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(OPS_MPI) poisson_ops.o constants.o \
        poisson_populate_kernel_seq_kernel.o \
        poisson_initialguess_kernel_seq_kernel.o \
        poisson_stencil_kernel_seq_kernel.o \
        poisson_update_kernel_seq_kernel.o \
        poisson_error_kernel_seq_kernel.o \
	-o poisson_mpi -lops_for_mpi $(HDF5_LIB)

poisson_mpi_openmp: poisson_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(OMPFLAGS) $(FMODS) $(OPS_MPI) -c constants.F90 \
	./MPI_OpenMP/poisson_initialguess_kernel_omp_kernel.F90 \
	./MPI_OpenMP/poisson_populate_kernel_omp_kernel.F90 \
	./MPI_OpenMP/poisson_stencil_kernel_omp_kernel.F90 \
	./MPI_OpenMP/poisson_update_kernel_omp_kernel.F90 \
	./MPI_OpenMP/poisson_error_kernel_omp_kernel.F90 \
	poisson_ops.F90
	$(MPIF90) $(FFLAGS) $(OMPFLAGS) $(FLINK) $(OPS_MPI) poisson_ops.o constants.o \
	poisson_initialguess_kernel_omp_kernel.o \
	poisson_populate_kernel_omp_kernel.o \
	poisson_stencil_kernel_omp_kernel.o \
	poisson_update_kernel_omp_kernel.o \
	poisson_error_kernel_omp_kernel.o \
        -o poisson_mpi_openmp -lops_for_mpi $(HDF5_LIB)

poisson_mpi_cuda: poisson_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) $(OPS_MPI) $(CUDAFOR) -c constants.F90 \
	./CUDA/poisson_initialguess_kernel_cuda_kernel.CUF \
        ./CUDA/poisson_populate_kernel_cuda_kernel.CUF \
        ./CUDA/poisson_stencil_kernel_cuda_kernel.CUF \
        ./CUDA/poisson_update_kernel_cuda_kernel.CUF \
        ./CUDA/poisson_error_kernel_cuda_kernel.CUF \
        poisson_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(OPS_MPI) $(CUDAFOR) poisson_ops.o constants.o \
        poisson_initialguess_kernel_cuda_kernel.o \
        poisson_populate_kernel_cuda_kernel.o \
        poisson_stencil_kernel_cuda_kernel.o \
        poisson_update_kernel_cuda_kernel.o \
        poisson_error_kernel_cuda_kernel.o \
        -o poisson_mpi_cuda -lops_for_mpi_cuda $(HDF5_LIB)

poisson_openacc: poisson_ops.F90 constants.F90
	$(FC) $(OPT) $(FMODS_OPENACC) $(OpenACCFLAGS) -c constants.F90 \
        ./MPI_OpenACC/poisson_initialguess_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_populate_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_stencil_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_update_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_error_kernel_openacc_kernel.F90 \
        poisson_ops.F90
	$(FC) $(FFLAGS) $(CUDAFOR) $(FLINK) $(OpenACCFLAGS) poisson_ops.o constants.o \
        poisson_initialguess_kernel_openacc_kernel.o \
        poisson_populate_kernel_openacc_kernel.o \
        poisson_stencil_kernel_openacc_kernel.o \
        poisson_update_kernel_openacc_kernel.o \
        poisson_error_kernel_openacc_kernel.o \
        -o poisson_openacc -lops_for_cuda $(HDF5_LIB)

poisson_mpi_openacc: poisson_ops.F90 constants.F90
	$(MPIF90) $(OPT) $(FMODS_OPENACC) $(OpenACCFLAGS) $(OPS_MPI) -c constants.F90 \
        ./MPI_OpenACC/poisson_initialguess_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_populate_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_stencil_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_update_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_error_kernel_openacc_kernel.F90 \
        poisson_ops.F90
	$(MPIF90) $(FFLAGS) $(CUDAFOR) $(FLINK) $(OpenACCFLAGS) $(OPS_MPI) poisson_ops.o constants.o \
        poisson_initialguess_kernel_openacc_kernel.o \
        poisson_populate_kernel_openacc_kernel.o \
        poisson_stencil_kernel_openacc_kernel.o \
        poisson_update_kernel_openacc_kernel.o \
        poisson_error_kernel_openacc_kernel.o \
        -o poisson_mpi_openacc -lops_for_mpi_cuda $(HDF5_LIB)

clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f poisson_seq
	rm -f poisson_openmp
	rm -f poisson_cuda
	rm -f poisson_mpi
	rm -f poisson_mpi_openmp
	rm -f poisson_mpi_cuda
	rm -f poisson_openacc
	rm -f poisson_mpi_openacc
