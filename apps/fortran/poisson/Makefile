OPS = $(OPS_INSTALL_PATH)
F_OPS = $(OPS)/fortran
F_OPS_LIB = $(F_OPS)/lib
F_OPS_MOD = $(F_OPS)/mod/$(OPS_COMPILER)
F_OPS_INC = $(F_OPS)/include

# flags for nvcc
# set NV_ARCH to select the correct one
ifndef NV_ARCH
  MESSAGE=select an NVIDA device to compile in CUDA, e.g. make NV_ARCH=KEPLER
  NV_ARCH=Kepler
endif
ifeq ($(NV_ARCH),Fermi)
  CODE_GEN_CUDA=-gencode arch=compute_20,code=sm_21
  PGI_CUDA_FORT_FLAGS=-Mcuda=cuda6.5,cc20
else
ifeq ($(NV_ARCH),Kepler)
  CODE_GEN_CUDA=-gencode arch=compute_35,code=sm_35
  PGI_CUDA_FORT_FLAGS=-Mcuda=cuda7.5,cc35
endif
endif

ifdef HDF5_INSTALL_PATH
  HDF5_INC              := -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB              := -L$(HDF5_INSTALL_PATH)/lib -lhdf5_hl -lhdf5 -lz
ifdef CHECKPOINTING
  HDF5_INC              := $(HDF5_INC) -DCHECKPOINTING
endif
endif


ifeq ($(OPS_COMPILER),intel)
  CC 		= icc -std=c99
ifdef DEBUG
  OPT	 	= -O0 -g
else
  OPT	 	= -O3 -g -xAVX -vec-report2
endif
  CCFLAGS       = $(OPT) -Wall -Werror -wd981
  CXX 		= icpc
  CXXFLAGS 	= $(CCFLAGS)
  CXXLINK	= -lstdc++
  FC		= ifort
  OMPFLAGS 	= -openmp -openmp-report2
  FFLAGS	= $(OPT)
  MPICC 	= $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX 	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIF90 	= $(MPI_INSTALL_PATH)/bin/mpif90
  MPIFLAGS 	= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  TARGETS       = clean poisson_seq poisson_openmp poisson_mpi poisson_mpi_openmp
  OPS_MPI	= -DOPS_MPI
else
ifeq ($(OPS_COMPILER),pgi)
ifdef DEBUG
  OPT           = -O0 -g -Minline -Kieee -Minform=inform -Minfo=all
else
  OPT           = -O3 -Kieee -Minline
endif
  CCFLAGS       = $(OPT) -Wall -Werror -wd981
  CXX           = pgc++
  CXXFLAGS      = $(CCFLAGS)
  CUDA_ALIGN_FLAG = -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CXXLINK       = -lstdc++ -pgcpplibs -lgcc_s
  OMPFLAGS 	= -mp
  OpenACCFLAGS  := -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 #-Minline
  FC            = pgfortran -fast $(PGI_CUDA_FORT_FLAGS) #,fastmath,ptxinfo
  FFLAGS        = $(OPT) $(PGFLAGS) -ldl
  MPICC         = $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX        = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIF90        = $(MPI_INSTALL_PATH)/bin/mpif90 $(PGI_CUDA_FORT_FLAGS)
  MPIFLAGS      = $(CXXFLAGS)
  TARGETS       = clean poisson_seq poisson_openmp poisson_cuda poisson_mpi poisson_mpi_openmp poisson_mpi_cuda
  OPS_MPI	= -DOPS_MPI
  CUDAFOR	= -DOPS_WITH_CUDAFOR
else
ifeq ($(OPS_COMPILER),xl)
ifdef DEBUG
  OPT           = -O0 -g
else
  OPT           = -O3 -qarch=pwr8 -qtune=pwr8
endif
  CCFLAGS       = $(OPT)
  CXX           = xlc++_r
  CXXFLAGS      = $(CCFLAGS)
  CUDA_ALIGN_FLAG = #-D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CXXLINK       = -lstdc++
  OMPFLAGS 	= -qsmp=omp -qthreaded
  FC            = xlf_r
  FFLAGS        = $(OPT) -ldl -libmc++
  MPICC         = $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX        = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIF90        = $(MPI_INSTALL_PATH)/bin/mpif90
  MPIFLAGS      = $(CXXFLAGS)
  TARGETS       = clean poisson_seq poisson_openmp poisson_cuda poisson_mpi poisson_mpi_openmp poisson_mpi_cuda
  OPS_MPI	= -WF,-DOPS_MPI
  CUDAFOR	= -WF,-DOPS_WITH_CUDAFOR -qcuda -qthreaded
else

print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif

FLINK = -L$(F_OPS_LIB) $(CXXLINK)
FMODS = -I$(F_OPS_MOD) -I$(F_OPS_INC)
FMODS_OPENACC = -I$(F_OPS_MOD)/openacc -I$(F_OPS_INC)

all: $(TARGETS)

poisson_seq: poisson_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) -c constants.F90 \
	./MPI/poisson_initialguess_kernel_seq_kernel.F90 \
	./MPI/poisson_populate_kernel_seq_kernel.F90 \
	./MPI/poisson_stencil_kernel_seq_kernel.F90 \
	./MPI/poisson_update_kernel_seq_kernel.F90 \
	./MPI/poisson_error_kernel_seq_kernel.F90 \
	poisson_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) poisson_ops.o constants.o \
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
ifeq ($(OPS_COMPILER),pgi)
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
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif

poisson_openacc: poisson_ops.F90 constants.F90
	$(MPIF90) $(OPT) $(FMODS_OPENACC) $(OpenACCFLAGS) -c constants.F90 \
        ./MPI_OpenACC/poisson_initialguess_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_populate_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_stencil_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_update_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/poisson_error_kernel_openacc_kernel.F90 \
        poisson_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(OpenACCFLAGS) poisson_ops.o constants.o \
        poisson_initialguess_kernel_openacc_kernel.o \
        poisson_populate_kernel_openacc_kernel.o \
        poisson_stencil_kernel_openacc_kernel.o \
        poisson_update_kernel_openacc_kernel.o \
        poisson_error_kernel_openacc_kernel.o \
        -o poisson_openacc -lops_for_openacc $(HDF5_LIB)


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
