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
  OPT	 	= -O3 -xAVX -vec-report2
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
  MPIF90        = $(MPI_INSTALL_PATH)/bin/mpif90
  MPIFLAGS 	= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  TARGETS	= clean mblock_seq mblock_openmp mblock_mpi mblock_mpi_openmp
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
  CXXLINK       = -lstdc++ -pgcpplibs
  OMPFLAGS 	= -mp
  FC            = pgfortran -fast $(PGI_CUDA_FORT_FLAGS) #,fastmath,ptxinfo
  FFLAGS        = $(OPT) $(PGFLAGS) -ldl
  MPICC         = $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX        = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIF90        = $(MPI_INSTALL_PATH)/bin/mpif90 $(PGI_CUDA_FORT_FLAGS)
  MPIFLAGS      = $(CXXFLAGS)
  TARGETS       = clean mblock_seq mblock_openmp mblock_cuda mblock_mpi mblock_mpi_openmp mblock_mpi_cuda
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif

FLINK = -L$(F_OPS_LIB) $(CXXLINK)
FMODS = -I$(F_OPS_MOD) -I$(F_OPS_INC)

all: $(TARGETS)

.ops_generated:
	$(OPS_INSTALL_PATH)/../translator/python/fortran/ops_fortran.py mblock.F90
	touch .ops_generated

mblock_seq: .ops_generated mblock_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(FMODS) -c constants.F90 ./MPI/mblock_populate_kernel_seq_kernel.F90 \
	mblock_ops.F90
	$(FC) $(FFLAGS) $(FLINK) mblock_ops.o constants.o mblock_populate_kernel_seq_kernel.o \
	-o mblock_seq -lops_for_seq

mblock_openmp: .ops_generated mblock_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(FMODS) $(OMPFLAGS) -c constants.F90 ./MPI_OpenMP/mblock_populate_kernel_omp_kernel.F90 \
        mblock_ops.F90
	$(FC) $(FFLAGS) $(FLINK) $(OMPFLAGS) mblock_ops.o constants.o mblock_populate_kernel_omp_kernel.o \
        -o mblock_openmp -lops_for_seq

mblock_cuda: .ops_generated mblock_ops.F90 constants.F90
ifeq ($(OPS_COMPILER),pgi)
	$(FC) $(FFLAGS) $(FMODS) -DOPS_WITH_CUDAFOR -c constants.F90 ./CUDA/mblock_populate_kernel_cuda_kernel.CUF \
        mblock_ops.F90
	$(FC) $(FFLAGS) $(FLINK) -DOPS_WITH_CUDAFOR mblock_ops.o constants.o mblock_populate_kernel_cuda_kernel.o \
        -o mblock_cuda -lops_for_cuda
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif

mblock_mpi: .ops_generated mblock_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) -c -DOPS_MPI constants.F90 ./MPI/mblock_populate_kernel_seq_kernel.F90 \
        mblock_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) -DOPS_MPI mblock_ops.o constants.o mblock_populate_kernel_seq_kernel.o \
        -o mblock_mpi -lops_for_mpi $(HDF5_LIB)

mblock_mpi_openmp: .ops_generated mblock_ops.F90 constants.F90
	$(MPIF90) $(FFLAGS) $(FMODS) $(OMPFLAGS) -c -DOPS_MPI constants.F90 ./MPI_OpenMP/mblock_populate_kernel_omp_kernel.F90 \
        mblock_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) $(OMPFLAGS) -DOPS_MPI mblock_ops.o constants.o mblock_populate_kernel_omp_kernel.o \
        -o mblock_mpi_openmp -lops_for_mpi $(HDF5_LIB)

mblock_mpi_cuda: .ops_generated mblock_ops.F90 constants.F90
ifeq ($(OPS_COMPILER),pgi)
	$(MPIF90) $(FFLAGS) $(FMODS) -DOPS_WITH_CUDAFOR -DOPS_MPI -c constants.F90 ./CUDA/mblock_populate_kernel_cuda_kernel.CUF \
        mblock_ops.F90
	$(MPIF90) $(FFLAGS) $(FLINK) -DOPS_WITH_CUDAFOR -DOPS_MPI mblock_ops.o constants.o mblock_populate_kernel_cuda_kernel.o \
        -o mblock_mpi_cuda -lops_for_mpi_cuda $(HDF5_LIB)
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif

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
	rm -rf CUDA MPI MPI_OpenMP *_ops.F90 .ops_generated MPI_OpenACC 
