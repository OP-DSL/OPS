OPS = $(OPS_INSTALL_PATH)
F_OPS = $(OPS)/fortran
F_OPS_LIB = $(F_OPS)/lib
F_OPS_MOD = $(F_OPS)/mod/$(OPS_COMPILER)
F_OPS_INC = $(F_OPS)/include

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
  MPIFLAGS 	= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  TARGETS	= clean multidim_seq
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
  FC            = pgfortran
  PGFFLAGS	= -fast -Mcuda=cuda5.5,cc35,fastmath,ptxinfo -DOPS_WITH_CUDAFOR
  FFLAGS        = $(OPT) $(PGFLAGS)
  MPICC         = $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX        = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS      = $(CXXFLAGS)
  TARGETS       = clean multidim_seq multidim_cuda
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif

FLINK = -L$(F_OPS_LIB) $(CXXLINK)
FMODS = -I$(F_OPS_MOD) -I$(F_OPS_INC)

all: $(TARGETS)

multidim_seq: multidim_ops.F90 constants.F90
	$(FC) $(FFLAGS) $(FMODS) -c constants.F90 ./MPI/multidim_print_kernel_seq_kernel.F90 \
	./MPI/multidim_copy_kernel_seq_kernel.F90 ./MPI/multidim_kernel_seq_kernel.F90 ./MPI/multidim_reduce_kernel_seq_kernel.F90 multidim_ops.F90
	$(FC) $(FFLAGS) $(FLINK) multidim_ops.o constants.o multidim_kernel_seq_kernel.o multidim_copy_kernel_seq_kernel.o \
	multidim_print_kernel_seq_kernel.o multidim_reduce_kernel_seq_kernel.o -o multidim_seq -lops_for_seq

multidim_cuda: multidim_ops.F90 constants.F90
ifeq ($(OPS_COMPILER),pgi)
	$(FC) $(FFLAGS) $(PGFFLAGS) $(FMODS) $(CUDA_ALIGN_FLAG) -DOPS_WITH_CUDAFOR -c constants.F90 ./CUDA/multidim_print_kernel_cuda_kernel.CUF \
	./CUDA/multidim_copy_kernel_cuda_kernel.CUF ./CUDA/multidim_kernel_cuda_kernel.CUF ./CUDA/multidim_reduce_kernel_cuda_kernel.CUF multidim_ops.F90
	$(FC) $(FFLAGS) $(PGFFLAGS) $(FLINK) $(CUDA_ALIGN_FLAG) -DOPS_WITH_CUDAFOR multidim_ops.o constants.o multidim_kernel_cuda_kernel.o \
	multidim_copy_kernel_cuda_kernel.o multidim_print_kernel_cuda_kernel.o \
	multidim_reduce_kernel_cuda_kernel.o -o multidim_cuda -lops_for_cuda
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif

clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f multidim_seq
