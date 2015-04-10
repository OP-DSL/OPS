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
  PGI_CUDA_FORT_FLAGS=-Mcuda=cuda6.0,cc35
endif
endif


ifeq ($(OPS_COMPILER),intel)
  CC 		:= icc -std=c99
ifdef DEBUG
  #OPT	 	:= -O3 -g -xAVX -vec-report2 #-xAVX
  OPT	 	:= -O0 -g
else
  OPT	 	:= -O3 -xAVX -vec-report2 #-xAVX
endif
  OMPFLAGS  	= -openmp -openmp-report2
  CCFLAGS       := $(OPT) -Wall -Werror -wd981
  CXX 		:= icpc
  CXXFLAGS 	:= $(CCFLAGS)
  CXXLINK	:= -lstdc++
  FC		:= ifort
  FFLAGS	:= $(OPT)
  MPICC 	:= $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX 	:= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIF90 	:= $(MPI_INSTALL_PATH)/bin/mpif90
  MPIFLAGS 	:= $(CXXFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
  TARGETS	:= clean shsgc_seq shsgc_mpi shsgc_openmp shsgc_mpi_openmp
else
ifeq ($(OPS_COMPILER),pgi)
ifdef DEBUG
  OPT           := -O0 -g -Minline -Kieee #-Minform=inform -Minfo=all
else
  OPT           := -O3 -Kieee -Minline -Minform=inform -Minfo=all
endif
  OMPFLAGS      := -mp
  OpenACCFLAGS	:= -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 #-Minline
  CCFLAGS       := $(OPT) -Wall -Werror -wd981
  CXX           := pgc++
  CXXFLAGS      := $(CCFLAGS)
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CXXLINK       := -lstdc++ -lgcc_s #-pgcpplibs
  FC            := pgfortran $(PGI_CUDA_FORT_FLAGS)
  FFLAGS        := $(OPT)
  MPICC         := $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX        := $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIF90 	:= $(MPI_INSTALL_PATH)/bin/mpif90 $(PGI_CUDA_FORT_FLAGS)
  MPIFLAGS      := $(CXXFLAGS)
  TARGETS       := clean shsgc_seq shsgc_mpi shsgc_openmp shsgc_mpi_openmp shsgc_cuda shsgc_mpi_cuda shsgc_openacc
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif

FLINK = -L$(F_OPS_LIB) $(CXXLINK)
FMODS = -I$(F_OPS_MOD) -I$(F_OPS_INC)

all: $(TARGETS)

shsgc_seq: shsgc_ops.F90 constants.F90
	$(FC) $(OPT) $(FMODS) -c constants.F90 ./MPI/initialize_kernel_seq_kernel.F90 \
	./MPI/drhouupdx_kernel_seq_kernel.F90 ./MPI/zerores_kernel_seq_kernel.F90 \
	./MPI/drhoudx_kernel_seq_kernel.F90 ./MPI/save_kernel_seq_kernel.F90 \
        ./MPI/drhoEpudx_kernel_seq_kernel.F90 ./MPI/updateRK3_kernel_seq_kernel.F90 \
	./MPI/Riemann_kernel_seq_kernel.F90 ./MPI/limiter_kernel_seq_kernel.F90 \
	./MPI/tvd_kernel_seq_kernel.F90 ./MPI/vars_kernel_seq_kernel.F90 \
	./MPI/calupwindeff_kernel_seq_kernel.F90 ./MPI/fact_kernel_seq_kernel.F90 \
	./MPI/update_kernel_seq_kernel.F90 ./MPI/test_kernel_seq_kernel.F90 shsgc_ops.F90
	$(FC) $(OPT) $(FLINK) shsgc_ops.o constants.o initialize_kernel_seq_kernel.o \
	drhouupdx_kernel_seq_kernel.o zerores_kernel_seq_kernel.o save_kernel_seq_kernel.o \
	drhoudx_kernel_seq_kernel.o drhoEpudx_kernel_seq_kernel.o updateRK3_kernel_seq_kernel.o \
	Riemann_kernel_seq_kernel.o limiter_kernel_seq_kernel.o tvd_kernel_seq_kernel.o \
	vars_kernel_seq_kernel.o calupwindeff_kernel_seq_kernel.o \
	fact_kernel_seq_kernel.o update_kernel_seq_kernel.o test_kernel_seq_kernel.o \
	-o shsgc_seq -lops_for_seq

shsgc_openmp: shsgc_ops.F90 constants.F90
	$(FC) $(OPT) $(FMODS) $(OMPFLAGS) -c constants.F90 ./MPI_OpenMP/initialize_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/drhouupdx_kernel_omp_kernel.F90 ./MPI_OpenMP/zerores_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/drhoudx_kernel_omp_kernel.F90 ./MPI_OpenMP/save_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/drhoEpudx_kernel_omp_kernel.F90 ./MPI_OpenMP/updateRK3_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/Riemann_kernel_omp_kernel.F90 ./MPI_OpenMP/limiter_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/tvd_kernel_omp_kernel.F90 ./MPI_OpenMP/vars_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/calupwindeff_kernel_omp_kernel.F90 ./MPI_OpenMP/fact_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/update_kernel_omp_kernel.F90 ./MPI_OpenMP/test_kernel_omp_kernel.F90 shsgc_ops.F90
	$(FC) $(OPT) $(FLINK) $(OMPFLAGS) shsgc_ops.o constants.o initialize_kernel_omp_kernel.o \
        drhouupdx_kernel_omp_kernel.o zerores_kernel_omp_kernel.o save_kernel_omp_kernel.o \
        drhoudx_kernel_omp_kernel.o drhoEpudx_kernel_omp_kernel.o updateRK3_kernel_omp_kernel.o \
        Riemann_kernel_omp_kernel.o limiter_kernel_omp_kernel.o tvd_kernel_omp_kernel.o \
        vars_kernel_omp_kernel.o calupwindeff_kernel_omp_kernel.o \
        fact_kernel_omp_kernel.o update_kernel_omp_kernel.o test_kernel_omp_kernel.o \
        -o shsgc_openmp -lops_for_seq

shsgc_cuda: shsgc_ops.F90 constants.F90
ifeq ($(OPS_COMPILER),pgi)
	$(FC) $(OPT) $(FMODS) -DOPS_WITH_CUDAFOR -c constants.F90 ./CUDA/initialize_kernel_cuda_kernel.CUF \
        ./CUDA/drhouupdx_kernel_cuda_kernel.CUF ./CUDA/zerores_kernel_cuda_kernel.CUF \
        ./CUDA/drhoudx_kernel_cuda_kernel.CUF ./CUDA/save_kernel_cuda_kernel.CUF \
        ./CUDA/drhoEpudx_kernel_cuda_kernel.CUF ./CUDA/updateRK3_kernel_cuda_kernel.CUF \
        ./CUDA/Riemann_kernel_cuda_kernel.CUF ./CUDA/limiter_kernel_cuda_kernel.CUF \
        ./CUDA/tvd_kernel_cuda_kernel.CUF ./CUDA/vars_kernel_cuda_kernel.CUF \
        ./CUDA/calupwindeff_kernel_cuda_kernel.CUF ./CUDA/fact_kernel_cuda_kernel.CUF \
        ./CUDA/update_kernel_cuda_kernel.CUF ./CUDA/test_kernel_cuda_kernel.CUF shsgc_ops.F90
	$(FC) $(OPT) $(FLINK) -DOPS_WITH_CUDAFOR shsgc_ops.o constants.o initialize_kernel_cuda_kernel.o \
        drhouupdx_kernel_cuda_kernel.o zerores_kernel_cuda_kernel.o save_kernel_cuda_kernel.o \
        drhoudx_kernel_cuda_kernel.o drhoEpudx_kernel_cuda_kernel.o updateRK3_kernel_cuda_kernel.o \
        Riemann_kernel_cuda_kernel.o limiter_kernel_cuda_kernel.o tvd_kernel_cuda_kernel.o \
        vars_kernel_cuda_kernel.o calupwindeff_kernel_cuda_kernel.o \
        fact_kernel_cuda_kernel.o update_kernel_cuda_kernel.o test_kernel_cuda_kernel.o \
        -o shsgc_cuda -lops_for_cuda
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif

shsgc_mpi: shsgc_ops.F90 constants.F90
	$(MPIF90) $(OPT) $(FMODS) -DOPS_MPI -c constants.F90 ./MPI/initialize_kernel_seq_kernel.F90 \
	./MPI/drhouupdx_kernel_seq_kernel.F90 ./MPI/zerores_kernel_seq_kernel.F90 \
	./MPI/drhoudx_kernel_seq_kernel.F90 ./MPI/save_kernel_seq_kernel.F90 \
        ./MPI/drhoEpudx_kernel_seq_kernel.F90 ./MPI/updateRK3_kernel_seq_kernel.F90 \
	./MPI/Riemann_kernel_seq_kernel.F90 ./MPI/limiter_kernel_seq_kernel.F90 \
	./MPI/tvd_kernel_seq_kernel.F90 ./MPI/vars_kernel_seq_kernel.F90 \
	./MPI/calupwindeff_kernel_seq_kernel.F90 ./MPI/fact_kernel_seq_kernel.F90 \
	./MPI/update_kernel_seq_kernel.F90 ./MPI/test_kernel_seq_kernel.F90 shsgc_ops.F90
	$(MPIF90) $(OPT) $(FLINK) -DOPS_MPI shsgc_ops.o constants.o initialize_kernel_seq_kernel.o \
	drhouupdx_kernel_seq_kernel.o zerores_kernel_seq_kernel.o save_kernel_seq_kernel.o \
	drhoudx_kernel_seq_kernel.o drhoEpudx_kernel_seq_kernel.o updateRK3_kernel_seq_kernel.o \
	Riemann_kernel_seq_kernel.o limiter_kernel_seq_kernel.o tvd_kernel_seq_kernel.o \
	vars_kernel_seq_kernel.o calupwindeff_kernel_seq_kernel.o \
	fact_kernel_seq_kernel.o update_kernel_seq_kernel.o test_kernel_seq_kernel.o \
	-o shsgc_mpi -lops_for_mpi

shsgc_mpi_openmp: shsgc_ops.F90 constants.F90
	$(MPIF90) $(OPT) $(FMODS) $(OMPFLAGS) -DOPS_MPI -c constants.F90 ./MPI_OpenMP/initialize_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/drhouupdx_kernel_omp_kernel.F90 ./MPI_OpenMP/zerores_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/drhoudx_kernel_omp_kernel.F90 ./MPI_OpenMP/save_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/drhoEpudx_kernel_omp_kernel.F90 ./MPI_OpenMP/updateRK3_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/Riemann_kernel_omp_kernel.F90 ./MPI_OpenMP/limiter_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/tvd_kernel_omp_kernel.F90 ./MPI_OpenMP/vars_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/calupwindeff_kernel_omp_kernel.F90 ./MPI_OpenMP/fact_kernel_omp_kernel.F90 \
        ./MPI_OpenMP/update_kernel_omp_kernel.F90 ./MPI_OpenMP/test_kernel_omp_kernel.F90 shsgc_ops.F90
	$(MPIF90) $(OPT) $(FLINK) $(OMPFLAGS) -DOPS_MPI shsgc_ops.o constants.o initialize_kernel_omp_kernel.o \
        drhouupdx_kernel_omp_kernel.o zerores_kernel_omp_kernel.o save_kernel_omp_kernel.o \
        drhoudx_kernel_omp_kernel.o drhoEpudx_kernel_omp_kernel.o updateRK3_kernel_omp_kernel.o \
        Riemann_kernel_omp_kernel.o limiter_kernel_omp_kernel.o tvd_kernel_omp_kernel.o \
        vars_kernel_omp_kernel.o calupwindeff_kernel_omp_kernel.o \
        fact_kernel_omp_kernel.o update_kernel_omp_kernel.o test_kernel_omp_kernel.o \
        -o shsgc_mpi_openmp -lops_for_mpi

shsgc_mpi_cuda: shsgc_ops.F90 constants.F90
ifeq ($(OPS_COMPILER),pgi)
	$(MPIF90) $(OPT) $(FMODS) -DOPS_MPI -DOPS_WITH_CUDAFOR -c constants.F90 ./CUDA/initialize_kernel_cuda_kernel.CUF \
        ./CUDA/drhouupdx_kernel_cuda_kernel.CUF ./CUDA/zerores_kernel_cuda_kernel.CUF \
        ./CUDA/drhoudx_kernel_cuda_kernel.CUF ./CUDA/save_kernel_cuda_kernel.CUF \
        ./CUDA/drhoEpudx_kernel_cuda_kernel.CUF ./CUDA/updateRK3_kernel_cuda_kernel.CUF \
        ./CUDA/Riemann_kernel_cuda_kernel.CUF ./CUDA/limiter_kernel_cuda_kernel.CUF \
        ./CUDA/tvd_kernel_cuda_kernel.CUF ./CUDA/vars_kernel_cuda_kernel.CUF \
        ./CUDA/calupwindeff_kernel_cuda_kernel.CUF ./CUDA/fact_kernel_cuda_kernel.CUF \
        ./CUDA/update_kernel_cuda_kernel.CUF ./CUDA/test_kernel_cuda_kernel.CUF shsgc_ops.F90
	$(MPIF90) $(OPT) $(FLINK) -DOPS_MPI -DOPS_WITH_CUDAFOR shsgc_ops.o constants.o initialize_kernel_cuda_kernel.o \
        drhouupdx_kernel_cuda_kernel.o zerores_kernel_cuda_kernel.o save_kernel_cuda_kernel.o \
        drhoudx_kernel_cuda_kernel.o drhoEpudx_kernel_cuda_kernel.o updateRK3_kernel_cuda_kernel.o \
        Riemann_kernel_cuda_kernel.o limiter_kernel_cuda_kernel.o tvd_kernel_cuda_kernel.o \
        vars_kernel_cuda_kernel.o calupwindeff_kernel_cuda_kernel.o \
        fact_kernel_cuda_kernel.o update_kernel_cuda_kernel.o test_kernel_cuda_kernel.o \
        -o shsgc_mpi_cuda -lops_for_mpi_cuda
else
	@echo "cuda cannot be built with $(OPS_COMPILER) compilers .. requires a CUDA FORTRAN compiler (e.g. pgi)"
endif


shsgc_openacc: shsgc_ops.F90 constants.F90
	$(FC) $(OPT) $(FMODS) $(OpenACCFLAGS) -c constants.F90 ./MPI_OpenACC/initialize_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/drhouupdx_kernel_openacc_kernel.F90 ./MPI_OpenACC/zerores_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/drhoudx_kernel_openacc_kernel.F90 ./MPI_OpenACC/save_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/drhoEpudx_kernel_openacc_kernel.F90 ./MPI_OpenACC/updateRK3_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/Riemann_kernel_openacc_kernel.F90 ./MPI_OpenACC/limiter_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/tvd_kernel_openacc_kernel.F90 ./MPI_OpenACC/vars_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/calupwindeff_kernel_openacc_kernel.F90 ./MPI_OpenACC/fact_kernel_openacc_kernel.F90 \
        ./MPI_OpenACC/update_kernel_openacc_kernel.F90 ./MPI_OpenACC/test_kernel_openacc_kernel.F90 shsgc_ops.F90
	$(FC) $(OPT) $(FLINK)  $(OpenACCFLAGS) shsgc_ops.o constants.o initialize_kernel_openacc_kernel.o \
        drhouupdx_kernel_openacc_kernel.o zerores_kernel_openacc_kernel.o save_kernel_openacc_kernel.o \
        drhoudx_kernel_openacc_kernel.o drhoEpudx_kernel_openacc_kernel.o updateRK3_kernel_openacc_kernel.o \
        Riemann_kernel_openacc_kernel.o limiter_kernel_openacc_kernel.o tvd_kernel_openacc_kernel.o \
        vars_kernel_openacc_kernel.o calupwindeff_kernel_openacc_kernel.o \
        fact_kernel_openacc_kernel.o update_kernel_openacc_kernel.o test_kernel_openacc_kernel.o \
        -o shsgc_openacc -lops_for_openacc

clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f shsgc_seq
	rm -f shsgc_mpi
	rm -f shsgc_openmp
	rm -f shsgc_mpi_openmp
	rm -f shsgc_cuda
	rm -f shsgc_mpi_cuda
	rm -f shsgc_openacc
