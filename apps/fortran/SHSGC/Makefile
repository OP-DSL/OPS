OPS = $(OPS_INSTALL_PATH)
F_OPS = $(OPS)/fortran
F_OPS_LIB = $(F_OPS)/lib
F_OPS_MOD = $(F_OPS)/mod/$(OPS_COMPILER)
F_OPS_INC = $(F_OPS)/include

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
  #OPT          := -O2 -g
  OPT           := -O0 -g -Minline -Kieee -Minform=inform -Minfo=all
else
  OPT           := -O3 -Kieee -Minline
endif
  OMPFLAGS      := -mp
  CCFLAGS       := $(OPT) -Wall -Werror -wd981
  CXX           := pgc++
  CXXFLAGS      := $(CCFLAGS)
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CXXLINK       := -lstdc++ #-pgcpplibs
  FC            := pgfortran
  FFLAGS        := $(OPT)
  MPICC         := $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPICXX        := $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIF90 	:= $(MPI_INSTALL_PATH)/bin/mpif90
  MPIFLAGS      := $(CXXFLAGS)
  TARGETS       := clean shsgc_seq shsgc_mpi shsgc_openmp shsgc_mpi_openmp
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


clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f shsgc_seq
	rm -f shsgc_mpi
	rm -f shsgc_openmp
	rm -f shsgc_mpi_openmp
