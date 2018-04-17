#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

#
# set paths for header files and libraries
#
OPS_INSTALL_PATH := $(OPS_INSTALL_PATH)/c
OPS_INC   = -I$(OPS_INSTALL_PATH)/include
OPS_LIB   = -L$(OPS_INSTALL_PATH)/lib
CUDA_LIB = -L$(CUDA_INSTALL_PATH)/lib64
CUDA_INC = -I$(CUDA_INSTALL_PATH)/include

OPENCL_INC = -I$(OPENCL_INSTALL_PATH)/include
OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lOpenCL

ifeq ($(OPS_COMPILER),gnu)
  CPP   = g++
ifdef DEBUG
  CPPFLAGS  = -O0 -g
else
  CPPFLAGS  = -Ofast -fPIC
endif
  OMPFLAGS  = -fopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpiCC
  MPICC  = $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS  = $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),intel)
  CPP   = icpc
ifdef DEBUG
  CCFLAGS = -O0 -g -no-prec-div -openmp -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK #-DOPS_DEBUG
else
  #CCFLAGS  = -O3 -ipo -no-prec-div -fp-model strict -fp-model source -prec-div -prec-sqrt -vec-report2 -xHost -parallel #-DCOMM_PERF #-DDEBUG
  CCFLAGS = -O3 -ipo -no-prec-div -restrict -fno-alias -fp-model fast=2 -fma -qopt-report #-vec-report
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -qopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPICC  = $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPIFLAGS  = $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),mic)
  CPP   = icpc
ifdef DEBUG
  CCFLAGS = -O0 -g -mmic -DMPICH_IGNORE_CXX_SEEK#-vec-report
else
  #CCFLAGS  = -O3 -ipo -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK#-vec-report
  #CCFLAGS  = -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -finline -inline-forceinline -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK#-vec-report
  CCFLAGS = -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK -finline#-vec-report
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -openmp -openmp-report2
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPICC  = $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS  = $(CCFLAGS)
  OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lintelocl -lOpenCL #for offload opencl execution on the phi
else
ifeq ($(OPS_COMPILER),pgi)
  CPP     = pgc++
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
ifdef DEBUG
  CCFLAGS   = -O0 -g -Minline -Kieee -Minform=inform -Minfo=all
else
  CCFLAGS   = -O3 -Kieee -Minline -nofma #-Minform=severe -Minfo=all
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -mp=nonuma
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPICC   = $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS  = $(CPPFLAGS)
else
ifeq ($(OPS_COMPILER),cray)
  CPP       = CC
    CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
ifdef DEBUG
  CCFLAGS   = -O2
else
  CCFLAGS   = -O3
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  =
  MPICPP    = CC
  MPICC     = cc
  MPIFLAGS  = $(CPPFLAGS)
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif
endif
endif

ifdef HDF5_INSTALL_PATH
ifdef CHECKPOINTING
  HDF5_INC 	  	:= -DCHECKPOINTING -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB 	  	:= -L$(HDF5_INSTALL_PATH)/lib -lhdf5_hl -lhdf5 -lz
  CC = $(MPICC)
  CPP = $(MPICPP)
  OPS_INC += $(HDF5_INC)
endif
endif


# flags for nvcc
# set NV_ARCH to select the correct one
ifndef NV_ARCH
  MESSAGE=select an NVIDA device to compile in CUDA, e.g. make NV_ARCH=KEPLER
  NV_ARCH=Kepler
endif
ifeq ($(NV_ARCH),Fermi)
  CODE_GEN_CUDA=-gencode arch=compute_20,code=sm_21
else
ifeq ($(NV_ARCH),Kepler)
  CODE_GEN_CUDA=-gencode arch=compute_35,code=sm_35
endif
endif


ifdef DEBUG
  NVCC_FLAGS = -O0 -g -G
else
  NVCC_FLAGS = -O3
endif

ifdef CUDA_HOST_COMPILER
	NVCC_FLAGS += -ccbin=$(CUDA_HOST_COMPILER)
endif


#
# master to make all versions
#

TARGETS = laplace2d_dev_seq laplace2d_dev_mpi laplace2d_mpi laplace2d_mpi_openmp laplace2d_openmp laplace2d_seq laplace2d_mpi_inline laplace2d_tiled laplace2d_mpi_tiled laplace2d_cuda laplace2d_mpi_cuda laplace2d_opencl laplace2d_mpi_opencl

ifeq ($(OPS_COMPILER),pgi)
TARGETS += laplace2d_openacc laplace2d_mpi_openacc
else
TARGETS += openacc_error
endif

all: clean $(TARGETS)

laplace2d_dev_seq: Makefile laplace2d.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(CPP) $(CPPFLAGS) $(OPS_INC)  $(OPS_LIB) laplace2d.cpp -lops_seq $(HDF5_LIB) -o laplace2d_dev_seq


laplace2d_dev_mpi: Makefile laplace2d.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC)  $(OPS_LIB) laplace2d.cpp -lops_mpi $(HDF5_LIB) -o laplace2d_dev_mpi

#
# mpi version
#

SEQ_KERNELS = laplace2d_seq_kernels.cpp

laplace2d_mpi: Makefile laplace2d_ops.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC)  $(OPS_LIB) laplace2d_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_mpi $(HDF5_LIB) -o laplace2d_mpi

#
# OpenMP version
#
OMP_KERNELS = laplace2d_omp_kernels.cpp

laplace2d_mpi_openmp: Makefile laplace2d_ops.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(OMPFLAGS) $(MPIFLAGS) -DOPS_MPI $(OPS_INC)  $(OPS_LIB) laplace2d_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_mpi $(HDF5_LIB) -o laplace2d_mpi_openmp


laplace2d_openmp: Makefile laplace2d_ops.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(CPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC)  $(OPS_LIB) laplace2d_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq $(HDF5_LIB) -o laplace2d_openmp

#
# Sequential version
#
laplace2d_seq: Makefile laplace2d_ops.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(CPP) $(MPIFLAGS)  $(OPS_INC) $(OPS_LIB) laplace2d_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_seq $(HDF5_LIB) -o laplace2d_seq

#
# Tiled version
#
laplace2d_tiled: Makefile laplace2d_ops.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(CPP) $(MPIFLAGS) $(OMPFLAGS)  $(OPS_INC) $(OPS_LIB) laplace2d_ops.cpp -I. ./Tiled/$(SEQ_KERNELS) -lops_seq $(HDF5_LIB) -o laplace2d_tiled 

laplace2d_mpi_tiled: Makefile laplace2d_ops.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OMPFLAGS)  $(OPS_INC) $(OPS_LIB) laplace2d_ops.cpp -I. ./Tiled/$(SEQ_KERNELS) -lops_mpi $(HDF5_LIB) -o laplace2d_mpi_tiled 


#
# CUDA version
#

laplace2d_cuda: Makefile ./CUDA/laplace2d_kernels_cu.o laplace2d_ops.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	$(CPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC)  $(OPS_LIB) $(CUDA_LIB) laplace2d_ops.cpp ./CUDA/laplace2d_kernels_cu.o -lcudart -lops_cuda $(HDF5_LIB) -o laplace2d_cuda

laplace2d_mpi_cuda: Makefile ./CUDA/laplace2d_kernels_mpi_cu.o laplace2d_ops.cpp laplace_kernels.h $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC)  $(OPS_LIB) $(CUDA_LIB) laplace2d_ops.cpp ./CUDA/laplace2d_kernels_mpi_cu.o -lcudart -lops_mpi_cuda $(HDF5_LIB) -o laplace2d_mpi_cuda

./CUDA/laplace2d_kernels_cu.o: ./CUDA/laplace2d_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC)  -I. -c -o ./CUDA/laplace2d_kernels_cu.o ./CUDA/laplace2d_kernels.cu

./CUDA/laplace2d_kernels_mpi_cu.o: ./CUDA/laplace2d_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI $(CODE_GEN_CUDA) $(OPS_INC)  -I. -I$(MPI_INSTALL_PATH)/include \
	-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/laplace2d_kernels_mpi_cu.o ./CUDA/laplace2d_kernels.cu

mpiinline_obj_list = $(shell find MPI_inline/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
mpiinline_c_obj_list = $(shell find MPI_inline/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")


./MPI_inline/laplace2d_kernels.o: ./MPI_inline/laplace2d_kernels.cpp ./MPI_inline/laplace2d_common.h Makefile
				$(MPICPP) $(MPIFLAGS) $(OMPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC)  \
 			       -I. MPI_inline/laplace2d_kernels.cpp -c -o MPI_inline/laplace2d_kernels.o

./MPI_inline/%.o: ./MPI_inline/%.cpp ./MPI_inline/%_c.c ./MPI_inline/laplace2d_common.h Makefile
				@echo "Compiling kernel" $*
				$(MPICC) $(MPIFLAGS) $(OMPFLAGS) -DOPS_MPI  $(OPS_INC) $(HDF5_INC) \
			        -I. MPI_inline/$*_c.c -c -o MPI_inline/$*_c.o
				$(MPICPP) $(MPIFLAGS) $(OMPFLAGS) -DOPS_MPI  $(OPS_INC) $(HDF5_INC) \
				-I. MPI_inline/$*.cpp -c -o MPI_inline/$*.o

laplace2d_mpi_inline: $(mpiinline_obj_list) laplace2d_ops.cpp \
                Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	        $(MPICPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB)  -DOPS_MPI \
		laplace2d_ops.cpp -I. $(mpiinline_obj_list) $(mpiinline_c_obj_list) -lops_mpi $(HDF5_LIB) -o laplace2d_mpi_inline


openacc_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
openacc_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")

./OpenACC/laplace2d_kernels.o: ./OpenACC/laplace2d_kernels.cpp Makefile
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC)  $(CUDA_INC) \
	-I. OpenACC/laplace2d_kernels.cpp -c -o OpenACC/laplace2d_kernels.o

./OpenACC/%.o: ./OpenACC/%.cpp ./OpenACC/%_c.c Makefile
	@echo "Compiling kernel" $*
	$(MPICC) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC)  $(CUDA_INC) \
              -I. OpenACC/$*_c.c -c -o OpenACC/$*_c.o
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC)  $(CUDA_INC) \
        -I. OpenACC/$*.cpp -c -o OpenACC/$*.o

laplace2d_openacc: $(openacc_obj_list) laplace2d_ops.cpp laplace_kernels.h Makefile $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(OPS_INC)  $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) \
    laplace2d_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_cuda $(HDF5_LIB) -o laplace2d_openacc

laplace2d_mpi_openacc: $(openacc_obj_list) laplace2d_ops.cpp laplace_kernels.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(OPS_INC)  $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) -DOPS_MPI \
    laplace2d_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_mpi_cuda $(HDF5_LIB) -o laplace2d_mpi_openacc

laplace2d_opencl: ./OpenCL/laplace2d_opencl_kernels.o laplace2d_ops.cpp laplace_kernels.h Makefile $(OPS_INSTALL_PATH)/lib/libops_opencl.a
	$(CPP) $(CPPFLAGS) $(OPS_INC)  $(OPS_LIB) $(OPENCL_LIB) laplace2d_ops.cpp ./OpenCL/laplace2d_opencl_kernels.o -I. -lops_opencl $(HDF5_LIB) -o laplace2d_opencl

./OpenCL/laplace2d_opencl_kernels.o: ./OpenCL/laplace2d_opencl_kernels.cpp Makefile
	$(MPICPP) $(CPPFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC)  -I. \
                -c -o ./OpenCL/laplace2d_opencl_kernels.o ./OpenCL/laplace2d_opencl_kernels.cpp

laplace2d_mpi_opencl: ./OpenCL/laplace2d_mpi_opencl_kernels.o laplace2d_ops.cpp laplace_kernels.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_opencl.a
	$(MPICPP) $(MPIFLAGS) $(OPS_INC)  $(OPS_LIB) $(OPENCL_LIB) -DOPS_MPI laplace2d_ops.cpp ./OpenCL/laplace2d_mpi_opencl_kernels.o -lops_mpi_opencl $(HDF5_LIB) -o laplace2d_mpi_opencl

./OpenCL/laplace2d_mpi_opencl_kernels.o: ./OpenCL/laplace2d_opencl_kernels.cpp Makefile
	$(MPICPP) $(MPIFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC)  -I. -DOPS_MPI\
                -c -o ./OpenCL/laplace2d_mpi_opencl_kernels.o ./OpenCL/laplace2d_opencl_kernels.cpp
#
#error
#
openacc_error:
	@echo "laplace2d_mpi_openacc cannot be built with $(OPS_COMPILER) compilers .. requires an OpenACC compiler (e.g. pgi)"


#
# cleanup
#

clean:
	rm -f laplace2d_dev_seq laplace2d_dev_mpi laplace2d_mpi laplace2d_seq laplace2d_mpi_inline laplace2d_tiled laplace2d_mpi_tiled laplace2d_openmp laplace2d_mpi_openmp laplace2d_cuda laplace2d_mpi_cuda laplace2d_openacc laplace2d_mpi_openacc ./CUDA/*.o ./OpenACC/*.o *.o laplace2d_opencl laplace2d_mpi_opencl ./OpenCL/*.o ./MPI_inline/*.o  *.o
