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

ifdef HDF5_INSTALL_PATH
  HDF5_INC 	  	:= -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB 	  	:= -L$(HDF5_INSTALL_PATH)/lib -lhdf5_hl -lhdf5 -lz
ifdef CHECKPOINTING
  HDF5_INC              := $(HDF5_INC) -DCHECKPOINTING
endif
endif

ifeq ($(OPS_COMPILER),gnu)
  CPP   = g++
ifdef DEBUG
  CPPFLAGS  = -O0 -g -DUNIX -Wall
else
  CPPFLAGS  = -O3 -fPIC -DUNIX -Wall
endif
  OMPFLAGS  = -fopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpiCC
  MPIFLAGS  = $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),intel)
  CPP   = icpc
ifdef DEBUG
  CCFLAGS = -O0 -g -no-prec-div -qopenmp -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK #-DOPS_DEBUG
else
  #CCFLAGS  = -O3 -ipo -no-prec-div -fp-model strict -fp-model source -prec-div -prec-sqrt -vec-report2 -xSSE4.2 -parallel #-DCOMM_PERF #-DDEBUG
  CCFLAGS = -O3 -ipo -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK#-vec-report
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -qopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
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
  MPIFLAGS  = $(CCFLAGS)
  OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lintelocl -lOpenCL #for offload opencl execution on the phi
else
ifeq ($(OPS_COMPILER),pgi)
  CPP     = pgCC
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
ifdef DEBUG
  CCFLAGS   = -O2 -Minline -Kieee -Minform=inform -Minfo=all
else
  CCFLAGS   = -O3 -Kieee -Minline -Minform=severe -Minfo=all
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -mp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpiCC
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


#
# master to make all versions
#

TARGETS = mgrid_seq
# mgrid_dev_mpi mgrid_mpi mgrid_mpi_openmp mgrid_openmp mgrid_seq mgrid_cuda mgrid_mpi_cuda mgrid_opencl #mgrid_openacc mgrid_mpi_openacc

ifeq ($(OPS_COMPILER),pgi)
TARGETS += mgrid_openacc mgrid_mpi_openacc
else
TARGETS += openacc_error
endif



all: clean $(TARGETS)

mgrid_dev_seq: Makefile mgrid.cpp mgrid_populate_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(OPS_LIB) mgrid.cpp -lops_seq -o mgrid_dev_seq


mgrid_dev_mpi: Makefile mgrid.cpp mgrid_populate_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(OPS_LIB) mgrid.cpp -lops_mpi -o mgrid_dev_mpi

#
# mpi version
#

SEQ_KERNELS = mgrid_seq_kernels.cpp

mgrid_mpi: Makefile mgrid_ops.cpp mgrid_populate_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(OPS_LIB) mgrid_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_mpi $(HDF5_LIB) -o mgrid_mpi

#
# OpenMP version
#
OMP_KERNELS = mgrid_omp_kernels.cpp

mgrid_mpi_openmp: Makefile mgrid_ops.cpp mgrid_populate_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(OMPFLAGS) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(OPS_LIB) mgrid_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_mpi -o mgrid_mpi_openmp


mgrid_openmp: Makefile mgrid_ops.cpp mgrid_populate_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) mgrid_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq  $(HDF5_LIB) -o mgrid_openmp

#
# Sequential version
#
mgrid_seq: Makefile mgrid_ops.cpp mgrid_populate_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) mgrid_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_seq $(HDF5_LIB) -o mgrid_seq

#
# CUDA version
#

mgrid_cuda: Makefile ./CUDA/mgrid_kernels_cu.o mgrid_ops.cpp mgrid_populate_kernel.h $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) mgrid_ops.cpp ./CUDA/mgrid_kernels_cu.o -lcudart -lops_cuda -o mgrid_cuda

mgrid_mpi_cuda: Makefile ./CUDA/mgrid_kernels_mpi_cu.o mgrid_ops.cpp mgrid_populate_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) mgrid_ops.cpp ./CUDA/mgrid_kernels_mpi_cu.o -lcudart -lops_mpi_cuda $(HDF5_LIB) -o mgrid_mpi_cuda

./CUDA/mgrid_kernels_cu.o: ./CUDA/mgrid_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC) -I. -c -o ./CUDA/mgrid_kernels_cu.o ./CUDA/mgrid_kernels.cu

./CUDA/mgrid_kernels_mpi_cu.o: ./CUDA/mgrid_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI $(CODE_GEN_CUDA) $(OPS_INC) -I. -I$(MPI_INSTALL_PATH)/include \
	-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/mgrid_kernels_mpi_cu.o ./CUDA/mgrid_kernels.cu

openacc_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
openacc_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")

./OpenACC/mgrid_kernels.o: ./OpenACC/mgrid_kernels.cpp Makefile
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
	-I. OpenACC/mgrid_kernels.cpp -c -o OpenACC/mgrid_kernels.o

./OpenACC/%.o: ./OpenACC/%.cpp ./OpenACC/%_c.c Makefile
	@echo "Compiling kernel" $*
	$(MPICC) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
              -I. OpenACC/$*_c.c -c -o OpenACC/$*_c.o
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 -DOPS_MPI $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
        -I. OpenACC/$*.cpp -c -o OpenACC/$*.o

mgrid_openacc: $(openacc_obj_list) mgrid_ops.cpp mgrid_populate_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) \
    mgrid_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_cuda -o mgrid_openacc

mgrid_mpi_openacc: $(openacc_obj_list) mgrid_ops.cpp mgrid_populate_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) -DOPS_MPI \
    mgrid_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_mpi_cuda -o mgrid_mpi_openacc

mgrid_opencl: ./OpenCL/mgrid_opencl_kernels.o mgrid_ops.cpp mgrid_populate_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_opencl.a
	$(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) $(OPENCL_LIB) mgrid_ops.cpp ./OpenCL/mgrid_opencl_kernels.o -I. ./OpenCL/$(SEQ_KERNELS) -lops_opencl -o mgrid_opencl

./OpenCL/mgrid_opencl_kernels.o: ./OpenCL/mgrid_opencl_kernels.cpp Makefile
	$(CPP) $(CPPFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/mgrid_opencl_kernels.o ./OpenCL/mgrid_opencl_kernels.cpp

#
#error
#
openacc_error:
	@echo "mgrid_mpi_openacc cannot be built with $(OPS_COMPILER) compilers .. requires an OpenACC compiler (e.g. pgi)"


#
# cleanup
#

clean:
	rm -f mgrid_dev_seq mgrid_dev_mpi mgrid_mpi mgrid_seq mgrid_openmp mgrid_mpi_openmp mgrid_cuda mgrid_mpi_cuda mgrid_openacc mgrid_mpi_openacc ./CUDA/*.o ./OpenACC/*.o *.o mgrid_opencl ./OpenCL/*.o *.o
