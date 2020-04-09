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
OPS_LIB   = -L$(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)
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
  CPPFLAGS  = -std=c++11 -O0 -g -DUNIX -Wall
else
  CPPFLAGS  = -std=c++11 -O3 -fPIC -DUNIX -Wall
endif
  OMPFLAGS  = -fopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpiCC
else
ifeq ($(OPS_COMPILER),intel)
  CPP   = icpc
ifdef DEBUG
  CCFLAGS = -std=c++11 -O0 -g -no-prec-div -qopenmp -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK #-DOPS_DEBUG
else
  #CCFLAGS  = -O3 -ipo -no-prec-div -fp-model strict -fp-model source -prec-div -prec-sqrt -vec-report2 -xSSE4.2 -parallel #-DCOMM_PERF #-DDEBUG
  CCFLAGS = -std=c++11 -O0 -g -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK#-vec-report
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -qopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
else
ifeq ($(OPS_COMPILER),mic)
  CPP   = icpc
ifdef DEBUG
  CCFLAGS = -O0 -g -mmic -DMPICH_IGNORE_CXX_SEEK#-vec-report
else
  #CCFLAGS  = -O3 -ipo -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK#-vec-report
  #CCFLAGS  = -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -finline -inline-forceinline -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK#-vec-report
  CCFLAGS = -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK -finline -vec-report
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -qopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
  OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lintelocl -lOpenCL #for offload opencl execution on the phi
else
ifeq ($(OPS_COMPILER),pgi)
  CPP     = pgc++
  CC      = pgcc
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
ifdef DEBUG
  CCFLAGS   = -O2 -Minline -Kieee #-Minform=inform -Minfo=all
else
  CCFLAGS   = -O3 -Kieee #-Minline -Minform=severe -Minfo=all
endif
  CPPFLAGS  = -std=c++11 $(CCFLAGS)
  OMPFLAGS  = -mp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPICC   = $(MPI_INSTALL_PATH)/bin/mpicc
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
  NVCC_FLAGS = -O0 -g -G -std=c++11
else
  NVCC_FLAGS = -O3 -std=c++11
endif


#
# master to make all versions
#

TARGETS = write_dev_seq write_seq write_dev_mpi write_mpi write_openmp write_mpi_openmp write_cuda write_mpi_cuda write_opencl write_mpi_opencl \
read_dev_seq read_seq read_dev_mpi read_mpi read_openmp read_mpi_openmp read_cuda read_mpi_cuda read_opencl read_mpi_opencl

ifeq ($(OPS_COMPILER),pgi)
TARGETS += #no open_acc backend to be build
else
TARGETS += openacc_error
endif



all: clean $(TARGETS)

write_dev_seq: Makefile write.cpp write_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) write.cpp -lops_seq -lops_hdf5_seq $(HDF5_LIB) -o write_dev_seq
read_dev_seq: Makefile read.cpp $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) read.cpp -lops_seq -lops_hdf5_seq $(HDF5_LIB) -o read_dev_seq


write_dev_mpi: Makefile write.cpp write_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) write.cpp -lops_hdf5_mpi -lops_mpi $(HDF5_LIB) -o write_dev_mpi
read_dev_mpi: Makefile read.cpp $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) read.cpp -lops_hdf5_mpi -lops_mpi $(HDF5_LIB) -o read_dev_mpi

#
# mpi version
#

SEQ_KERNELS = write_cpu_kernels.cpp

write_mpi: Makefile write_ops.cpp write_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) write_ops.cpp -I. ./MPI_OpenMP/$(SEQ_KERNELS) -lops_hdf5_mpi -lops_mpi $(HDF5_LIB) -o write_mpi
read_mpi: Makefile read_ops.cpp $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) read_ops.cpp -I. ./MPI_OpenMP/$(SEQ_KERNELS) -lops_hdf5_mpi -lops_mpi $(HDF5_LIB) -o read_mpi

#
# OpenMP version
#
OMP_KERNELS = write_cpu_kernels.cpp

write_mpi_openmp: Makefile write_ops.cpp write_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) write_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_hdf5_mpi -lops_mpi $(HDF5_LIB) -o write_mpi_openmp
read_mpi_openmp: Makefile read_ops.cpp $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) read_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_hdf5_mpi -lops_mpi $(HDF5_LIB) -o read_mpi_openmp


write_openmp: Makefile write_ops.cpp write_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) $(OPS_INC) $(OPS_LIB) write_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq -lops_hdf5_seq $(HDF5_LIB) -o write_openmp
read_openmp: Makefile read_ops.cpp $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) $(OPS_INC) $(OPS_LIB) read_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq -lops_hdf5_seq $(HDF5_LIB) -o read_openmp

#
# Sequential version
#
write_seq: Makefile write_ops.cpp write_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(OPS_LIB) write_ops.cpp -I. ./MPI_OpenMP/$(SEQ_KERNELS) -lops_seq -lops_hdf5_seq $(HDF5_LIB) -o write_seq
read_seq: Makefile read_ops.cpp $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(OPS_LIB) read_ops.cpp -I. ./MPI_OpenMP/$(SEQ_KERNELS) -lops_seq -lops_hdf5_seq $(HDF5_LIB) -o read_seq

#
# CUDA version
#

write_cuda: Makefile ./CUDA/write_kernels_cu.o write_ops.cpp write_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) write_ops.cpp ./CUDA/write_kernels_cu.o -lcudart -lops_cuda -lops_hdf5_seq $(HDF5_LIB) -o write_cuda
read_cuda: Makefile ./CUDA/read_kernels_cu.o read_ops.cpp $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) read_ops.cpp ./CUDA/read_kernels_cu.o -lcudart -lops_cuda -lops_hdf5_seq $(HDF5_LIB) -o read_cuda


write_mpi_cuda: Makefile ./CUDA/write_kernels_mpi_cu.o write_ops.cpp write_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) write_ops.cpp ./CUDA/write_kernels_mpi_cu.o -lcudart -lops_hdf5_mpi -lops_mpi_cuda $(HDF5_LIB) -o write_mpi_cuda
read_mpi_cuda: Makefile ./CUDA/read_kernels_mpi_cu.o read_ops.cpp $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) read_ops.cpp ./CUDA/read_kernels_mpi_cu.o -lcudart -lops_hdf5_mpi -lops_mpi_cuda $(HDF5_LIB) -o read_mpi_cuda


./CUDA/write_kernels_cu.o: ./CUDA/write_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC) -I. -c -o ./CUDA/write_kernels_cu.o ./CUDA/write_kernels.cu
./CUDA/read_kernels_cu.o: ./CUDA/read_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC) -I. -c -o ./CUDA/read_kernels_cu.o ./CUDA/read_kernels.cu

./CUDA/write_kernels_mpi_cu.o: ./CUDA/write_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI $(CODE_GEN_CUDA) $(OPS_INC) -I. -I$(MPI_INSTALL_PATH)/include \
	-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/write_kernels_mpi_cu.o ./CUDA/write_kernels.cu
./CUDA/read_kernels_mpi_cu.o: ./CUDA/read_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI $(CODE_GEN_CUDA) $(OPS_INC) -I. -I$(MPI_INSTALL_PATH)/include \
	-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/read_kernels_mpi_cu.o ./CUDA/read_kernels.cu


#
#OpenCL version
#
write_opencl: ./OpenCL/write_opencl_kernels.o write_ops.cpp write_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_opencl.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(OPS_LIB) $(OPENCL_LIB) write_ops.cpp ./OpenCL/write_opencl_kernels.o  -lops_opencl -lops_hdf5_seq $(HDF5_LIB) -o write_opencl
read_opencl: ./OpenCL/read_opencl_kernels.o read_ops.cpp Makefile $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_opencl.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(OPS_LIB) $(OPENCL_LIB) read_ops.cpp ./OpenCL/read_opencl_kernels.o  -lops_opencl -lops_hdf5_seq $(HDF5_LIB) -o read_opencl


write_mpi_opencl: ./OpenCL/write_mpi_opencl_kernels.o write_ops.cpp write_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi_opencl.a
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) write_ops.cpp ./OpenCL/write_mpi_opencl_kernels.o -lops_hdf5_mpi -lops_mpi_opencl $(HDF5_LIB) -o write_mpi_opencl
read_mpi_opencl: ./OpenCL/read_mpi_opencl_kernels.o read_ops.cpp Makefile $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi_opencl.a
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) read_ops.cpp ./OpenCL/read_mpi_opencl_kernels.o -lops_hdf5_mpi -lops_mpi_opencl $(HDF5_LIB) -o read_mpi_opencl

./OpenCL/write_opencl_kernels.o: ./OpenCL/write_opencl_kernels.cpp Makefile
	$(CPP) $(CPPFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/write_opencl_kernels.o ./OpenCL/write_opencl_kernels.cpp
./OpenCL/read_opencl_kernels.o: ./OpenCL/read_opencl_kernels.cpp Makefile
	$(CPP) $(CPPFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/read_opencl_kernels.o ./OpenCL/read_opencl_kernels.cpp

./OpenCL/write_mpi_opencl_kernels.o: ./OpenCL/write_opencl_kernels.cpp Makefile
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/write_mpi_opencl_kernels.o ./OpenCL/write_opencl_kernels.cpp
./OpenCL/read_mpi_opencl_kernels.o: ./OpenCL/read_opencl_kernels.cpp Makefile
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/read_mpi_opencl_kernels.o ./OpenCL/read_opencl_kernels.cpp

#
#error
#
openacc_error:
	@echo "write_mpi_openacc cannot be built with $(OPS_COMPILER) compilers .. requires an OpenACC compiler (e.g. pgi)"
	@echo "read_mpi_openacc cannot be built with $(OPS_COMPILER) compilers .. requires an OpenACC compiler (e.g. pgi)"


#
# cleanup
#

clean:
	rm -f write_dev_seq write_seq write_dev_seq write_mpi write_dev_mpi write_openmp write_mpi_openmp write_cuda write_mpi_cuda write_openacc write_mpi_openacc write_opencl write_mpi_opencl ./CUDA/*.o ./OpenACC/*.o *.o write_opencl ./OpenCL/*.o *.o
	rm -f read_dev_seq read_seq read_dev_seq read_mpi read_dev_mpi read_openmp read_mpi_openmp read_cuda read_mpi_cuda read_openacc read_mpi_openacc read_opencl read_mpi_opencl ./CUDA/*.o ./OpenACC/*.o *.o read_opencl ./OpenCL/*.o *.o
