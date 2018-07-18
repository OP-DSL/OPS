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
ifdef CHECKPOINTING
  HDF5_INC 	  	:= -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB 	  	:= -L$(HDF5_INSTALL_PATH)/lib -lhdf5_hl -lhdf5 -lz
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
  MPICC = $(MPI_INSTALL_PATH)/bin/mpicc
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
  CCFLAGS = -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK -finline -vec-report
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -qopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS  = $(CCFLAGS)
  OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lintelocl -lOpenCL #for offload opencl execution on the phi
else
ifeq ($(OPS_COMPILER),pgi)
  CPP     = pgc++
  CC      = pgcc
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
ifdef DEBUG
  CCFLAGS   = -O2 -Minline -Kieee #-Minform=inform -Minfo=all
else
  CCFLAGS   = -O3 -Kieee -Minline #-Minform=severe -Minfo=all
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -mp
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

TARGETS = random_dev_seq random_seq random_dev_mpi random_mpi random_openmp random_mpi_openmp random_cuda random_mpi_cuda random_opencl random_mpi_opencl

ifeq ($(OPS_COMPILER),pgi)
TARGETS += random_openacc random_mpi_openacc
else
TARGETS += openacc_error
endif



all: clean $(TARGETS)

random_dev_seq: Makefile random.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) random.cpp -lops_seq $(HDF5_LIB) -o random_dev_seq


random_dev_mpi: Makefile random.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) random.cpp -lops_mpi $(HDF5_LIB) -o random_dev_mpi

#
# mpi version
#

SEQ_KERNELS = random_seq_kernels.cpp

random_mpi: Makefile random_ops.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) random_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_mpi $(HDF5_LIB) -o random_mpi

#
# OpenMP version
#
OMP_KERNELS = random_omp_kernels.cpp

random_mpi_openmp: Makefile random_ops.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(OMPFLAGS) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) random_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_mpi $(HDF5_LIB) -o random_mpi_openmp

random_mpi_tiled: Makefile random_ops.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(OMPFLAGS) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) random_ops.cpp -I. ./Tiled/$(SEQ_KERNELS) -lops_mpi $(HDF5_LIB) -o random_mpi_tiled

random_openmp: Makefile random_ops.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) random_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq $(HDF5_LIB) -o random_openmp

random_tiled: Makefile random_ops.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) random_ops.cpp -I. ./Tiled/$(SEQ_KERNELS) -lops_seq $(HDF5_LIB) -o random_tiled

#
# Sequential version
#
random_seq: Makefile random_ops.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) random_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_seq $(HDF5_LIB) -o random_seq

#
# CUDA version
#

random_cuda: Makefile ./CUDA/random_kernels_cu.o random_ops.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) random_ops.cpp ./CUDA/random_kernels_cu.o -lcudart -lcurand -lops_cuda $(HDF5_LIB) -o random_cuda

random_mpi_cuda: Makefile ./CUDA/random_kernels_mpi_cu.o random_ops.cpp random_print_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) random_ops.cpp ./CUDA/random_kernels_mpi_cu.o -lcudart -lcurand -lops_mpi_cuda $(HDF5_LIB) -o random_mpi_cuda

./CUDA/random_kernels_cu.o: ./CUDA/random_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC) -I. -c -o ./CUDA/random_kernels_cu.o ./CUDA/random_kernels.cu

./CUDA/random_kernels_mpi_cu.o: ./CUDA/random_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI $(CODE_GEN_CUDA) $(OPS_INC) -I. -I$(MPI_INSTALL_PATH)/include \
	-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/random_kernels_mpi_cu.o ./CUDA/random_kernels.cu

openacc_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
openacc_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")

openacc_mpi_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\_mpi.o/g | tr "\n" " ")
openacc_mpi_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\_mpi.o/g | tr "\n" " ")


./OpenACC/random_kernels.o: ./OpenACC/random_kernels.cpp Makefile
	$(CPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(CPPFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
	-I. OpenACC/random_kernels.cpp -c -o OpenACC/random_kernels.o


./OpenACC/random_kernels_mpi.o: ./OpenACC/random_kernels.cpp Makefile
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 -DOPS_MPI $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
	-I. OpenACC/random_kernels.cpp -c -o OpenACC/random_kernels_mpi.o

./OpenACC/%_mpi.o: ./OpenACC/%.cpp ./OpenACC/%_c.c Makefile
	@echo "Compiling kernel" $*
	$(MPICC) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
              -I. OpenACC/$*_c.c -c -o OpenACC/$*_c_mpi.o
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
        -I. OpenACC/$*.cpp -c -o OpenACC/$*_mpi.o

./OpenACC/%.o: ./OpenACC/%.cpp ./OpenACC/%_c.c Makefile
	@echo "Compiling kernel" $*
	$(CC) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(CCFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
              -I. OpenACC/$*_c.c -c -o OpenACC/$*_c.o
	$(CPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(CPPFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
        -I. OpenACC/$*.cpp -c -o OpenACC/$*.o

random_openacc: $(openacc_obj_list) ./OpenACC/random_kernels.o random_ops.cpp random_print_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) -DOPS_MPI $(CUDA_INC) $(CUDA_LIB) \
    random_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lcurand -lops_cuda $(HDF5_LIB) -o random_openacc


random_mpi_openacc: $(openacc_mpi_obj_list) ./OpenACC/random_kernels_mpi.o random_ops.cpp random_print_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) -DOPS_MPI \
    random_ops.cpp -I. $(openacc_mpi_obj_list) $(openacc_mpi_c_obj_list) -lcudart -lcurand -lops_mpi_cuda $(HDF5_LIB) -o random_mpi_openacc


#
#OpenCL version
#
random_opencl: ./OpenCL/random_opencl_kernels.o random_ops.cpp random_print_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_opencl.a
	$(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) $(OPENCL_LIB) random_ops.cpp ./OpenCL/random_opencl_kernels.o  -lops_opencl $(HDF5_LIB) -o random_opencl


random_mpi_opencl: ./OpenCL/random_mpi_opencl_kernels.o random_ops.cpp random_print_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_opencl.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) random_ops.cpp ./OpenCL/random_opencl_kernels.o -lops_mpi_opencl $(HDF5_LIB) -o random_mpi_opencl

./OpenCL/random_opencl_kernels.o: ./OpenCL/random_opencl_kernels.cpp Makefile
	$(CPP) $(CPPFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/random_opencl_kernels.o ./OpenCL/random_opencl_kernels.cpp

./OpenCL/random_mpi_opencl_kernels.o: ./OpenCL/random_opencl_kernels.cpp Makefile
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/random_opencl_kernels.o ./OpenCL/random_opencl_kernels.cpp

mpiinline_obj_list = $(shell find MPI_inline/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
mpiinline_c_obj_list = $(shell find MPI_inline/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")


./MPI_inline/random_kernels.o: ./MPI_inline/random_kernels.cpp ./MPI_inline/random_common.h Makefile
				$(MPICPP) $(MPIFLAGS) $(OMPFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC)  \
 			       -I. MPI_inline/random_kernels.cpp -c -o MPI_inline/random_kernels.o

./MPI_inline/%.o: ./MPI_inline/%.cpp ./MPI_inline/%_c.c ./MPI_inline/random_common.h Makefile
				@echo "Compiling kernel" $*
				$(MPICC) $(MPIFLAGS) $(OMPFLAGS) -DOPS_MPI  $(OPS_INC) $(HDF5_INC) \
			        -I. MPI_inline/$*_c.c -c -o MPI_inline/$*_c.o
				$(MPICPP) $(MPIFLAGS) $(OMPFLAGS) -DOPS_MPI  $(OPS_INC) $(HDF5_INC) \
				-I. MPI_inline/$*.cpp -c -o MPI_inline/$*.o

random_mpi_inline: $(mpiinline_obj_list) random_ops.cpp random_print_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	        $(MPICPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB)  -DOPS_MPI \
		random_ops.cpp -I. $(mpiinline_obj_list) $(mpiinline_c_obj_list) -lops_mpi $(HDF5_LIB) -o random_mpi_inline


#
#error
#
openacc_error:
	@echo "random_mpi_openacc cannot be built with $(OPS_COMPILER) compilers .. requires an OpenACC compiler (e.g. pgi)"


#
# cleanup
#

clean:
	rm -f random_dev_seq random_seq random_dev_seq random_mpi random_dev_mpi random_openmp random_mpi_openmp random_cuda random_mpi_cuda random_openacc random_mpi_openacc random_opencl random_mpi_opencl ./CUDA/*.o ./OpenACC/*.o *.o random_opencl ./OpenCL/*.o *.o
