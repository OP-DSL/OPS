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

TRID_INSTALL_PATH ?= /home/mudalige/TriDiagonal/Tridiag-Git/tridsolver/scalar/build
TRID_INC := -I$(TRID_INSTALL_PATH)/include
TRID_LIB := -L$(TRID_INSTALL_PATH)/lib

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
  CCFLAGS = -O0 -g -no-prec-div -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK
else
  CCFLAGS = -fp-model strict -fp-model source -prec-div -prec-sqrt #-xCORE_AVX2 -m64 -parallel -no-offload -O3 -ip -restrict
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -openmp -openmp-report2
  INTEL_LIB = -L/opt/compilers/intel/intelPS-2015/lib/intel64/ -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts
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
  CCFLAGS = -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK -finline -vec-report
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -openmp -openmp-report2
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
  NVCC_FLAGS = -O3 -arch sm_35 --ptxas-options=-v -Xptxas -dlcm=cg
endif


#
# generate code
#
.generated: $(OPS_INSTALL_PATH)/../../ops_translator/c/*.py 
	$(OPS_INSTALL_PATH)/../../ops_translator/c/ops.py adi.cpp
	rm -f .generated
	touch .generated
	touch ./OpenCL/adi_seq_kernels.cpp


#
# master to make all versions
#

TARGETS = adi_dev_seq adi_seq adi_openmp adi_cuda # adi_opencl adi_mpi_openmp adi_dev_mpi adi_mpi adi_mpi_cuda adi_opencl adi_mpi_opencl

ifeq ($(OPS_COMPILER),pgi)
#TARGETS += adi_openacc adi_mpi_openacc
else
#TARGETS += openacc_error
endif



all: clean $(TARGETS)

adi_dev_seq: Makefile .generated adi.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) adi.cpp -lops_seq -lops_hdf5_seq $(TRID_LIB) -ltridcpu $(HDF5_LIB) -o adi_dev_seq


adi_dev_mpi: Makefile .generated adi.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) adi.cpp $(TRID_LIB) -ltridcpu -lops_mpi $(HDF5_LIB) -o adi_dev_mpi

#
# mpi version
#

SEQ_KERNELS = adi_seq_kernels.cpp

adi_mpi: Makefile .generated adi_ops.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) adi_ops.cpp -I. ./MPI/$(SEQ_KERNELS) $(TRID_LIB) -ltridcpu -lops_mpi $(HDF5_LIB) -o adi_mpi

#
# OpenMP version
#
OMP_KERNELS = adi_omp_kernels.cpp

adi_mpi_openmp: Makefile .generated adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	$(MPICPP) $(OMPFLAGS) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) adi_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_mpi $(HDF5_LIB) -o adi_mpi_openmp

adi_openmp: Makefile .generated adi_ops.cpp init_kernel.h preproc_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(MPIFLAGS) $(OMPFLAGS) $(OPS_INC) $(OPS_LIB) adi_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq -lops_hdf5_seq $(TRID_LIB) -ltridcpu $(INTEL_LIB) \
 $(HDF5_LIB) -o adi_openmp

#
# Sequential version
#
adi_seq: Makefile .generated adi_ops.cpp init_kernel.h preproc_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	$(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) adi_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_seq -lops_hdf5_seq $(TRID_LIB) -ltridcpu  $(HDF5_LIB) -o adi_seq

#
# CUDA version
#

adi_cuda: Makefile .generated ./CUDA/adi_kernels_cu.o adi_ops.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	$(MPICPP) $(MPIFLAGS) $(CUDA_INC) $(TRID_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) adi_ops.cpp ./CUDA/adi_kernels_cu.o -lcudart -lops_cuda -lops_hdf5_seq $(TRID_LIB) -ltridcuda $(HDF5_LIB) -o adi_cuda

adi_mpi_cuda: Makefile .generated ./CUDA/adi_kernels_mpi_cu.o adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) adi_ops.cpp ./CUDA/adi_kernels_mpi_cu.o -lcudart -lops_mpi_cuda $(HDF5_LIB) -o adi_mpi_cuda

./CUDA/adi_kernels_cu.o: ./CUDA/adi_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC) -I. -c -o ./CUDA/adi_kernels_cu.o ./CUDA/adi_kernels.cu

./CUDA/adi_kernels_mpi_cu.o: ./CUDA/adi_kernels.cu Makefile
	nvcc $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI $(CODE_GEN_CUDA) $(OPS_INC) -I. -I$(MPI_INSTALL_PATH)/include \
	-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/adi_kernels_mpi_cu.o ./CUDA/adi_kernels.cu

openacc_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
openacc_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")

openacc_mpi_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\_mpi.o/g | tr "\n" " ")
openacc_mpi_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\_mpi.o/g | tr "\n" " ")


./OpenACC/adi_kernels.o: ./OpenACC/adi_kernels.cpp Makefile
	$(CPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(CPPFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
	-I. OpenACC/adi_kernels.cpp -c -o OpenACC/adi_kernels.o


./OpenACC/adi_kernels_mpi.o: ./OpenACC/adi_kernels.cpp Makefile
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 -DOPS_MPI $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
	-I. OpenACC/adi_kernels.cpp -c -o OpenACC/adi_kernels_mpi.o

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

adi_openacc: $(openacc_obj_list) .generated ./OpenACC/adi_kernels.o adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) -DOPS_MPI $(CUDA_INC) $(CUDA_LIB) \
    adi_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_cuda -lops_hdf5_seq $(HDF5_LIB) -o adi_openacc


adi_mpi_openacc: $(openacc_mpi_obj_list) .generated ./OpenACC/adi_kernels_mpi.o adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) -DOPS_MPI \
    adi_ops.cpp -I. $(openacc_mpi_obj_list) $(openacc_mpi_c_obj_list) -lcudart -lops_mpi_cuda $(HDF5_LIB) -o adi_mpi_openacc


#
#OpenCL version
#
adi_opencl: .generated ./OpenCL/adi_opencl_kernels.o adi_ops.cpp Makefile $(OPS_INSTALL_PATH)/lib/libops_opencl.a
	$(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) $(OPENCL_LIB) adi_ops.cpp ./OpenCL/adi_opencl_kernels.o  -lops_opencl -lops_hdf5_seq $(HDF5_LIB) -o adi_opencl


adi_mpi_opencl: .generated ./OpenCL/adi_mpi_opencl_kernels.o adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_opencl.a
	$(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) adi_ops.cpp ./OpenCL/adi_opencl_kernels.o -lops_mpi_opencl $(HDF5_LIB) -o adi_mpi_opencl

./OpenCL/adi_opencl_kernels.o: ./OpenCL/adi_opencl_kernels.cpp Makefile
	$(CPP) $(CPPFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/adi_opencl_kernels.o ./OpenCL/adi_opencl_kernels.cpp

./OpenCL/adi_mpi_opencl_kernels.o: ./OpenCL/adi_opencl_kernels.cpp Makefile
	$(MPICPP) $(CPPFLAGS) -DOPS_MPI $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) -I. \
                -c -o ./OpenCL/adi_opencl_kernels.o ./OpenCL/adi_opencl_kernels.cpp

#
#error
#
openacc_error:
	@echo "adi_mpi_openacc cannot be built with $(OPS_COMPILER) compilers .. requires an OpenACC compiler (e.g. pgi)"


#
# cleanup
#

clean:
	rm -f adi_dev_seq adi_seq adi_dev_seq adi_mpi adi_dev_mpi adi_openmp adi_mpi_openmp adi_cuda adi_mpi_cuda adi_openacc adi_mpi_openacc adi_opencl adi_mpi_opencl ./CUDA/*.o ./OpenACC/*.o *.o adi_opencl ./OpenCL/*.o *.o
