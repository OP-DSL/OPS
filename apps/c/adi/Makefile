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

TRID_INSTALL_PATH ?= $(TDMA_INSTALL_PATH)
TRID_INC := -I$(TRID_INSTALL_PATH)/include
TRID_LIB := -L$(TRID_INSTALL_PATH)/lib

ifeq ($(OPS_COMPILER),gnu)
  CPP   = g++
ifdef DEBUG
  CPPFLAGS  = -O0 -std=c++11 -g -DUNIX -Wall
else
  CPPFLAGS  = -O3 -std=c++11 -fPIC -DUNIX -Wall
endif
  OMPFLAGS  = -fopenmp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpiCC
else
ifeq ($(OPS_COMPILER),intel)
  CPP   = icpc
ifdef DEBUG
  CCFLAGS = -O0 -std=c++11 -g -no-prec-div -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK
else
  CCFLAGS = -std=c++11 -fp-model strict -fp-model source -prec-div -prec-sqrt #-xCORE_AVX2 -m64 -parallel -no-offload -O3 -ip -restrict
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -qopenmp
  CXXFLAGS  = $(CCFLAGS)
  INTEL_LIB = -L/opt/compilers/intel/intelPS-2015/lib/intel64/ -limf -lintlc -lsvml -lirng -liomp5 -lcilkrts
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
  OMPFLAGS  = -openmp -openmp-report2
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
  OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lintelocl -lOpenCL #for offload opencl execution on the phi
else
ifeq ($(OPS_COMPILER),pgi)
  CPP     = pgc++
  CC      = pgcc
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
ifdef DEBUG
  CCFLAGS   = -O2 -Minline -Kieee -std=c++11 #-Minform=inform -Minfo=all
else
  CCFLAGS   = -O3 -Kieee -Minline -std=c++11 #-Minform=severe -Minfo=all
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -mp
  MPICPP  = $(MPI_INSTALL_PATH)/bin/mpicxx
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
  NVCC_FLAGS = -O3 -arch sm_35 -std=c++11 --ptxas-options=-v -Xptxas -dlcm=cg
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

TARGETS = adi_dev_seq adi_seq adi_openmp adi_cuda adi_mpi adi_mpi_cuda # adi_opencl adi_mpi_openmp adi_dev_mpi adi_mpi adi_mpi_cuda adi_opencl adi_mpi_opencl

ifeq ($(OPS_COMPILER),pgi)
#TARGETS += adi_openacc adi_mpi_openacc
else
#TARGETS += openacc_error
endif



all: clean $(TARGETS)

adi_dev_seq: Makefile .generated adi.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) adi.cpp -lops_seq -lops_hdf5_seq $(TRID_LIB) -ltridcpu -lops_trid_seq $(HDF5_LIB) -o adi_dev_seq


adi_dev_mpi: Makefile adi.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(CXXFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) adi.cpp $(TRID_LIB) -ltridcpu -lops_hdf5_mpi -lops_mpi -lops_trid_mpi $(HDF5_LIB) -o adi_dev_mpi

#
# mpi version
#

SEQ_KERNELS = adi_cpu_kernels.cpp

adi_mpi: Makefile .generated adi_ops.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(CXXFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) adi_ops.cpp -I. ./MPI_OpenMP/$(SEQ_KERNELS) $(TRID_LIB) -ltridcpu -lops_hdf5_mpi -lops_mpi -lops_trid_mpi $(HDF5_LIB) -o adi_mpi

#
# OpenMP version
#
OMP_KERNELS = adi_cpu_kernels.cpp

adi_mpi_openmp: Makefile .generated adi_ops.cpp adi_print_kernel.h adi_copy_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi.a
	$(MPICPP) $(OMPFLAGS) $(CXXFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) adi_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_hdf5_mpi -lops_mpi -lops_trid_mpi $(HDF5_LIB) -o adi_mpi_openmp

adi_openmp: Makefile .generated adi_ops.cpp init_kernel.h preproc_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(CXXFLAGS) $(OMPFLAGS) $(OPS_INC) $(OPS_LIB) adi_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq -lops_hdf5_seq -lops_trid_seq $(TRID_LIB) -ltridcpu $(INTEL_LIB) \
 $(HDF5_LIB) -o adi_openmp

#
# Sequential version
#
adi_seq: Makefile .generated adi_ops.cpp init_kernel.h preproc_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_seq.a
	$(MPICPP) $(CXXFLAGS) $(OPS_INC) $(OPS_LIB) adi_ops.cpp -I. ./MPI_OpenMP/$(SEQ_KERNELS) -lops_seq -lops_hdf5_seq -lops_trid_seq $(TRID_LIB) -ltridcpu  $(HDF5_LIB) -o adi_seq

#
# CUDA version
#

adi_cuda: Makefile .generated ./CUDA/adi_kernels_cu.o adi_ops.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_cuda.a
	$(MPICPP) $(CXXFLAGS) $(CUDA_INC) $(TRID_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) adi_ops.cpp ./CUDA/adi_kernels_cu.o -lcudart -lcublas -lops_cuda -lops_trid_cuda -lops_hdf5_seq $(TRID_LIB) -ltridcuda $(HDF5_LIB) -o adi_cuda

adi_mpi_cuda: Makefile .generated ./CUDA/adi_kernels_mpi_cu.o adi_ops.cpp preproc_kernel.h init_kernel.h $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi_cuda.a
	$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) adi_ops.cpp ./CUDA/adi_kernels_mpi_cu.o $(TRID_LIB) -ltridcuda_mpi -lops_hdf5_mpi -lops_mpi_cuda -lops_trid_mpi_cuda -lcudart -lcublas $(HDF5_LIB) -o adi_mpi_cuda

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
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 -DOPS_MPI $(CXXFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
	-I. OpenACC/adi_kernels.cpp -c -o OpenACC/adi_kernels_mpi.o

./OpenACC/%_mpi.o: ./OpenACC/%.cpp ./OpenACC/%_c.c Makefile
	@echo "Compiling kernel" $*
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(CXXFLAGS) -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
              -I. OpenACC/$*_c.c -c -o OpenACC/$*_c_mpi.o
	$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(CXXFLAGS) -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
        -I. OpenACC/$*.cpp -c -o OpenACC/$*_mpi.o

./OpenACC/%.o: ./OpenACC/%.cpp ./OpenACC/%_c.c Makefile
	@echo "Compiling kernel" $*
	$(CC) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(CCFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
              -I. OpenACC/$*_c.c -c -o OpenACC/$*_c.o
	$(CPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(CPPFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
        -I. OpenACC/$*.cpp -c -o OpenACC/$*.o

adi_openacc: $(openacc_obj_list) .generated ./OpenACC/adi_kernels.o adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(CXXFLAGS) $(OPS_INC) $(OPS_LIB) -DOPS_MPI $(CUDA_INC) $(CUDA_LIB) \
    adi_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lcublas -lops_cuda -lops_hdf5_seq $(HDF5_LIB) -o adi_openacc


adi_mpi_openacc: $(openacc_mpi_obj_list) .generated ./OpenACC/adi_kernels_mpi.o adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi_cuda.a
	$(MPICPP) -acc -ta=tesla:cc35 $(CXXFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) -DOPS_MPI \
    adi_ops.cpp -I. $(openacc_mpi_obj_list) $(openacc_mpi_c_obj_list) -lcudart -lcublas -lops_mpi_cuda -lops_hdf5_mpi $(HDF5_LIB) -o adi_mpi_openacc


#
#OpenCL version
#
adi_opencl: .generated ./OpenCL/adi_opencl_kernels.o adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_opencl.a
	$(MPICPP) $(CXXFLAGS) $(OPS_INC) $(OPS_LIB) $(OPENCL_LIB) adi_ops.cpp ./OpenCL/adi_opencl_kernels.o  -lops_opencl -lops_hdf5_seq $(HDF5_LIB) -o adi_opencl


adi_mpi_opencl: .generated ./OpenCL/adi_mpi_opencl_kernels.o adi_ops.cpp adi_kernel.h adi_print_kernel.h adi_copy_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/$(OPS_COMPILER)/libops_mpi_opencl.a
	$(MPICPP) $(CXXFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) adi_ops.cpp ./OpenCL/adi_opencl_kernels.o -lops_mpi_opencl -lops_hdf5_mpi $(HDF5_LIB) -o adi_mpi_opencl

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
