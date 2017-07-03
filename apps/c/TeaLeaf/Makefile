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
OPS_INC		= -I$(OPS_INSTALL_PATH)/include
OPS_LIB		= -L$(OPS_INSTALL_PATH)/lib
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

#HDF5_INC             := -I/usr/include/hdf5/openmpi/
#HDF5_LIB             := -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ -lhdf5_hl -lhdf5 -lz

ifeq ($(OPS_COMPILER),gnu)
  CPP		= g++
ifdef DEBUG
  CPPFLAGS	= -O0 -g -DUNIX -Wall
else
  CPPFLAGS	= -O3 -fPIC -DUNIX -Wall
endif
  OMPFLAGS	= -fopenmp
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPIFLAGS	= $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),intel)
  CPP		= icpc
ifdef DEBUG
  CCFLAGS	= -O0 -g -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK #-DOPS_DEBUG
else
  CCFLAGS	= -O3 -ipo -fp-model strict -fp-model source -no-prec-div -prec-sqrt -vec-report2 -xHost -parallel -restrict -fno-alias -inline-forceinline #-DCOMM_PERF #-DDEBUG
  #CCFLAGS	= -O3 -no-prec-div -ip  -fp-model strict -fp-model source -prec-div -prec-sqrt
endif
  CPPFLAGS	= $(CCFLAGS)
  OMPFLAGS	= -qopenmp #-openmp-report2
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPICC	= $(MPI_INSTALL_PATH)/bin/mpicc -std=c99
  MPIFLAGS	= $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),mic)
  CPP		= icpc
ifdef DEBUG
  CCFLAGS	= -O0 -g -mmic -DMPICH_IGNORE_CXX_SEEK#-vec-report
else
  #CCFLAGS	= -O3 -ipo -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK#-vec-report
  #CCFLAGS	= -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -finline -inline-forceinline -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK#-vec-report
  CCFLAGS	= -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK -finline#-vec-report
endif
  CPPFLAGS	= $(CCFLAGS)
  OMPFLAGS	= -qopenmp
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS	= $(CCFLAGS)
  OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lintelocl -lOpenCL #for offload opencl execution on the phi
else
ifeq ($(OPS_COMPILER),pgi)
  CPP 		= pgc++
ifdef DEBUG
  CCFLAGS 	= -O0 -g -Minline -Kieee #-Minform=inform -Minfo=all
else
  CCFLAGS 	= -O2 -Kieee -Minline #-fastsse -gopt -Mipa=fast -Mlist -Kieee #-Minline #-Minform=severe -Minfo=all
endif
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS 	= -mp=nonuma
  MPICPP 	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPICC 	= $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS 	= $(CPPFLAGS) -ldl
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
ifeq ($(OPS_COMPILER),xl)
  CPP       = mpixlc
  CUDA_ALIGN_FLAG :=
ifdef DEBUG
  CCFLAGS   = -O2 -g
else
  CCFLAGS   = -O5 -qnoeh -qipa=partition=large -g -qfullpath -Q -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036 -qsrcmsg -qfloat=nomaf
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -qsmp=omp  -qthreaded
  MPICPP    = mpixlcxx
  MPICC     = mpixlc
  MPIFLAGS  = $(CPPFLAGS)
else
ifeq ($(OPS_COMPILER),clang)
CPP           = clang++
ifdef DEBUG 
CPPFLAGS      = -O0 -g -DUNIX -Wall
else
CPPFLAGS      = -O3 -DUNIX -Wall
endif
MPICPP        = $(MPI_INSTALL_PATH)/bin/mpiCC
MPIFLAGS      = $(CPPFLAGS)
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif
endif
endif
endif
endif

ifdef HDF5_INSTALL_PATH
  CC = $(MPICC)
  CPP = $(MPICPP)
endif

NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc
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
  NVCC_FLAGS = -O3 -restrict --use_fast_math #--fmad false
endif

#switch on FMA instructions in OpenCL
OCL_FLAGS=#-DOCL_FMA_SWITCH_ON

#
# master to make all versions
#

TARGETS = tealeaf_dev_seq tealeaf_dev_mpi tealeaf_mpi tealeaf_mpi_openmp tealeaf_openmp tealeaf_seq tealeaf_tiled tealeaf_cuda tealeaf_mpi_cuda tealeaf_mpi_cuda_tiled tealeaf_opencl tealeaf_mpi_opencl tealeaf_mpi_tiled #tealeaf_openacc tealeaf_mpi_openacc

ifeq ($(OPS_COMPILER),pgi)
TARGETS += tealeaf_openacc tealeaf_mpi_openacc
else
TARGETS += openacc_error
endif

PLAIN_FILES = build_field.cpp diffuse.cpp initialise.cpp read_input.cpp tea_solve.cpp timestep.cpp
OPS_FILES_NOGEN = tea_leaf.cpp field_summary.cpp  generate.cpp initialise_chunk.cpp start.cpp set_field.cpp  tea_leaf_cg.cpp tea_leaf_cheby.cpp tea_leaf_common.cpp tea_leaf_jacobi.cpp tea_leaf_ppcg.cpp update_halo.cpp
OPS_FILES_GEN = tea_leaf_ops.cpp field_summary_ops.cpp  generate_ops.cpp initialise_chunk_ops.cpp start_ops.cpp set_field_ops.cpp  tea_leaf_cg_ops.cpp tea_leaf_cheby_ops.cpp tea_leaf_common_ops.cpp tea_leaf_jacobi_ops.cpp tea_leaf_ppcg_ops.cpp update_halo_ops.cpp
HEADERS = data.h generate_chunk_kernel.h tea_leaf_cg_kernels.h tea_leaf.h tea_leaf_ops_vars.h user_types.h \
		definitions.h initialise_chunk_kernel.h tea_leaf_cheby_kernels.h tea_leaf_jacobi_kernels.h tea_leaf_ppcg_kernels.h \
		field_summary_kernel.h set_field_kernels.h tea_leaf_common_kernels.h tea_leaf_kernels.h update_halo_kernels.h

all: clean $(TARGETS)

tealeaf_dev_seq: Makefile $(PLAIN_FILES) $(OPS_FILES_NOGEN) $(HEADERS) \
						$(OPS_INSTALL_PATH)/lib/libops_seq.a
		$(CPP) $(CPPFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) \
					$(PLAIN_FILES) $(OPS_FILES_NOGEN) -lops_seq $(HDF5_LIB) -o tealeaf_dev_seq

tealeaf_dev_mpi: Makefile $(PLAIN_FILES) $(OPS_FILES_NOGEN) $(HEADERS) $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	        $(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(PLAIN_FILES) $(OPS_FILES_NOGEN) -lops_mpi $(HDF5_LIB) -o tealeaf_dev_mpi

#
# mpi version
#

SEQ_KERNELS = tea_leaf_seq_kernels.cpp

tealeaf_mpi: Makefile $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	        $(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) -I. ./MPI/$(SEQ_KERNELS) -lops_mpi $(HDF5_LIB) -o tealeaf_mpi

#
# OpenMP version
#
OMP_KERNELS = tea_leaf_omp_kernels.cpp

tealeaf_mpi_openmp: Makefile $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	        $(MPICPP) $(OMPFLAGS) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_mpi $(HDF5_LIB) -o tealeaf_mpi_openmp


tealeaf_openmp: Makefile $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_seq.a
	        $(CPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq $(HDF5_LIB) -o tealeaf_openmp

#
# Sequential version
#
tealeaf_seq: Makefile $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_seq.a
		$(CPP) $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) -I. ./MPI/$(SEQ_KERNELS) -lops_seq $(HDF5_LIB) -o tealeaf_seq

#
# Tiled version
#
tealeaf_tiled: Makefile $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_seq.a
		$(CPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) -I. ./Tiled/$(SEQ_KERNELS) -lops_seq $(HDF5_LIB) -o tealeaf_tiled

tealeaf_mpi_tiled: Makefile $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_mpi.a
		$(CPP) $(OMPFLAGS) -DOPS_MPI $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) -I. ./Tiled/$(SEQ_KERNELS) -lops_mpi $(HDF5_LIB) -o tealeaf_mpi_tiled

#
#
# CUDA version
#

tealeaf_cuda: Makefile ./CUDA/tealeaf_kernels_cu.o $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_cuda.a
		$(CPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) \
				 ./CUDA/tealeaf_kernels_cu.o -lcudart -lops_cuda $(HDF5_LIB) -o tealeaf_cuda

tealeaf_mpi_cuda: Makefile ./CUDA/tealeaf_kernels_mpi_cu.o $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
		$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) \
				 ./CUDA/tealeaf_kernels_mpi_cu.o -lcudart -lops_mpi_cuda $(HDF5_LIB) -o tealeaf_mpi_cuda

tealeaf_mpi_cuda_tiled: Makefile ./CUDA/tealeaf_kernels_mpi_tiled_cu.o $(PLAIN_FILES) $(OPS_FILES_GEN) $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
		$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) \
				 ./CUDA/tealeaf_kernels_mpi_tiled_cu.o -lcudart -lops_mpi_cuda $(HDF5_LIB) -o tealeaf_mpi_cuda_tiled

./CUDA/tealeaf_kernels_cu.o:	./CUDA/tea_leaf_kernels.cu \
				Makefile
				$(NVCC) $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC) $(HDF5_INC) -I. \
				-c -o ./CUDA/tealeaf_kernels_cu.o ./CUDA/tea_leaf_kernels.cu

./CUDA/tealeaf_kernels_mpi_cu.o:	./CUDA/tea_leaf_kernels.cu \
				Makefile
				$(NVCC) $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI $(CODE_GEN_CUDA) $(OPS_INC) $(HDF5_INC) -I. -I$(MPI_INSTALL_PATH)/include \
				-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/tealeaf_kernels_mpi_cu.o ./CUDA/tea_leaf_kernels.cu

./CUDA/tealeaf_kernels_mpi_tiled_cu.o:	./CUDA/tea_leaf_kernels.cu \
				Makefile
				$(NVCC) $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI -DOPS_LAZY $(CODE_GEN_CUDA) $(OPS_INC) $(HDF5_INC) -I. -I$(MPI_INSTALL_PATH)/include \
				-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/tealeaf_kernels_mpi_tiled_cu.o ./CUDA/tea_leaf_kernels.cu

openacc_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
openacc_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")

./OpenACC/tea_leaf_kernels.o: ./OpenACC/tea_leaf_kernels.cpp ./OpenACC/tea_leaf_common.h Makefile
		$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) -DOPS_MPI -DOPS_GPU $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenACC/tea_leaf_kernels.cpp -c -o OpenACC/tea_leaf_kernels.o

./OpenACC/%.o: ./OpenACC/%.cpp ./OpenACC/%_c.c ./OpenACC/tea_leaf_common.h Makefile
		@echo "Compiling kernel" $*
		$(MPICC) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) -DOPS_GPU -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenACC/$*_c.c -c -o OpenACC/$*_c.o
		$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) -DOPS_GPU -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenACC/$*.cpp -c -o OpenACC/$*.o

tealeaf_openacc: $(openacc_obj_list) $(PLAIN_FILES) $(OPS_FILES_GEN) Makefile $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	        $(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) -DOPS_MPI -DOPS_GPU $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) \
		$(PLAIN_FILES) $(OPS_FILES_GEN) -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_mpi_cuda $(HDF5_LIB) -o tealeaf_openacc

tealeaf_mpi_openacc: $(openacc_obj_list) $(PLAIN_FILES) $(OPS_FILES_GEN) Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	        $(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) -DOPS_MPI -DOPS_GPU $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) -DOPS_MPI \
		$(PLAIN_FILES) $(OPS_FILES_GEN) -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_mpi_cuda $(HDF5_LIB) -o tealeaf_mpi_openacc


openmp4_obj_list = $(shell find ./OpenMP4/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
openmp4_c_obj_list = $(shell find ./OpenMP4/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")

./OpenMP4/tea_leaf_kernels.o: ./OpenMP4/tea_leaf_kernels.cpp ./OpenMP4/tea_leaf_common.h Makefile
		$(MPICPP) -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda $(MPIFLAGS) -DOPS_GPU $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenMP4/tea_leaf_kernels.cpp -c -o OpenMP4/tea_leaf_kernels.o

./OpenMP4/%.o: ./OpenMP4/%.cpp ./OpenMP4/%_c.c ./OpenMP4/tea_leaf_common.h Makefile
		@echo "Compiling kernel" $*
		$(MPICPP) -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda $(MPIFLAGS) -DOPS_GPU -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenMP4/$*_c.c -c -o OpenMP4/$*_c.o
		$(MPICPP) -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda $(MPIFLAGS) -DOPS_GPU $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenMP4/$*.cpp -c -o OpenMP4/$*.o

tealeaf_openmp4: $(openmp4_obj_list) $(PLAIN_FILES) $(OPS_FILES_GEN) \
                $(HEADERS) Makefile 
	        $(MPICPP) -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda $(MPIFLAGS) -DOPS_GPU $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) \
		$(PLAIN_FILES) $(OPS_FILES_GEN) -I. $(openmp4_obj_list) -lcudart -lops_openmp4 $(HDF5_LIB) -o tealeaf_openmp4



tealeaf_opencl: ./OpenCL/tealeaf_opencl_kernels.o $(PLAIN_FILES) $(OPS_FILES_GEN) Makefile $(OPS_INSTALL_PATH)/lib/libops_opencl.a
		$(CPP) $(MPIFLAGS) $(OCL_FLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) ./OpenCL/tealeaf_opencl_kernels.o -I. ./OpenCL/$(SEQ_KERNELS) -lops_opencl $(HDF5_LIB) -o tealeaf_opencl

./OpenCL/tealeaf_opencl_kernels.o: ./OpenCL/tea_leaf_opencl_kernels.cpp Makefile
		$(MPICPP) $(CPPFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) $(HDF5_INC) -I. \
                -c -o ./OpenCL/tealeaf_opencl_kernels.o ./OpenCL/tea_leaf_opencl_kernels.cpp

tealeaf_mpi_opencl: ./OpenCL/tealeaf_mpi_opencl_kernels.o $(PLAIN_FILES) $(OPS_FILES_GEN) Makefile $(OPS_INSTALL_PATH)/lib/libops_opencl.a
	        $(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OCL_FLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) $(PLAIN_FILES) $(OPS_FILES_GEN) \
					 ./OpenCL/tealeaf_mpi_opencl_kernels.o -I. ./OpenCL/$(SEQ_KERNELS) -lops_mpi_opencl $(HDF5_LIB) -o tealeaf_mpi_opencl

./OpenCL/tealeaf_mpi_opencl_kernels.o: ./OpenCL/tea_leaf_opencl_kernels.cpp Makefile
		$(MPICPP) $(MPIFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) $(HDF5_INC) -I. -DOPS_MPI \
                -c -o ./OpenCL/tealeaf_mpi_opencl_kernels.o ./OpenCL/tea_leaf_opencl_kernels.cpp


#
#error
#
openacc_error:
	@echo "tealeaf_mpi_openacc cannot be built with $(OPS_COMPILER) compilers .. requires an OpenACC compiler (e.g. pgi)"


#
# cleanup
#

clean:
		rm -f tealeaf_dev_seq tealeaf_dev_mpi tealeaf_mpi tealeaf_seq tealeaf_tiled tealeaf_mpi_tiled tealeaf_openmp tealeaf_mpi_openmp tealeaf_cuda tealeaf_mpi_cuda tealeaf_mpi_cuda_tiled tealeaf_openacc tealeaf_mpi_openacc ./CUDA/*.o ./OpenACC/*.o *.o tealeaf_opencl tealeaf_mpi_opencl ./OpenCL/*.o *.o *.lst
