# if using GNU programming environment on cray machine
ifneq ($(findstring GNU,$(PE_ENV)),)
	CC  := cc
	CXX := CC
	FC  := ftn
else
	CC  := gcc
	CXX := g++
	FC  := gfortran
endif

ifdef DEBUG
	CCFLAGS   := -O0 -std=c99 -fPIC -Wall -ffloat-store -g
	CXXFLAGS   := -O0 -fPIC -Wall -ffloat-store -g -std=c++11 #-fsanitize=address -fsanitize=undefined
	FFLAGS += -O0 -g -ffree-form -ffree-line-length-none
else
	CCFLAGS   := -O3 -std=c99 -fPIC -Wall -ffloat-store -g -ftree-vectorize -fopenmp
	CXXFLAGS   := -O3 -fPIC -Wall -ffloat-store -g -std=c++11 -fopenmp
	FFLAGS += -O3 -g -ffree-form -ffree-line-length-none -fopenmp
endif

OMPFLAGS := -fopenmp
ifdef THREADED
	THREADING_FLAGS ?= -fopenmp
endif

FMODS   := -J$(F_INC_MOD)
FMODS_F2C_CUDA    := -J$(F_INC_MOD)/f2c_cuda
FMODS_OMPOFFLOAD    := -J$(F_INC_MOD)/ompoffload
FMODS_F2C_HIP    := -J$(F_INC_MOD)/f2c_hip

CXXLINK := -lstdc++
FTNLINK := -lgfortran

ifneq ($(findstring GNU,$(PE_ENV)),)
MPI_INSTALL_PATH = $(CRAY_MPICH_DIR)

MPICC=cc
MPICPP=CC
MPICXX=CC
MPIFC=ftn
MPIF90=ftn

MPI_INC = $(CRAY_MPICH_DIR)/include
MPI_LIB = $(CRAY_MPICH_DIR)/lib $(PE_MPICH_GTL_DIR_amd_gfx90a)
MPI_LINK = -lmpi
endif

ifeq ($(AMD_ARCH),MI100)
HIP_ARCH=gfx908
else
ifeq ($(AMD_ARCH),MI200)
#valid for ARCHER2 AMD Instinct MI210 and LUMI AMD MI250X GPU
HIP_ARCH=gfx90a
endif
endif

ifneq ($(HIP_INSTALL_PATH),)
HIPCC=hipcc
HIPMPICXX=hipcc
#ROCM_PATH environment variable provided by rocm module loaded
HIPFLAGS = -x hip -std=c++17 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH} --offload-arch=${HIP_ARCH}
HIP_LINK = -L${HIP_INSTALL_PATH}/lib -lamdhip64
HIP_LINK+= -lrocm_smi64

CXXFLAGS := $(filter-out -ffloat-store, $(CXXFLAGS))

#$PE_MPICH_GTL_LIBS_amd_gfx90a = -lmpi_gtl_hsa
MPI_HIP_LINK = -lmpi_gtl_hsa

OMPOFFLOADFLAGS= -fopenmp --offload-arch=${HIP_ARCH}
OMPOFFLOADFOR = -DOPS_WITH_OMPOFFLOADFOR

#$PE_MPICH_GTL_LIBS_amd_gfx90a = -lmpi_gtl_hsa
MPI_OMPOFFLOAD_LINK = -lmpi_gtl_hsa
endif

