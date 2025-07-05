HIPCC ?= hipcc
CC      := hipcc
CXX     := hipcc

ifdef DEBUG
        CCFLAGS   := -O0 -std=c99 -fPIC -Wall -g #-ffloat-store
        CXXFLAGS   := -O0 -fPIC -Wall -g -std=c++11 #-ffloat-store
else
        CCFLAGS   := -O3 -std=c99 -fPIC -Wall -g -fopenmp #-ffloat-store
        CXXFLAGS  := -O3 -fPIC -Wall -g -std=c++11 -fopenmp
endif

ifdef IEEE
        CCFLAGS += -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -fno-associative-math
        CXXFLAGS += -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee -fno-associative-math
endif

OMPFLAGS = -fopenmp
OMPOFFLOADFLAGS = -fopenmp

HIP_INC ?= $(HIP_INSTALL_PATH)/include -I$(HIP_INSTALL_PATH)/../include
HIP_LIB ?= $(HIP_INSTALL_PATH)/lib
MPI_HIP_LINK ?= -L$(MPI_LIB) -lmpi

HIP_LINK += -lrocm_smi64

#HIP_LINK ?= $(shell mpicxx --showme:compile) -L$(HIP_INSTALL_PATH)/lib -L$(HIP_INSTALL_PATH)/lib64 -L$(HIP_INSTALL_PATH)/hsa/lib64 -L$(HIP_INSTALL_PATH)/hsa/lib -L$(HIP_INSTALL_PATH)/hcc/lib -lhc_am -lmcwamp -lhip_hcc -lhsa-runtime64 -lamd_comgr -lhsakmt $(shell mpicxx --showme:link)

HIPMPICPP ?= $(HIPCC)

ifneq ($(strip $(NV_ARCH)),)
HIPFLAGS += -D__HIP_PLATFORM_NVIDIA__
endif

ifneq ($(strip $(AMD_ARCH)),)
HIPFLAGS += -D__HIP_PLATFORM_AMD__
endif
