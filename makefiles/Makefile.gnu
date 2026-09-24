CC      := gcc
CXX       := g++
FC := gfortran

ifdef DEBUG
	CCFLAGS   := -O0 -std=c99 -fPIC -Wall -g
	CXXFLAGS   := -O0 -fPIC -Wall -g -std=c++11 #-fsanitize=address -fsanitize=undefined
	FFLAGS += -O0 -g -ffree-form -ffree-line-length-none
else
	CCFLAGS   := -O3 -std=c99 -fPIC -Wall -g -ftree-vectorize -fopenmp
	CXXFLAGS   := -O3 -fPIC -Wall -g -std=c++11 -fopenmp
	FFLAGS += -O3 -g -ffree-form -ffree-line-length-none -fopenmp
endif

ifdef IEEE
	CCFLAGS += -fno-fast-math -ffp-contract=off -fno-associative-math
	CXXFLAGS += -fno-fast-math -ffp-contract=off -fno-associative-math
	FFLAGS += -fno-fast-math -ffp-contract=off -fno-associative-math
endif

# -ffloat-store spills every FP intermediate to memory. It only guards against
# x87 excess precision, which x86-64 SSE2 does not have, so on x86-64 it buys
# nothing and costs about 2x on cache-blocked (tiled) runs, where the working
# set is in L2/L3 and the extra stores are no longer hidden by DRAM stalls.
ifdef FLOAT_STORE
	CCFLAGS += -ffloat-store
	CXXFLAGS += -ffloat-store
endif

OMPFLAGS := -fopenmp
ifdef THREADED
	THREADING_FLAGS ?= -fopenmp
endif

FMODS   := -J$(F_INC_MOD)
FMODS_F2C_CUDA    := -J$(F_INC_MOD)/f2c_cuda

CXXLINK := -lstdc++
FTNLINK := -lgfortran
