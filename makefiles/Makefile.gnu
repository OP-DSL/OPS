CC      := gcc
CXX       := g++
FC := gfortran

ifdef DEBUG
	CCFLAGS   := -O0 -std=c99 -fPIC -Wall -ffloat-store -g
	CXXFLAGS   := -O0 -fPIC -Wall -ffloat-store -g -std=c++11
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

CXXLINK := -lstdc++
FTNLINK := -lgfortran
#MPI_LINK = -lmpi_cxx
