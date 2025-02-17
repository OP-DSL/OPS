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
CXXLINK := -lstdc++
FMODS   := -J$(F_INC_MOD)
OMPFLAGS := -fopenmp
ifdef THREADED
	THREADING_FLAGS ?= -fopenmp
endif
MPI_LINK = -lmpi_cxx
