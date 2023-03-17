CC      := gcc
CXX       := g++
FC := gfortran
ifdef DEBUG
	CCFLAGS   := -O0 -std=c99 -fPIC -DUNIX -Wall -ffloat-store -g
	CXXFLAGS   := -O0 -fPIC -DUNIX -Wall -ffloat-store -g -std=c++11
else
	CCFLAGS   := -O3 -std=c99 -fPIC -Wall -ffloat-store -g -ftree-vectorize
	CXXFLAGS   := -O3 -fPIC -Wall -ffloat-store -g -std=c++11
endif
FFLAGS := $(CFLAGS) -O3 -ffree-form -ffree-line-length-none -J$(F_INC_MOD)
CXXLINK := -lstdc++
OMPFLAGS := -fopenmp
ifdef THREADED
	THREADING_FLAGS ?= -fopenmp
endif
