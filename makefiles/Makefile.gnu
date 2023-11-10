CC      := gcc
CXX       := g++
FC := gfortran

ifdef DEBUG
	CCFLAGS   := -O0 -std=c99 -fPIC -DUNIX -Wall -ffloat-store -g
	CXXFLAGS   := -O0 -fPIC -DUNIX -Wall -ffloat-store -g -std=c++11
	FFLAGS += -O0 -g -ffree-form -ffree-line-length-none -J$(F_INC_MOD)
else
	CCFLAGS   := -O3 -std=c99 -fPIC -Wall -ffloat-store -g -ftree-vectorize -fopenmp
	CXXFLAGS   := -O3 -fPIC -Wall -ffloat-store -g -std=c++11
	FFLAGS += -O3 -g -ffree-form -ffree-line-length-none -fopenmp -J$(F_INC_MOD)
endif
CXXLINK := -lstdc++
OMPFLAGS := -fopenmp
ifdef THREADED
	THREADING_FLAGS ?= -fopenmp
endif
