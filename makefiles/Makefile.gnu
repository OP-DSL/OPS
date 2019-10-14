CC      := gcc
CXX       := g++
FC := gfortran
ifdef DEBUG
	CCFLAGS   := -O0 -std=c99 -fPIC -DUNIX -Wall -ffloat-store -g
	CXXFLAGS   := -O0 -fPIC -DUNIX -Wall -ffloat-store -g
else
	CCFLAGS   := -O3 -std=c99 -fPIC -DUNIX -Wall -ffloat-store -g -ftree-vectorize
	CXXFLAGS   := -O3 -fPIC -DUNIX -Wall -ffloat-store -g
endif
FFLAGS := $(CFLAGS) -ffree-form -ffree-line-length-none -J$(F_INC_MOD)
CXXLINK := -lstdc++
OMPFLAGS := -fopenmp
