#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

#
# set paths for header files and libraries
#
OPS_INC		= -I$(OPS_INSTALL_PATH)/include
OPS_LIB		= -L$(OPS_INSTALL_PATH)/lib


ifeq ($(OPS_COMPILER),gnu)
  CPP		= g++
  CPPFLAGS	= -g -fPIC -DUNIX -Wall
  OMPFLAGS	= -fopenmp
  MPICPP	= /usr/bin/mpiCC
  MPIFLAGS	= $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),intel)
  CPP		= icpc
  CCFLAGS	= -g -O0 #-vec-report -xSSE4.2 #-DCOMM_PERF #-DDEBUG
  CPPFLAGS	= $(CCFLAGS)
  OMPFLAGS	= -openmp -openmp-report2
  MPICPP	= mpiCC
  MPIFLAGS	= $(CCFLAGS)
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif



#
# master to make all versions
#

all: clean cloverleaf_seq


#
# simple sequential version
#

cloverleaf_seq:	clover_leaf.cpp test_kernel.h
		$(CPP) $(CPPFLAGS) clover_leaf.cpp $(OPS_INC) $(OPS_LIB) -lops_seq -o cloverleaf_seq
#
# cleanup
#

clean:
		rm -f cloverleaf_seq *.o

