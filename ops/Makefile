#
# The following environment variables should be predefined:
#
# OPS_COMPILER (gnu,intel,etc)
#

#
# set paths for various files
#

SRC = src
INC = include
LIB = lib
OBJ = obj

#
# compiler settings
#

ifeq ($(OPS_COMPILER),gnu)
  CC 	  	:= gcc
  CCFLAGS 	:= -std=c99 -fPIC -DUNIX -Wall #-g
  CXX 	  	:= g++
  CXXFLAGS	:= -std=c99 -fPIC -DUNIX -Wall #-g -Wextra
  MPICXX  	:= /usr/bin/mpiCC
  MPIFLAGS	:= $(CXXFLAGS)
else
ifeq ($(OPS_COMPILER),intel)
  CC 		:= icc
  CCFLAGS 	:= -O0 -g  #-vec-report -xSSE4.2
  CXX 		:= icpc
  CXXFLAGS 	:= $(CCFLAGS)
  MPICXX 	:= mpiCC
  MPIFLAGS 	:= $(CXXFLAGS)
else
ifeq ($(OPS_COMPILER),pgi)
  CC 		:= pgcc
  CCFLAGS 	:= -O3
  CXX 		:= pgCC
  CXXFLAGS 	:= $(CCFLAGS)
  MPICXX 	:= /home/gihan/openmpi-pgi/bin/mpiCC
  MPIFLAGS 	:= $(CXXFLAGS)
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif

.PHONY: clean mklib

all: clean core seq
mklib:
	@mkdir -p $(LIB) $(OBJ)

core: mklib $(INC)/ops_lib_core.h $(SRC)/core/ops_lib_core.c $(SRC)/externlib/ops_util.c
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/core/ops_lib_core.c -o $(OBJ)/ops_lib_core.o
	$(CXX) $(CXXFLAGS) -I$(INC) -c $(SRC)/externlib/ops_util.c -o $(OBJ)/ops_util.o

seq: mklib core $(OBJ)/ops_lib_core.o
	ar -r $(LIB)/libops_seq.a $(OBJ)/ops_lib_core.o $(OBJ)/ops_util.o

clean:
	-rm -if $(OBJ)/*.o
	-rm -if $(OBJ)/*~
	-rm -if $(LIB)/*.a
	-rm -if $(SRC)/*~
	-rm -if $(INC)/*~
	-rm -if *~
