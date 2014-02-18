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
  CCFLAGS	= -O0 -g -no-prec-div -openmp -fp-model strict -fp-model source -prec-div -prec-sqrt
  #CCFLAGS	= -O3 -ipo -no-prec-div -fp-model strict -fp-model source -prec-div -prec-sqrt -vec-report2 -xSSE4.2 -parallel #-DCOMM_PERF #-DDEBUG
  #CCFLAGS	= -O3 -ipo -no-prec-div -fp-model strict -fp-model source -prec-div -prec-sqrt -vec-report
  #CCFLAGS	= -O3 -ipo -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -vec-report
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

all: clean cloverleaf_mpi


#
# mpi version
#
SEQ_KERNELS = clover_leaf_seq_kernels.cpp

cloverleaf_mpi: clover_leaf.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk.cpp \
	        initialise.cpp generate.cpp ideal_gas.cpp \
                update_halo.cpp field_summary.cpp time_step.cpp viscosity.cpp \
                calc_dt.cpp PdV_ops.cpp revert.cpp accelerate_ops.cpp flux_calc.cpp \
                advection.cpp advec_cell.cpp advec_mom.cpp reset_field.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
	        $(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) clover_leaf.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk.cpp initialise.cpp generate.cpp \
                ideal_gas.cpp update_halo.cpp field_summary.cpp time_step.cpp \
                viscosity.cpp calc_dt.cpp PdV_ops.cpp revert.cpp accelerate_ops.cpp \
                flux_calc.cpp advection.cpp reset_field.cpp advec_cell.cpp \
		advec_mom.cpp $(SEQ_KERNELS) -lops_mpi -o cloverleaf_mpi
#
# cleanup
#

clean:
		rm -f cloverleaf_mpi *.o
