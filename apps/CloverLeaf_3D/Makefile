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
CUDA_LIB = -L$(CUDA_INSTALL_PATH)/lib64
CUDA_INC = -I$(CUDA_INSTALL_PATH)/include


ifeq ($(OPS_COMPILER),gnu)
  CPP		= g++
ifdef DEBUG
  CPPFLAGS	= -O0 -g -DUNIX -Wall
else
  CPPFLAGS	= -O3 -fPIC -DUNIX -Wall
endif
  OMPFLAGS	= -fopenmp
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPIFLAGS	= $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),intel)
  CPP		= icpc
ifdef DEBUG
  CCFLAGS	= -O0 -g -no-prec-div -openmp -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK #-DOPS_DEBUG
else
  #CCFLAGS	= -O3 -ipo -no-prec-div -fp-model strict -fp-model source -prec-div -prec-sqrt -vec-report2 -xSSE4.2 -parallel #-DCOMM_PERF #-DDEBUG
  CCFLAGS	= -O3 -g -ipo -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK -vec-report
endif
  CPPFLAGS	= $(CCFLAGS)
  OMPFLAGS	= -openmp -openmp-report2
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPIFLAGS	= $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),pgi)
  CPP 		= pgCC
	CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
ifdef DEBUG
  CCFLAGS 	= -O2 -Minline -Kieee -Minform=inform -Minfo=all
else
  CCFLAGS 	= -O3 -Kieee -Minline -Minform=severe -Minfo=all
endif
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS 	= -mp
  MPICPP 	= $(MPI_INSTALL_PATH)/bin/mpiCC
  MPICC 	= $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS 	= $(CPPFLAGS)
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif


# flags for nvcc
# set NV_ARCH to select the correct one
ifndef NV_ARCH
  MESSAGE=select an NVIDA device to compile in CUDA, e.g. make NV_ARCH=KEPLER
  NV_ARCH=Kepler
endif
ifeq ($(NV_ARCH),Fermi)
	CODE_GEN_CUDA=-gencode arch=compute_20,code=sm_21
else
ifeq ($(NV_ARCH),Kepler)
	CODE_GEN_CUDA=-gencode arch=compute_35,code=sm_35
endif
endif


ifdef DEBUG
  NVCC_FLAGS = -O0 -g -G
else
	NVCC_FLAGS = -O3 
endif


#
# master to make all versions
#

all: clean cloverleaf_dev_seq cloverleaf_dev_mpi cloverleaf_mpi cloverleaf_mpi_openmp cloverleaf_openmp cloverleaf_seq cloverleaf_cuda cloverleaf_openacc


cloverleaf_dev_seq: clover_leaf.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk.cpp \
                initialise.cpp generate.cpp ideal_gas.cpp \
                update_halo.cpp field_summary.cpp time_step.cpp viscosity.cpp \
                calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp flux_calc.cpp \
                advection.cpp advec_cell.cpp advec_mom.cpp reset_field.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
		$(MPICPP) $(CPPFLAGS) $(OPS_INC) $(OPS_LIB) clover_leaf.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk.cpp initialise.cpp generate.cpp \
                ideal_gas.cpp update_halo.cpp field_summary.cpp time_step.cpp \
                viscosity.cpp calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp \
                flux_calc.cpp advection.cpp reset_field.cpp advec_cell.cpp \
                advec_mom.cpp -lops_seq -o cloverleaf_dev_seq


cloverleaf_dev_mpi: clover_leaf.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk.cpp \
	        initialise.cpp generate.cpp ideal_gas.cpp \
                update_halo.cpp field_summary.cpp time_step.cpp viscosity.cpp \
                calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp flux_calc.cpp \
                advection.cpp advec_cell.cpp advec_mom.cpp reset_field.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
	        $(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) clover_leaf.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk.cpp initialise.cpp generate.cpp \
                ideal_gas.cpp update_halo.cpp field_summary.cpp time_step.cpp \
                viscosity.cpp calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp \
                flux_calc.cpp advection.cpp reset_field.cpp advec_cell.cpp \
		advec_mom.cpp -lops_mpi -o cloverleaf_dev_mpi

#
# mpi version
#

SEQ_KERNELS = clover_leaf_seq_kernels.cpp

cloverleaf_mpi: clover_leaf.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
	        initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
	        $(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) clover_leaf.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_mpi -o cloverleaf_mpi

#
# OpenMP version
#
OMP_KERNELS = clover_leaf_omp_kernels.cpp

cloverleaf_mpi_openmp: clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
	        initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
	        $(MPICPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_mpi -o cloverleaf_mpi_openmp


cloverleaf_openmp: clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
                initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
	        $(MPICPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
                advec_mom_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_openmp -o cloverleaf_openmp

#
# Sequential version
#
cloverleaf_seq: clover_leaf.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
                initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
		$(MPICPP) $(MPIFLAGS) $(OPS_INC) $(OPS_LIB) clover_leaf.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
                advec_mom_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_seq -o cloverleaf_seq

#
# CUDA version
#

cloverleaf_cuda: ./CUDA/cloverleaf_kernels_cu.o clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
		initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
		update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
		calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
		advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
		initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
		update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
		PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
		reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
		$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
		initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
		ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
		viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
		flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp ./CUDA/cloverleaf_kernels_cu.o -lcudart -lops_cuda -o cloverleaf_cuda

cloverleaf_mpi_cuda: ./CUDA/cloverleaf_kernels_cu.o clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
		initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
		update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
		calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
		advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
		initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
		update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
		PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
		reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
		$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC) $(OPS_LIB) $(CUDA_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
		initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
		ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
		viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
		flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp ./CUDA/cloverleaf_kernels_cu.o -lcudart -lops_mpi_cuda -o cloverleaf_mpi_cuda

./CUDA/cloverleaf_kernels_cu.o:	./CUDA/clover_leaf_kernels.cu \
				Makefile
				nvcc $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC) -I. \
				-c -o ./CUDA/cloverleaf_kernels_cu.o ./CUDA/clover_leaf_kernels.cu

openacc_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
openacc_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")


./OpenACC/clover_leaf_kernels.o: ./OpenACC/clover_leaf_kernels.cpp ./OpenACC/clover_leaf_common.h Makefile
				$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
        -I. OpenACC/clover_leaf_kernels.cpp -c -o OpenACC/clover_leaf_kernels.o

./OpenACC/%.o: ./OpenACC/%.cpp ./OpenACC/%_c.c ./OpenACC/clover_leaf_common.h Makefile
				@echo "Compiling kernel" $*
				$(MPICC) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
        -I. OpenACC/$*_c.c -c -o OpenACC/$*_c.o
				$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(CUDA_INC) \
				-I. OpenACC/$*.cpp OpenACC/$*_c.o -c -o OpenACC/$*.o

cloverleaf_openacc: $(openacc_obj_list) clover_leaf.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
	        initialise.cpp generate.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h
	        $(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) $(CUDA_ALIGN_FLAG) $(OPS_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) \
								 clover_leaf.cpp start.cpp build_field.cpp \
                read_input.cpp initialise_chunk_ops.cpp initialise.cpp generate.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
								advec_mom_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_cuda -o cloverleaf_openacc
#
# cleanup
#

clean:
		rm -f cloverleaf_dev_seq cloverleaf_dev_mpi cloverleaf_mpi cloverleaf_seq cloverleaf_openmp cloverleaf_mpi_openmp cloverleaf_cuda cloverleaf_openacc ./CUDA/*.o ./OpenACC/*.o *.o
