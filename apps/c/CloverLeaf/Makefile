#
# The following environment variables should be predefined:
#
# OPS_INSTALL_PATH
# OPS_COMPILER (gnu,intel,etc)
#

#
# set paths for header files and libraries
#
OPS_INSTALL_PATH := $(OPS_INSTALL_PATH)/c
OPS_INC		= -I$(OPS_INSTALL_PATH)/include
OPS_LIB		= -L$(OPS_INSTALL_PATH)/lib
CUDA_LIB = -L$(CUDA_INSTALL_PATH)/lib64
CUDA_INC = -I$(CUDA_INSTALL_PATH)/include

OPENCL_INC = -I$(OPENCL_INSTALL_PATH)/include
OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lOpenCL

ifdef HDF5_INSTALL_PATH
  HDF5_INC 	  	:= -I$(HDF5_INSTALL_PATH)/include
  HDF5_LIB 	  	:= -L$(HDF5_INSTALL_PATH)/lib -lhdf5_hl -lhdf5 -lz
ifdef CHECKPOINTING
  HDF5_INC              := $(HDF5_INC) -DCHECKPOINTING
endif
endif

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
  CCFLAGS	= -O3 -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK #-qopt-report #-qopt-report=5 # -vec-report
endif
  CPPFLAGS	= $(CCFLAGS)
  OMPFLAGS	= -openmp #-openmp-report2
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS	= $(CCFLAGS)
else
ifeq ($(OPS_COMPILER),mic)
  CPP		= icpc
ifdef DEBUG
  CCFLAGS	= -O0 -g -mmic -DMPICH_IGNORE_CXX_SEEK#-vec-report
else
  #CCFLAGS	= -O3 -ipo -no-prec-div -restrict -fno-alias -fp-model strict -fp-model source -prec-div -prec-sqrt -DMPICH_IGNORE_CXX_SEEK#-vec-report
  #CCFLAGS	= -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -finline -inline-forceinline -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK#-vec-report
  CCFLAGS	= -O3 -ipo -no-prec-div -restrict -fno-alias -mmic -vec-report -DSIMD_VEC=32 -DMPICH_IGNORE_CXX_SEEK -finline#-vec-report
endif
  CPPFLAGS	= $(CCFLAGS)
  OMPFLAGS	= -openmp -openmp-report2
  MPICPP	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPIFLAGS	= $(CCFLAGS)
  OPENCL_LIB = -L$(OPENCL_INSTALL_PATH)/lib64 -lintelocl -lOpenCL #for offload opencl execution on the phi
else
ifeq ($(OPS_COMPILER),pgi)
  CPP 		= pgc++
ifdef DEBUG
  CCFLAGS 	= -O0 -g -Minline -Kieee #-Minform=inform -Minfo=all
else
  CCFLAGS 	= -O2 -Kieee -nofma -Minline #-fastsse -gopt -Mipa=fast -Mlist -Kieee #-Minline #-Minform=severe -Minfo=all
endif
  CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
  CPPFLAGS 	= $(CCFLAGS)
  OMPFLAGS 	= -mp=nonuma
  MPICPP 	= $(MPI_INSTALL_PATH)/bin/mpicxx
  MPICC 	= $(MPI_INSTALL_PATH)/bin/mpicc
  MPIFLAGS 	= $(CPPFLAGS) -ldl
else
ifeq ($(OPS_COMPILER),cray)
  CPP       = CC
    CUDA_ALIGN_FLAG := -D__x86_64 -D__align__\(n\)=__attribute__\(\(aligned\(n\)\)\) -D__location__\(a\)=__annotate__\(a\) -DCUDARTAPI=
ifdef DEBUG
  CCFLAGS   = -O2
else
  CCFLAGS   = -O3
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  =
  MPICPP    = CC
  MPICC     = cc
  MPIFLAGS  = $(CPPFLAGS)
else
ifeq ($(OPS_COMPILER),xl)
  CPP       = mpixlc
  CUDA_ALIGN_FLAG :=
ifdef DEBUG
  CCFLAGS   = -O2 -g
else
  CCFLAGS   = -O5 -qnoeh -qipa=partition=large -g -qfullpath -Q -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036 -qsrcmsg -qfloat=nomaf
endif
  CPPFLAGS  = $(CCFLAGS)
  OMPFLAGS  = -qsmp=omp  -qthreaded
  MPICPP    = mpixlcxx
  MPICC     = mpixlc
  MPIFLAGS  = $(CPPFLAGS)
else
print:
	@echo "unrecognised value for OPS_COMPILER"
endif
endif
endif
endif
endif
endif

ifdef HDF5_INSTALL_PATH
  CC = $(MPICC)
  CPP = $(MPICPP)
endif

NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc
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
  NVCC_FLAGS = -O3 -restrict --fmad false
endif

#switch on FMA instructions in OpenCL
OCL_FLAGS=#-DOCL_FMA_SWITCH_ON

#
# master to make all versions
#

TARGETS = cloverleaf_dev_seq cloverleaf_dev_mpi cloverleaf_mpi cloverleaf_mpi_openmp cloverleaf_openmp cloverleaf_seq cloverleaf_cuda cloverleaf_mpi_cuda cloverleaf_opencl cloverleaf_mpi_opencl #cloverleaf_openacc cloverleaf_mpi_openacc

ifeq ($(OPS_COMPILER),pgi)
TARGETS += cloverleaf_openacc cloverleaf_mpi_openacc
else
TARGETS += openacc_error
endif



all: clean $(TARGETS)

cloverleaf_dev_seq: Makefile clover_leaf.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk.cpp \
                initialise.cpp generate.cpp ideal_gas.cpp \
                update_halo.cpp field_summary.cpp time_step.cpp viscosity.cpp \
                calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp flux_calc.cpp \
                advection.cpp advec_cell.cpp advec_mom.cpp reset_field.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
		$(CPP) $(CPPFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) clover_leaf.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk.cpp initialise.cpp generate.cpp \
                ideal_gas.cpp update_halo.cpp field_summary.cpp time_step.cpp \
                viscosity.cpp calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp \
                flux_calc.cpp advection.cpp reset_field.cpp advec_cell.cpp \
                advec_mom.cpp -lops_seq $(HDF5_LIB) -o cloverleaf_dev_seq


cloverleaf_dev_mpi: Makefile clover_leaf.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk.cpp \
	        initialise.cpp generate.cpp ideal_gas.cpp \
                update_halo.cpp field_summary.cpp time_step.cpp viscosity.cpp \
                calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp flux_calc.cpp \
                advection.cpp advec_cell.cpp advec_mom.cpp reset_field.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	        $(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) clover_leaf.cpp start.cpp  \
                build_field.cpp read_input.cpp initialise_chunk.cpp initialise.cpp generate.cpp \
                ideal_gas.cpp update_halo.cpp field_summary.cpp time_step.cpp \
                viscosity.cpp calc_dt.cpp PdV.cpp revert.cpp accelerate.cpp \
                flux_calc.cpp advection.cpp reset_field.cpp advec_cell.cpp \
		advec_mom.cpp -lops_mpi $(HDF5_LIB) -o cloverleaf_dev_mpi

#
# mpi version
#

SEQ_KERNELS = clover_leaf_seq_kernels.cpp

cloverleaf_mpi: Makefile clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
	        initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	        $(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_mpi $(HDF5_LIB) -o cloverleaf_mpi

#
# OpenMP version
#
OMP_KERNELS = clover_leaf_omp_kernels.cpp

cloverleaf_mpi_openmp: Makefile clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
	        initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi.a
	        $(MPICPP) $(OMPFLAGS) $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) clover_leaf_ops.cpp start.cpp \
                build_field.cpp read_input.cpp initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_mpi $(HDF5_LIB) -o cloverleaf_mpi_openmp


cloverleaf_openmp: Makefile clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
                initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
	        $(CPP) $(OMPFLAGS) $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
                advec_mom_ops.cpp -I. ./MPI_OpenMP/$(OMP_KERNELS) -lops_seq $(HDF5_LIB) -o cloverleaf_openmp

#
# Sequential version
#
cloverleaf_seq: Makefile clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
                initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h $(OPS_INSTALL_PATH)/lib/libops_seq.a
		$(CPP) $(MPIFLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
                advec_mom_ops.cpp -I. ./MPI/$(SEQ_KERNELS) -lops_seq $(HDF5_LIB) -o cloverleaf_seq

#
# CUDA version
#

cloverleaf_cuda: Makefile ./CUDA/cloverleaf_kernels_cu.o clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
		initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
		update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
		calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
		advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
		initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
		update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
		PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
		reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h $(OPS_INSTALL_PATH)/lib/libops_cuda.a
		$(CPP) $(OMPFLAGS) $(CPPFLAGS) $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
		initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
		ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
		viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
		flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp ./CUDA/cloverleaf_kernels_cu.o -lcudart -lops_cuda $(HDF5_LIB) -o cloverleaf_cuda

cloverleaf_mpi_cuda: Makefile ./CUDA/cloverleaf_kernels_mpi_cu.o clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
		initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
		update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
		calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
		advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
		initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
		update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
		PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
		reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
		$(MPICPP) $(OMPFLAGS) $(CPPFLAGS) -DOPS_MPI $(CUDA_INC) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_LIB) clover_leaf_ops.cpp start.cpp \
		build_field.cpp read_input.cpp initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
		ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
		viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
		flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp ./CUDA/cloverleaf_kernels_mpi_cu.o -lcudart -lops_mpi_cuda $(HDF5_LIB) -o cloverleaf_mpi_cuda

./CUDA/cloverleaf_kernels_cu.o:	./CUDA/clover_leaf_kernels.cu \
				Makefile
				$(NVCC) $(VAR) $(INC) $(NVCC_FLAGS) $(CODE_GEN_CUDA) $(OPS_INC) $(HDF5_INC) -I. \
				-c -o ./CUDA/cloverleaf_kernels_cu.o ./CUDA/clover_leaf_kernels.cu

./CUDA/cloverleaf_kernels_mpi_cu.o:	./CUDA/clover_leaf_kernels.cu \
				Makefile
				$(NVCC) $(VAR) $(INC) $(NVCC_FLAGS) -DOPS_MPI $(CODE_GEN_CUDA) $(OPS_INC) $(HDF5_INC) -I. -I$(MPI_INSTALL_PATH)/include \
				-DMPICH_IGNORE_CXX_SEEK -c -o ./CUDA/cloverleaf_kernels_mpi_cu.o ./CUDA/clover_leaf_kernels.cu

openacc_obj_list = $(shell find OpenACC/ -name "*.cpp" | sed s/\\.cpp/\\.o/g | tr "\n" " ")
openacc_c_obj_list = $(shell find OpenACC/ -name "*_c.c" | sed s/\\.c/\\.o/g | tr "\n" " ")

./OpenACC/clover_leaf_kernels.o: ./OpenACC/clover_leaf_kernels.cpp ./OpenACC/clover_leaf_common.h Makefile
		$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenACC/clover_leaf_kernels.cpp -c -o OpenACC/clover_leaf_kernels.o

./OpenACC/%.o: ./OpenACC/%.cpp ./OpenACC/%_c.c ./OpenACC/clover_leaf_common.h Makefile
		@echo "Compiling kernel" $*
		$(MPICC) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenACC/$*_c.c -c -o OpenACC/$*_c.o
		$(MPICPP) -acc -ta=tesla:cc35 -ta=tesla:maxregcount:64 $(MPIFLAGS) -DOPS_MPI $(CUDA_ALIGN_FLAG) $(OPS_INC) $(HDF5_INC) $(CUDA_INC) \
		-I. OpenACC/$*.cpp -c -o OpenACC/$*.o

cloverleaf_openacc: $(openacc_obj_list) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
	        initialise.cpp generate.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_cuda.a
	        $(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) \
		clover_leaf_ops.cpp start.cpp build_field.cpp \
                read_input.cpp initialise_chunk_ops.cpp initialise.cpp generate.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_mpi_cuda $(HDF5_LIB) -o cloverleaf_openacc

cloverleaf_mpi_openacc: $(openacc_obj_list) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
	        initialise.cpp generate.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_mpi_cuda.a
	        $(MPICPP) -acc -ta=tesla:cc35 $(MPIFLAGS) -DOPS_MPI $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(CUDA_INC) $(CUDA_LIB) -DOPS_MPI \
		clover_leaf_ops.cpp start.cpp build_field.cpp \
                read_input.cpp initialise_chunk_ops.cpp initialise.cpp generate.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
		advec_mom_ops.cpp -I. $(openacc_obj_list) $(openacc_c_obj_list) -lcudart -lops_mpi_cuda $(HDF5_LIB) -o cloverleaf_mpi_openacc

cloverleaf_opencl: ./OpenCL/cloverleaf_opencl_kernels.o clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
		initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
		update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
		calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
		advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
		initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
		update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
		PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
		reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_opencl.a
		$(CPP) $(MPIFLAGS) $(OCL_FLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
                advec_mom_ops.cpp ./OpenCL/cloverleaf_opencl_kernels.o -I. ./OpenCL/$(SEQ_KERNELS) -lops_opencl $(HDF5_LIB) -o cloverleaf_opencl

./OpenCL/cloverleaf_opencl_kernels.o: ./OpenCL/clover_leaf_opencl_kernels.cpp Makefile
		$(MPICPP) $(CPPFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) $(HDF5_INC) -I. \
                -c -o ./OpenCL/cloverleaf_opencl_kernels.o ./OpenCL/clover_leaf_opencl_kernels.cpp

cloverleaf_mpi_opencl: ./OpenCL/cloverleaf_mpi_opencl_kernels.o clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp initialise_chunk_ops.cpp \
                initialise.cpp generate_ops.cpp ideal_gas_ops.cpp \
                update_halo_ops.cpp field_summary_ops.cpp time_step.cpp viscosity_ops.cpp \
                calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp flux_calc_ops.cpp \
                advection.cpp advec_cell_ops.cpp advec_mom_ops.cpp reset_field_ops.cpp \
                initialise_chunk_kernel.h generate_chunk_kernel.h ideal_gas_kernel.h \
                update_halo_kernel.h field_summary_kernel.h viscosity_kernel.h \
                PdV_kernel.h revert_kernel.h accelerate_kernel.h flux_calc_kernel.h \
                reset_field_kernel.h advec_cell_kernel.h advec_mom_kernel.h Makefile $(OPS_INSTALL_PATH)/lib/libops_opencl.a
	        $(MPICPP) $(MPIFLAGS) -DOPS_MPI $(OCL_FLAGS) $(OPS_INC) $(HDF5_INC) $(OPS_LIB) $(OPENCL_LIB) clover_leaf_ops.cpp start.cpp build_field.cpp read_input.cpp \
                initialise_chunk_ops.cpp initialise.cpp generate_ops.cpp \
                ideal_gas_ops.cpp update_halo_ops.cpp field_summary_ops.cpp time_step.cpp \
                viscosity_ops.cpp calc_dt_ops.cpp PdV_ops.cpp revert_ops.cpp accelerate_ops.cpp \
                flux_calc_ops.cpp advection.cpp reset_field_ops.cpp advec_cell_ops.cpp \
                advec_mom_ops.cpp ./OpenCL/cloverleaf_mpi_opencl_kernels.o -I. ./OpenCL/$(SEQ_KERNELS) -lops_mpi_opencl $(HDF5_LIB) -o cloverleaf_mpi_opencl

./OpenCL/cloverleaf_mpi_opencl_kernels.o: ./OpenCL/clover_leaf_opencl_kernels.cpp Makefile
		$(MPICPP) $(MPIFLAGS) $(VAR) $(OPENCL_INC) $(INC) $(OPS_INC) $(HDF5_INC) -I. -DOPS_MPI \
                -c -o ./OpenCL/cloverleaf_mpi_opencl_kernels.o ./OpenCL/clover_leaf_opencl_kernels.cpp


#
#error
#
openacc_error:
	@echo "cloverleaf_mpi_openacc cannot be built with $(OPS_COMPILER) compilers .. requires an OpenACC compiler (e.g. pgi)"


#
# cleanup
#

clean:
		rm -f cloverleaf_dev_seq cloverleaf_dev_mpi cloverleaf_mpi cloverleaf_seq cloverleaf_openmp cloverleaf_mpi_openmp cloverleaf_cuda cloverleaf_mpi_cuda cloverleaf_openacc cloverleaf_mpi_openacc ./CUDA/*.o ./OpenACC/*.o *.o cloverleaf_opencl cloverleaf_mpi_opencl ./OpenCL/*.o *.o *.lst *.optrpt
