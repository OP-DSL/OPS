# Compiler settings
MPIF90 = /opt/mpi/openmpi/1.10.6/gnu/6.3.0/bin/mpif90
MPICC  = /opt/mpi/openmpi/1.10.6/gnu/6.3.0/bin/mpic++

# Flags
FFLAGS = -O3 -g -ffree-form -ffree-line-length-none -fopenmp -DOPS_F2C
CFLAGS = -O3 -fPIC -Wall -ffloat-store -g -std=c++11 -fopenmp -DOPS_FTN

# Directories
FORTRAN_MOD = /ext-home/asl/OPS_cg/f2c/OPS/ops/fortran/mod/gnu
FORTRAN_LIB = /ext-home/asl/OPS_cg/f2c/OPS/ops/fortran/lib/gnu
C_INCLUDE = /ext-home/asl/OPS_cg/f2c/OPS/ops/c/include

# Source and object files
FORTRAN_SRCS = constants.F90 apply_stencil_kernel_seq_kernel.F90 left_bndcon_kernel_seq_kernel.F90 \
               right_bndcon_kernel_seq_kernel.F90 set_zero_kernel_seq_kernel.F90 \
               copy_kernel_seq_kernel.F90 laplace2d_ops.F90
FORTRAN_OBJS = $(FORTRAN_SRCS:.F90=.o)

CXX_SRCS = mpi_openmp_master_kernel.cpp 
CXX_OBJS = $(CXX_SRCS:.cpp=_c.o)

# Target executable
TARGET = laplace2d_c_seq

# Rules
all: $(TARGET)

# Compile Fortran source files
%.o: %.F90
	$(MPIF90) $(FFLAGS) -J$(FORTRAN_MOD) -I$(FORTRAN_MOD) -c $< -o $@

# Compile C++ source files
%_c.o: %.cpp
	$(MPICC) $(CFLAGS) -I$(C_INCLUDE) -c $< -o $@

# Link everything together
$(TARGET): $(FORTRAN_OBJS) $(CXX_OBJS)
	$(MPIF90) $(FFLAGS) -J$(FORTRAN_MOD) -I$(C_INCLUDE) -I$(FORTRAN_MOD) \
	$(CXX_OBJS) $(FORTRAN_OBJS) -L$(FORTRAN_LIB) -lops_for_seq -lstdc++ -o $@

# Clean up
clean:
	rm -f $(FORTRAN_OBJS) $(CXX_OBJS) $(TARGET)

