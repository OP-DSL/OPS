CC       = gfortran
OMPFLAGS = -Ofast -lm

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_step1.F90 Makefile
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ laplace2d_step1.F90

clean:
	$(RM) $(BIN)
