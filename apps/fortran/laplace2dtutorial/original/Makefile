FC       = ifort
OMPFLAGS = -Ofast

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_original.F90 Makefile
	$(FC) $(OMPFLAGS) -o $@ laplace2d_original.F90 -lm

clean:
	$(RM) $(BIN)
