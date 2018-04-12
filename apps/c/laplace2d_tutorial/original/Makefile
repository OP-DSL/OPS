CC       = g++
OMPFLAGS = -Ofast -lm

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_original.cpp Makefile
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ laplace2d_original.cpp

clean:
	$(RM) $(BIN)

