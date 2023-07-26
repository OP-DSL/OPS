CC       = g++
OMPFLAGS = -Ofast -lm  -std=c++11 -fopenmp

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_step4.cpp Makefile
	$(CC) $(CCFLAGS) -I$(OPS_INSTALL_PATH)/c/include $(OMPFLAGS) -o $@ laplace2d_step4.cpp -L$(OPS_INSTALL_PATH)/c/lib/$(OPS_COMPILER) -lops_seq

clean:
	$(RM) $(BIN)

