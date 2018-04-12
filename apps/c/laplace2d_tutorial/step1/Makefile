CC       = g++
OMPFLAGS = -Ofast -lm

BIN =  laplace2d

all: $(BIN)

laplace2d: laplace2d_step1.cpp Makefile
	$(CC) $(CCFLAGS) -I$(OPS_INSTALL_PATH)/c/include $(OMPFLAGS) -o $@ laplace2d_step1.cpp -L$(OPS_INSTALL_PATH)/c/lib -lops_seq

clean:
	$(RM) $(BIN)

