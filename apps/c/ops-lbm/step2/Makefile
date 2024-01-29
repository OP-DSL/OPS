CC       = g++
OMPFLAGS = -O3 -lm -fopenmp -std=c++11

BIN =  lattboltz2d

all: $(BIN)

lattboltz2d: lattboltz2d.cpp Makefile
	$(CC) $(CCFLAGS) -I$(OPS_INSTALL_PATH)/c/include $(OMPFLAGS) -o $@ lattboltz2d.cpp -L$(OPS_INSTALL_PATH)/c/lib/$(OPS_COMPILER) -lops_seq

clean:
	$(RM) $(BIN)
