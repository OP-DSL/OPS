CC       = g++
OMPFLAGS = -O3 -lm

BIN =  lattboltz2d

all: $(BIN)

lattboltz2d: lattboltz2d.cpp Makefile
	$(CC) $(CCFLAGS) $(OMPFLAGS) -o $@ lattboltz2d.cpp

clean:
	$(RM) $(BIN)
