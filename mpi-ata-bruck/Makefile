TARGET = mpi-ata-bruck
COMMON = ../common
FLAGS = -march=native -mtune=native -O3 -flto -fomit-frame-pointer -pipe -Wall
DEBUG_FLAGS = -Og -Wall
SRC = $(TARGET).cpp

build:
	mpic++ $(FLAGS) -o $(TARGET).out $(SRC) 

test:
	mpiexec -n 2 ./$(TARGET).out

run: build test

debug:
	mpic++ $(DEBUG_FLAGS) -o $(TARGET).out $(SRC)
	mpiexec -n 2 konsole -e gdb ./$(TARGET).out

clean:
	rm -f $(TARGET).out
	rm -f *.log

.PHONY: build
