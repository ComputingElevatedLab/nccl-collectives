TARGET = nccl-ata-bruck
COMMON = ../common
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl
SRC = $(TARGET).cu

build: Makefile
	nvcc $(NVCC_FLAGS) $(SRC) -o $(TARGET).out $(LDFLAGS)

test:
	./$(TARGET).out

run: build test

debug:
	mpiexec -n 2 konsole -e gdb ./$(TARGET).out

clean:
	rm -f $(TARGET).out

.PHONY.: build