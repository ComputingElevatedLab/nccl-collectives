TARGET = nccl-ata-built-in
COMMON = ../common
CUDA_PATH ?= /opt/cuda
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl
SRC = $(TARGET).cu $(COMMON)/error-catch.cu

build:
	nvcc $(NVCC_FLAGS) $(SRC) -o $(TARGET).out $(LDFLAGS)

test:
	./$(TARGET).out

run: build test

debug:
	mpiexec -n 2 konsole -e gdb ./$(TARGET).out

clean:
	rm -f $(TARGET).out

.PHONY.: build