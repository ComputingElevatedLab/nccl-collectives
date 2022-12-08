TARGET = nccl-ata-bruck
COMMON = ../common
CUDA_PATH ?= /opt/cuda
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl
SRC = nccl-ata-bruck.cu $(COMMON)/error-catch.cu

build: Makefile nccl-ata-bruck.cu
	nvcc $(NVCC_FLAGS) $(SRC) -o $(TARGET).out $(LDFLAGS)

test:
	./$(TARGET).out

run: build test

clean:
	rm -f $(TARGET).out

.PHONY.: build