export NCCL_DEBUG=WARN

TARGET = nccl-ata
COMMON = ../common
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl
THETA_FLAGS = -L/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.12.12-1_gcc-9.4.0-1ubuntu1-20.04/lib -L/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0/lib
SRC = $(TARGET).cu

build:
	nvcc $(NVCC_FLAGS) $(SRC) -o $(TARGET).out $(LDFLAGS) $(THETA_FLAGS)

test:
	mpiexec -n 2 ./$(TARGET).out

run: build test

debug:
	mpiexec -n 2 konsole -e gdb ./$(TARGET).out

clean:
	rm -f $(TARGET).out

.PHONY.: build