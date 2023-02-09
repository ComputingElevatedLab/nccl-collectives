export NCCL_DEBUG=WARN

TARGET = verify-nccl-bruck
NVCC_FLAGS += -dlto -lineinfo
LDFLAGS = -lmpi -lnccl -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib
SRC = $(TARGET).cu

build:
	nvcc $(NVCC_FLAGS) $(SRC) -o $(TARGET).out $(LDFLAGS)

test:
	mpirun -n 2 ./$(TARGET).out

run: build test

debug:
	mpirun -n 2 konsole -e gdb ./$(TARGET).out

clean:
	rm -f $(TARGET).out
	rm -f *.log

.PHONY.: build
