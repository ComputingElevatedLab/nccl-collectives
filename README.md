# nccl-collectives
Implementing more optimized collectives for NCCL.

# Setup thetaGPU
```
ssh thetagpusn1
qsub -I -n 1 -t 10 -q full-node -A dist_relational_alg --attrs filesystems=home,grand,theta-fs0
cd <nccl-collectives-path>/
module load nccl
```

# Build
There are 4 all-to-all implementations, each stored in their own folder within the root directory. Each implementation has a Makefile that should compile on a local machine as well as on theta. `cd` into the implementation that you want to run and type `make`. The generated binary is called <implementation-name>.out and can be found in the same directory.

# Run
The command is the same for all implementations: `mpiexec -n <N> ./<implementation-name>.out`.

# Input and Output
For a given number of MPI processes (n), each process allocates an array of size n and fills each element with it's rank. So for `n = 2` the input looks like:
```
p1 = [0 0]
p2 = [1 1]
```

Then each implementation uses these arrays to perform the same operation: sending each rank to every other rank. Therefore the expected output when `n = 2` is:
```
p1 = [0 1]
p2 = [0 2]
```

For `n = 4` the input would be:
```
p1 = [0 0 0 0]
p2 = [1 1 1 1]
p3 = [2 2 2 2]
p4 = [3 3 3 3]
```

and the expected output would be:
```
p1 = [0 1 2 3]
p2 = [0 1 2 3]
p3 = [0 1 2 3]
p4 = [0 1 2 3]
```

# Debug
The relevant files to look at for the NCCL Bruck implementation are:
```
nccl-collectives/common/bruck.cu
nccl-collectives/nccl-ata-bruck/nccl-ata-bruck.cu
```

The files for the corresponding MPI-only Bruck implementation are:
```
nccl-collectives/common/bruck.cpp
nccl-collectives/mpi-ata-bruck/mpi-ata-bruck.cpp
```

The MPI-only Bruck implementation is confirmed to give the correct output, and can be used to as a reference when debugging the NCCL Bruck implementation.
