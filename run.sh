#! /bin/bash
#COBALT -n 10 -t 0:30:00 -q full-node -A dist_relational_alg

NODES=10
GPUS=8
NPROC=$(($NODES * $GPUS))

module load nccl
rm run.log
make clean
make all

# CPU
# mpirun -n $NPROC ./mpi-ata/mpi-ata.out
# mpirun -n $NPROC ./mpi-ata-bruck/mpi-ata-bruck.out
# mpirun -n $NPROC ./mpi-ata-spreadout/mpi-ata-spreadout.out

# GPU
mpirun -hostfile $COBALT_NODEFILE -n $NPROC -npernode $GPUS ./nccl-ata/nccl-ata.out
mpirun -hostfile $COBALT_NODEFILE -n $NPROC -npernode $GPUS ./nccl-ata-bruck/nccl-ata-bruck.out