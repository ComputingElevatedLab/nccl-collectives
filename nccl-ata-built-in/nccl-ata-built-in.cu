// Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#communicator-creation-and-destruction-examples
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "../common/error-catch.cu"
#include "../common/hostname.cu"

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_CALL(MPI_Init(&argc, &argv));
    int world_rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    int world_size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    // Figure out what host the current MPI process is running on
    uint64_t hostHashs[world_size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[world_rank] = getHostHash(hostname);
    MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    // Determining local_rank is necessary for selecting a GPU to associate the current MPI process with
    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm));
        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    // Initialize send and recieve arrays for every GPU device, for every node
    int num_devices = 0;
    CUDA_CALL(cudaGetDeviceCount(&num_devices));
    CUDA_CALL(cudaSetDevice(local_rank % num_devices));

    if (world_rank == 0 && num_devices < world_size) {
        printf("This code assumes that one MPI process is used per CUDA device\n");
        printf("There are %d MPI processes but only %d CUDA devices\n", world_size, num_devices);
        return 1;
    }

    int** d_send_data = (int**) malloc(num_devices * sizeof(int*));
    int** d_recv_data = (int**) malloc(num_devices * sizeof(int*));
    cudaStream_t* stream = (cudaStream_t*) malloc(num_devices * sizeof(cudaStream_t));

    for (int i = 0; i < num_devices; ++i) {
        CUDA_CALL(cudaSetDevice(local_rank % num_devices + i));
        CUDA_CALL(cudaMalloc(d_send_data + i, world_size * sizeof(float)));
        CUDA_CALL(cudaMalloc(d_recv_data + i, world_size * sizeof(float)));
        CUDA_CALL(cudaMemset(d_send_data[i], 1, world_size * sizeof(float)));
        CUDA_CALL(cudaMemset(d_recv_data[i], 0, world_size * sizeof(float)));
        CUDA_CALL(cudaStreamCreate(stream + i));
    }

    ncclUniqueId id;
    ncclComm_t comms[num_devices];

    if (world_rank == 0) {
        NCCL_CALL(ncclGetUniqueId(&id));
    }
    MPI_CALL(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    NCCL_CALL(ncclGroupStart());
    for (int i = 0; i < num_devices; i++) {
        CUDA_CALL(cudaSetDevice(local_rank * num_devices + i));
        NCCL_CALL(ncclCommInitRank(comms + i, world_size * num_devices, id, world_rank * num_devices + i));
    }
    NCCL_CALL(ncclGroupEnd());

    NCCL_CALL(ncclGroupStart());
    for (int i = 0; i < num_devices; i++) {
        NCCL_CALL(ncclAllReduce((const void*) d_send_data[i], (void*) d_recv_data[i], world_size, ncclFloat, ncclSum, comms[i], stream[i]));
    }
    NCCL_CALL(ncclGroupEnd());

    // Wait for all streams to synchronize
    for (int i = 0; i < num_devices; i++) {
        CUDA_CALL(cudaStreamSynchronize(stream[i]));
    }

    // Free all device variables
    for (int i = 0; i < num_devices; i++) {
        CUDA_CALL(cudaFree(d_send_data[i]));
        CUDA_CALL(cudaFree(d_recv_data[i]));
    }

    // Destroy the NCCL communicator
    for (int i = 0; i < num_devices; i++) {
        NCCL_CALL(ncclCommDestroy(comms[i]));
    }

    // Finalize MPI
    MPI_CALL(MPI_Finalize());

    printf("[MPI Rank %d] Success \n", world_rank);
    return 0;
}