// Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#communicator-creation-and-destruction-examples
#include <iostream>

#include <mpi.h>
#include <nccl.h>

#include "../common/bruck.cu"
#include "../common/error-catch.cu"
#include "../common/hostname.cu"

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_CALL(MPI_Init(&argc, &argv));

    // Set MPI size and rank
    int size;
    int rank;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    

    // Figure out what host the current MPI process is running on
    uint64_t hostHashs[size];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[rank] = getHostHash(hostname);
    MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    // Compute and set the local rank based on the hostname
    int local_rank = 0;
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            break;
        }
        if (hostHashs[i] == hostHashs[rank]) {
            local_rank++;
        }
    }

    // Initialize a unique NCCL ID at process 0 and broadcast it to all others
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_CALL(MPI_Bcast((void*) &id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Allocate memory for host variables
    int h_send_data[size];
    int h_recv_data[size];

    // TODO try setting to 0s
    // Fill the send buffer with each process rank
    for (int i = 0; i < size; i++) {
        h_send_data[i] = rank;
    }

    // Allocate memory for device variables
    cudaStream_t stream;
    int* d_send_data = nullptr;
    int* d_recv_data = nullptr;
    CUDA_CALL(cudaSetDevice(local_rank));
    CUDA_CALL(cudaMalloc((void**) &d_send_data, size * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**) &d_recv_data, size * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_send_data, h_send_data, size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaStreamCreate(&stream));

    // Initialize NCCL
    ncclComm_t comm;
    NCCL_CALL(ncclCommInitRank(&comm, size, id, rank));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Perform all-to-all to send and receive
    cudaEventRecord(start, 0);
    ncclBruck(2, (char*) d_send_data, 1, ncclInt, (char*) d_recv_data, 1, ncclInt, comm, stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Rank " << rank << ": elapsed all-to-all time: " << elapsedTime << " ms" << std::endl;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify that all processes have the same thing in their recieve buffer
    CUDA_CALL(cudaMemcpy(h_recv_data, d_recv_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Rank " << rank << ": received data: [";
    for (int i = 0; i < size; i++) {
        std::cout << " r" << rank << " " << h_recv_data[i] << " ";
    }
    std::cout << "]" << std::endl;

    // Free all device variables
    CUDA_CALL(cudaFree(d_send_data));
    CUDA_CALL(cudaFree(d_recv_data));

    // Destroy NCCL communicator
    ncclCommDestroy(comm);

    // Finalize MPI
    MPI_CALL(MPI_Finalize());
    return 0;
}