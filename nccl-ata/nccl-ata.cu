// Source:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <nccl.h>

#include "../common/error-catch.cpp"
#include "../common/error-catch.cu"
#include "../common/hostname.cu"

int main(int argc, char *argv[]) {
  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));

  // Set MPI size and rank
  int size;
  int rank;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int count;
  CUDACHECK(cudaGetDeviceCount(&count));
  if (rank == 0) {
    std::cout << "There are " << count << " CUDA devices available"
              << std::endl;
  }

  // Figure out what host the current MPI process is running on
  uint64_t hostHashs[size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
                         sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

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
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Allocate memory for host variables
  int bytes = sizeof(int);
  int *h_send_data = new int[size];
  int *h_recv_data = new int[size];

  // Fill the send buffer with each process rank
  for (int i = 0; i < size; i++) {
    h_send_data[i] = rank;
  }

  // Allocate memory for device variables
  cudaStream_t stream;
  int *d_send_data = nullptr;
  int *d_recv_data = nullptr;
  CUDACHECK(cudaSetDevice(local_rank));
  CUDACHECK(cudaMalloc((void **)&d_send_data, size * bytes));
  CUDACHECK(cudaMalloc((void **)&d_recv_data, size * bytes));
  CUDACHECK(
      cudaMemcpy(d_send_data, h_send_data, size * bytes, cudaMemcpyDefault));
  CUDACHECK(cudaStreamCreate(&stream));

  // Initialize NCCL
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

  // Warm-up loop
  for (int i = 0; i < 5; i++) {
    CUDACHECK(
        cudaMemcpy(d_send_data, h_send_data, size * bytes, cudaMemcpyDefault));
    CUDACHECK(cudaMemset(d_recv_data, 0, size * bytes));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    ncclGroupStart();
    for (int j = 0; j < size; j++) {
      ncclSend((void *)&d_send_data[j], 1, ncclInt, j, comm, stream);
      ncclRecv((void *)&d_recv_data[j], 1, ncclInt, j, comm, stream);
    }
    ncclGroupEnd();
  }

  // Benchmark loop
  const int num_executions = 100;
  std::vector<float> times(num_executions);
  for (int i = 0; i < num_executions; i++) {
    // Reset send data
    CUDACHECK(
        cudaMemcpy(d_send_data, h_send_data, size * bytes, cudaMemcpyDefault));
    CUDACHECK(cudaMemset(d_recv_data, 0, size * bytes));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Perform all-to-all to send and receive each rank
    cudaEventRecord(start, 0);
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < size; j++) {
      ncclSend((void *)&d_send_data[j], 1, ncclInt, j, comm, stream);
      ncclRecv((void *)&d_recv_data[j], 1, ncclInt, j, comm, stream);
    }
    NCCLCHECK(ncclGroupEnd());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute elapsed time
    float localElapsedTime;
    cudaEventElapsedTime(&localElapsedTime, start, stop);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    float elapsedTime;
    MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_FLOAT, MPI_MAX,
                        0, MPI_COMM_WORLD));
    if (rank == 0) {
      times[i] = localElapsedTime;
    }
  }

  // Verify that all ranks have the same thing in their recieve buffer
  CUDACHECK(cudaMemcpy(h_recv_data, d_recv_data, size * sizeof(int),
                       cudaMemcpyDeviceToHost));
  std::cout << "Rank " << rank << " received data: [";
  for (int i = 0; i < size; i++) {
    std::cout << " " << h_recv_data[i] << " ";
  }
  std::cout << "]" << std::endl;

  if (rank == 0) {
    float sum = 0;
    for (int i = 0; i < num_executions; i++) {
      sum += times[i];
    }
    std::cout << "Average elapsed all-to-all time across " << num_executions
              << " executions: " << sum / num_executions << " ms" << std::endl;
  }

  // Free all host variables
  delete[] h_send_data;
  delete[] h_recv_data;

  // Free all device variables
  CUDACHECK(cudaFree(d_send_data));
  CUDACHECK(cudaFree(d_recv_data));

  // Destroy NCCL communicator
  ncclCommDestroy(comm);

  // Finalize MPI
  MPICHECK(MPI_Finalize());
  return 0;
}