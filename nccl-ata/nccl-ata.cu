// Source:
// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
#include <fstream>
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
    std::cout << "nccl-ata" << std::endl;
    std::cout << "CUDA devices available: " << count << std::endl;
  }

  // Figure out what host the current MPI process is running on
  uint64_t hostHashs[size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

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
  MPICHECK(MPI_Bcast((void*) &id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Initialize NCCL
  ncclComm_t comm;
  cudaStream_t stream;
  CUDACHECK(cudaSetDevice(local_rank));
  CUDACHECK(cudaStreamCreate(&stream));
  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

  // Host variables
  int *h_send_data;
  int *h_recv_data;

  // Device variables
  int *d_send_data;
  int *d_recv_data;

  // Benchmark loop
  const int num_executions = 100;
  for (int i = 1; i <= 1000; i *= 10) {
    // Send and recieve buffers must be the same size
    const int buffer_size = size * i;
    const int bytes = buffer_size * sizeof(int);

    h_send_data = new int[buffer_size];
    h_recv_data = new int[buffer_size];
    CUDACHECK(cudaMalloc((void**) &d_send_data, bytes));
    CUDACHECK(cudaMalloc((void**) &d_recv_data, bytes));

    // Fill the send buffer with each process rank
    for (int j = 0; j < buffer_size; j++) {
      h_send_data[j] = rank;
    }
    
    CUDACHECK(cudaMemcpy(d_send_data, h_send_data, bytes, cudaMemcpyDefault));
    CUDACHECK(cudaMemset(d_recv_data, 0, bytes));
    if (rank == 0) {
      std::cout << "Finished setting buffers" << std::endl;
    }

    // Warm-up loop
    for (int j = 0; j < 5; j++) {
      CUDACHECK(cudaMemcpy(d_send_data, h_send_data, bytes, cudaMemcpyDefault));
      CUDACHECK(cudaMemset(d_recv_data, 0, bytes));
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      NCCLCHECK(ncclGroupStart());
      for (int k = 0; k < buffer_size; k++) {
        NCCLCHECK(ncclSend((void*) &d_send_data[k], i, ncclInt, k % size, comm, stream));
        NCCLCHECK(ncclRecv((void*) &d_recv_data[k], i, ncclInt, k % size, comm, stream));
      }
      NCCLCHECK(ncclGroupEnd());
    }

    if (rank == 0) {
      std::cout << "Finished warming up" << std::endl;
    }

    std::vector<float> times(num_executions);
    for (int j = 0; j < num_executions; j++) {
      // Reset buffers
      CUDACHECK(cudaMemcpy(d_send_data, h_send_data, bytes, cudaMemcpyDefault));
      CUDACHECK(cudaMemset(d_recv_data, 0, bytes));
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Create CUDA events
      cudaEvent_t start, stop;
      CUDACHECK(cudaEventCreate(&start));
      CUDACHECK(cudaEventCreate(&stop));

      // Perform all-to-all
      cudaEventRecord(start, 0);
      NCCLCHECK(ncclGroupStart());
      for (int k = 0; k < buffer_size; k++) {
        NCCLCHECK(ncclSend((void*) &d_send_data[k], i, ncclInt, k % size, comm, stream));
        NCCLCHECK(ncclRecv((void*) &d_recv_data[k], i, ncclInt, k % size, comm, stream));
      }
      NCCLCHECK(ncclGroupEnd());
      CUDACHECK(cudaEventRecord(stop, 0));
      CUDACHECK(cudaEventSynchronize(stop));

      // Compute elapsed time
      float localElapsedTime;
      CUDACHECK(cudaEventElapsedTime(&localElapsedTime, start, stop));

      // Destroy CUDA events
      CUDACHECK(cudaEventDestroy(start));
      CUDACHECK(cudaEventDestroy(stop));

      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      float elapsedTime;
      MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD));
      if (rank == 0) {
        times[j] = localElapsedTime;
      }
    }

    if (rank == 0) {
      std::cout << "Finished benchmark loop" << std::endl;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
      float sum = 0;
      for (int i = 0; i < num_executions; i++) {
        sum += times[i];
      }
      float average = sum / num_executions;

      std::ofstream log;
      log.open("run.log", std::ios_base::app);
      log << "nccl-ata w/ " << bytes << " byte buffer: " << average << " ms" << std::endl;
      log.close();

      std::cout << "Finished " << bytes << "-size buffer benchmark" << std::endl;
    }
    
    // Verify that all ranks have the same thing in their recieve buffer
    // CUDACHECK(cudaMemcpy(h_recv_data, d_recv_data, bytes, cudaMemcpyDefault));
    // std::cout << "Rank " << rank << " received data: [";
    // for (int i = 0; i < buffer_size; i++) {
    //   std::cout << " " << h_recv_data[i] << " ";
    // }
    // std::cout << "]" << std::endl;

    // Free all allocated variables
    delete[] h_send_data;
    delete[] h_recv_data;
    CUDACHECK(cudaFree(d_send_data));
    CUDACHECK(cudaFree(d_recv_data));
  }

  // Destroy NCCL communicator
  ncclCommDestroy(comm);

  // Finalize MPI
  MPICHECK(MPI_Finalize());
  return 0;
}