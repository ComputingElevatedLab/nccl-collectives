// Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <nccl.h>

#include "../common/error-catch.cpp"
#include "../common/error-catch.cu"
#include "../common/hostname.cu"

int main(int argc, char *argv[])
{
  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));

  // Set MPI size and rank
  int size;
  int rank;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int count;
  CUDACHECK(cudaGetDeviceCount(&count));
  if (rank == 0)
  {
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
  for (int i = 0; i < size; i++)
  {
    if (i == rank)
    {
      break;
    }
    if (hostHashs[i] == hostHashs[rank])
    {
      local_rank++;
    }
  }

  // Initialize a unique NCCL ID at process 0 and broadcast it to all others
  ncclUniqueId id;
  if (rank == 0)
  {
    ncclGetUniqueId(&id);
  }
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

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
  for (int i = 100; i <= 2000; i += 100)
  {
    // Send and recieve buffers must be the same size
    const int64_t buffer_size = size * i;
    const int64_t send_bytes = i * sizeof(int);
    const int64_t recv_bytes = buffer_size * sizeof(int);

    h_send_data = new int[i];
    h_recv_data = new int[buffer_size];
    CUDACHECK(cudaMalloc((void **)&d_send_data, send_bytes));
    CUDACHECK(cudaMalloc((void **)&d_recv_data, recv_bytes));

    // Fill the send buffer with each process rank
    for (int j = 0; j < i; j++)
    {
      h_send_data[j] = rank;
    }

    CUDACHECK(cudaMemcpy(d_send_data, h_send_data, send_bytes, cudaMemcpyDefault));
    CUDACHECK(cudaMemset(d_recv_data, 0, recv_bytes));
    if (rank == 0)
    {
      std::cout << "Finished setting buffers" << std::endl;
    }

    // Warm-up loop
    for (int j = 0; j < 5; j++)
    {
      CUDACHECK(cudaMemcpy(d_send_data, h_send_data, send_bytes, cudaMemcpyDefault));
      CUDACHECK(cudaMemset(d_recv_data, 0, recv_bytes));
      CUDACHECK(cudaDeviceSynchronize());
      NCCLCHECK(ncclGroupStart());
      for (int k = 0; k < buffer_size; k++)
      {
        NCCLCHECK(ncclSend((void *)&d_send_data[k], i, ncclInt, k % size, comm, stream));
        NCCLCHECK(ncclRecv((void *)&d_recv_data[k], i, ncclInt, k % size, comm, stream));
      }
      NCCLCHECK(ncclGroupEnd());
    }

    if (rank == 0)
    {
      std::cout << "Finished warming up" << std::endl;
    }

    std::vector<double> times(num_executions);
    for (int j = 0; j < num_executions; j++)
    {
      // Reset buffers
      CUDACHECK(cudaMemcpy(d_send_data, h_send_data, send_bytes, cudaMemcpyDefault));
      CUDACHECK(cudaMemset(d_recv_data, 0, recv_bytes));
      CUDACHECK(cudaDeviceSynchronize());

      // Perform all-to-all
      auto start = std::chrono::high_resolution_clock::now();
      NCCLCHECK(ncclGroupStart());
      for (int k = 0; k < buffer_size; k++)
      {
        NCCLCHECK(ncclSend((void *)&d_send_data[k], i, ncclInt, k % size, comm, stream));
        NCCLCHECK(ncclRecv((void *)&d_recv_data[k], i, ncclInt, k % size, comm, stream));
      }
      NCCLCHECK(ncclGroupEnd());
      CUDACHECK(cudaStreamSynchronize(stream));
      auto stop = std::chrono::high_resolution_clock::now();

      // Compute elapsed time
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      const double localElapsedTime = duration.count();

      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      double elapsedTime;
      MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
      if (rank == 0)
      {
        times[j] = localElapsedTime;
      }
    }

    if (rank == 0)
    {
      std::cout << "Finished benchmark loop" << std::endl;
    }

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0)
    {
      double sum = 0;
      for (int j = 0; j < num_executions; j++)
      {
        sum += times[j];
      }
      double average = sum / num_executions;

      std::ofstream log;
      log.open("run.log", std::ios_base::app);
      log << "nccl-ata w/ " << i * sizeof(int) << " bytes sent per GPU: " << average << " ns" << std::endl;
      log.close();

      std::cout << "Finished " << i * sizeof(int) << "-size byte benchmark" << std::endl;
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