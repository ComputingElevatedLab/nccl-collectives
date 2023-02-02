// Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <nccl.h>

#include "../common/bruck.cu"
#include "../common/error-catch.cpp"
#include "../common/error-catch.cu"
#include "../common/hostname.cu"
#include "../common/synchronize.cu"

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
    std::cout << "bruck-verify" << std::endl;
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
  int *h_ata_recv_data;
  int *h_bruck_recv_data;

  // Device variables
  int *d_ata_send_data;
  int *d_ata_recv_data;
  int *d_bruck_send_data;
  int *d_bruck_recv_data;

  std::ofstream log;
  log.open("test.txt", std::ofstream::out | std::ofstream::trunc);
  log.close();

  // Verification loop
  for (int i = 10; i <= 80000; i += 10)
  {
    // Send and recieve buffers must be the same size for bruck
    const int64_t buffer_size = size * i;
    const int64_t bytes = buffer_size * sizeof(int);
    size_t rankOffset = i * ncclTypeSize(ncclInt);

    // Allocate host memory
    h_send_data = new int[i];
    h_ata_recv_data = new int[buffer_size];
    h_bruck_recv_data = new int[buffer_size];

    // Allocate device memory
    CUDACHECK(cudaMalloc((void **)&d_ata_send_data, bytes));
    CUDACHECK(cudaMalloc((void **)&d_ata_recv_data, bytes));
    CUDACHECK(cudaMalloc((void **)&d_bruck_send_data, bytes));
    CUDACHECK(cudaMalloc((void **)&d_bruck_recv_data, bytes));

    // Fill the send buffer with each process rank
    for (int j = 0; j < i; j++)
    {
      h_send_data[j] = rank;
    }

    // Copy host memory to device memory
    CUDACHECK(cudaMemcpy(d_ata_send_data, h_send_data, bytes, cudaMemcpyDefault));
    CUDACHECK(cudaMemset(d_ata_recv_data, 0, bytes));
    CUDACHECK(cudaMemcpy(d_bruck_send_data, h_send_data, bytes, cudaMemcpyDefault));
    CUDACHECK(cudaMemset(d_bruck_recv_data, 0, bytes));
    CUDACHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
      std::cout << "Finished setting buffers" << std::endl;
    }

    // Perform ata w/ synchronization
    ncclResult_t state;
    NCCLCHECK(ncclGroupStart());
    for (int j = 0; j < size; j++)
    {
      NCCLCHECK(ncclSend(((char *)d_ata_send_data) + j * rankOffset, i, ncclInt, j, comm, stream));
      NCCLCHECK(ncclRecv(((char *)d_ata_recv_data) + j * rankOffset, i, ncclInt, j, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    do
    {
      NCCLCHECK(ncclCommGetAsyncError(comm, &state));
    } while (state == 7);
    ncclStreamSynchronize(stream, comm);

    // Perform bruck w/ synchronization
    ncclBruck(2, (char *)d_bruck_send_data, i, ncclInt, (char *)d_bruck_recv_data, i, ncclInt, comm, stream);
    do
    {
      NCCLCHECK(ncclCommGetAsyncError(comm, &state));
    } while (state == 7);
    ncclStreamSynchronize(stream, comm);

    // Verify that d_bruck_recv_data == d_ata_recv_data
    bool same = true;
    CUDACHECK(cudaMemcpy(h_ata_recv_data, d_ata_recv_data, bytes, cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(h_bruck_recv_data, d_bruck_recv_data, bytes, cudaMemcpyDefault));
    CUDACHECK(cudaDeviceSynchronize());

    for (int j = 0; j < buffer_size; j++)
    {
      if (h_ata_recv_data[j] != h_bruck_recv_data[j])
      {
        same = false;
      }
    }

    if (same == true)
    {
      std::cout << "Rank " << rank << ": verified for " << i * sizeof(int) << " bytes sent" << std::endl;
    }
    else
    {
      std::cout << "Rank " << rank << ": failed for " << i * sizeof(int) << " bytes sent" << std::endl;
    }

    if (rank == 0)
    {
      log.open("verify.log", std::ios_base::app);
      log << std::fixed << "ata w/ " << i * sizeof(int) << " bytes:";
      for (int j = 0; j < buffer_size; j++)
      {
        log << std::fixed << " " << h_ata_recv_data[j];
      }
      log << "\n";
      log << std::fixed << "bruck w/ " << i * sizeof(int) << " bytes:";
      for (int j = 0; j < buffer_size; j++)
      {
        log << std::fixed << " " << h_bruck_recv_data[j];
      }
      log << "\n";
      log.close();
    }

    // Free all allocated variables
    MPI_Barrier(MPI_COMM_WORLD);
    delete[] h_send_data;
    delete[] h_ata_recv_data;
    delete[] h_bruck_recv_data;
    CUDACHECK(cudaFree(d_ata_send_data));
    CUDACHECK(cudaFree(d_ata_recv_data));
    CUDACHECK(cudaFree(d_bruck_send_data));
    CUDACHECK(cudaFree(d_bruck_recv_data));
  }

  // Destroy NCCL communicator
  ncclCommDestroy(comm);

  // Finalize MPI
  MPICHECK(MPI_Finalize());
  return 0;
}