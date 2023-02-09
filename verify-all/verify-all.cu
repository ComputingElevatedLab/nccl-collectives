// Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <nccl.h>

// CPU
#include "../common/bruck.cpp"
#include "../common/error-catch.cpp"

// CUDA
#include "../common/bruck.cu"
#include "../common/error-catch.cu"
#include "../common/hostname.cu"
#include "../common/synchronize.cu"

int main(int argc, char *argv[])
{
  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));

  // Set MPI size and rank
  int world_size;
  int mpi_rank;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  if (mpi_rank == 0)
  {
    std::cout << "verify-all" << std::endl;
  }

  // Figure out what host the current MPI process is running on
  uint64_t hostHashs[world_size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[mpi_rank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

  // Compute and set the local rank based on the hostname
  int local_host_rank = 0;
  for (int i = 0; i < world_size; i++)
  {
    if (i == mpi_rank)
    {
      break;
    }
    if (hostHashs[i] == hostHashs[mpi_rank])
    {
      local_host_rank++;
    }
  }

  // Initialize a unique NCCL ID at process 0 and broadcast it to all others
  ncclUniqueId id;
  if (mpi_rank == 0)
  {
    ncclGetUniqueId(&id);
  }
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // Initialize NCCL
  ncclComm_t comm;
  cudaStream_t stream;
  std::cout << "mpi_rank " << mpi_rank << " - local_host_rank " << local_host_rank << " - id " << id.internal << std::endl;
  CUDACHECK(cudaSetDevice(local_host_rank));
  CUDACHECK(cudaStreamCreate(&stream));
  NCCLCHECK(ncclCommInitRank(&comm, world_size, id, mpi_rank));

  // Host variables
  int *h_send_data;
  int *h_verify_data;
  int *h_mpi_bruck_recv_data;
  int *h_nccl_builtin_recv_data;

  // Device variables
  int *d_builtin_send_data;
  int *d_builtin_recv_data;

  // Misc variables
  const int ncclInProgress = 7;
  ncclResult_t nccl_ata_bruck_r2_state;

  // Verification loop
  // for (int send_count = 1; send_count <= 100; send_count++)
  {
    // Send and recieve buffers must be the same size for bruck
    const int send_count = 1;
    const int64_t buffer_size = send_count * world_size;
    const int64_t buffer_bytes = buffer_size * sizeof(int);

    // Allocate host memory
    h_send_data = new int[buffer_size];
    h_verify_data = new int[buffer_size];
    h_mpi_bruck_recv_data = new int[buffer_size];
    h_nccl_builtin_recv_data = new int[buffer_size];

    // Allocate device memory
    CUDACHECK(cudaMalloc((void **)&d_builtin_send_data, buffer_bytes));
    CUDACHECK(cudaMalloc((void **)&d_builtin_recv_data, buffer_bytes));

    // Force all test buffers to be 0 - initialized
    for (int i = 0; i < buffer_bytes; i++)
    {
      h_mpi_bruck_recv_data[i] = 0;
    }

    CUDACHECK(cudaMemset(d_builtin_send_data, 0, buffer_bytes));
    CUDACHECK(cudaMemset(d_builtin_recv_data, 0, buffer_bytes));

    // Prepare the send buffer
    // [ 0 0 0]
    for (int i = 0; i < buffer_size; i++)
    {
      h_send_data[i] = mpi_rank;
    }

    // Prepare the verification buffer
    for (int i = 0; i < world_size; i++)
    {
      for (int j = 0; j < send_count; j++)
      {
        h_verify_data[j + i * send_count] = i;
      }
    }

    std::cout << "Iteration " << send_count << "- rank " << mpi_rank << " send buffer: [";
    for (int i = 0; i < send_count; i++)
    {
      std::cout << " " << h_send_data[i] << " ";
    }
    std::cout << "]" << std::endl;
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (mpi_rank == 0)
    {
      std::cout << "Iteration " << send_count << " verify buffer: [";
      for (int i = 0; i < buffer_size; i++)
      {
        std::cout << " " << h_verify_data[i] << " ";
      }
      std::cout << "]" << std::endl;
    }

    // Copy host memory to device memory
    CUDACHECK(cudaMemcpy(d_builtin_send_data, h_send_data, buffer_bytes, cudaMemcpyDefault));

    // Synchronize before tests
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK(cudaDeviceSynchronize());

    // MPI: Bruck r = 2
    uniform_radix_r_bruck(2, (char *)h_send_data, send_count, MPI_INT, (char *)h_mpi_bruck_recv_data, send_count, MPI_INT, MPI_COMM_WORLD);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // NCCL: Builtin
    const size_t rankOffset = send_count * ncclTypeSize(ncclInt);
    NCCLCHECK(ncclGroupStart());
    for (int k = 0; k < world_size; k++)
    {
      NCCLCHECK(ncclSend(((char *)d_builtin_send_data) + k * rankOffset, send_count, ncclInt, k, comm, stream));
      NCCLCHECK(ncclRecv(((char *)d_builtin_recv_data) + k * rankOffset, send_count, ncclInt, k, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    do
    {
      NCCLCHECK(ncclCommGetAsyncError(comm, &nccl_ata_bruck_r2_state));
    } while (nccl_ata_bruck_r2_state == ncclInProgress);
    ncclStreamSynchronize(stream, comm);

    // // NCCL: Bruck r = 2
    // ncclBruck(2, (char *)d_bruck_r2_send_data, send_count, ncclInt, (char *)d_bruck_r2_recv_data, send_count, ncclInt, comm, stream);

    // Verify all implementations against the verification data
    CUDACHECK(cudaMemcpy(h_nccl_builtin_recv_data, d_builtin_recv_data, buffer_bytes, cudaMemcpyDefault));
    CUDACHECK(cudaDeviceSynchronize());

    bool passed[2] = {true};

    // MPI: Bruck r = 2
    for (int i = 0; i < buffer_size; i++)
    {
      if (h_mpi_bruck_recv_data[i] != h_verify_data[i])
      {
        passed[0] = false;

        std::cout << "Rank " << mpi_rank << " - mpi-bruck-r2: received data: [";
        for (int j = 0; j < buffer_size; j++)
        {
          std::cout << " " << h_mpi_bruck_recv_data[j] << " ";
        }
        std::cout << "]" << std::endl;
      }
    }

    // NCCL: Builtin
    for (int i = 0; i < buffer_size; i++)
    {
      if (h_nccl_builtin_recv_data[i] != h_verify_data[i])
      {
        passed[1] = false;

        std::cout << "Rank " << mpi_rank << " - nccl-bruck-r2: received data: [";
        for (int j = 0; j < buffer_size; j++)
        {
          std::cout << " " << h_nccl_builtin_recv_data[j] << " ";
        }
        std::cout << "]" << std::endl;
      }
    }

    bool any_failed = false;
    for (int i = 0; i < 6; i++)
    {
      if (passed[i] == false)
      {
        any_failed = true;
        break;
      }
    }

    std::string result[2] = {"failed", "passed"};

    if (any_failed)
    {
      std::cout << "Iteration " << send_count << " - mpi-bruck-r2 - rank " << mpi_rank << ": " << result[passed[0]] << std::endl;
      std::cout << "Iteration " << send_count << " - nccl-bruck-r2 - rank " << mpi_rank << ": " << result[passed[1]] << std::endl;
    }
    else
    {
      std::cout << "Iteration " << send_count << " - rank " << mpi_rank << ": all passed";
    }

    CUDACHECK(cudaDeviceSynchronize());

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Free all allocated memory

  delete[] h_mpi_bruck_recv_data;
  delete[] h_nccl_builtin_recv_data;
  CUDACHECK(cudaFree(d_builtin_send_data));
  CUDACHECK(cudaFree(d_builtin_recv_data));

  // Destroy NCCL communicator
  NCCLCHECK(ncclCommDestroy(comm));

  // Finalize MPI
  MPICHECK(MPI_Finalize());
  return 0;
}