// Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
#include <chrono>
#include <iostream>

#include <mpi.h>
#include <nccl.h>

#include "../common/bruck.cu"
#include "../common/hostname.cu"
#include "../common/typesize.cu"

int main(int argc, char *argv[])
{
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Set MPI size and rank
  int world_size;
  int mpi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // Figure out what host the current MPI process is running on
  uint64_t hostHashs[world_size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[mpi_rank] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);

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
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  // Initialize NCCL
  ncclComm_t comm;
  cudaStream_t stream;
  cudaSetDevice(local_host_rank);
  cudaStreamCreate(&stream);
  ncclCommInitRank(&comm, world_size, id, mpi_rank);

  // Host variables
  int *h_send_data;
  int *h_verify_data;
  int *h_recv_data;

  // Device variables
  int *d_send_data;
  int *d_recv_data;

  std::vector<int> test_sizes{1, 64, 256, 1024};

  for (int i = 0; i < test_sizes.size(); i++)
  {
    // Send and recieve buffers
    const int send_count = test_sizes[i];
    const int buffer_size = send_count * world_size;
    const int buffer_bytes = buffer_size * sizeof(int);

    // Allocate host memory
    h_send_data = new int[buffer_size];
    h_verify_data = new int[buffer_size];
    h_recv_data = new int[buffer_size];

    // Allocate device memory
    cudaMalloc((void **)&d_send_data, buffer_bytes);
    cudaMalloc((void **)&d_recv_data, buffer_bytes);
    cudaMemset(d_send_data, 0, buffer_bytes);
    cudaMemset(d_recv_data, 0, buffer_bytes);

    // Prepare the send buffer
    for (int j = 0; j < buffer_size; j++)
    {
      h_send_data[j] = mpi_rank;
    }

    // Prepare the verification buffer
    for (int j = 0; j < world_size; j++)
    {
      for (int k = 0; k < send_count; k++)
      {
        h_verify_data[k + j * send_count] = j;
      }
    }

    // Copy host memory to device memory
    cudaMemcpy(d_send_data, h_send_data, buffer_bytes, cudaMemcpyHostToDevice);

    // NCCL all to all
    const size_t rankOffset = send_count * ncclTypeSize(ncclInt);
    auto start = std::chrono::high_resolution_clock::now();
    ncclGroupStart();
    for (int r = 0; r < world_size; r++)
    {
      ncclSend(((char *)d_send_data) + r * rankOffset, send_count, ncclInt, r, comm, stream);
      ncclRecv(((char *)d_recv_data) + r * rankOffset, send_count, ncclInt, r, comm, stream);
    }
    ncclGroupEnd();
    auto stop = std::chrono::high_resolution_clock::now();

    // Compute elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    const float localElapsedTime = duration.count();
    float elapsedTime;
    MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Verify against the verification data
    cudaMemcpy(h_recv_data, d_recv_data, buffer_bytes, cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int j = 0; j < buffer_size; j++)
    {
      if (h_recv_data[j] != h_verify_data[j])
      {
        passed = false;
      }
    }

    if (passed)
    {
      std::cout << "Rank " << mpi_rank << " passed" << std::endl;
    }
    else
    {
      std::cout << "Rank " << mpi_rank << " failed:\t[";
      for (int j = 0; j < buffer_size; j++)
      {
        std::cout << " " << h_recv_data[j] << " ";
      }
      std::cout << "]" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::cout << std::fixed << "Elapsed time: " << elapsedTime << " μs" << std::endl;
    }

    // Free all allocated memory
    delete[] h_recv_data;
    cudaFree(d_send_data);
    cudaFree(d_recv_data);
  }

  // Destroy NCCL communicator
  ncclCommDestroy(comm);

  // Finalize MPI
  MPI_Finalize();
  return 0;
}