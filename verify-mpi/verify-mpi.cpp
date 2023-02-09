#include <iostream>
#include <vector>

#include <mpi.h>

#include "../common/bruck.cpp"
#include "../common/error-catch.cpp"

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
    std::cout << "verify-mpi" << std::endl;
  }

  // Verification loop
  for (int send_count = 1; send_count <= 100; send_count++)
  {
    // Send and recieve buffers must be the same size for bruck
    const int buffer_size = send_count * world_size;

    // Allocate memory
    int *builtin_send_data = new int[send_count];
    int *bruck_r2_send_data = new int[send_count];
    int *bruck_rws_send_data = new int[send_count];
    int *builtin_recv_data = new int[buffer_size];
    int *bruck_r2_recv_data = new int[buffer_size];
    int *bruck_rws_recv_data = new int[buffer_size];
    int *verify_data = new int[buffer_size];

    // Force all test buffers to be 0 - initialized
    for (int i = 0; i < buffer_size; i++)
    {
      builtin_recv_data[i] = 0;
      bruck_r2_recv_data[i] = 0;
      bruck_rws_recv_data[i] = 0;
    }

    // Prepare the send buffers
    for (int i = 0; i < send_count; i++)
    {
      builtin_send_data[i] = mpi_rank;
      bruck_r2_send_data[i] = mpi_rank;
      bruck_rws_send_data[i] = mpi_rank;
    }

    // Prepare the verification buffer
    for (int i = 0; i < world_size; i++)
    {
      for (int j = 0; j < send_count; j++)
      {
        verify_data[i * send_count + j] = i;
      }
    }

    // Synchronize before tests
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // MPI: Built-in
    MPICHECK(MPI_Alltoall(builtin_send_data, send_count, MPI_INT, builtin_recv_data, send_count, MPI_INT, MPI_COMM_WORLD));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // MPI: Bruck r = 2
    uniform_radix_r_bruck(2, (char *)bruck_r2_send_data, send_count, MPI_INT, (char *)bruck_r2_recv_data, send_count, MPI_INT, MPI_COMM_WORLD);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // MPI: Bruck r = world_size
    uniform_radix_r_bruck(world_size, (char *)bruck_rws_send_data, send_count, MPI_INT, (char *)bruck_rws_recv_data, send_count, MPI_INT, MPI_COMM_WORLD);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    bool passed[3] = {true, true, true};

    // MPI: Built-in
    for (int i = 0; i < buffer_size; i++)
    {
      if (builtin_recv_data[i] != verify_data[i])
      {
        passed[0] = false;
      }

      if (bruck_r2_recv_data[i] != verify_data[i])
      {
        passed[1] = false;
      }

      if (bruck_rws_recv_data[i] != verify_data[i])
      {
        passed[2] = false;
      }
    }

    std::string result[2] = {"failed", "passed"};

    if (mpi_rank == 1)
    {
      std::cout << send_count << " ints - mpi-ata - rank " << mpi_rank << ": " << result[passed[0]] << std::endl;
      std::cout << send_count << " ints - mpi-bruck-r2 - rank " << mpi_rank << ": " << result[passed[1]] << std::endl;
      std::cout << send_count << " ints - mpi-bruck-rws - rank " << mpi_rank << ": " << result[passed[2]] << std::endl;
    }
  }

  // Finalize MPI
  MPICHECK(MPI_Finalize());
  return 0;
}