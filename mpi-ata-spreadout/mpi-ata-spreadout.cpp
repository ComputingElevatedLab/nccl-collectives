#include <chrono>
#include <iostream>

#include <mpi.h>

#include "../common/error-catch.cpp"
#include "../common/spreadout.cpp"

// Distribute each process rank using MPI_Alltoall
int main(int argc, char** argv) {

  // Initialize MPI
  MPICHECK(MPI_Init(&argc, &argv));

  // Set MPI size and rank
  int size;
  int rank;
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // Send and recieve buffers must be the same size
  int send_data[size];
  int recv_data[size];

  // Fill the send buffer with each process rank
  for (int i = 0; i < size; i++) {
    send_data[i] = rank;
  }

  // Use MPI_Alltoall to send and receive each rank
  auto start = std::chrono::high_resolution_clock::now();
  spreadout_alltoall((char*) send_data, 1, MPI_INT, (char*) recv_data, 1, MPI_INT,  MPI_COMM_WORLD);
  auto stop = std::chrono::high_resolution_clock::now();

  // Compute elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  const float localElapsedTime = duration.count();
  // std::cout << "Rank " << rank << " elapsed all-to-all time: " << localElapsedTime << " ms" << std::endl;

  // Verify that all ranks have the same thing in their recieve buffer
  std::cout << "Rank " << rank << ": received data: [";
  for (int i = 0; i < size; i++) {
    std::cout << " " << recv_data[i] << " ";
  }
  std::cout << "]" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  float elapsedTime;
  MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "Max elapsed all-to-all time across ranks: " << elapsedTime << " ms" << std::endl;
  }

  // Finalize MPI
  MPICHECK(MPI_Finalize());
  return 0;
}
