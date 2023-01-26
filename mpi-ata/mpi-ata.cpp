#include <chrono>
#include <fstream>
#include <iostream>

#include <mpi.h>

#include "../common/error-catch.cpp"

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
  MPICHECK(MPI_Alltoall(send_data, 1, MPI_INT, recv_data, 1, MPI_INT, MPI_COMM_WORLD));
  auto stop = std::chrono::high_resolution_clock::now();

  // Compute elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  const float localElapsedTime = duration.count();
  // std::cout << "Rank " << rank << " elapsed all-to-all time: " << localElapsedTime << " ms" << std::endl;

  // Verify that all ranks have the same thing in their recieve buffer
  std::cout << "Rank " << rank << " received data: [";
  for (int i = 0; i < size; i++) {
    std::cout << " " << recv_data[i] << " ";
  }
  std::cout << "]" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  float elapsedTime;
  MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    std::ofstream log;
    log.open ("run.log", std::ios_base::app);
    log << "mpi-ata: " << elapsedTime << " ms" << std::endl;
    log.close();
  }

  // Finalize MPI
  MPICHECK(MPI_Finalize());
  return 0;
}
