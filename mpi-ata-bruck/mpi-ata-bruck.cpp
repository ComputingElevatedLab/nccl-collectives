#include <chrono>
#include <iostream>

#include <mpi.h>

#include "../common/bruck.cpp"
#include "../common/error-catch.cpp"

// Distribute each process rank using MPI_Alltoall
int main(int argc, char** argv) {

  // Initialize MPI
  MPI_CALL(MPI_Init(&argc, &argv));

  // Set MPI size and rank
  int size;
  int rank;
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // Send and recieve buffers must be the same size
  int send_data[size];
  int recv_data[size];

  // Fill the send buffer with each process rank
  for (int i = 0; i < size; i++) {
    send_data[i] = rank;
  }

  // Use MPI_Alltoall to send and receive each rank
  auto start = std::chrono::high_resolution_clock::now();
  uniform_radix_r_bruck(2, (char*) send_data, 1, MPI_INT, (char*) recv_data, 1, MPI_INT,  MPI_COMM_WORLD);
  auto stop = std::chrono::high_resolution_clock::now();

  // Compute elapsed time
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Rank " << rank << ": elapsed all-to-all time: " << duration.count() << " ms" << std::endl;

  // Verify that all processes have the same thing in their recieve buffer
  std::cout << "Rank " << rank << ": received data: [";
  for (int i = 0; i < size; i++) {
    std::cout << " r" << rank << " " << recv_data[i] << " ";
  }
  std::cout << "]" << std::endl;

  // Finalize MPI
  MPI_CALL(MPI_Finalize());
  return 0;
}
