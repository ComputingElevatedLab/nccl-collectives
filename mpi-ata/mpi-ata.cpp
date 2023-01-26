#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

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

  // Allocated variables
  int* send_data;
  int* recv_data;

  // Benchmark loop
  const int num_executions = 100;
  for (int i = 1; i <= 1000000; i *= 10) {
    // Send and recieve buffers must be the same size
    int multiplier = i;
    const int buffer_size = size * multiplier;
    send_data = new int[buffer_size];
    recv_data = new int[buffer_size];

    // Fill the send buffer with each process rank
    for (int i = 0; i < buffer_size; i++) {
      send_data[i] = rank;
    }

    // Warm-up loop
    for (int j = 0; j < 5; j++) {
      for (int j = 0; j < buffer_size; j++) {
        send_data[j] = rank;
        recv_data[j] = 0;
      }
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      MPICHECK(MPI_Alltoall(send_data, multiplier, MPI_INT, recv_data, multiplier, MPI_INT, MPI_COMM_WORLD));
    }

    std::vector<float> times(num_executions);
    for (int j = 0; j < num_executions; j++) {
      // Reset buffers
      for (int k = 0; k < buffer_size; k++) {
        send_data[k] = rank;
        recv_data[k] = 0;
      }
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Perform all to all
      auto start = std::chrono::high_resolution_clock::now();
      MPICHECK(MPI_Alltoall(send_data, multiplier, MPI_INT, recv_data, multiplier, MPI_INT, MPI_COMM_WORLD));
      auto stop = std::chrono::high_resolution_clock::now();

      // Compute elapsed time
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      const float localElapsedTime = duration.count();

      MPI_Barrier(MPI_COMM_WORLD);
      float elapsedTime;
      MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        times[j] = localElapsedTime;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      float sum = 0;
      for (int i = 0; i < num_executions; i++) {
        sum += times[i];
      }
      float average = sum / num_executions;

      std::ofstream log;
      log.open ("run.log", std::ios_base::app);
      log << "mpi-ata w/ " << buffer_size * sizeof(int)
          << " byte buffer: " << average << " ms" << std::endl;
      log.close();
    }

    // Verify that all ranks have the same thing in their recieve buffer
    // std::cout << "Rank " << rank << " received data: [";
    // for (int i = 0; i < buffer_size; i++) {
    //   std::cout << " " << recv_data[i] << " ";
    // }
    // std::cout << "]" << std::endl;

    // Free allocated memory
    delete[] send_data;
    delete[] recv_data;
  }
  
  // Finalize MPI
  MPICHECK(MPI_Finalize());
  return 0;
}