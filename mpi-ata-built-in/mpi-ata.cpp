#include <stdio.h>

#include <mpi.h>

#include "../common/error-catch.cpp"

// Distribute each process rank using MPI_Alltoall
int main(int argc, char** argv) {
  MPI_CALL(MPI_Init(&argc, &argv));
  int size;
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
  int rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  // Send and recieve buffers must be the same size
  int send_data[size];
  int recv_data[size];

  // Fill the send buffer with each process rank
  for (int i = 0; i < size; i++) {
    send_data[i] = rank;
  }

  // Use MPI_Alltoall to send and receive each rank
  MPI_CALL(MPI_Alltoall(send_data, 1, MPI_INT, recv_data, 1, MPI_INT, MPI_COMM_WORLD));

  // Verify that all processes have the same thing in their recieve buffer
  printf("Process %d received data: [", rank);
  for (int i = 0; i < size ; i++) {
    printf(" %d ", recv_data[i]);
  }
  printf("]\n");

  // Finalize MPI
  MPI_CALL(MPI_Finalize());
}
