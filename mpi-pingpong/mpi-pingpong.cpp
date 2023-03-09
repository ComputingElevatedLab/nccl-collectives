#include <iomanip>
#include <iostream>

#include "mpi.h"

#include "../common/error-catch.cpp"

int main(int argc, char **argv)
{
	// Initialize MPI
	MPICHECK(MPI_Init(&argc, &argv));

	// Set MPI size and rank
	int size;
	int rank;
	MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
	MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	std::cout << std::setiosflags(std::ios_base::fixed);

	int ite_count = 2;

	// Warm up
	{
		int send_data[4];
		int recv_data[4];
		std::fill_n(send_data, 4, rank);
		std::fill_n(recv_data, 4, -1);

		for (int i = 0; i < ite_count; i++)
		{
			double start = MPI_Wtime();
			if (rank == size - 1)
			{
				MPICHECK(MPI_Recv(&recv_data, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
			}

			if (rank == 0)
			{
				MPICHECK(MPI_Send(&send_data, 4, MPI_INT, size - 1, 0, MPI_COMM_WORLD));
			}
			double stop = MPI_Wtime();

			const double localElapsedTime = stop - start;
			double elapsedTime;
			MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
			if (rank == 0)
			{
				std::cout << "Warm up: " << 4 << " elements sent in " << elapsedTime << " seconds" << std::endl;
			}
		}
		MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
	}

	// Actual test
	for (int count = 4; count <= 2048; count *= 2)
	{
		int send_data[count];
		int recv_data[count];
		std::fill_n(send_data, count, rank);
		std::fill_n(recv_data, count, -1);

		for (int i = 0; i < ite_count; i++)
		{
			double start = MPI_Wtime();
			if (rank == size - 1)
			{
				MPICHECK(MPI_Recv(&recv_data, count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
			}

			if (rank == 0)
			{
				MPICHECK(MPI_Send(&send_data, count, MPI_INT, size - 1, 0, MPI_COMM_WORLD));
			}
			double stop = MPI_Wtime();

			const double localElapsedTime = stop - start;
			double elapsedTime;
			MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
			if (rank == 0)
			{
				std::cout << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
			}
		}
	}
	MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

	MPICHECK(MPI_Finalize());
	return 0;
}