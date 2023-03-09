#include <cmath>
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

	int ite_count = 1;
	int loopCount = std::ceil(std::log2(size));

	char *send_data;
	char *recv_data;

	// Warm up
	{
		int count = 32;
		int bytes = count * sizeof(char);
		send_data = (char *)malloc(bytes);
		recv_data = (char *)malloc(bytes);
		std::fill_n(send_data, count, rank);
		std::fill_n(recv_data, 4, -1);

		for (int i = 0; i < ite_count; i++)
		{
			double start = MPI_Wtime();
			int distance = std::pow(2, loopCount - 1);
			for (int i = 0; i < loopCount; i++)
			{
				int sendrank = (rank + distance) % size;
				int recvrank = (rank - distance + size) % size;
				MPI_Sendrecv(send_data, count, MPI_CHAR, sendrank, 0, recv_data, count, MPI_CHAR, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				distance /= 2;
			}
			double stop = MPI_Wtime();
			const double localElapsedTime = stop - start;
			double elapsedTime;
			MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
			if (rank == 0)
			{
				std::cout << "Warm up descending: " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
			}
		}

		MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

		for (int i = 0; i < ite_count; i++)
		{
			double start = MPI_Wtime();
			int distance = 1;
			for (int i = 0; i < loopCount; i++)
			{
				int sendrank = (rank + distance) % size;
				int recvrank = (rank - distance + size) % size;
				MPI_Sendrecv(send_data, bytes, MPI_CHAR, sendrank, 0, recv_data, bytes, MPI_CHAR, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				distance *= 2;
			}
			double stop = MPI_Wtime();

			const double localElapsedTime = stop - start;
			double elapsedTime;
			MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
			if (rank == 0)
			{
				std::cout << "Warm up ascending: " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
			}
		}

		MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
		delete[] send_data;
		delete[] recv_data;
	}

	// Actual test
	for (int count = 32; count <= 8192; count *= 2)
	{
		int bytes = count * sizeof(char);
		send_data = (char *)malloc(bytes);
		recv_data = (char *)malloc(bytes);
		std::fill_n(send_data, count, rank);
		std::fill_n(recv_data, 4, -1);

		for (int i = 0; i < ite_count; i++)
		{
			double start = MPI_Wtime();
			int distance = std::pow(2, loopCount - 1);
			for (int i = 0; i < loopCount; i++)
			{
				int sendrank = (rank + distance) % size;
				int recvrank = (rank - distance + size) % size;
				MPI_Sendrecv(send_data, bytes, MPI_CHAR, sendrank, 0, recv_data, bytes, MPI_CHAR, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				distance /= 2;
			}
			double stop = MPI_Wtime();

			const double localElapsedTime = stop - start;
			double elapsedTime;
			MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
			if (rank == 0)
			{
				std::cout << "descending: " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
			}
		}

		MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

		for (int i = 0; i < ite_count; i++)
		{
			double start = MPI_Wtime();
			int distance = 1;
			for (int i = 0; i < loopCount; i++)
			{
				int sendrank = (rank + distance) % size;
				int recvrank = (rank - distance + size) % size;

				MPI_Sendrecv(send_data, bytes, MPI_CHAR, sendrank, 0, recv_data, bytes, MPI_CHAR, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				distance *= 2;
			}
			double stop = MPI_Wtime();

			const double localElapsedTime = stop - start;
			double elapsedTime;
			MPICHECK(MPI_Reduce(&localElapsedTime, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
			if (rank == 0)
			{
				std::cout << "ascending: " << count << " elements sent in " << elapsedTime << " seconds" << std::endl;
			}
		}

		MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
		delete[] send_data;
		delete[] recv_data;
	}

	MPI_Finalize();
	return 0;
}