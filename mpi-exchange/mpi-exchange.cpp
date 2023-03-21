#include <cmath>
#include "mpi.h"

static int rank, nprocs;

void running_test(int loopCount, int iteCount, int warmup);
void exchange_ascending(int loopCount, int mesgsize, char* sendbuf, char* recvbuf);
void exchange_descending(int loopCount, int mesgsize, char* sendbuf, char* recvbuf);

int main(int argc, char **argv)
{
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        std::cout << "ERROR: MPI_Init error\n" << std::endl;
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_size error\n" << std::endl;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    	std::cout << "ERROR: MPI_Comm_rank error\n" << std::endl;

    int loopCount = std::ceil(std::log2(nprocs));


    // running test
    running_test(loopCount, 1, 0);

	MPI_Finalize();
    return 0;
}

void running_test(int loopCount, int iteCount, int warmup) {

    for (int mesgsize = 2; mesgsize <= 4; mesgsize *= 2) {

        char * sendbuf = (char*)malloc(mesgsize*sizeof(char));
    	for (int i = 0; i < mesgsize; i++)
    		sendbuf[i] = rank;
    	char * recvbuf = (char*)malloc(mesgsize*sizeof(char));

		for (int i = 0; i < iteCount; i++) {
			double start = MPI_Wtime();
			exchange_ascending(loopCount, mesgsize, sendbuf, recvbuf);
			double end = MPI_Wtime();
			double time = end - start;
		}

		free(sendbuf);
		free(recvbuf);
    }
}

void exchange_ascending(int loopCount, int mesgsize, char* sendbuf, char* recvbuf) {

	int distance = 1;
	for (int i = 0; i < loopCount; i++) {
		int sendrank = (rank + distance) % nprocs;
		int recvrank = (rank - distance + nprocs) % nprocs;

		MPI_Sendrecv(sendbuf, mesgsize, MPI_CHAR, sendrank, 0, recvbuf, mesgsize, MPI_CHAR, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		distance *= 2;
	}

    std::cout << std::fixed << "Rank " << rank << " recv buffer on line " << __LINE__ << ":\t[";
    for (int j = 0; j < mesgsize; j++)
    {
        std::cout << " " << (int) recvbuf[j] << " ";
    }
    std::cout << "]" << std::endl;
}

void exchange_descending(int loopCount, int mesgsize, char* sendbuf, char* recvbuf) {
	int distance = std::pow(2, loopCount-1);
	for (int i = 0; i < loopCount; i++) {
		int sendrank = (rank + distance) % nprocs;
		int recvrank = (rank - distance + nprocs) % nprocs;

		MPI_Sendrecv(sendbuf, mesgsize, MPI_CHAR, sendrank, 0, recvbuf, mesgsize, MPI_CHAR, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		distance /= 2;
	}
}