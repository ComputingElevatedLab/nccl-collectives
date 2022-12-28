// Source: https://github.com/harp-lab/rbruck_alltoall
#include <cmath>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/error-catch.cpp"

int myPow(int x, unsigned int p) {
    if (p == 0) {
        return 1;
    } else if (p == 1) {
        return x;
    }

    int tmp = myPow(x, p / 2);
    if (p % 2 == 0) {
        return tmp * tmp;
    } else {
        return x * tmp * tmp;
    }
}

std::vector<int> convert10tob(int w, int N, int b) {
	std::vector<int> v(w);
	int i = 0;
	while(N) {
        v[i++] = (N % b);
	    N /= b;
	}
	return v;
}

void uniform_radix_r_bruck(int r, char *sendbuf, int sendcount, MPI_Datatype sendtype, char *recvbuf, int recvcount, MPI_Datatype recvtype,  MPI_Comm comm) {
    int rank, nprocs;
    MPI_CALL(MPI_Comm_rank(comm, &rank))
    MPI_CALL(MPI_Comm_size(comm, &nprocs))
    std::cout << "Rank " << rank << ": at the beginning of uniform_radix_r_bruck" << std::endl;
    int typesize;
    MPI_CALL(MPI_Type_size(sendtype, &typesize))

    int unit_size = sendcount * typesize;
    int w = std::ceil(std::log(nprocs) / std::log(r)); // calculate the number of digits when using r-representation
	int nlpow = myPow(r, w - 1);
	int d = (myPow(r, w) - nprocs) / nlpow; // calculate the number of highest digits
    std::cout << "Rank " << rank << ": unit_size=" << unit_size << ", w=" << w << ", nlpow=" << nlpow << ", d=" << d << std::endl;

    // local rotation
    std::memcpy(recvbuf, sendbuf, nprocs * unit_size);
    std::memcpy(&sendbuf[(nprocs - rank) * unit_size], recvbuf, rank * unit_size);
    std::memcpy(sendbuf, &recvbuf[rank * unit_size], (nprocs - rank) * unit_size);

    // convert rank to base r representation
    std::vector<std::vector<int>> rank_r_reps(nprocs * w);
	for (int i = 0; i < nprocs; i++) {
		rank_r_reps[i] = convert10tob(w, i, r);
	}
	std::cout << "Rank " << rank << ": rank_r_reps: [";
	for (int i = 0; i < nprocs; i++) {
		for (int j = 0; j < w; j++) {
			std::cout << " r" << rank << " " << rank_r_reps[i][j] << " ";
		}
	}
	std::cout << "]" << std::endl;

	int sent_blocks[nlpow];
	int di = 0;
	int ci = 0;

	char* temp_buffer = (char*)malloc(nlpow * unit_size); // temporary buffer
    std::cout << "Rank " << rank << ": temp_buffer allocated with (" << nlpow * unit_size << "b)" << std::endl;

	// communication steps = (r - 1)w - d
    for (int x = 0; x < w; x++) {
    	int ze = (x == w - 1)? r - d: r;
    	for (int z = 1; z < ze; z++) {
    		// get the sent data-blocks
    		// copy blocks which need to be sent at this step
    		di = 0;
    		ci = 0;
    		for (int i = 0; i < nprocs; i++) {
    			if (rank_r_reps[i][x] == z) {
                    std::cout << "Rank " << rank << ": before di=" << di << ", ci=" << ci << std::endl;
    				sent_blocks[di++] = i;
                    std::cout << "Rank " << rank << ": rank_r_reps[" << i << "][" << x << "]=" << z << " (" << rank_r_reps[i][x]<< "), sent_blocks=" << i << std::endl;
    				memcpy(&temp_buffer[unit_size*ci++], &sendbuf[i*unit_size], unit_size);
                    std::cout << "Rank " << rank << ": after di=" << di << ", ci=" << ci << std::endl;
    			}
    		}

    		// send and receive
    		int distance = z * myPow(r, x); // pow(1, 51) = 51, int d = pow(1, 51); // 50
    		int recv_proc = (rank - distance + nprocs) % nprocs; // receive data from rank - 2^step process
    		int send_proc = (rank + distance) % nprocs; // send data from rank + 2^k process
    		long long comm_size = di * unit_size;
            std::cout << "Rank " << rank << ": distance=" << distance << ", recv_rank=" << recv_proc << ", send_rank=" << send_proc << std::endl;
            MPI_CALL(MPI_Sendrecv(temp_buffer, comm_size, MPI_CHAR, send_proc, 0, recvbuf, comm_size, MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE))

    		// replace with received data
    		for (int i = 0; i < di; i++) {
    			long long offset = sent_blocks[i] * unit_size;
    			memcpy(sendbuf + offset, recvbuf + (i * unit_size), unit_size);
                std::cout << "Rank " << rank << ": copying from recv_data[" << (i * unit_size) << "] to send_data[" << offset << "]" << std::endl;
    		}
    	}
    }

    // local rotation
	for (int i = 0; i < nprocs; i++) {
		int index = (rank - i + nprocs) % nprocs;
		memcpy(&recvbuf[index * unit_size], &sendbuf[i * unit_size], unit_size);
        std::cout << "Rank " << rank << ": copying from send_data[" << i * unit_size << "] to recv_data[" << index * unit_size << "]" << std::endl;
	}

    free(temp_buffer);
}
