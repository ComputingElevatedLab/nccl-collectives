// Source: https://github.com/harp-lab/rbruck_alltoall
#include <cmath>
#include <cstring>
#include <vector>

#include "mpi.h"
#include "nccl.h"

#include "error-catch.cu"

int myPow(int x, unsigned int p) {
    if (p == 0) {
		return 1;
	} else if (p == 1) {
		return x;
	}

    int tmp = myPow(x, p / 2);
    if (p % 2 == 0) {
		return tmp * tmp;
	}
	return x * tmp * tmp;
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

void ncclBruck(int r, char* d_send_data, int send_count, ncclDataType_t send_type, char* d_recv_data, int recv_count, ncclDataType_t recv_type,  ncclComm_t comm, cudaStream_t stream) {
    int size;
    int rank;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size))
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank))
	std::cout << "Rank " << rank << ": at the beginning of ncclBruck" << std::endl;
	
	if (r < 2 || size < 2) {
		std::cout << "Error: ncclBruck requires r >= 2 and nProc >= 2" << std::endl;
	}

    // TODO: should be sizeof(type)
    int unit_size = send_count * sizeof(int);
	int w = std::ceil(std::log(size) / std::log(r));
	int nlpow = myPow(r, w - 1);
	int d = (myPow(r, w) - size) / nlpow;
	std::cout << "Rank " << rank << ": unit_size=" << unit_size << ", w=" << w << ", nlpow=" << nlpow << ", d=" << d << std::endl;

	int h_send_debug[size];
	int h_recv_debug[size];
	int h_temp_debug[nlpow];

	// CUDA_CALL(cudaMemcpy(h_send_debug, d_send_data, size * sizeof(int), cudaMemcpyDefault))
	// std::cout << "Rank " << rank << ": d_send_data: [";
	// for (int i = 0; i < size; i++) {
	// 	printf(" %02X", h_send_debug[i]);
	// }
	// std::cout << "]" << std::endl;
	// CUDA_CALL(cudaMemcpy(h_recv_debug, d_recv_data, size * sizeof(int), cudaMemcpyDefault))
	// std::cout << "Rank " << rank << ": d_recv_data: [";
	// for (int i = 0; i < size; i++) {
	// 	printf(" %02X", h_recv_debug[i]);
	// }
	// std::cout << "]" << std::endl;

    CUDA_CALL(cudaMemcpy(d_recv_data, d_send_data, size * unit_size, cudaMemcpyDefault))

	// CUDA_CALL(cudaMemcpy(h_recv_debug, d_recv_data, size * sizeof(int), cudaMemcpyDefault))
	// std::cout << "Rank " << rank << ": d_recv_data: [";
	// for (int i = 0; i < size; i++) {
	// 	printf(" %02X", h_recv_debug[i]);
	// }
	// std::cout << "]" << std::endl;

	CUDA_CALL(cudaMemcpy(&d_send_data[(size - rank) * unit_size], d_recv_data, rank * unit_size, cudaMemcpyDefault))

	// CUDA_CALL(cudaMemcpy(h_send_debug, d_send_data, size * sizeof(int), cudaMemcpyDefault))
	// std::cout << "Rank " << rank << ": d_send_data: [";
	// for (int i = 0; i < size; i++) {
	// 	printf(" %02X", h_send_debug[i]);
	// }
	// std::cout << "]" << std::endl;

	CUDA_CALL(cudaMemcpy(d_send_data, &d_recv_data[rank * unit_size], (size - rank) * unit_size, cudaMemcpyDefault))

	// CUDA_CALL(cudaMemcpy(h_send_debug, d_send_data, size * sizeof(int), cudaMemcpyDefault))
	// std::cout << "Rank " << rank << ": d_send_data: [";
	// for (int i = 0; i < size; i++) {
	// 	printf(" %02X", h_send_debug[i]);
	// }
	// std::cout << "]" << std::endl;

    std::vector<std::vector<int>> rank_r_reps(size * w);
	for (int i = 0; i < size; i++) {
		rank_r_reps[i] = convert10tob(w, i, r);
	}
	std::cout << "]" << std::endl;

	std::cout << "Rank " << rank << ": rank_r_reps: [";
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < w; j++) {
			std::cout << " r" << rank << " " << rank_r_reps[i][j] << " ";
		}
	}
	std::cout << "]" << std::endl;

	int sent_blocks[nlpow];
	int di = 0;
	int ci = 0;

	int* temp_buffer;
    CUDA_CALL(cudaMalloc((void **) &temp_buffer, nlpow * unit_size))
    std::cout << "Rank " << rank << ": temp_buffer allocated with (" << nlpow * unit_size << "b)" << std::endl;

    for (int x = 0; x < w; x++) {
    	int ze = (x == w - 1)? r - d: r;
    	for (int z = 1; z < ze; z++) {
    		di = 0;
    		ci = 0;
    		for (int i = 0; i < size; i++) {
    			if (rank_r_reps[i][x] == z) {
                    std::cout << "Rank " << rank << ": before di=" << di << ", ci=" << ci << std::endl;
    				sent_blocks[di] = i;
                    std::cout << "Rank " << rank << ": rank_r_reps[" << i << "][" << x << "]=" << z << " (" << rank_r_reps[i][x]<< "), sent_blocks=" << i << std::endl;
    				CUDA_CALL(cudaMemcpy(&temp_buffer[unit_size * ci], &d_send_data[unit_size * i], unit_size, cudaMemcpyDefault))

					CUDA_CALL(cudaMemcpy(h_temp_debug, temp_buffer, nlpow * sizeof(int), cudaMemcpyDefault))
					// std::cout << "Rank " << rank << ": temp_buffer: [";
					// for (int i = 0; i < nlpow; i++) {
					// 	printf(" %02X", temp_buffer[i]);
					// }
					// std::cout << "]" << std::endl;

                    di += 1;
                    ci += 1;
                    std::cout << "Rank " << rank << ": after di=" << di << ", ci=" << ci << std::endl;
				}
    		}

    		int distance = z * myPow(r, x);
    		int recv_rank = (rank - distance + size) % size;
    		int send_rank = (rank + distance) % size;
            std::cout << "Rank " << rank << ": distance=" << distance << ", recv_rank=" << recv_rank << ", send_rank=" << send_rank << std::endl;

            NCCL_CALL(ncclGroupStart())
            NCCL_CALL(ncclSend(temp_buffer, di * unit_size, send_type, send_rank, comm, stream))
            NCCL_CALL(ncclRecv(d_recv_data, di * unit_size, recv_type, recv_rank, comm, stream))
            NCCL_CALL(ncclGroupEnd())

    		for (int i = 0; i < di; i++) {
    			CUDA_CALL(cudaMemcpy(&d_send_data[sent_blocks[i] * unit_size], &d_recv_data[i * unit_size], unit_size, cudaMemcpyDefault))
				CUDA_CALL(cudaMemcpy(h_send_debug, d_send_data, size * sizeof(int), cudaMemcpyDefault))
				// std::cout << "Rank " << rank << ": d_send_data: [";
				// for (int i = 0; i < size; i++) {
				// 	printf(" %02X", h_send_debug[i]);
				// }
				// std::cout << "]" << std::endl;
                std::cout << "Rank " << rank << ": copying from d_recv_data[" << i * unit_size << "] to d_send_data[" << sent_blocks[i] * unit_size << "]" << std::endl;
    		}
    	}
    }

	for (int i = 0; i < size; i++) {
		CUDA_CALL(cudaMemcpy(&d_recv_data[((rank - i + size) % size) * unit_size], &d_send_data[i * unit_size], unit_size, cudaMemcpyDefault))
        CUDA_CALL(cudaMemcpy(h_recv_debug, d_recv_data, size * sizeof(int), cudaMemcpyDefault))
		// std::cout << "Rank " << rank << ": d_recv_data: [";
		// for (int i = 0; i < size; i++) {
		// 	printf(" %02X", h_recv_debug[i]);
		// }
		// std::cout << "]" << std::endl;
		std::cout << "Rank " << rank << ": copying from d_send_data[" << i * unit_size << "] to d_recv_data[" << ((rank - i + size) % size) * unit_size << "]" << std::endl;
	}

    CUDA_CALL(cudaFree(temp_buffer))
}