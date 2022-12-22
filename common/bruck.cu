// Source: https://github.com/harp-lab/rbruck_alltoall
#include <cmath>
#include <cstring>
#include <vector>

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
	// Set MPI size and rank
    int size;
    int rank;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	std::cout << "I am rank " << rank << std::endl;
	
	if (r < 2 || size < 2) {
		std::cout << "Error: ncclBruck requires r >= 2 and nProc >= 2" << std::endl;
		return;
	}

    int send_size = send_count * sizeof(int);
	std::cout << "My send size is " << send_size << std::endl;

	int w = ceil(std::log(size) / std::log(r));
	int nlpow = myPow(r, w - 1);
	int d = (myPow(r, w) - size) / nlpow;
	std::cout << "My w is " << w << std::endl;
	std::cout << "My nlpow is " << nlpow << std::endl;
	std::cout << "My d is " << d << std::endl;

    CUDA_CALL(cudaMemcpy(d_recv_data, d_send_data, size * send_size, cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaMemcpy(&d_send_data[(size - rank) * send_size], d_recv_data, rank * send_size, cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaMemcpy(d_send_data, &d_recv_data[rank * send_size], (size - rank) * send_size, cudaMemcpyDeviceToDevice));

    std::vector<std::vector<int>> rank_r_reps(size * w);
	for (int i = 0; i < size; i++) {
		rank_r_reps[i] = convert10tob(w, i, r);
	}
	std::cout << "Process " << rank << " rank_r_reps: [";
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < w; j++) {
			std::cout << " " << rank_r_reps[i][j] << " ";
		}
	}
	std::cout << "]" << std::endl;

	int sent_blocks[nlpow];
	int di = 0;
	int ci = 0;

	int* temp_buffer;
    CUDA_CALL(cudaMalloc((void **) &temp_buffer, nlpow * send_size));

    for (int x = 0; x < w; x++) {
    	int ze = (x == w - 1)? r - d: r;
    	for (int z = 1; z < ze; z++) {
    		di = 0;
    		ci = 0;
    		for (int i = 0; i < size; i++) {
    			if (rank_r_reps[i][x] == z) {
					std::cout << "Process " << rank << " here" << std::endl;
    				sent_blocks[di++] = i;
					std::cout << "Process " << rank << " sentblocks " << i << std::endl;
    				CUDA_CALL(cudaMemcpy(&temp_buffer[send_size * ci++], &d_send_data[send_size * i], send_size, cudaMemcpyDeviceToDevice));
				}
    		}

    		int distance = z * myPow(r, x);
    		int recv_rank = (rank - distance + size) % size;
    		int send_rank = (rank + distance) % size;
			std::cout << "My distance is " << distance << std::endl;
			std::cout << "My recv_proc is " << recv_rank << std::endl;
			std::cout << "My send_proc is " << send_rank << std::endl;
            NCCL_CALL(ncclGroupStart());
            NCCL_CALL(ncclSend(temp_buffer, di * send_count * sizeof(int), send_type, send_rank, comm, stream));
            NCCL_CALL(ncclRecv(temp_buffer, di * recv_count * sizeof(int), recv_type, recv_rank, comm, stream));
            NCCL_CALL(ncclGroupEnd());

    		for (int i = 0; i < di; i++) {
    			CUDA_CALL(cudaMemcpy(&d_send_data[sent_blocks[i] * send_size], &d_recv_data[i * send_size], send_size, cudaMemcpyDeviceToDevice));
    		}
    	}
    }

	for (int i = 0; i < size; i++) {
		CUDA_CALL(cudaMemcpy(&d_recv_data[((rank - i + size) % size) * send_size], &d_send_data[i * send_size], send_size, cudaMemcpyDeviceToDevice));
	}

    cudaFree(temp_buffer);
}