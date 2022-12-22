// Source: https://github.com/harp-lab/rbruck_alltoall
#include <cmath>
#include <cstring>
#include <vector>

#include <stdio.h>

#include "nccl.h"

#include "error-catch.cu"

int myPow(int x, unsigned int p) {
    if (p == 0) return 1;
    if (p == 1) return x;
    int tmp = myPow(x, p/2);
    if (p%2 == 0) return tmp * tmp;
    else return x * tmp * tmp;
}

std::vector<int> convert10tob(int w, int N, int b) {
	std::vector<int> v(w);
	int i = 0;
	while(N) {
	  v.at(i++) = (N % b);
	  N /= b;
	}
	return v;
}

void ncclBruck(int r, int* d_send_data, int send_count, ncclDataType_t send_type, int* d_recv_data, int recv_count, ncclDataType_t recv_type,  ncclComm_t comm, cudaStream_t stream) {
	// Set MPI size and rank
    int size;
    int rank;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	
	if (r < 2 || size < 2) {
		std::cout << "Error: ncclBruck requires r >= 2 and nProc >= 2" << std::endl;
		return;
	}

    int send_size = send_count * sizeof(int);

	int w = ceil(std::log(size) / std::log(r));
	int nlpow = myPow(r, w - 1);
	int d = (myPow(r, w) - size) / nlpow;

    CUDA_CALL(cudaMemcpy(d_recv_data, d_send_data, size * send_size, cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaMemcpy(d_send_data + (size - rank) * send_count, d_recv_data, rank * send_size, cudaMemcpyDeviceToDevice));
	CUDA_CALL(cudaMemcpy(d_send_data, d_recv_data + rank * send_count, (size - rank) * send_size, cudaMemcpyDeviceToDevice));


    std::vector<std::vector<int>> rank_r_reps(size * w);
	for (int i = 0; i < size; i++) {
		rank_r_reps[i] = convert10tob(w, i, r);
	}

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
    				sent_blocks[di++] = i;
    				CUDA_CALL(cudaMemcpy(temp_buffer + send_count * ci++, d_send_data + send_count * i, send_size, cudaMemcpyDeviceToDevice));
    			}
    		}

    		int distance = z * myPow(r, x);
    		int recv_rank = (rank - distance + size) % size;
    		int send_rank = (rank + distance) % size;
    		long long comm_size = di * send_size;

            NCCL_CALL(ncclGroupStart());
            NCCL_CALL(ncclSend(temp_buffer, comm_size, send_type, send_rank, comm, stream));
            NCCL_CALL(ncclRecv(temp_buffer, recv_count, recv_type, recv_rank, comm, stream));
            NCCL_CALL(ncclGroupEnd());

    		for (int i = 0; i < di; i++) {
    			CUDA_CALL(cudaMemcpy(d_send_data + sent_blocks[i] * send_count, d_recv_data + i * send_count, send_size, cudaMemcpyDeviceToDevice));
    		}
    	}
    }

	for (int i = 0; i < size; i++) {
		CUDA_CALL(cudaMemcpy(d_recv_data + ((rank - i + size) % size) * send_count, d_send_data + i * send_count, send_size, cudaMemcpyDeviceToDevice));
	}

    cudaFree(temp_buffer);
}