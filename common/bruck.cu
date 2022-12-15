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
	  v[i++] = (N % b);
	  N /= b;
	}
	return v;
}

void ncclBruck(int r, char* sendbuf, int sendcount, ncclDataType_t sendtype, char* recvbuf, int recvcount, ncclDataType_t recvtype,  ncclComm_t comm, cudaStream_t stream, int rank, int size) {
    int typesize = sizeof(sendtype);
    int unit_size = sendcount * typesize;

	int w = 1;
	if (size != 0 && r == 0) {
		int w = ceil(std::log(size) / std::log(r));
	}
	int nlpow = myPow(r, w-1);
	int d = (myPow(r, w) - size) / nlpow; // calculate the number of highest digits

    // local rotation
    CUDA_CALL(cudaMemcpy(recvbuf, sendbuf, size * unit_size, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(&sendbuf[(size - rank) * unit_size], recvbuf, rank * unit_size, cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(sendbuf, &recvbuf[rank * unit_size], (size - rank) * unit_size, cudaMemcpyDeviceToDevice));

    // convert rank to base r representation
    int* rank_r_reps;
    CUDA_CALL(cudaMalloc((void **) &rank_r_reps, size * w * sizeof(int)));
	for (int i = 0; i < size; i++) {
		std::vector<int> r_rep = convert10tob(w, i, r);
		CUDA_CALL(cudaMemcpy(&rank_r_reps[i * w], r_rep.data(), w * sizeof(int), cudaMemcpyDeviceToDevice));
	}

	int sent_blocks[nlpow];
	int sent_blocks_comp[nlpow];
	int di = 0;
	int ci = 0;

	int comm_steps = (r - 1) * w - d;
	char* temp_buffer;
    CUDA_CALL(cudaMalloc((void **) &temp_buffer, nlpow * unit_size));

	// communication steps = (r - 1)w - d
    for (int x = 0; x < w; x++) {
    	int ze = (x == w - 1)? r - d: r;
    	for (int z = 1; z < ze; z++) {
    		// get the sent data-blocks
    		// copy blocks which need to be sent at this step
    		di = 0;
    		ci = 0;
    		for (int i = 0; i < size; i++) {
    			if (rank_r_reps[i*w + x] == z) {
    				sent_blocks[di++] = i;
    				CUDA_CALL(cudaMemcpy(&temp_buffer[unit_size*ci++], &sendbuf[i*unit_size], unit_size, cudaMemcpyDeviceToDevice));
    			}
    		}

    		int distance = z * myPow(r, x); // pow(1, 51) = 51, int d = pow(1, 51); // 50
    		int recv_proc = (rank - distance + size) % size; // receive data from rank - 2^step process
    		int send_proc = (rank + distance) % size; // send data from rank + 2^k process
    		long long comm_size = di * unit_size;

            NCCL_CALL(ncclGroupStart());
            NCCL_CALL(ncclSend(temp_buffer, comm_size, ncclChar, send_proc, comm, stream));
            NCCL_CALL(ncclRecv(temp_buffer, recvcount, ncclChar, recv_proc, comm, stream));
            NCCL_CALL(ncclGroupEnd());

    		// replace with received data
    		for (int i = 0; i < di; i++) {
    			long long offset = sent_blocks[i] * unit_size;
    			CUDA_CALL(cudaMemcpy(sendbuf + offset, recvbuf+(i*unit_size), unit_size, cudaMemcpyDeviceToDevice));
    		}
    	}
    }

    cudaFree(rank_r_reps);
    cudaFree(temp_buffer);

    // local rotation
	for (int i = 0; i < size; i++) {
		int index = (rank - i + size) % size;
		CUDA_CALL(cudaMemcpy(&recvbuf[index*unit_size], &sendbuf[i*unit_size], unit_size, cudaMemcpyDeviceToDevice));
	}
}