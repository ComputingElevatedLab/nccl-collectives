// Source: https://github.com/harp-lab/rbruck_alltoall
#include <cmath>
#include <cstring>
#include <vector>

#include "nccl.h"

#include "error-catch.cu"

size_t ncclTypeSize(ncclDataType_t type) {
	switch(type) {
		case ncclChar:
		#if NCCL_MAJOR >= 2
			case ncclUint8:
		#endif
		return 1;
		case ncclHalf:
		#if defined(__CUDA_BF16_TYPES_EXIST__)
			case ncclBfloat16:
		#endif
		return 2;
		case ncclInt:
		case ncclFloat:
		#if NCCL_MAJOR >= 2
			case ncclUint32:
		#endif
		return 4;
		case ncclInt64:
		case ncclUint64:
		case ncclDouble:
		return 8;
		default: return 0;
	}
}

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

void ncclBruck(int r, char* d_send_data, int send_count, ncclDataType_t send_type, char* d_recv_data, int recv_count, ncclDataType_t recv_type,  ncclComm_t comm, cudaStream_t stream) {
    int size;
	int rank;
    NCCLCHECK(ncclCommCount(comm, &size));
	NCCLCHECK(ncclCommUserRank(comm, &rank));

	if (r < 2 || size < 2) {
		std::cout << "Error: ncclBruck requires r >= 2 and nProc >= 2" << std::endl;
		return;
	}

    int unit_size = send_count * ncclTypeSize(send_type);
	int w = std::ceil(std::log(size) / std::log(r));
	int nlpow = myPow(r, w - 1);
	int d = (myPow(r, w) - size) / nlpow;

    CUDACHECK(cudaMemcpy(d_recv_data, d_send_data, size * unit_size, cudaMemcpyDefault));
	CUDACHECK(cudaMemcpy(&d_send_data[(size - rank) * unit_size], d_recv_data, rank * unit_size, cudaMemcpyDefault));
	CUDACHECK(cudaMemcpy(d_send_data, &d_recv_data[rank * unit_size], (size - rank) * unit_size, cudaMemcpyDefault));

    std::vector<std::vector<int>> rank_r_reps(size * w);
	for (int i = 0; i < size; i++) {
		std::vector<int> v(w);
		int N = i;
		int j = 0;
		while(N) {
			v[j++] = (N % r);
			N /= r;
		}
		rank_r_reps[i] = v;
	}

	int sent_blocks[nlpow];
	int di = 0;
	int ci = 0;

	char* temp_buffer;
    CUDACHECK(cudaMalloc((void **) &temp_buffer, nlpow * unit_size));

    for (int x = 0; x < w; x++) {
    	int ze = (x == w - 1)? r - d: r;
    	for (int z = 1; z < ze; z++) {
    		di = 0;
    		ci = 0;
    		for (int i = 0; i < size; i++) {
    			if (rank_r_reps[i][x] == z) {
    				sent_blocks[di] = i;
    				CUDACHECK(cudaMemcpy(&temp_buffer[unit_size * ci], &d_send_data[unit_size * i], unit_size, cudaMemcpyDefault));
                    di += 1;
                    ci += 1;
				}
    		}

    		int distance = z * myPow(r, x);
    		int recv_rank = (rank - distance + size) % size;
    		int send_rank = (rank + distance) % size;

            NCCLCHECK(ncclGroupStart());
            NCCLCHECK(ncclSend(temp_buffer, di * unit_size, send_type, send_rank, comm, stream));
            NCCLCHECK(ncclRecv(d_recv_data, di * unit_size, recv_type, recv_rank, comm, stream));
            NCCLCHECK(ncclGroupEnd());

    		for (int i = 0; i < di; i++) {
    			CUDACHECK(cudaMemcpy(&d_send_data[sent_blocks[i] * unit_size], &d_recv_data[i * unit_size], unit_size, cudaMemcpyDefault));
    		}
    	}
    }

	for (int i = 0; i < size; i++) {
		CUDACHECK(cudaMemcpy(&d_recv_data[((rank - i + size) % size) * unit_size], &d_send_data[i * unit_size], unit_size, cudaMemcpyDefault));
	}

    CUDACHECK(cudaFree(temp_buffer));
}
