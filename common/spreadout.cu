#include <iostream>

#include "cuda_runtime.h"
#include "nccl.h"

#include "error-catch.cu"
#include "typesize.cu"

void ncclSpreadout(char* d_send_data, int send_count, ncclDataType_t send_type, char* d_recv_data, int recv_count, ncclDataType_t recv_type,  ncclComm_t comm, cudaStream_t stream) {
    int size;
	int rank;
    NCCLCHECK(ncclCommCount(comm, &size));
	NCCLCHECK(ncclCommUserRank(comm, &rank));

	int unit_size = send_count * ncclTypeSize(send_type);
    std::cout << "Rank " << rank << ": unit_size=" << unit_size << std::endl;

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < size; i++) {
        int src = (rank + i) % size;
        std::cout << "Rank " << rank << ": src=" << src << std::endl;
        NCCLCHECK(ncclRecv(&d_recv_data[src * recv_count * unit_size], recv_count * unit_size, recv_type, src, comm, stream));
    }
    for (int i = 0; i < size; i++) {
        int dst = (rank - i + size) % size;
        std::cout << "Rank " << rank << ": dst=" << dst << std::endl;
        NCCLCHECK(ncclSend(&d_send_data[dst * send_count * unit_size], send_count * unit_size, send_type, dst, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
}