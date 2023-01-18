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

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < size; i++) {
        int send_rank = (rank - i + size) % size;
        int recv_rank = (rank + i) % size;
        NCCLCHECK(ncclSend(&d_send_data[send_rank * send_count * unit_size], send_count * unit_size, send_type, send_rank, comm, stream));
        NCCLCHECK(ncclRecv(&d_recv_data[recv_rank * recv_count * unit_size], recv_count * unit_size, recv_type, recv_rank, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
}