// Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
#include <sched.h>

#include "nccl.h"

int ncclStreamSynchronize(cudaStream_t stream, ncclComm_t comm)
{
  cudaError_t cudaErr;
  ncclResult_t ncclErr, ncclAsyncErr;
  while (1)
  {
    cudaErr = cudaStreamQuery(stream);
    if (cudaErr == cudaSuccess)
      return 0;

    if (cudaErr != cudaErrorNotReady)
    {
      printf("CUDA Error : cudaStreamQuery returned %d\n", cudaErr);
      return 1;
    }

    ncclErr = ncclCommGetAsyncError(comm, &ncclAsyncErr);
    if (ncclErr != ncclSuccess)
    {
      printf("NCCL Error : ncclCommGetAsyncError returned %d\n", ncclErr);
      return 1;
    }

    if (ncclAsyncErr != ncclSuccess)
    {
      // An asynchronous error happened. Stop the operation and destroy
      // the communicator
      ncclErr = ncclCommAbort(comm);
      if (ncclErr != ncclSuccess)
        printf("NCCL Error : ncclCommDestroy returned %d\n", ncclErr);
      // Caller may abort or try to re-create a new communicator.
      return 2;
    }

    // We might want to let other threads (including NCCL threads) use the CPU.
    sched_yield();
  }
}