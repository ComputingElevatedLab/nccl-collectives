// Source: https://github.com/FZJ-JSC/tutorial-multi-gpu
#include "nccl.h"

#include "hostname.cu"

#define CUDACHECK(cmd)                                     \
  do                                                       \
  {                                                        \
    cudaError_t err = cmd;                                 \
    if (err != cudaSuccess)                                \
    {                                                      \
      char hostname[1024];                                 \
      getHostName(hostname, 1024);                         \
      printf("%s: CUDA failure %s:%d '%s'\n",              \
             hostname,                                     \
             __FILE__, __LINE__, cudaGetErrorString(err)); \
    }                                                      \
  } while (0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,13,0)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d "           \
           "'%s / %s'\n",                           \
           hostname,__FILE__,__LINE__,              \
           ncclGetErrorString(res),                 \
           ncclGetLastError(NULL));                 \
    return testNcclError;                           \
  }                                                 \
} while(0)
#else
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    return testNcclError;                           \
  }                                                 \
} while(0)
#endif