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

#define NCCLCHECK(cmd)                     \
  do                                       \
  {                                        \
    ncclResult_t res = cmd;                \
    if (res != ncclSuccess)                \
    {                                      \
      char hostname[1024];                 \
      getHostName(hostname, 1024);         \
      printf("%s: NCCL failure %s:%d "     \
             "'%s / %s'\n",                \
             hostname, __FILE__, __LINE__, \
             ncclGetErrorString(res),      \
             ncclGetLastError(NULL));      \
    }                                      \
  } while (0)
