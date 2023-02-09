#pragma once
#include "nccl.h"

size_t ncclTypeSize(ncclDataType_t type)
{
	switch (type)
	{
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
	default:
		return 0;
	}
}