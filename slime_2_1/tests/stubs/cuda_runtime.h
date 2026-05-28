// Host-only test stub. Found via -Itests/stubs before the system path. Defines
// the minimum surface area that the 2.1 module headers reference so the
// __host__ __device__ inlines can be compiled and exercised without CUDA.

#ifndef SLIME_2_1_STUBS_CUDA_RUNTIME_H
#define SLIME_2_1_STUBS_CUDA_RUNTIME_H

#include <cstddef>

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

using cudaStream_t = void*;
using cudaError_t  = int;
#define cudaSuccess 0

#endif  // SLIME_2_1_STUBS_CUDA_RUNTIME_H
