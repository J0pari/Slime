// Host-only test stub. Found via -Itests/stubs before the system path. Defines
// the minimum surface area that the 2.1 module headers reference so the
// __host__ __device__ inlines can be compiled and exercised without CUDA.

#ifndef COEVO_STUBS_CUDA_RUNTIME_H
#define COEVO_STUBS_CUDA_RUNTIME_H

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

inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }

// Graph API stubs (enough for phase_graphs.cu to parse in host-only mode).
using cudaGraph_t     = void*;
using cudaGraphExec_t = void*;
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0 };
inline cudaError_t cudaStreamCreate(cudaStream_t*)          { return 0; }
inline cudaError_t cudaStreamDestroy(cudaStream_t)          { return 0; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { return 0; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*)            { return 0; }
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t)         { return 0; }
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t)              { return 0; }
inline void        cudaGraphExecDestroy(cudaGraphExec_t)    {}
inline void        cudaGraphDestroy(cudaGraph_t)            {}

// Memory stubs.
inline cudaError_t cudaMallocManaged(void**, size_t, unsigned = 0) { return 0; }
inline cudaError_t cudaDeviceSynchronize()                         { return 0; }
inline cudaError_t cudaGetLastError()                              { return 0; }

// Dim3 / launch config stubs.
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

#endif  // COEVO_STUBS_CUDA_RUNTIME_H
