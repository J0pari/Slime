#ifndef SLIME_2_1_STUBS_CUDA_FP16_H
#define SLIME_2_1_STUBS_CUDA_FP16_H

struct __half { unsigned short bits; };
inline float  __half2float(const __half& h) { (void)h; return 0.f; }
inline __half __float2half(float)           { return __half{0}; }
inline float atomicAdd(float* p, float v)   { float o = *p; *p = o + v; return o; }

#endif  // SLIME_2_1_STUBS_CUDA_FP16_H
