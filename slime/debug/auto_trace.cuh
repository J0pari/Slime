#ifndef AUTO_TRACE_CUH
#define AUTO_TRACE_CUH




#include "kernel_trace.cu"







#ifdef __CUDACC__









#define LAUNCH(kernel, grid, block, shared, stream, ...) \
    traced_kernel_launch(kernel, #kernel, __FILE__, __LINE__, grid, block, shared, stream, ##__VA_ARGS__)

#define LAUNCH_SIMPLE(kernel, grid, block, ...) \
    traced_kernel_launch(kernel, #kernel, __FILE__, __LINE__, grid, block, 0, 0, ##__VA_ARGS__)

#endif 

#endif 
