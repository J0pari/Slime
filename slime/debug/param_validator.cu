#ifndef PARAM_VALIDATOR_CU
#define PARAM_VALIDATOR_CU

#include <cuda_runtime.h>
#include <stdio.h>


inline bool validate_pointer(const char* name, void* ptr, bool must_be_device, const char* file, int line) {
    if (ptr == nullptr) {
        return false;
    }

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const char* type_str;
    bool is_valid = true;

    switch(attr.type) {
        case cudaMemoryTypeUnregistered: type_str = "unregistered"; is_valid = !must_be_device; break;
        case cudaMemoryTypeHost: type_str = "host"; is_valid = !must_be_device; break;
        case cudaMemoryTypeDevice: type_str = "device"; break;
        case cudaMemoryTypeManaged: type_str = "managed"; break;
        default: type_str = "unknown"; is_valid = false; break;
    }

    if (!is_valid) {
        return false;
    }
    return true;
}


template<typename T>
inline void print_struct_layout(const char* name) {
}


inline bool validate_int_range(const char* name, int value, int min, int max, const char* file, int line) {
    if (value < min || value > max) {
        return false;
    }
    return true;
}


inline bool validate_launch_config(dim3 grid, dim3 block, size_t shared_mem, const char* kernel_name, const char* file, int line) {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    bool valid = true;

    if (block.x * block.y * block.z > (unsigned)prop.maxThreadsPerBlock) {
        valid = false;
    }

    if (grid.x > (unsigned)prop.maxGridSize[0] ||
        grid.y > (unsigned)prop.maxGridSize[1] ||
        grid.z > (unsigned)prop.maxGridSize[2]) {
        valid = false;
    }

    if (shared_mem > prop.sharedMemPerBlock) {
        valid = false;
    }

    if (valid) {
    }

    return valid;
}

#define VALIDATE_DEVICE_PTR(ptr) \
    validate_pointer(#ptr, (void*)(ptr), true, __FILE__, __LINE__)

#define VALIDATE_HOST_PTR(ptr) \
    validate_pointer(#ptr, (void*)(ptr), false, __FILE__, __LINE__)

#define VALIDATE_INT_RANGE(val, min, max) \
    validate_int_range(#val, val, min, max, __FILE__, __LINE__)

#define VALIDATE_LAUNCH_CONFIG(grid, block, shared, kernel) \
    validate_launch_config(grid, block, shared, #kernel, __FILE__, __LINE__)

#define PRINT_STRUCT(type) \
    print_struct_layout<type>(#type)

#define BEGIN_KERNEL_VALIDATION(kernel_name) \
    do { \
\
\
        cudaError_t _pending_err = cudaGetLastError(); \
        if (_pending_err != cudaSuccess) { \
\
        } \
    } while(0)

#define END_KERNEL_VALIDATION() \
    do { \
\
    } while(0)

#endif
