#ifndef CUDA_FIX_H
#define CUDA_FIX_H

// Fix CUDA's broken deprecation macros for MSVC device code
// The system headers use __pragma which doesn't work in device code
// We need to redefine these macros to use _Pragma instead

#if defined(__CUDACC__) && defined(_MSC_VER)

// Undefine the broken macros
#undef __NV_SILENCE_DEPRECATION_BEGIN
#undef __NV_SILENCE_DEPRECATION_END

// Redefine with working _Pragma syntax
#define __NV_SILENCE_DEPRECATION_BEGIN \
  __NV_SILENCE_HOST_DEPRECATION_BEGIN \
  _Pragma("nv_diagnostic push") \
  _Pragma("nv_diag_suppress 1444")

#define __NV_SILENCE_DEPRECATION_END \
  __NV_SILENCE_HOST_DEPRECATION_END \
  _Pragma("nv_diagnostic pop")

#endif

#endif // CUDA_FIX_H
