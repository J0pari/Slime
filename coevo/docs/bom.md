# M-001: Bill of Materials

Hardware and toolchain:

- NVIDIA GPU sm_86+ (Ampere/Hopper) with Tensor Cores, 6 GB+ VRAM
- CUDA Toolkit 12.0+
- Host compiler: MSVC 2022 Build Tools on Windows, GCC 11+ on Linux
- Optional: Nsight Systems for profiling captured graphs

No third-party runtime dependencies beyond the CUDA toolkit.

Memory note: BTRAJ capture adds 24 KB of managed-memory writes per generation
(4 samples × 32 floats × FP32 × POOL_SIZE = 24576 bytes); no additional memory
class or device feature is required for it.
