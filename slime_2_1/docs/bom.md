# M-001: Bill of Materials

Identical to 2.0. No new dependencies introduced in 2.1.

Carried forward from 2.0:

- NVIDIA GPU sm_86+ (Ampere/Hopper) with Tensor Cores, 6 GB+ VRAM
- CUDA Toolkit 12.0+
- Host compiler: MSVC 2022 Build Tools on Windows, GCC 11+ on Linux
- Optional: Nsight Systems for profiling captured graphs

The 2.1 BTRAJ change adds 24 KB of managed-memory writes per generation
(4 samples * 32 floats * FP32 * POOL_SIZE = 24576 bytes); no additional
memory class or device feature is needed.
