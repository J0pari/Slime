#!/usr/bin/env bash
# Source this from Git Bash to set up the MSVC + CUDA + make build environment.
#
#   source env.sh
#
# Safe to source multiple times.  After sourcing, cl.exe / nvcc / make are
# all in PATH and INCLUDE / LIB point at the MSVC + Windows SDK headers/libs.

# ---- Paths (edit these if your install locations differ) --------------------
MSVC_VER="14.44.35207"
MSVC_ROOT="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/${MSVC_VER}"
WINSDK_VER="10.0.22621.0"
WINSDK_ROOT="/c/Program Files (x86)/Windows Kits/10"
CUDA_ROOT="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
MAKE_DIR="/c/Program Files (x86)/GnuWin32/bin"

# ---- Validate ---------------------------------------------------------------
_fail=0
[[ ! -f "${MSVC_ROOT}/bin/Hostx64/x64/cl.exe" ]] && echo "env.sh: cl.exe not found at ${MSVC_ROOT}" >&2 && _fail=1
[[ ! -f "${CUDA_ROOT}/bin/nvcc.exe" ]]            && echo "env.sh: nvcc not found at ${CUDA_ROOT}" >&2 && _fail=1
[[ ! -f "${MAKE_DIR}/make.exe" ]]                  && echo "env.sh: make not found at ${MAKE_DIR}" >&2 && _fail=1
if [[ $_fail -ne 0 ]]; then
    echo "env.sh: Fix the paths at the top of this file." >&2
    unset _fail
    return 1 2>/dev/null || exit 1
fi
unset _fail

# ---- PATH -------------------------------------------------------------------
_add_path() { [[ ":${PATH}:" != *":$1:"* ]] && export PATH="$1:${PATH}"; }

_add_path "${MSVC_ROOT}/bin/Hostx64/x64"
_add_path "${CUDA_ROOT}/bin"
_add_path "${MAKE_DIR}"

unset -f _add_path

# ---- INCLUDE (semicolon-delimited Windows paths, as cl.exe expects) ---------
_W_MSVC="$(cygpath -w "${MSVC_ROOT}")"
_W_SDK="$(cygpath -w "${WINSDK_ROOT}")"
export INCLUDE="${_W_MSVC}\\include;${_W_SDK}\\Include\\${WINSDK_VER}\\ucrt;${_W_SDK}\\Include\\${WINSDK_VER}\\shared;${_W_SDK}\\Include\\${WINSDK_VER}\\um"

# ---- LIB --------------------------------------------------------------------
export LIB="${_W_MSVC}\\lib\\x64;${_W_SDK}\\Lib\\${WINSDK_VER}\\ucrt\\x64;${_W_SDK}\\Lib\\${WINSDK_VER}\\um\\x64"

unset _W_MSVC _W_SDK

# ---- Sanity check -----------------------------------------------------------
echo "env.sh: cl    -> $(which cl.exe 2>/dev/null || echo 'NOT FOUND')"
echo "env.sh: nvcc  -> $(which nvcc 2>/dev/null || echo 'NOT FOUND')"
echo "env.sh: make  -> $(which make 2>/dev/null || echo 'NOT FOUND')"
