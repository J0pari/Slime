# Build: each entry point is a single translation unit that #includes the rest,
# so the dependency graph stays simple while construction (C-001) is in flight.
# Targets touching device code require nvcc + an NVIDIA GPU and have not been
# compiled yet; `check` runs the host-only unit tests and needs no CUDA.

NVCC      ?= nvcc
ARCH      ?= sm_86
CXXFLAGS  ?= -std=c++17 -O2
NVCCFLAGS ?= -arch=$(ARCH) -rdc=true --extended-lambda --expt-relaxed-constexpr

BUILD_DIR ?= build
BIN          := $(BUILD_DIR)/coevo
HOST_TESTS   := $(BUILD_DIR)/host_tests
FORWARD_SMOKE := $(BUILD_DIR)/forward_smoke

SRC := integration/host_main.cu
HOST_SRC := tests/host_unit_tests.cpp

CXX      ?= g++
HOST_CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Itests/stubs -I.

.PHONY: all clean host-tests check forward-smoke

all: $(BIN)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(SRC) -o $@ -lcudadevrt

# Host-only unit tests for the math inlines (hybrid blending, role
# multipliers, SOT gate, PT swap probabilities, genome bit layout, etc.).
# Builds and runs without CUDA installed - useful for CI smoke and for
# web-sandbox sessions.
$(HOST_TESTS): $(HOST_SRC) config/constants.cuh tests/stubs/cuda_runtime.h tests/stubs/cuda_fp16.h | $(BUILD_DIR)
	$(CXX) $(HOST_CXXFLAGS) $(HOST_SRC) -o $@

host-tests: $(HOST_TESTS)
	./$(HOST_TESTS)

# Default 'check' target uses host tests so it passes without CUDA.
check: host-tests

# Stage-1 forward smoke test (A-201). Requires nvcc + an NVIDIA GPU. This is
# the first device-side build target; expect to fix compile errors on the
# first real build. Runs one organism through a forward pass and checks the
# BTRAJ is finite and non-constant.
$(FORWARD_SMOKE): tests/forward_smoke.cu nca/engine.cu nca/reaction_diffusion.cu config/constants.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) tests/forward_smoke.cu -o $@ -lcudadevrt

forward-smoke: $(FORWARD_SMOKE)
	./$(FORWARD_SMOKE)

clean:
	rm -rf $(BUILD_DIR)
