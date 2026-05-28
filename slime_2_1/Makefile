# Slime 2.1 build. Mirrors the 2.0 convention (single translation unit
# including the rest), so the dependency graph stays simple while
# construction sequence C-001 is in flight.

NVCC      ?= nvcc
ARCH      ?= sm_86
CXXFLAGS  ?= -std=c++17 -O2
NVCCFLAGS ?= -arch=$(ARCH) -rdc=true --extended-lambda --expt-relaxed-constexpr

BUILD_DIR ?= build
BIN       := $(BUILD_DIR)/slime_2_1
HOST_TESTS := $(BUILD_DIR)/host_tests

SRC := integration/host_main.cu
HOST_SRC := tests/host_unit_tests.cpp

CXX      ?= g++
HOST_CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Itests/stubs -I.

.PHONY: all clean host-tests check

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

clean:
	rm -rf $(BUILD_DIR)
