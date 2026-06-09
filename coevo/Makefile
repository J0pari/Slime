# Build system for Slime Evolution.
#
# Each entry point is a single translation unit that #includes the rest.
# Prerequisites: source env.sh once per terminal session.

NVCC      ?= nvcc
ARCH      ?= sm_86
CXXFLAGS  ?= -std=c++17 -O2
NVCCFLAGS ?= -arch=$(ARCH) -rdc=true --extended-lambda --expt-relaxed-constexpr

BUILD_DIR ?= build
BIN          := $(BUILD_DIR)/coevo
HOST_TESTS   := $(BUILD_DIR)/host_tests
FORWARD_SMOKE := $(BUILD_DIR)/forward_smoke
WAVE1_TEST   := $(BUILD_DIR)/wave1_test

SRC := integration/host_main.cu
HOST_SRC := tests/host_unit_tests.cpp

HOST_CXX      ?= $(NVCC)
HOST_CXXFLAGS := -Itests/stubs -I. -std=c++17 -Xcompiler /EHsc -Xcompiler /W3

.PHONY: all run clean host-tests check forward-smoke wave1-test

all: $(BIN)

run: $(BIN)
	./$(BIN)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN): $(SRC) integration/main_loop.cu safety/alignment.cu autodiff/warp_tape.cu optimizer/came.cu nca/engine.cu nca/reaction_diffusion.cu genome/codec.cu archive/soft_qd_archive.cu curriculum/problem_generator.cu safety/monitoring.cu safety/parallel_tempering.cu safety/structural.cu config/constants.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(SRC) -o $@ -lcudadevrt

# Host-only unit tests (math inlines: blending, SOT gate, PT, genome, etc.).
$(HOST_TESTS): $(HOST_SRC) config/constants.cuh tests/stubs/cuda_runtime.h tests/stubs/cuda_fp16.h | $(BUILD_DIR)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_SRC) -o $@

host-tests: $(HOST_TESTS)
	./$(HOST_TESTS)

check: host-tests

# Forward smoke test (A-201). One organism, 64-step CA, verify BTRAJ.
$(FORWARD_SMOKE): tests/forward_smoke.cu nca/engine.cu nca/reaction_diffusion.cu config/constants.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) tests/forward_smoke.cu -o $@ -lcudadevrt

forward-smoke: $(FORWARD_SMOKE)
	./$(FORWARD_SMOKE)

# Wave 1: Autodiff + CAME. Forward with checkpoints, backward with full
# stencil adjoint, gradient aggregation, CAME step, loss decreases.
$(WAVE1_TEST): tests/wave1_autodiff.cu autodiff/warp_tape.cu optimizer/came.cu nca/engine.cu nca/reaction_diffusion.cu config/constants.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) tests/wave1_autodiff.cu -o $@ -lcudadevrt

wave1-test: $(WAVE1_TEST)
	./$(WAVE1_TEST)

clean:
	rm -rf $(BUILD_DIR)
