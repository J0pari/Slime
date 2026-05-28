# Slime 2.1 build. Mirrors the 2.0 convention (single translation unit
# including the rest), so the dependency graph stays simple while
# construction sequence C-001 is in flight.

NVCC      ?= nvcc
ARCH      ?= sm_86
CXXFLAGS  ?= -std=c++17 -O2
NVCCFLAGS ?= -arch=$(ARCH) -rdc=true --extended-lambda --expt-relaxed-constexpr

BUILD_DIR ?= build
BIN       := $(BUILD_DIR)/slime_2_1

SRC := integration/host_main.cu

.PHONY: all clean syntax-check

all: $(BIN)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN): $(SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(SRC) -o $@ -lcudadevrt

# Host-only syntax check. Requires clang++ and stubs <cuda_runtime.h> et al.
# Used in environments where CUDA isn't installed (CI smoke, web sandboxes).
syntax-check:
	@echo "Host-only syntax check is not wired yet; install CUDA toolkit and"
	@echo "use 'make all' for full validation."

clean:
	rm -rf $(BUILD_DIR)
