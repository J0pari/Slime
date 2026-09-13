# Build system for Slime Evolution.
#
# Each entry point is a single translation unit that #includes the rest.
# Prerequisites: source env.sh once per terminal session.

NVCC      ?= nvcc
ARCH      ?= sm_86
CXXFLAGS  ?= -std=c++17 -O2
NVCCFLAGS ?= -arch=$(ARCH) -rdc=true --extended-lambda --expt-relaxed-constexpr

BUILD_DIR ?= build
BIN          := $(BUILD_DIR)/coevo.exe
HOST_TESTS   := $(BUILD_DIR)/host_tests.exe
FORWARD_SMOKE := $(BUILD_DIR)/forward_smoke.exe
AUTODIFF_TEST   := $(BUILD_DIR)/autodiff_test.exe
EVOLUTION_TEST  := $(BUILD_DIR)/evolution_test.exe

SRC := integration/host_main.cu
HOST_SRC := tests/host_unit_tests.cpp

HOST_CXX      ?= g++
HOST_CXXFLAGS := -Itests/stubs -I. -std=c++17 -Wall -Wextra -Werror

.PHONY: all run run-10 clean host-tests check forward-smoke autodiff-test evolution-test \
	task-conditioning-test architecture-check architecture-test architecture-status architecture-report \
	gpu-status gpu-contract gpu-evolution gpu-run10

all: $(BIN)

run: $(BIN)
	./$(BIN)

# Acceptance run: the integration binary accepts a generation count.
run-10: $(BIN)
	./$(BIN) 10

# ---- Architecture-control layer (see AGENTS.md) ---------------------------
architecture-check:
	python architecture/compiler.py check --golden
	python architecture/source_gates.py
	python -m unittest discover -s tests/architecture -v

architecture-test:
	python -m unittest discover -s tests/architecture -v

architecture-status:
	python architecture/compiler.py status

architecture-report:
	python architecture/compiler.py report

# ---- GPU scheduler (training-architecture, gpu-scheduler/v1) --------------
# All GPU work goes through the cross-repo scheduler. Requires
# TRAINING_ARCH_ROOT (the training-architecture repo root); the client
# refuses loudly when it is unset. Scheduled jobs are covered by the
# scheduler's GPU lock; use --direct only for an explicit manual launch.
gpu-status:
	python architecture/gpu_client.py status

gpu-contract:
	python architecture/gpu_client.py contract

gpu-evolution:
	python architecture/gpu_client.py run --name slime-evolution-regression --vram 2048 \
		-- build/evolution_test.exe

gpu-run10:
	python architecture/gpu_client.py run --name slime-run10 --vram 2048 \
		--total 10 -- build/coevo.exe 10

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN): $(SRC) integration/main_loop.cu safety/alignment.cu autodiff/warp_tape.cu optimizer/came.cu optimizer/came_math.cuh nca/engine.cu nca/reaction_diffusion.cu genome/codec.cu archive/soft_qd_archive.cu curriculum/problem_generator.cu safety/monitoring.cu safety/parallel_tempering.cu safety/pt_ladder.cuh safety/structural.cu config/constants.cuh | $(BUILD_DIR)
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

# Autodiff + CAME: forward with checkpoints, backward with full
# stencil adjoint, gradient aggregation, CAME step, loss decreases.
$(AUTODIFF_TEST): tests/autodiff_acceptance.cu autodiff/warp_tape.cu optimizer/came.cu optimizer/came_math.cuh nca/engine.cu nca/reaction_diffusion.cu genome/codec.cu config/constants.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) tests/autodiff_acceptance.cu -o $@ -lcudadevrt

autodiff-test: $(AUTODIFF_TEST)
	./$(AUTODIFF_TEST)

# Regression: effective-weight causality, materialization, PT
# transaction, finite-difference gradient validation. Links with a larger
# stack: the test functions hold several 32KB DeltaWeights/GradBuffers frames.
$(EVOLUTION_TEST): tests/evolution_regression.cu safety/parallel_tempering.cu safety/pt_ladder.cuh optimizer/came.cu optimizer/came_math.cuh autodiff/warp_tape.cu nca/engine.cu nca/reaction_diffusion.cu genome/codec.cu config/constants.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) tests/evolution_regression.cu -o $@ -lcudadevrt -Xlinker /STACK:33554432

evolution-test: $(EVOLUTION_TEST)
	./$(EVOLUTION_TEST)

# Task-conditioning witness (A201.task-conditioning-complete): perturbing a
# single task-embedding dimension must change the descriptor.
TASK_CONDITIONING := $(BUILD_DIR)/task_conditioning.exe
$(TASK_CONDITIONING): tests/task_conditioning.cu autodiff/warp_tape.cu nca/engine.cu nca/reaction_diffusion.cu genome/codec.cu config/constants.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) tests/task_conditioning.cu -o $@ -lcudadevrt

task-conditioning-test: $(TASK_CONDITIONING)
	./$(TASK_CONDITIONING)

# S-001 checkpoint state roundtrip (GPU component test).
CHECKPOINT_STATE := $(BUILD_DIR)/checkpoint_state.exe
$(CHECKPOINT_STATE): tests/checkpoint_state.cu integration/host_main.cu integration/checkpointing.cu integration/main_loop.cu autodiff/warp_tape.cu optimizer/came.cu nca/engine.cu nca/reaction_diffusion.cu genome/codec.cu archive/soft_qd_archive.cu curriculum/problem_generator.cu safety/monitoring.cu safety/parallel_tempering.cu safety/pt_ladder.cuh safety/structural.cu safety/alignment.cu safety/operator_cmds.cuh predictor/hybrid_surprise.cu config/constants.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) tests/checkpoint_state.cu -o $@ -lcudadevrt

checkpoint-state-test: $(CHECKPOINT_STATE)
	./$(CHECKPOINT_STATE)

clean:
	rm -rf $(BUILD_DIR)

stability-run: $(BIN)
	python tests/long_run_check.py --binary $(BIN) --gens 20 --chunk 5

mutation-check:
	python tests/mutation_check.py

rebuild-gpu:
	python tests/rebuild_binaries.py
