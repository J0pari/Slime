# Slime GPU-native build system
NVCC = nvcc
ARCH = sm_86  # RTX 3060
CUDAFLAGS = -arch=$(ARCH) -O3 --use_fast_math -rdc=true -lcudadevrt --expt-relaxed-constexpr
CUDALIBS = -lcudart -lcuda -lcudadevrt -lnvrtc

KERNELS = slime/kernels/warp_ca.cu
TARGET = slime

all: $(TARGET)

$(TARGET): $(KERNELS)
	$(NVCC) $(CUDAFLAGS) $(KERNELS) -o $(TARGET) $(CUDALIBS)

clean:
	rm -f $(TARGET) *.o

test: $(TARGET)
	./$(TARGET) --test

bench: $(TARGET)
	./$(TARGET) --benchmark

.PHONY: all clean test bench