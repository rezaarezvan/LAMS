NVCC = nvcc
CC = gcc
NVCCFLAGS = -O3 -arch=sm_60  # Adjust the architecture flag as needed
CFLAGS = -O3 -fPIC

CPU_SRC = src/linear_algebra.c src/stats.c
CUDA_SRC = src/cuda_linear_algebra.cu
TEST_SRC = tests/tests.c
CUDA_TEST_SRC = tests/cuda_tests.cu

CPU_OUTPUT = output_cpu
CUDA_OUTPUT = output_cuda
CPU_LIB = libcpucode.a

all: $(CPU_OUTPUT) $(CUDA_OUTPUT)

$(CPU_OUTPUT): $(TEST_SRC) $(CPU_LIB)
	$(CC) $(CFLAGS) -o $(CPU_OUTPUT) $(TEST_SRC) -L. -lcpucode -lm

$(CUDA_OUTPUT): $(CUDA_SRC) $(CUDA_TEST_SRC) $(CPU_LIB)
	$(NVCC) $(NVCCFLAGS) -o $(CUDA_OUTPUT) $(CUDA_SRC) $(CUDA_TEST_SRC) -L. -lcpucode -lcudart

$(CPU_LIB): $(CPU_SRC)
	$(CC) $(CFLAGS) -c $(CPU_SRC)
	ar rcs $(CPU_LIB) *.o
	rm *.o

run_cpu: $(CPU_OUTPUT)
	./$(CPU_OUTPUT)

run_gpu: $(CUDA_OUTPUT)
	./$(CUDA_OUTPUT)

run: run_cpu run_gpu

clean:
	rm -f $(CPU_OUTPUT) $(CUDA_OUTPUT) $(CPU_LIB) *.o

.PHONY: all run_cpu run_gpu run clean
