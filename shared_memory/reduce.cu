#include <cstdlib>
#include <iostream>
#include <math.h>

#define THREADS 512
#define BLOCKS 4096

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void reduce(float *x, float *y, int n) {
  __shared__ int shared[THREADS];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  shared[threadIdx.x] = x[tid];

  for (int i = 1; i < blockDim.x; i *= 2) {
    __syncthreads();
    int index = 2 * i * threadIdx.x;
    if (index < blockDim.x) {
      shared[index] += shared[index + i];
    }
  }

  if (threadIdx.x == 0)
    y[blockIdx.x] = shared[0];
}

__global__ void reduce2(float *x, float *y, int n) {
  __shared__ int shared[THREADS];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  shared[threadIdx.x] = x[tid];

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    __syncthreads();
    if (threadIdx.x < i) {
      shared[threadIdx.x] += shared[threadIdx.x + i];
    }
  }

  if (threadIdx.x == 0)
    y[blockIdx.x] = shared[0];
}

int main(int argc, char *argv[]) {
  // check command param
  if (argc < 2) {
    printf("Usage: ./reduce mode\n");
    return 0;
  }
  int mode = argc > 1 ? atoi(argv[1]) : 0;

  int N = BLOCKS * THREADS;
  int nBytes1 = N * sizeof(float);
  int nBytes2 = BLOCKS * sizeof(float);

  // malloc host memory
  float *x, *y;
  x = (float *)malloc(nBytes1);
  y = (float *)malloc(nBytes2);

  // initial data
  for (int i = 0; i < N; ++i) {
    x[i] = 1.0;
  }

  // malloc device memory
  float *d_x, *d_y;
  cudaMalloc((void **)&d_x, nBytes1);
  cudaMalloc((void **)&d_y, nBytes2);

  // memcpy data from host to device
  cudaMemcpy((void *)d_x, (void *)x, nBytes1, cudaMemcpyHostToDevice);

  // setup param of launch kernel
  int blockSize = THREADS;
  int gridSize = (N + blockSize - 1) / blockSize;
  // warm up
  reduce<<<gridSize, blockSize>>>(d_x, d_y, N);
  float ms;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  checkCuda(cudaEventRecord(startEvent, 0));
  switch (mode) {
  case 0:
    reduce<<<gridSize, blockSize>>>(d_x, d_y, N);
    break;
  case 1:
    reduce2<<<gridSize, blockSize>>>(d_x, d_y, N);
    break;
  default:
    reduce<<<gridSize, blockSize>>>(d_x, d_y, N);
    break;
  }
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("mode: %d, time: %f ms, bw: %f\n", mode, ms,
         (nBytes1 + nBytes2) / ms / 1000000);

  // memcpy result from device to host
  cudaMemcpy((void *)y, (void *)d_y, nBytes2, cudaMemcpyDeviceToHost);

  // compare result
  float maxError = 0.0;
  for (int i = 0; i < BLOCKS; i++)
    maxError = fmax(maxError, fabs(y[i] - THREADS));
  std::cout << "Max Error: " << maxError << std::endl;

  // free device memory
  cudaFree(d_x);
  cudaFree(d_y);
  // free host memory
  free(x);
  free(y);

  return 0;
}
