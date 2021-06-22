#include <cstdlib>
#include <iostream>
#include <math.h>

#define N 800
#define BLOCK 256
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void naiveMatmul(float *a, float *b, float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int row = tid / N;
  int col = tid % N;
  float sum = 0.0f;
  if (row < N && col < N) {
    for (int i = 0; i < N; ++i) {
      sum += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
  }
}

int main(int argc, char *argv[]) {
  printf("Usage: ./matmul\n");

  // malloc host memory
  float *a, *b, *c;
  a = (float *)malloc(N * N * sizeof(float));
  b = (float *)malloc(N * N * sizeof(float));
  c = (float *)malloc(N * N * sizeof(float));

  // initial data
  for (int i = 0; i < N * N; ++i) {
    a[i] = 1.0f;
    b[i] = 1.0f;
  }

  // malloc device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, N * N * sizeof(float));
  cudaMalloc((void **)&d_b, N * N * sizeof(float));
  cudaMalloc((void **)&d_c, N * N * sizeof(float));

  // memcpy data from host to device
  cudaMemcpy((void *)d_a, (void *)a, N * N * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_b, (void *)b, N * N * sizeof(float),
             cudaMemcpyHostToDevice);

  // setup param of launch kernel
  dim3 blockSize(BLOCK);
  dim3 gridSize(N * (N + BLOCK - 1) / BLOCK);

  // launch kernel
  // warm up
  naiveMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c);
  float ms;
  int repeat = 100;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < repeat; ++i) {
    naiveMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c);
  }
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("time: %f ms\n", ms / repeat);

  // memcpy result from device to host
  cudaMemcpy((void *)c, (void *)d_c, N * N * sizeof(float),
             cudaMemcpyDeviceToHost);

  // compare result
  float maxError = 0.0;
  for (int i = 0; i < N * N; i++)
    maxError = fmax(maxError, fabs(c[i] - N));
  std::cout << "Max Error: " << maxError << std::endl;

  // free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  // free host memory
  free(a);
  free(b);
  free(c);

  return 0;
}
