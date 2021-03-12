#include <math.h>
#include <iostream>
#include <cstdlib>

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void naiveMatmul(float *a, float *b, float *c, int M, int K, int N) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  float sum = 0.0f;
  for (int i = 0; i < K; ++i) {
    sum += a[row * K + i] *b[i * N + col];
  }
  c[row * N + col] = sum;
}

int main(int argc, char * argv[]) {
  // check command param
  if (argc < 4) {
    printf("Usage: ./matmul M K N\n");
    return 0;
  }

  int M = atoi(argv[1]);
  int K = atoi(argv[2]);
  int N = atoi(argv[3]);

  // malloc host memory
  float *a, *b, *c;
  a = (float*)malloc(M * K * sizeof(float));
  b = (float*)malloc(K * N * sizeof(float));
  c = (float*)malloc(M * N * sizeof(float));

  // initial data
  for (int i = 0; i < M * K; ++i) {
    a[i] = 1.0f;
  }
  for (int i = 0; i < K * N; ++i) {
    b[i] = 1.0f;
  }

  // malloc device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, M * K * sizeof(float));
  cudaMalloc((void**)&d_b, K * N * sizeof(float));
  cudaMalloc((void**)&d_c, M * N * sizeof(float));

  // memcpy data from host to device
  cudaMemcpy((void*)d_a, (void*)a, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_b, (void*)b, K * N * sizeof(float), cudaMemcpyHostToDevice);

  // setup param of launch kernel
  dim3 blockSize(32, 32);
  dim3 gridSize((M + 31) / 32, (N + 31) / 32);

  // launch kernel
  // warm up
  naiveMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, K, N);
  float ms;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  checkCuda(cudaEventRecord(startEvent,0));
  naiveMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, K, N);
  checkCuda(cudaEventRecord(stopEvent,0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));

  // memcpy result from device to host
  cudaMemcpy((void*)c, (void*)d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  // compare result
  float maxError = 0.0;
  for (int i = 0; i < M * N; i++)
    maxError = fmax(maxError, fabs(c[i] - K));
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
