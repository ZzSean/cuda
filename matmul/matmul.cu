#include <cstdlib>
#include <iostream>
#include <math.h>

#define K 32
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void naiveMatmul(float *a, float *b, float *c, int M, int N) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  float sum = 0.0f;
  for (int i = 0; i < K; ++i) {
    sum += a[row * K + i] * b[i * N + col];
  }
  c[row * N + col] = sum;
}

__global__ void coalescedMatmul(float *a, float *b, float *c, int M, int N) {
  __shared__ float aShared[K][K];
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  float sum = 0.0f;
  aShared[threadIdx.y][threadIdx.x] = a[row * K + threadIdx.x];
  __syncwarp();
  for (int i = 0; i < K; ++i) {
    sum += aShared[threadIdx.y][i] * b[i * N + col];
  }
  c[row * N + col] = sum;
}

__global__ void sharedABMatmul(float *a, float *b, float *c, int M, int N) {
  __shared__ float aShared[K][K];
  __shared__ float bShared[K][K];
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  float sum = 0.0f;
  aShared[threadIdx.y][threadIdx.x] = a[row * K + threadIdx.x];
  bShared[threadIdx.y][threadIdx.x] = b[threadIdx.y * N + col];
  __syncthreads();
  for (int i = 0; i < K; ++i) {
    sum += aShared[threadIdx.y][i] * bShared[i][threadIdx.x];
  }
  c[row * N + col] = sum;
}

int main(int argc, char *argv[]) {
  // check command param
  if (argc < 3) {
    printf("Usage: ./matmul M N mode(optional)\n");
    return 0;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int mode = argc > 3 ? atoi(argv[3]) : 0;

  // malloc host memory
  float *a, *b, *c;
  a = (float *)malloc(M * K * sizeof(float));
  b = (float *)malloc(K * N * sizeof(float));
  c = (float *)malloc(M * N * sizeof(float));

  // initial data
  for (int i = 0; i < M * K; ++i) {
    a[i] = 1.0f;
  }
  for (int i = 0; i < K * N; ++i) {
    b[i] = 1.0f;
  }

  // malloc device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, M * K * sizeof(float));
  cudaMalloc((void **)&d_b, K * N * sizeof(float));
  cudaMalloc((void **)&d_c, M * N * sizeof(float));

  // memcpy data from host to device
  cudaMemcpy((void *)d_a, (void *)a, M * K * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_b, (void *)b, K * N * sizeof(float),
             cudaMemcpyHostToDevice);

  // setup param of launch kernel
  dim3 blockSize(K, K);
  dim3 gridSize((M + K - 1) / K, (N + K - 1) / K);

  // launch kernel
  // warm up
  naiveMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N);
  float ms;
  int IO = (M * K + K * N + M * N) * 4;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  checkCuda(cudaEventRecord(startEvent, 0));
  switch (mode) {
  case 0:
    naiveMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N);
    break;
  case 1:
    coalescedMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N);
    break;
  case 2:
    sharedABMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N);
    break;
  default:
    naiveMatmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N);
    break;
  }
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("mode: %d, time: %f ms, bw: %f\n", mode, ms, IO / ms / 1000000);

  // memcpy result from device to host
  cudaMemcpy((void *)c, (void *)d_c, M * N * sizeof(float),
             cudaMemcpyDeviceToHost);

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
