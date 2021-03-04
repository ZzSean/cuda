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

__global__ void add(float *x, float *y, float *z, int n, int offset) {
  int index = threadIdx.x + blockIdx.x * blockDim.x + offset;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n + offset; i += stride) {
    z[i] = x[i] + y[i];
  }
}

int main(int argc, char * argv[]) {
  // check command param
  if (argc < 3) {
    printf("Usage: ./add_cuda is_offset block_num grid_num\n");
    return 0;
  }

  int N = 1 << 20;
  int nBytes = (N + 128) * sizeof(float);

  // malloc host memory
  float *x, *y, *z;
  x = (float*)malloc(nBytes);
  y = (float*)malloc(nBytes);
  z = (float*)malloc(nBytes);

  // initial data
  for (int i = 0; i < N + 128; ++i) {
    x[i] = 10.0;
    y[i] = 20.0;
  }

  // malloc device memory
  float *d_x, *d_y, *d_z;
  cudaMalloc((void**)&d_x, nBytes);
  cudaMalloc((void**)&d_y, nBytes);
  cudaMalloc((void**)&d_z, nBytes);

  // memcpy data from host to device
  cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);

  // setup param of launch kernel
  int blockSize = atoi(argv[2]);
  int gridSize;
  if (argc > 3) {
    gridSize = atoi(argv[3]);
  } else {
    gridSize = (N + blockSize - 1) / blockSize;
  }
  // launch kernel
  int is_offset = atoi(argv[1]);
  int offset = 0;
  if (is_offset > 0) {
    offset = 64;
    printf("Offset, Bandwidth (GB/s):\n");
  } else {
    printf("Time (ms), Bandwidth (GB/s):\n");
  }
  // warm up
  add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N, 0);
  float ms;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  // loop for offset
  for (int i = 0; i <= offset; ++i) {
    checkCuda(cudaEventRecord(startEvent,0));
    add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N, i);
    checkCuda(cudaEventRecord(stopEvent,0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    if (is_offset > 0) {
      printf("%d, %f\n", i, 12 / ms);
    } else {
      printf("%f, %f\n", ms, 12 / ms);
    }
  }

  // memcpy result from device to host
  cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

  // compare result
  float maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(z[i] - 30.0));
  std::cout << "Max Error: " << maxError << std::endl;

  // free device memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  // free host memory
  free(x);
  free(y);
  free(z);

  return 0;
}
