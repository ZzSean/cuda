#include <cstdlib>
#include <iostream>
#include <math.h>

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void loop_unroll1(float *x, float *y, float *z, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    z[i] = x[i] + y[i];
  }
}

__global__ void loop_unroll2(float *x, float *y, float *z, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
// #pragma unroll 2
  for (int i = index; i < n; i += 2 * stride) {
    z[i] = x[i] + y[i];
    z[i + stride] = x[i + stride] + y[i + stride];
  }
}

__global__ void loop_unroll4(float *x, float *y, float *z, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
// #pragma unroll 4
  for (int i = index; i < n; i += 4 * stride) {
    z[i] = x[i] + y[i];
    z[i + 1 * stride] = x[i + 1 * stride] + y[i + 1 * stride];
    z[i + 2 * stride] = x[i + 2 * stride] + y[i + 2 * stride];
    z[i + 3 * stride] = x[i + 3 * stride] + y[i + 3 * stride];
  }
}

int main(int argc, char *argv[]) {
  // check command param
  if (argc < 2) {
    printf("Usage: ./loop_unroll block_num grid_num(optional)\n");
    return 0;
  }

  int N = 1 << 25;
  int nBytes = N * sizeof(float);

  // malloc host memory
  float *x, *y, *z;
  x = (float *)malloc(nBytes);
  y = (float *)malloc(nBytes);
  z = (float *)malloc(nBytes);

  // initial data
  for (int i = 0; i < N; ++i) {
    x[i] = 10.0;
    y[i] = 20.0;
  }

  // malloc device memory
  float *d_x, *d_y, *d_z;
  cudaMalloc((void **)&d_x, nBytes);
  cudaMalloc((void **)&d_y, nBytes);
  cudaMalloc((void **)&d_z, nBytes);

  // memcpy data from host to device
  cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_y, (void *)y, nBytes, cudaMemcpyHostToDevice);

  // setup param of launch kernel
  int blockSize = atoi(argv[1]);
  int gridSize;
  if (argc > 2) {
    gridSize = atoi(argv[2]);
  } else {
    gridSize = (N + blockSize - 1) / blockSize;
  }
  // launch kernel
  printf("Time (ms), Bandwidth (GB/s):\n");
  // warm up
  loop_unroll1<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
  float ms;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  // loop for unroll_num
  for (int i = 0; i < 3; ++i) {
    int unroll_num = pow(2, i);
    printf("loop unroll %d\n", unroll_num);
    checkCuda(cudaEventRecord(startEvent, 0));
    switch (unroll_num) {
    case 1:
      for (int j = 0; j < 1000; ++j) {
        loop_unroll1<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
      }
      break;
    case 2:
      for (int j = 0; j < 1000; ++j) {
        loop_unroll2<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
      }
      break;
    case 4:
      for (int j = 0; j < 1000; ++j) {
        loop_unroll4<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
      }
      break;
    default:
      break;
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("time: %f\n", ms / 1000);
  }

  // memcpy result from device to host
  cudaMemcpy((void *)z, (void *)d_z, nBytes, cudaMemcpyDeviceToHost);

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
