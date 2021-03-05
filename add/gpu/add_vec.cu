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

__global__ void add_scale(float *x, float *y, float *z, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    z[i] = x[i] + y[i];
  }
}

__global__ void add_vector2(float *x, float *y, float *z, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  float2 x_tmp, y_tmp, z_tmp;
  for (int i = index; i < n / 2; i += stride) {
    x_tmp = reinterpret_cast<float2 *>(x)[i];
    y_tmp = reinterpret_cast<float2 *>(y)[i];
    z_tmp.x = x_tmp.x + y_tmp.x;
    z_tmp.y = x_tmp.y + y_tmp.y;
    reinterpret_cast<float2 *>(z)[i] = z_tmp;
  }
}

__global__ void add_vector4(float *x, float *y, float *z, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  float4 x_tmp, y_tmp, z_tmp;
  for (int i = index; i < n / 4; i += stride) {
    x_tmp = reinterpret_cast<float4 *>(x)[i];
    y_tmp = reinterpret_cast<float4 *>(y)[i];
    z_tmp.x = x_tmp.x + y_tmp.x;
    z_tmp.y = x_tmp.y + y_tmp.y;
    z_tmp.z = x_tmp.z + y_tmp.z;
    z_tmp.w = x_tmp.w + y_tmp.w;
    reinterpret_cast<float4 *>(z)[i] = z_tmp;
  }
}

int main(int argc, char * argv[]) {
  // check command param
  if (argc < 2) {
    printf("Usage: ./add_cuda block_num grid_num(optional)\n");
    return 0;
  }

  int N = 1 << 20;
  int nBytes = N * sizeof(float);

  // malloc host memory
  float *x, *y, *z;
  x = (float*)malloc(nBytes);
  y = (float*)malloc(nBytes);
  z = (float*)malloc(nBytes);

  // initial data
  for (int i = 0; i < N; ++i) {
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
  int vector_num = 1;
  int blockSize = atoi(argv[1]);
  int gridSize;
  if (argc > 2) {
    gridSize = atoi(argv[2]);
  } else {
    gridSize = (N / vector_num + blockSize - 1) / blockSize;
  }
  // launch kernel
  // warm up
  add_scale<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
  float ms;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  // loop for vector_num
  for (int i = 0; i < 3; ++i) {
    vector_num = pow(2, i);
    printf("vector_%d\n", vector_num);
    checkCuda(cudaEventRecord(startEvent,0));
    switch (vector_num) {
      case 1:
        add_scale<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
        break;
      case 2:
        add_vector2<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
        break;
      case 4:
        add_vector4<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
        break;
      default:
        add_scale<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
        break;
    }
    checkCuda(cudaEventRecord(stopEvent,0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("%f, %f\n", ms, 12 / ms);
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
