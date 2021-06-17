#include <cstdlib>
#include <iostream>
#include <math.h>
#include <vector>

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

__device__ __forceinline__ int shared_memory_idx(int idx) {
  return (threadIdx.y + idx) * blockDim.x + threadIdx.x;
}

__global__ void ReduceAny(float *x, float *y, int reduce_num, int left_num) {
                          // int block_size, int x_strides[4], int reduce_dim[2],
                          // int reduce_strides[2], int left_dim[2],
                          // int left_strides[2]) {
  __shared__ float shared_memory[512];
  int input_idx = blockIdx.y * blockDim.x * blockDim.y +
                  threadIdx.y * blockDim.x + threadIdx.x;
  int left_idx = blockIdx.x;
  int input_stride = gridDim.y * blockDim.y * blockDim.x;
  float reduce_var = 0.0f;
  int x_strides[4] = {32768, 1024, 32, 1};
  int reduce_dim[2] = {1, 3};
  int reduce_strides[2] = {32, 1};
  int left_dim[2] = {0, 2};
  int left_strides[2] = {32, 1};
  int sub_index[4] = {0, 0, 0, 0};
  int cnt = 1;
  for (int i = 0; i < 2; ++i) {
    sub_index[left_dim[i]] = left_idx / left_strides[i];
    left_idx %= left_strides[i];
  }
  for (int i = input_idx; i < reduce_num; i += input_stride)
  {
    int reduce_idx = i;
    for (int j = 0; j < 2; ++j)
    {
      sub_index[reduce_dim[j]] = reduce_idx / reduce_strides[j];
      reduce_idx %= reduce_strides[j];
    }
    int idx_x = 0;
    for (int k = 0; k < 4; ++k)
    {
      idx_x += (sub_index[k] * x_strides[k]);
    }
    cnt++;
    reduce_var = reduce_var + x[idx_x];
  }
  shared_memory[shared_memory_idx(0)] = reduce_var;
  for (int stride = blockDim.y / 2; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (threadIdx.y < stride && threadIdx.y + stride < blockDim.y)
    {
      float other = shared_memory[shared_memory_idx(stride)];
      reduce_var = reduce_var + other;
    }
    shared_memory[shared_memory_idx(0)] = reduce_var;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.y == 0)
  {
    // printf("loop: %d, input_idx: %d, idx: %d\n", cnt, input_idx, idx_x);
    printf("%f\n", reduce_var);
  }
}

int main(int argc, char *argv[]) {
  // check command param
  if (argc < 1) {
    printf("Usage: ./reduce \n");
    return 0;
  }

  // setup param
  int input_size = 1;
  int output_size = 1;
  int reduce_size = 1;
  std::vector<int> input_dim = {16, 32, 32, 32};

  int x_strides[4] = {32768, 1024, 32, 1};
  int reduce_dim[2] = {1, 3};
  int reduce_strides[2] = {32, 1};
  int left_dim[2] = {0, 2};
  int left_strides[2] = {32, 1};

  for (int i = 0; i < input_dim.size(); ++i) {
    input_size *= input_dim[i];
  }

  for (int i = 0; i < 2; ++i) {
    reduce_size *= input_dim[reduce_dim[i]];
  }
  output_size = input_size / reduce_size;
  printf("input_size: %d, output_size: %d\n", input_size, output_size);

  // malloc host memory
  float *x, *y;
  x = (float *)malloc(input_size * sizeof(float));
  y = (float *)malloc(output_size * sizeof(float));

  // initial data
  for (int i = 0; i < input_size; ++i) {
    x[i] = static_cast<float>(1 / 100.0);
  }

  // malloc device memory
  float *d_x, *d_y;
  cudaMalloc((void **)&d_x, input_size * sizeof(float));
  cudaMalloc((void **)&d_y, output_size * sizeof(float));

  // memcpy data from host to device
  cudaMemcpy((void *)d_x, (void *)x, input_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // setup param of launch kernel
  dim3 blockSize(128, 4, 1);
  dim3 gridSize(512, 1, 1);

  // launch kernel
  // warm up
  ReduceAny<<<gridSize, blockSize>>>(d_x, d_y, reduce_size, output_size);
  //  reduce_size, x_strides, reduce_dim,
  //  reduce_strides, left_dim, left_strides);
  float ms;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  checkCuda(cudaEventRecord(startEvent, 0));
  // ReduceAny<<<gridSize, blockSize>>>(x, y, reduce_size, output_size, 33,
  //                                    x_strides, reduce_dim, reduce_strides,
  //                                    left_dim, left_strides);
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("time: %f ms\n", ms);

  // memcpy result from device to host
  cudaMemcpy((void *)y, (void *)d_y, output_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  // compare result
  float maxError = 0.0;
  for (int i = 0; i < output_size; i++) {
    maxError = fmax(maxError, fabs(y[i] - 10240));
    if (maxError > 0.00001) {
      std::cout << "Max Error[" << i << "]: " << maxError << std::endl;
      break;
    }
  }

  // free device memory
  cudaFree(d_x);
  cudaFree(d_y);
  // free host memory
  free(x);
  free(y);

  return 0;
}
