#include "cuda_fp16.h"
#include <cstdlib>
#include <iostream>
#include <math.h>

// inline cudaError_t checkCuda(cudaError_t result) {
// #if defined(DEBUG) || defined(_DEBUG)
//   if (result != cudaSuccess) {
//     fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//     assert(result == cudaSuccess);
//   }
// #endif
//   return result;
// }

// template <typename T, int Size>
// struct alignas(sizeof(T) * Size) CudaAlignedVector {
//   T val[Size];
// };

// __global__ void add_scale(float *x, float *y, float *z, int n) {
//   int index = threadIdx.x + blockIdx.x * blockDim.x;
//   int stride = blockDim.x * gridDim.x;
//   for (int i = index; i < n; i += stride) {
//     z[i] = x[i] + y[i];
//   }
// }

// __global__ void add_vector2(float *x, float *y, float *z, int n) {
//   int index = threadIdx.x + blockIdx.x * blockDim.x;
//   int stride = blockDim.x * gridDim.x;
//   float2 x_tmp, y_tmp, z_tmp;
//   for (int i = index; i < n / 2; i += stride) {
//     x_tmp = reinterpret_cast<float2 *>(x)[i];
//     y_tmp = reinterpret_cast<float2 *>(y)[i];
//     z_tmp.x = x_tmp.x + y_tmp.x;
//     z_tmp.y = x_tmp.y + y_tmp.y;
//     reinterpret_cast<float2 *>(z)[i] = z_tmp;
//   }
// }

// __global__ void add_vector4(float *x, float *y, float *z, int n) {
//   int index = threadIdx.x + blockIdx.x * blockDim.x;
//   int stride = blockDim.x * gridDim.x;
//   float4 x_tmp, y_tmp, z_tmp;
//   for (int i = index; i < n / 4; i += stride) {
//     x_tmp = reinterpret_cast<float4 *>(x)[i];
//     y_tmp = reinterpret_cast<float4 *>(y)[i];
//     z_tmp.x = x_tmp.x + y_tmp.x;
//     z_tmp.y = x_tmp.y + y_tmp.y;
//     z_tmp.z = x_tmp.z + y_tmp.z;
//     z_tmp.w = x_tmp.w + y_tmp.w;
//     reinterpret_cast<float4 *>(z)[i] = z_tmp;
//   }
// }

// template <typename T, int vec_size>
// __global__ void add(T *x, T *y, T *z, int n) {
//   using VecT = CudaAlignedVector<T, vec_size>;
//   VecT *x_vec = reinterpret_cast<VecT *>(x);
//   VecT *y_vec = reinterpret_cast<VecT *>(y);
//   VecT *z_vec = reinterpret_cast<VecT *>(z);
//   VecT x_var, y_var, z_var;
//   T *x_ptr = reinterpret_cast<T *>(&x_var);
//   T *y_ptr = reinterpret_cast<T *>(&y_var);
//   T *z_ptr = reinterpret_cast<T *>(&z_var);

//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = gridDim.x * blockDim.x;
//   int loop = n / vec_size;
//   int remainder = n % vec_size;

//   for (int i = tid; i < loop; i += stride) {
//     x_var = x_vec[i];
//     y_var = y_vec[i];
// #pragma unroll
//     for (int j = 0; j < vec_size; ++j) {
//       z_ptr[j] = x_ptr[j] + y_ptr[j];
//     }
//     z_vec[i] = z_var;
//   }

//   if (tid == loop && remainder != 0) {
//     while (remainder) {
//       int idx = n - remainder;
//       remainder--;
//       z[idx] = x[idx] + y[idx];
//     }
//   }
// }

// template <typename T> void add_op() {
//   int N = 1 << 24;
//   int nBytes = N * sizeof(T);

//   // malloc host memory
//   T *x = (T *)malloc(nBytes);
//   T *y = (T *)malloc(nBytes);
//   T *z = (T *)malloc(nBytes);

//   // initial data
//   for (int i = 0; i < N; ++i) {
//     x[i] = static_cast<T>(10.0f);
//     y[i] = static_cast<T>(20.0f);
//   }

//   // malloc device memory
//   T *d_x, *d_y, *d_z;
//   cudaMalloc((void **)&d_x, nBytes);
//   cudaMalloc((void **)&d_y, nBytes);
//   cudaMalloc((void **)&d_z, nBytes);

//   // memcpy data from host to device
//   cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);
//   cudaMemcpy((void *)d_y, (void *)y, nBytes, cudaMemcpyHostToDevice);

//   // setup param of launch kernel
//   int vector_num = 1;
//   int blockSize = 512;
//   int gridSize = (N / vector_num + blockSize - 1) / blockSize;
//   // launch kernel
//   // warm up
//   add<T, 1><<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
//   float ms;
//   cudaEvent_t startEvent, stopEvent;
//   checkCuda(cudaEventCreate(&startEvent));
//   checkCuda(cudaEventCreate(&stopEvent));
//   // loop for vector_num
//   for (int i = 0; i < 4; ++i) {
//     vector_num = pow(2, i);
//     printf("vector_%d\n", vector_num);
//     checkCuda(cudaEventRecord(startEvent, 0));
//     switch (vector_num) {
//     case 1:
//       for (int j = 0; j < 1000; ++j) {
//         add<T, 1><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
//       }
//       break;
//     case 2:
//       for (int j = 0; j < 1000; ++j) {
//         add<T, 2><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
//       }
//       break;
//     case 4:
//       for (int j = 0; j < 1000; ++j) {
//         add<T, 4><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
//       }
//       break;
//     case 8:
//       for (int j = 0; j < 1000; ++j) {
//         add<T, 8><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
//       }
//       break;
//     default:
//       break;
//     }
//     checkCuda(cudaEventRecord(stopEvent, 0));
//     checkCuda(cudaEventSynchronize(stopEvent));
//     checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
//     printf("time: %f, bandwidth: %f\n", ms / 1000,
//            static_cast<float>(3 * 16 * sizeof(T)) / ms * 1000);
//   }

//   // memcpy result from device to host
//   cudaMemcpy((void *)z, (void *)d_z, nBytes, cudaMemcpyDeviceToHost);

//   // compare result
//   float maxError = 0.0;
//   for (int i = 0; i < N; i++)
//     maxError =
//         fmax(maxError, fabs(static_cast<float>(z[i] - static_cast<T>(30.0f))));
//   std::cout << "Max Error: " << maxError << std::endl;

//   // free device memory
//   cudaFree(d_x);
//   cudaFree(d_y);
//   cudaFree(d_z);
//   // free host memory
//   free(x);
//   free(y);
//   free(z);
// }

int main(int argc, char *argv[]) {
  // // check command param
  // if (argc < 2) {
  //   printf("Usage: ./add_vec data_type(0: half, 1: float, 2: double)\n");
  //   return 0;
  // }
  // int data_type = atoi(argv[1]);
  // switch (data_type) {
  // case 0:
  //   add_op<half>();
  //   break;
  // case 1:
  //   add_op<float>();
  //   break;
  // case 2:
  //   add_op<char>();
  //   break;
  // default:
  //   add_op<float>();
  //   break;
  // }
  return 0;
}