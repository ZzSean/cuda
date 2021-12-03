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

template <typename T, int Size>
struct alignas(sizeof(T) * Size) CudaAlignedVector {
  T val[Size];
};

template <typename T, int vec_size>
__global__ void relu(T *x, T *y, bool *z, int n) {
  using VecT = CudaAlignedVector<T, vec_size>;
  using VecBool = CudaAlignedVector<bool, vec_size>;
  VecT *x_vec = reinterpret_cast<VecT *>(x);
  VecT *y_vec = reinterpret_cast<VecT *>(y);
  // VecBool *z_vec = reinterpret_cast<VecBool *>(z);
  VecT x_var, y_var;
  // VecBool z_var;
  T *x_ptr = reinterpret_cast<T *>(&x_var);
  T *y_ptr = reinterpret_cast<T *>(&y_var);
  // bool *z_ptr = reinterpret_cast<bool *>(&z_var);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int loop = n / vec_size;
  int remainder = n % vec_size;

  for (int i = tid; i < loop; i += stride) {
    x_var = x_vec[i];
#pragma unroll
    for (int j = 0; j < vec_size; ++j) {
      y_ptr[j] = x_ptr[j] > 0 ? x_ptr[j] : 0;
    }
    y_vec[i] = y_var;
  }

  if (tid == 0 && remainder != 0) {
    while (remainder) {
      int idx = n - remainder;
      remainder--;
      T in = x[idx];
      y[idx] = in > 0 ? in : 0;
    }
  }
}

template <typename T, int vec_size>
__global__ void relu_grad(T *dy, T *y, bool *z, T *dx, int n) {
  using VecT = CudaAlignedVector<T, vec_size>;
  using VecBool = CudaAlignedVector<bool, vec_size>;
  VecT *dy_vec = reinterpret_cast<VecT *>(dy);
  VecT *y_vec = reinterpret_cast<VecT *>(y);
  VecT *dx_vec = reinterpret_cast<VecT *>(dx);
  VecT dy_var, y_var, dx_var;
  T *dy_ptr = reinterpret_cast<T *>(&dy_var);
  T *y_ptr = reinterpret_cast<T *>(&y_var);
  T *dx_ptr = reinterpret_cast<T *>(&dx_var);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int loop = n / vec_size;
  int remainder = n % vec_size;

  for (int i = tid; i < loop; i += stride) {
    dy_var = dy_vec[i];
    y_var = y_vec[i];
#pragma unroll
    for (int j = 0; j < vec_size; ++j) {
      dx_ptr[j] = y_ptr[j] > 0 ? dy_ptr[j] : 0;
    }
    dx_vec[i] = dx_var;
  }

  if (tid == 0 && remainder != 0) {
    while (remainder) {
      int idx = n - remainder;
      remainder--;
      T in1 = dy[idx];
      T in2 = y[idx];
      dx[idx] = in2 > 0 ? in1 : 0;
    }
  }
}

template <typename T, int vec_size>
__global__ void relu_mask(T *x, T *y, bool *z, int n) {
  using VecT = CudaAlignedVector<T, vec_size>;
  using VecBool = CudaAlignedVector<bool, vec_size>;
  VecT *x_vec = reinterpret_cast<VecT *>(x);
  VecT *y_vec = reinterpret_cast<VecT *>(y);
  VecBool *z_vec = reinterpret_cast<VecBool *>(z);
  VecT x_var, y_var;
  VecBool z_var;
  T *x_ptr = reinterpret_cast<T *>(&x_var);
  T *y_ptr = reinterpret_cast<T *>(&y_var);
  bool *z_ptr = reinterpret_cast<bool *>(&z_var);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int loop = n / vec_size;
  int remainder = n % vec_size;

  for (int i = tid; i < loop; i += stride) {
    x_var = x_vec[i];
#pragma unroll
    for (int j = 0; j < vec_size; ++j) {
      y_ptr[j] = x_ptr[j] > 0 ? x_ptr[j] : 0;
      z_ptr[j] = x_ptr[j] > 0;
    }
    y_vec[i] = y_var;
    z_vec[i] = z_var;
  }

  if (tid == 0 && remainder != 0) {
    while (remainder) {
      int idx = n - remainder;
      remainder--;
      T in = x[idx];
      y[idx] = in > 0 ? in : 0;
      z[idx] = in > 0;
    }
  }
}

template <typename T, int vec_size>
__global__ void relu_mask_grad(T *dy, T *y, bool *z, T *dx, int n) {
  using VecT = CudaAlignedVector<T, vec_size>;
  using VecBool = CudaAlignedVector<bool, vec_size>;
  VecT *dy_vec = reinterpret_cast<VecT *>(dy);
  VecT *dx_vec = reinterpret_cast<VecT *>(dx);
  VecBool *z_vec = reinterpret_cast<VecBool *>(z);
  VecT dy_var, dx_var;
  VecBool z_var;
  T *dy_ptr = reinterpret_cast<T *>(&dy_var);
  T *dx_ptr = reinterpret_cast<T *>(&dx_var);
  bool *z_ptr = reinterpret_cast<bool *>(&z_var);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int loop = n / vec_size;
  int remainder = n % vec_size;

  for (int i = tid; i < loop; i += stride) {
    dy_var = dy_vec[i];
    z_var = z_vec[i];
#pragma unroll
    for (int j = 0; j < vec_size; ++j) {
      dx_ptr[j] = z_ptr[j] ? dy_ptr[j] : 0;
    }
    dx_vec[i] = dx_var;
  }

  if (tid == 0 && remainder != 0) {
    while (remainder) {
      int idx = n - remainder;
      remainder--;
      T in1 = dy[idx];
      T in2 = z[idx];
      dx[idx] = in2 ? in1 : 0;
    }
  }
}

template <typename T> void relu_op() {
  int N = 128 * 64 * 112 * 112;
  int nBytes = N * sizeof(T);

  // malloc host memory
  T *x = (T *)malloc(nBytes);
  T *y = (T *)malloc(nBytes);
  T *dx = (T *)malloc(nBytes);
  T *dy = (T *)malloc(nBytes);
  bool *z = (bool *)malloc(N * sizeof(bool));

  // initial data
  for (int i = 0; i < N; ++i) {
    x[i] = static_cast<T>(1.0f);
    dy[i] = static_cast<T>(1.0f);
  }

  // malloc device memory
  T *d_x, *d_y, *d_dx, *d_dy;
  bool *d_z;
  cudaMalloc((void **)&d_x, nBytes);
  cudaMalloc((void **)&d_y, nBytes);
  cudaMalloc((void **)&d_dx, nBytes);
  cudaMalloc((void **)&d_dy, nBytes);
  cudaMalloc((void **)&d_z, N * sizeof(bool));

  // memcpy data from host to device
  cudaMemcpy((void *)d_x, (void *)x, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_dy, (void *)dy, nBytes, cudaMemcpyHostToDevice);

  // setup param of launch kernel
  int vector_num = 1;
  int blockSize = 1024;
  // int gridSize = (N / vector_num + blockSize - 1) / blockSize;
  int gridSize = 256;
  // launch kernel
  // warm up
  relu<T, 1><<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
  float ms;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  // loop for vector_num
  for (int i = 0; i < 3; ++i) {
    vector_num = pow(2, i);
    printf("vector_%d\n", vector_num);
    checkCuda(cudaEventRecord(startEvent, 0));
    switch (vector_num) {
    case 1:
      for (int j = 0; j < 1000; ++j) {
        relu<T, 1><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
        relu_grad<T, 1><<<gridSize / vector_num, blockSize>>>(d_dy, d_y, d_z, d_dx, N);
        // relu_mask<T, 1><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
        // relu_mask_grad<T, 1><<<gridSize / vector_num, blockSize>>>(d_dy, d_y, d_z, d_dx, N);
      }
      break;
    case 2:
      for (int j = 0; j < 1000; ++j) {
        relu<T, 2><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
        relu_grad<T, 2><<<gridSize / vector_num, blockSize>>>(d_dy, d_y, d_z, d_dx, N);
        // relu_mask<T, 2><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
        // relu_mask_grad<T, 2><<<gridSize / vector_num, blockSize>>>(d_dy, d_y, d_z, d_dx, N);
      }
      break;
    case 4:
      for (int j = 0; j < 1000; ++j) {
        relu<T, 4><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
        relu_grad<T, 4><<<gridSize / vector_num, blockSize>>>(d_dy, d_y, d_z, d_dx, N);
        // relu_mask<T, 4><<<gridSize / vector_num, blockSize>>>(d_x, d_y, d_z, N);
        // relu_mask_grad<T, 4><<<gridSize / vector_num, blockSize>>>(d_dy, d_y, d_z, d_dx, N);
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
  cudaMemcpy((void *)y, (void *)d_y, nBytes, cudaMemcpyDeviceToHost);

  // compare result
  float maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError =
        fmax(maxError, fabs(static_cast<float>(y[i] - static_cast<T>(1.0f))));
  std::cout << "Max Error: " << maxError << std::endl;

  // free device memory
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  // free host memory
  free(x);
  free(y);
  free(z);
}

int main(int argc, char *argv[]) {
  // check command param
  if (argc < 2) {
    printf("Usage: ./relu data_type(0: half, 1: float, 2: char)\n");
    return 0;
  }
  int data_type = atoi(argv[1]);
  switch (data_type) {
  case 0:
    relu_op<float>();
    break;
  case 1:
    relu_op<float>();
    break;
  case 2:
    relu_op<char>();
    break;
  default:
    relu_op<float>();
    break;
  }
  return 0;
}