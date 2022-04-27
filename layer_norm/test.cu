#include "cuda_fp16.h"
#include <cstdlib>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>

#define HOSTDEVICE __host__ __device__

template <int VecSize>
__forceinline__ __device__ void RandVec(curandStatePhilox4_32_10_t *state,
                                        float *data);

template <>
__forceinline__ __device__ void RandVec<1>(curandStatePhilox4_32_10_t *state,
                                           float *data) {
  data[0] = curand_uniform(state);
}

template <>
__forceinline__ __device__ void RandVec<2>(curandStatePhilox4_32_10_t *state,
                                           float *data) {
  data[0] = curand_uniform(state);
  data[1] = curand_uniform(state);
}

template <>
__forceinline__ __device__ void RandVec<4>(curandStatePhilox4_32_10_t *state,
                                           float *data) {
  float4 rand4 = curand_uniform4(state);
  data[0] = rand4.x;
  data[1] = rand4.y;
  data[2] = rand4.w;
  data[3] = rand4.z;
}

template <>
__forceinline__ __device__ void RandVec<8>(curandStatePhilox4_32_10_t *state,
                                           float *data) {
  RandVec<4>(state, data);
  RandVec<4>(state, data + 4);
}

template <typename T, int Size> struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];

  HOSTDEVICE inline const T &operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T &operator[](int i) { return val[i]; }
};

template <typename T, int Size>
HOSTDEVICE inline void Load(const T *addr, AlignedVector<T, Size> *vec) {
  const AlignedVector<T, Size> *addr_vec =
      reinterpret_cast<const AlignedVector<T, Size> *>(addr);
  *vec = *addr_vec;
}

template <typename T, int Size>
HOSTDEVICE inline void Store(const AlignedVector<T, Size> &vec, T *addr) {
  AlignedVector<T, Size> *addr_vec =
      reinterpret_cast<AlignedVector<T, Size> *>(addr);
  *addr_vec = vec;
}

template <typename T>
inline __device__ T GetFactor(const float dropout_prob,
                              const bool is_upscale_in_train,
                              const bool is_test) {
  T factor = is_upscale_in_train ? static_cast<T>(1.0f / (1.0f - dropout_prob))
                                 : static_cast<T>(1.0f);
  if (is_test) {
    factor = is_upscale_in_train ? static_cast<T>(1.0f)
                                 : static_cast<T>(1.0f - dropout_prob);
  }
  return factor;
}

/*
 * @brief layernorm(residual + dropout(x));
 * Conditions:
 * (1) The number of cols is 768/1024/4096;
 * (2) layer_norm scale and bias is not null;
 * (3) linear bias is null;
 * @param
 * rows: batch_size * seq_len
 * cols: 1024
 * x_: [rows, cols], inputs
 * residual_:[rows, cols]
 * bias_: [cols], linear bias, can be null
 * gamma_: [cols]: layernorm scale, not null
 * beta_: [cols], layernorm bias, not null
 * mask_out_: [rows, cols], dropout result
 * residual_out_: [rows, cols], residual + dropout(src)
 * y_: [rows, cols], layernorm result
 * mean_out_: [rows]: layernorm means
 * var_out_: [rows]: layernorm vars
 */
template <
    typename T, typename U, typename ScaleT = U, typename MaskType = uint8_t,
    int VecSize = 8, int WARPS_M = 4, int WARPS_N = 1, int BYTES_PER_LDG = 16,
    int ELTS_PER_ROW = 1024, int THREADS_PER_WARP = 32,
    int THREADS_PER_ROW = WARPS_N *THREADS_PER_WARP,
    int THREADS_PER_CTA = WARPS_M *THREADS_PER_ROW, int ROWS_PER_CTA = WARPS_M,
    int ELTS_PER_ROW_PER_CTA = THREADS_PER_ROW *VecSize,
    int LDGS = ELTS_PER_ROW / ELTS_PER_ROW_PER_CTA>
__global__ __launch_bounds__(THREADS_PER_CTA) void fused_fast_ln_fwd_kernel(
    int rows, int cols, uint64_t seed, const float dropout_prob,
    const bool is_upscale_in_train, const bool is_test,
    const uint64_t increment, const float epsilon, const T *__restrict__ x_ptr,
    const T *__restrict__ residual_ptr, const T *__restrict__ bias_ptr,
    const ScaleT *__restrict__ gamma_ptr, const ScaleT *__restrict__ beta_ptr,
    MaskType *__restrict__ mask_out_ptr, U *__restrict__ mean_out_ptr,
    U *__restrict__ var_out_ptr, T *__restrict__ residual_out_ptr,
    T *__restrict__ y_ptr) {
  __shared__ U smem[WARPS_M * WARPS_N];
  using Vec = AlignedVector<T, VecSize>;
  using Vec_scale = AlignedVector<ScaleT, VecSize>;
  using MaskStoreT = AlignedVector<MaskType, VecSize>;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP; // 0, 1, ..., 31
  const int warp = tidx / THREADS_PER_WARP; // 0, 1, 2, 3
  const int warp_n = warp % WARPS_N;        // 0
  const int warp_m = warp / WARPS_N;        // 0, 1, 2, 3

  const int c = warp_n * THREADS_PER_WARP + lane; // lane
  const int r = bidx * ROWS_PER_CTA + warp_m;     // row id

  int idx = r * ELTS_PER_ROW + c;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);

  T factor = GetFactor<T>(dropout_prob, is_upscale_in_train, is_test);

  // bias
  Vec bias[LDGS];
  if (bias_ptr != nullptr) {
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      Load<T, VecSize>(bias_ptr + col * VecSize, &bias[it]);
      col += THREADS_PER_ROW;
    }
  }

  Vec_scale gamma[LDGS];
  Vec_scale beta[LDGS];
#pragma unroll
  for (int it = 0, col = c; it < LDGS; it++) {
    Load<ScaleT, VecSize>(gamma_ptr + col * VecSize, &gamma[it]);
    Load<ScaleT, VecSize>(beta_ptr + col * VecSize, &beta[it]);
    col += THREADS_PER_ROW;
  }

  constexpr U rn = 1.f / U(ELTS_PER_ROW);
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    Vec x[LDGS];
    Vec residual[LDGS];
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      Load<T, VecSize>(x_ptr + row * ELTS_PER_ROW + col * VecSize, &x[it]);
      Load<T, VecSize>(residual_ptr + row * ELTS_PER_ROW + col * VecSize,
                       &residual[it]);
      col += THREADS_PER_ROW;
    }

    MaskStoreT mask_vec[LDGS];
    if (!is_test) {
#pragma unroll
      for (int it = 0; it < LDGS; it++) {
        float rand[VecSize];
        RandVec<VecSize>(&state, rand);
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          mask_vec[it][jt] = static_cast<MaskType>(rand[jt] >= dropout_prob);
        }
      }
    } else {
#pragma unroll
      for (int it = 0; it < LDGS; it++) {
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          mask_vec[it][jt] = static_cast<MaskType>(1);
        }
      }
    }

    // 4 * 8
    U xf[LDGS * VecSize];
    if (bias_ptr != nullptr) {
#pragma unroll
      for (int it = 0; it < LDGS; it++) {
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          // dropout(x) + residual
          x[it][jt] = (x[it][jt] + bias[it][jt]) *
                          static_cast<T>(mask_vec[it][jt]) * factor +
                      residual[it][jt];
          xf[it * VecSize + jt] = U(x[it][jt]);
        }
      }
    } else {
#pragma unroll
      for (int it = 0; it < LDGS; it++) {
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          // dropout(x) + residual
          x[it][jt] = x[it][jt] * static_cast<T>(mask_vec[it][jt]) * factor +
                      residual[it][jt];
          xf[it * VecSize + jt] = U(x[it][jt]);
        }
      }
    }

// store dropout_residual_out and mask_out
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      Store<T, VecSize>(x[it],
                        residual_out_ptr + row * ELTS_PER_ROW + col * VecSize);
      Store<MaskType, VecSize>(mask_vec[it], mask_out_ptr + row * ELTS_PER_ROW +
                                                 col * VecSize);
      col += THREADS_PER_ROW;
    }

    U mu_local = 0.f;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        mu_local += xf[it * VecSize + jt];
      }
    }

#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
      mu_local += __shfl_xor_sync(uint32_t(-1), mu_local, it);
    }
    if (WARPS_N > 1) {
      if (lane == 0) {
        smem[warp_m * WARPS_N + warp_n] = mu_local;
      }
      __syncthreads();
      if (tidx == 0) {
        mu_local = 0.f;
#pragma unroll
        for (int it = 0; it < WARPS_N; ++it) {
          mu_local += smem[warp_m * WARPS_N + it];
        }
        smem[warp_m] = mu_local;
      }
      __syncthreads();
      mu_local = smem[warp_m];
    }
    mu_local *= rn;
    if (lane == 0) {
      mean_out_ptr[row] = mu_local;
    }
    U var_local = 0.f;

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        U diff = xf[it * VecSize + jt] - mu_local;
        var_local += diff * diff;
      }
    }

#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
      var_local += __shfl_xor_sync(uint32_t(-1), var_local, it);
    }
    if (WARPS_N > 1) {
      if (lane == 0) {
        smem[warp_m * WARPS_N + warp_n] = var_local;
      }
      __syncthreads();
      if (tidx == 0) {
        var_local = 0.f;
#pragma unroll
        for (int it = 0; it < WARPS_N; ++it) {
          var_local += smem[warp_m * WARPS_N + it];
        }
        smem[warp_m] = var_local;
      }
      __syncthreads();
      var_local = smem[warp_m];
    }
    U rsigma = rsqrtf(var_local * rn + epsilon);
    if (lane == 0) {
      // Note: the stored var is different for paddle(ln) and apex (fast ln).
      // var_out_ptr[row] = rsigma;
      var_out_ptr[row] = var_local * rn;
    }

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        // use fp16 to compute
        // ScaleT tmp = static_cast<ScaleT>(rsigma * (xf[it * VecSize + jt] -
        // mu_local));
        // x[it][jt] = gamma[it][jt] *  tmp + beta[it][jt];
        // cast to fp32 to compute
        U tmp = rsigma * (static_cast<U>(xf[it * VecSize + jt]) - mu_local);
        x[it][jt] = static_cast<T>(static_cast<U>(gamma[it][jt]) * tmp +
                                   static_cast<U>(beta[it][jt]));
      }
    }

#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      Store<T, VecSize>(x[it], y_ptr + row * ELTS_PER_ROW + col * VecSize);
      col += THREADS_PER_ROW;
    }
  }
}

int main(int argc, char *argv[]) {
  // setup param
  constexpr int rows = 1024;
  constexpr int cols = 1024;
  size_t data_bytes = rows * cols * sizeof(half);
  size_t param_bytes_in = cols * sizeof(float);
  size_t param_bytes_in_h = param_bytes_in / 2;
  size_t param_bytes_out = rows * sizeof(float);
  size_t mask_bytes = rows * cols * sizeof(uint8_t);

  // memory malloc
  half *x, *residual, *bias, *residual_out, *y;
  float *gamma, *beta, *mean, *var;
  uint8_t *mask;
  cudaMalloc((void **)&x, data_bytes);
  cudaMalloc((void **)&residual, data_bytes);
  cudaMalloc((void **)&y, data_bytes);
  cudaMalloc((void **)&residual_out, data_bytes);
  cudaMalloc((void **)&bias, param_bytes_in_h);
  cudaMalloc((void **)&gamma, param_bytes_in);
  cudaMalloc((void **)&beta, param_bytes_in);
  cudaMalloc((void **)&mean, param_bytes_out);
  cudaMalloc((void **)&var, param_bytes_out);
  cudaMalloc((void **)&mask, mask_bytes);

  // launch kernel
  constexpr int WARPS_N = cols < 1024 ? 1 : (cols / 1024);
  constexpr int WARPS_M = 4 / WARPS_N;
  const int THREADS_PER_WARP = 32;
  const int BYTES_PER_LDG = 16;
  const int VecSize = BYTES_PER_LDG / sizeof(half);
  const int THREADS_PER_CTA = WARPS_N * THREADS_PER_WARP * WARPS_M;
  const int ROWS_PER_CTA = WARPS_M;
  const int grid =
      static_cast<int>(std::ceil(rows / static_cast<float>(ROWS_PER_CTA)));
  fused_fast_ln_fwd_kernel<half, float, float, uint8_t, VecSize, WARPS_M,
                           WARPS_N, BYTES_PER_LDG,
                           cols><<<grid, THREADS_PER_CTA>>>(
      rows, cols, 1, 0.6, true, false, 32, 0.000001, x, residual, bias, gamma,
      beta, mask, mean, var, residual_out, y);

  cudaFree(x);
  cudaFree(residual);
  cudaFree(y);
  cudaFree(residual_out);
  cudaFree(bias);
  cudaFree(gamma);
  cudaFree(beta);
  cudaFree(mean);
  cudaFree(var);
  cudaFree(mask);
}