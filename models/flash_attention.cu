#include "flash_attention.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <limits>
#include <cooperative_groups.h>
#include "cublas_v2.h"

namespace cg = cooperative_groups;

using namespace torch::indexing;

const int BLOCKSIZE = 32;
#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

__global__ void forward_get_m(int M, const float *rowmax_S, const float *prev_m, float *m)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M)
  {
    auto m_idx = batch * M + x;
    m[m_idx] = max(prev_m[m_idx], rowmax_S[m_idx]);
    // printf("m %d %d %f %f %f\n", x, y, prev_m[m_idx], rowmax_S[m_idx], m[m_idx]);
  }
}

__global__ void forward_get_P(int M, int N, const float *S, const float *m,
                              const float *mask, float *P, float *rowsum_P)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto m_idx = batch * M + x;
    auto S_idx = batch * M * N + x * N + y;

    auto val = expf(S[S_idx] - m[m_idx]) * mask[batch * N + y];
    P[S_idx] = val;
    rowsum_P[m_idx] += val;
    // printf("P %d %d %f %f %f\n", x, y, S[S_idx], m[m_idx], rowsum_P[m_idx]);
  }
}

__global__ void forward_get_l(int M,
                              const float *prev_m, const float *m, const float *rowsum_P, const float *prev_l, float *l)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M)
  {
    auto lm_idx = batch * M + x;

    l[lm_idx] = expf(prev_m[lm_idx] - m[lm_idx]) * prev_l[lm_idx] + rowsum_P[lm_idx];
    // printf("l %d %d %f %f %f %f\n", x, y, prev_m[lm_idx], m[lm_idx], prev_l[lm_idx], rowsum_P[lm_idx]);
  }
}

__global__ void forward_get_O(int M, int N,
                              const float *prev_m, const float *m,
                              const float *PV, const float *prev_O, float *O)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto m_idx = batch * M + x;
    auto O_idx = batch * M * N + x * N + y;

    O[O_idx] = (1 / expf(prev_m[m_idx] - m[m_idx])) * prev_O[O_idx];
    O[O_idx] += PV[O_idx];
    // printf("O %d %d %f %f %f %f %f\n", x, y, prev_m[m_idx], m[m_idx], prev_O[O_idx], PV[O_idx], O[O_idx]);
  }
}

__global__ void forward_set_Ol(int M, int N, const float *m,
                               const float *O, const float *l, float *O_list, float *l_list)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto ml_idx = batch * M + x;
    auto O_idx = batch * M * N + x * N + y;

    O_list[O_idx] = (1 / l[ml_idx]) * O[O_idx];
    l_list[ml_idx] = m[ml_idx] + logf(l[ml_idx]);
  }
}

__global__ void forward_get_S(int M, int N, int K,
                              float *A, float *B, float *rowmax_S, float *S)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += A[batch * M * K + x * K + i] * B[batch * K * N + i * N + y];
    }

    S[batch * M * N + x * N + y] = tmp;
    rowmax_S[batch * M + x] = max(rowmax_S[batch * M + x], tmp);
  }
}

__global__ void sgemm(int M, int N, int K,
                      float *A, float *B, float *C)
{
  auto batch = blockIdx.x * BLOCKSIZE;
  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += A[batch * M * K + x * K + i] * B[batch * K * N + i * N + y];
    }
    // C = A@B
    C[batch * M * N + x * N + y] = tmp;
  }
}

__global__ void sgemm_add(int M, int N, int K,
                          float *A, float *B, float *C)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += A[batch * M * K + x * K + i] * B[batch * K * N + i * N + y];
    }
    // C = C + A@B
    // printf("%u %u %u %d %d %u\n", batch, x, y, M, N, batch * M * N + x * N + y);
    C[batch * M * N + x * N + y] += tmp;
  }
}

__global__ void backward_get_P(int M, int N,
                               const float *S, const float *L, float *P)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    P[batch * M * N + x * N + y] = expf(S[batch * M * N + x * N + y] - L[batch * M + x]);
  }
}

__global__ void backward_get_dS(int M, int N,
                                const float *P, const float *dP, const float *D, float *dS)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto index = batch * M * N + x * N + y;
    dS[index] = P[index] * (dP[index] - D[batch * M + x]);
  }
}

__global__ void backward_set_dKV(int M, int N,
                                 const float *dK, const float *dV, float *dK_list, float *dV_list)
{
  auto batch = blockIdx.x * BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto index = batch * M * N + x * N + y;
    dK_list[index] = dK[index];
    dV_list[index] = dV[index];
  }
}

std::vector<torch::Tensor> flash_attention_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const int Br, const int Bc,
    torch::Tensor mask)
{
  // cudaDeviceProp device_prop;
  // cudaGetDeviceProperties(&device_prop, 0);
  // std::cout << "name : " << device_prop.name << '\n';

  // std::cout
  //     << "max threads size per block: "
  //     << device_prop.maxThreadsPerBlock << "\n";

  // std::cout
  //     << "max thread size           : "
  //     << device_prop.maxThreadsDim[0] << " x "
  //     << device_prop.maxThreadsDim[1] << " x "
  //     << device_prop.maxThreadsDim[2] << '\n';

  // std::cout
  //     << "max grid size             : "
  //     << device_prop.maxGridSize[0] << " x "
  //     << device_prop.maxGridSize[1] << " x "
  //     << device_prop.maxGridSize[2] << '\n';

  int B = Q.size(0);
  int nh = Q.size(1);
  int N = Q.size(2);
  const int hd = Q.size(3);

  // 切り上げ除算
  const int Tr = (N + Br - 1) / Br;
  const int Tc = (N + Bc - 1) / Bc;

  const auto dtype = Q.dtype();

  auto Q_list = Q.reshape({B * nh, Tr, Br, hd}).contiguous();
  auto K_list = K.reshape({B * nh, Tc, Bc, hd}).transpose(-1, -2).contiguous();
  auto V_list = V.reshape({B * nh, Tc, Bc, hd}).contiguous();

  auto mask_list = mask.reshape({B * nh, Tc, Bc}).contiguous();

  auto O_list = torch::empty({B * nh, Tr, Br, hd}).to(dtype).to(Q.device());
  auto L_list = torch::empty({B * nh, Tr, Br}).to(dtype).to(Q.device());

  auto S = torch::empty({B * nh, Br, Bc}).to(dtype).to(Q.device());
  auto P = S;
  auto PV = torch::empty({B * nh, Br, hd}).to(dtype).to(Q.device());

  auto prev_O = torch::zeros({B * nh, Br, hd}).to(dtype).to(Q.device());
  auto prev_l = torch::zeros({B * nh, Br}).to(dtype).to(Q.device());
  auto prev_m = torch::full({B * nh, Br}, -INFINITY).to(dtype).to(Q.device());

  auto O = torch::zeros({B * nh, Br, hd}).to(dtype).to(Q.device());
  auto l = torch::zeros({B * nh, Br}).to(dtype).to(Q.device());
  auto m = torch::zeros({B * nh, Br}).to(dtype).to(Q.device());

  auto rowmax_S = torch::zeros({B * nh, Br}).to(dtype).to(Q.device());
  auto rowsum_P = torch::zeros({B * nh, Br}).to(dtype).to(Q.device());

  dim3 gridDim(CEIL_DIV(B * nh, BLOCKSIZE), CEIL_DIV(Br, BLOCKSIZE), CEIL_DIV(Bc, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf("CUBLAS initialization failed\n");
    return {};
  }

  size_t mem_size;

  for (int i = 0; i < Tr; i++)
  {
    auto Qi = Q_list.index({Slice(), i}).contiguous().data_ptr<float>();
    prev_O.fill_(0);
    prev_l.fill_(0);
    prev_m.fill_(-INFINITY);

    for (int j = 0; j < Tc; j++)
    {
      gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);
      rowmax_S.fill_(-INFINITY);
      // printf("%d %d %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x);
      forward_get_S<<<gridDim, blockDim>>>(
          Br, Bc, hd,
          Qi,
          K_list.index({Slice(), j}).contiguous().data_ptr<float>(),
          rowmax_S.data_ptr<float>(),
          S.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.z = 1;
      forward_get_m<<<gridDim, blockDim>>>(
          Br,
          rowmax_S.data_ptr<float>(),
          prev_m.data_ptr<float>(), m.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);
      rowsum_P.fill_(0);
      forward_get_P<<<gridDim, blockDim>>>(
          Br, Bc,
          S.data_ptr<float>(),
          m.data_ptr<float>(),
          mask_list.index({Slice(), j}).contiguous().data_ptr<float>(),
          P.data_ptr<float>(),
          rowsum_P.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.z = 1;
      forward_get_l<<<gridDim, blockDim>>>(
          Br,
          prev_m.data_ptr<float>(),
          m.data_ptr<float>(),
          rowsum_P.data_ptr<float>(),
          prev_l.data_ptr<float>(),
          l.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.z = CEIL_DIV(hd, BLOCKSIZE);
      // printf("%d %d %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x);
      sgemm<<<gridDim, blockDim>>>(
          Br, hd, Bc,
          P.data_ptr<float>(),
          V_list.index({Slice(), j}).contiguous().data_ptr<float>(),
          PV.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);
      forward_get_O<<<gridDim, blockDim>>>(
          Br, Bc,
          prev_m.data_ptr<float>(),
          m.data_ptr<float>(),
          PV.data_ptr<float>(),
          prev_O.data_ptr<float>(),
          O.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      prev_m.copy_(m);
      prev_l.copy_(l);
      prev_O.copy_(O);
    }

    gridDim.z = CEIL_DIV(hd, BLOCKSIZE);
    forward_set_Ol<<<gridDim, blockDim>>>(
        Br, hd,
        m.data_ptr<float>(),
        O.data_ptr<float>(),
        l.data_ptr<float>(),
        O_list.index({Slice(), i}).contiguous().data_ptr<float>(),
        L_list.index({Slice(), i}).contiguous().data_ptr<float>());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  return {
      O_list.reshape({B, nh, N, hd}).contiguous(),
      L_list.reshape({B, nh, N}).contiguous()};
}

std::vector<torch::Tensor> flash_attention_cuda_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L,
    const int Br, const int Bc)
{
  auto B = Q.size(0);
  auto nh = Q.size(1);
  auto N = Q.size(2);
  auto hd = Q.size(3);

  // 切り上げ除算
  auto Tr = (N + Br - 1) / Br;
  auto Tc = (N + Bc - 1) / Bc;

  auto dO_list = dO.reshape({B * nh, Tr, Br, hd});
  auto Q_list = Q.reshape({B * nh, Tr, Br, hd}).contiguous();
  auto K_list = K.reshape({B * nh, Tc, Bc, hd}).contiguous();
  auto V_list = V.reshape({B * nh, Tc, Bc, hd}).contiguous();
  auto L_list = L.reshape({B * nh, Tr, Br}).contiguous();
  auto dQ_list = torch::zeros_like(Q_list);
  auto dK_list = torch::zeros_like(K_list);
  auto dV_list = torch::zeros_like(V_list);

  auto S = torch::empty({B * nh, Br, Bc}).to(Q.dtype()).to(Q.device());
  auto dP = torch::empty({B * nh, Br, Bc}).to(Q.dtype()).to(Q.device());
  auto dS = dP;

  // 𝐷 = rowsum(dO ◦ O) ∈ R𝑑 (pointwise multiply)
  auto D = torch::sum(dO * O, 3);
  auto D_list = D.reshape({B * nh, Tr, Br});

  dim3 gridDim(CEIL_DIV(B * nh, BLOCKSIZE), CEIL_DIV(Br, BLOCKSIZE), CEIL_DIV(Bc, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

  size_t mem_size;

  for (int j = 0; j < Tc; j++)
  {
    // Initialize dK𝑗 = (0)𝐵𝑐×𝑑, dV𝑗 = (0)𝐵𝑐×𝑑
    auto dK = torch::empty({B * nh, Br, hd}).to(Q.dtype()).to(Q.device());
    auto dV = torch::empty({B * nh, Br, hd}).to(Q.dtype()).to(Q.device());

    auto KjT = K_list.index({Slice(), j}).transpose(-1, -2).contiguous().data_ptr<float>();
    auto Kj = K_list.index({Slice(), j}).contiguous().data_ptr<float>();
    auto VjT = V_list.index({Slice(), j}).transpose(-1, -2).contiguous().data_ptr<float>();

    for (int i = 0; i < Tr; i++)
    {
      auto Qi = Q_list.index({Slice(), i}).contiguous().data_ptr<float>();
      auto dOi = dO_list.index({Slice(), i}).contiguous().data_ptr<float>();

      gridDim.y = CEIL_DIV(Br, BLOCKSIZE);
      gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);

      // S( 𝑗 )𝑖 = Q𝑖K𝑇𝑗∈ R𝐵𝑟 ×𝐵𝑐
      sgemm<<<gridDim, blockDim>>>(
          Br, Bc, hd,
          Qi,
          KjT,
          S.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      // P( 𝑗 )𝑖 = exp(S𝑖 𝑗 − 𝐿𝑖) ∈ R𝐵𝑟 ×𝐵𝑐
      auto P = S;
      backward_get_P<<<gridDim, blockDim>>>(
          Br, Bc,
          S.data_ptr<float>(),
          L_list.index({Slice(), i}).contiguous().data_ptr<float>(),
          P.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.y = CEIL_DIV(Bc, BLOCKSIZE);
      gridDim.z = CEIL_DIV(hd, BLOCKSIZE);
      // dV𝑗 ← dV𝑗 + (P( 𝑗 )𝑖 )⊤dO𝑖 ∈ R𝐵𝑐×𝑑.
      sgemm_add<<<gridDim, blockDim>>>(
          Bc, hd, Br,
          P.transpose(-1, -2).contiguous().data_ptr<float>(),
          dOi,
          dV.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.y = CEIL_DIV(Br, BLOCKSIZE);
      gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);

      // dP( 𝑗 )𝑖 = dO𝑖V⊤𝑗 ∈ R𝐵𝑟 ×𝐵𝑐
      sgemm<<<gridDim, blockDim>>>(
          Br, Bc, hd,
          dOi,
          VjT,
          dP.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      // dS( 𝑗 )𝑖 = P( 𝑗 )𝑖 ◦ (dP( 𝑗 )𝑖 − 𝐷𝑖) ∈ R𝐵𝑟 ×𝐵𝑐
      backward_get_dS<<<gridDim, blockDim>>>(
          Br, Bc,
          P.data_ptr<float>(),
          dP.data_ptr<float>(),
          D_list.index({Slice(), i}).contiguous().data_ptr<float>(),
          dS.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.z = CEIL_DIV(hd, BLOCKSIZE);
      // Load dQ𝑖 from HBM to SRAM, then on chip,
      // update dQ𝑖 ← dQ𝑖 + dS( 𝑗 )𝑖 K𝑗 ∈ R𝐵𝑟 ×𝑑, and write back to HBM.
      sgemm_add<<<gridDim, blockDim>>>(
          Br, hd, Bc,
          dS.data_ptr<float>(),
          Kj,
          dQ_list.index({Slice(), i}).contiguous().data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.y = CEIL_DIV(Bc, BLOCKSIZE);
      gridDim.z = CEIL_DIV(hd, BLOCKSIZE);
      // dK𝑗 ← dK𝑗 + dS( 𝑗 )𝑖⊤Q𝑖 ∈ R𝐵𝑐×𝑑
      sgemm_add<<<gridDim, blockDim>>>(
          Bc, hd, Br,
          dS.transpose(-1, -2).contiguous().data_ptr<float>(),
          Qi,
          dK.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
    }

    gridDim.y = CEIL_DIV(Bc, BLOCKSIZE);
    gridDim.z = CEIL_DIV(Br, BLOCKSIZE);
    backward_set_dKV<<<gridDim, blockDim>>>(
        Bc, Br,
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        dK_list.index({Slice(), j}).contiguous().data_ptr<float>(),
        dV_list.index({Slice(), j}).contiguous().data_ptr<float>());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  torch::Tensor undef;
  return {
      dQ_list.reshape({B, nh, N, hd}),
      dK_list.reshape({B, nh, N, hd}),
      dV_list.reshape({B, nh, N, hd}),
      undef};
}