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

__device__ float reduce_sum(cg::thread_group g, float *temp, float val)
{
  int lane = g.thread_rank();

  for (int i = g.size() / 2; i > 0; i /= 2)
  {
    temp[lane] = val;
    g.sync();
    if (lane < i)
    {
      val += temp[lane + i];
    }
    g.sync();
  }
  return val;
}

__device__ float reduce_max(cg::thread_group g, float *temp, float val)
{
  int lane = g.thread_rank();

  for (int i = g.size() / 2; i > 0; i /= 2)
  {
    temp[lane] = val;
    g.sync();
    if (lane < i)
    {
      val = max(temp[lane], temp[lane + i]);
    }
    g.sync();
  }
  return val;
}

__global__ void forward_get_m(int M, int N, int Tr, int tr_idx, const float *S, const float *prev_m, float *m)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto m_idx = batch * Tr * M + (tr_idx + 1) * M + x;

    m[m_idx] = max(m[m_idx], max(prev_m[batch * M + x], S[batch * M * N + x * N + y]));
  }
}

__global__ void forward_get_P(int M, int N, int Tr, int tr_idx, const float *S, const float *m, float *P, float *rowsum_P)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto m_idx = batch * M + x;
    auto S_idx = batch * M * N + x * N + y;

    auto val = expf(S[S_idx] - m[m_idx]);
    P[S_idx] = val;
    rowsum_P[batch * M + x] += val;
  }
}

__global__ void forward_get_l(int M, int N, int Tr, int tr_idx,
                              const float *prev_m, const float *m, const float *rowsum_P, const float *prev_l, float *l)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto lm_idx = batch * M + x;

    l[lm_idx] = expf(prev_m[lm_idx] - m[lm_idx]) * prev_l[lm_idx] + rowsum_P[lm_idx];
  }
}

__global__ void forward_get_O(int M, int N, int Tr, int tr_idx,
                              const float *prev_m, const float *m,
                              const float *PV, const float *prev_O, float *O)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto m_idx = batch * M + x;
    auto O_idx = batch * M * N + x * N + y;

    O[O_idx] = (1 / (prev_m[m_idx] - m[m_idx])) * prev_O[O_idx];
    // 非数値(nanなど)の場合は0設定
    if (O[O_idx] != O[O_idx])
    {
      O[O_idx] = 0;
    }
    O[O_idx] += PV[O_idx];
  }
}

__global__ void forward_set_Ol(int M, int N, int Tr, int tr_idx, const float *m, const float *O, const float *l, float *O_list, float *l_list)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    auto ml_idx = batch * M + x;
    auto O_idx = batch * M * N + x * N + y;

    O_list[batch * Tr * M * N + tr_idx * M * N + x * N + y] = (1 / l[ml_idx]) * O[O_idx];
    l_list[batch * Tr * M + tr_idx * M + x] = m[ml_idx] + logf(l[ml_idx]);
  }
}

__global__ void sgemm_plus(int M, int N, int K,
                           float *A, float *B, const float *C, float *D)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    extern __shared__ float mbm_block[];
    float *a = mbm_block;
    float *b = a + M * K;

    // load shared memory
    a = &A[batch * M * K];
    b = &B[batch * K * N];

    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += a[x * K + i] * b[i * N + y];
    }
    // D = A@B + C
    // printf("%u %u %u %d %d %u\n", batch, x, y, M, N, batch * M * N + x * N + y);
    D[batch * M * N + x * N + y] = tmp + C[batch * N + y];
  }
}

__global__ void sgemm(int M, int N, int K,
                      float *A, float *B, float *C)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    extern __shared__ float mbm_block[];
    float *a = mbm_block;
    float *b = a + M * K;

    // load shared memory
    a = &A[batch * M * K];
    b = &B[batch * K * N];

    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += a[x * K + i] * b[i * N + y];
    }
    // C = A@B
    // printf("%u %u %u %d %d %u\n", batch, x, y, M, N, batch * M * N + x * N + y);
    C[batch * M * N + x * N + y] = tmp;
  }
}

__global__ void backward_get_P(int M, int N,
                               const float *S, const float *L, float *P)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    P[batch * M * N + x * N + y] = expf(S[batch * M * N + x * N + y] - L[batch * M + x]);
  }
}

__global__ void sgemm_add(int M, int N, int K,
                          float *A, float *B, float *C)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    extern __shared__ float mbm_block[];
    float *a = mbm_block;
    float *b = a + M * K;

    // load shared memory
    a = &A[batch * M * K];
    b = &B[batch * K * N];

    float tmp = 0.0;
    for (int i = 0; i < K; ++i)
    {
      tmp += a[x * K + i] * b[i * N + y];
    }
    // C = C + A@B
    // printf("%u %u %u %d %d %u\n", batch, x, y, M, N, batch * M * N + x * N + y);
    C[batch * M * N + x * N + y] += tmp;
  }
}

__global__ void backward_get_dS(int M, int N,
                                const float *P, const float *dP, const float *D, float *dS)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    dS[batch * M * N + x * N + y] = P[batch * M * N + x * N + y] * (dP[batch * M * N + x * N + y] - D[batch * M + x]);
  }
}

__global__ void backward_set_dKV(int M, int N,
                                 const float *dK, const float *dV, float *dK_list, float *dV_list)
{
  auto batch = blockIdx.x * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

  auto x = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  auto y = blockIdx.z * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N)
  {
    dK_list[batch * M * N + x * N + y] = dK_list[x * N + y];
    dV_list[batch * M * N + x * N + y] = dV_list[x * N + y];
  }
}

__global__ void flash_attention_cuda_backward_kernel(
    const float *dO_list,
    const float *Q_list,
    const float *K_list,
    const float *V_list,
    const float *L_list,
    const float *D_list,
    float *dQ_list,
    float *dK_list,
    float *dV_list,
    const int Tr, const int Tc, const int Br, const int Bc, const int hd)

{
  const auto d_idx = threadIdx.x;

  const auto q_ofst = blockIdx.x * Tr * Br * hd;

  const auto kv_ofst = blockIdx.x * Tc * Bc * hd +
                       blockIdx.y * Bc * hd;

  const auto lm_ofst = blockIdx.x * Tr * Br;

  extern __shared__ float mbm_block[];
  float *Qi = mbm_block;
  float *Kj = Qi + Br;
  float *Vj = Kj + Bc;
  float *Sij = Vj + Bc;
  float *dPij = Sij + Br * Bc;
  float *dOi = dPij + Br * Bc;
  float *dQi = dOi + Br;
  float *dKj = dOi + Br;
  float *dVj = dKj + Br;
  float *temp = dVj + Br;

  cg::thread_block g = cg::this_thread_block();

  for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
  {
    // Load Kj to SRAM
    Kj[bc_idx * hd + d_idx] = K_list[kv_ofst + bc_idx * hd + d_idx];

    // Load VjT to SRAM
    Vj[bc_idx * hd + d_idx] = V_list[kv_ofst + bc_idx * hd + d_idx];
  }

  auto Pij = Sij;
  auto dSij = dPij;

  for (int i = 0; i < Tr; i++)
  {
    for (int br_idx = 0; br_idx < Br; br_idx++)
    {
      // Load Qi to SRAM
      Qi[br_idx] = Q_list[q_ofst + i * Br + br_idx * hd + d_idx];

      // Load dOi to SRAM
      dOi[br_idx] = dO_list[q_ofst + i * Br + br_idx * hd + d_idx];

      auto Li = L_list[lm_ofst + i * Br + br_idx];

      // d次元の完了を待つ
      g.sync();

      for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
      {
        // S( 𝑗 )𝑖 = Q𝑖K𝑇𝑗∈ R𝐵𝑟 ×𝐵𝑐
        // d次元足し合わせ
        Sij[br_idx * Bc + bc_idx] = reduce_sum(g, temp, Qi[br_idx] * Kj[d_idx * Bc + bc_idx]);

        // P( 𝑗 )𝑖 = exp(S𝑖 𝑗 − 𝐿𝑖) ∈ R𝐵𝑟 ×𝐵𝑐
        Pij[br_idx * Bc + bc_idx] = expf(Sij[br_idx * Bc + bc_idx] - Li);
      }
    }

    float sum = 0;
    for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
    {
      for (int br_idx = 0; br_idx < Br; br_idx++)
      {
        // dV𝑗 ← dV𝑗 + (P( 𝑗 )𝑖 )⊤dO𝑖 ∈ R𝐵𝑐×𝑑
        // Br次元足し合わせ
        sum += Pij[bc_idx * Br + br_idx] * dOi[br_idx];
      }
      dVj[bc_idx] += sum;
    }

    for (int br_idx = 0; br_idx < Br; br_idx++)
    {
      auto Di = D_list[lm_ofst + i * Br + br_idx];

      for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
      {
        // d次元足し合わせ
        // dP( 𝑗 )𝑖 = dO𝑖V⊤𝑗 ∈ R𝐵𝑟 ×𝐵𝑐
        dPij[br_idx * Bc + d_idx] = reduce_sum(g, temp, dOi[br_idx] * Vj[d_idx * Bc + bc_idx]);

        // dS( 𝑗 )𝑖 = P( 𝑗 )𝑖 ◦ (dP( 𝑗 )𝑖 − 𝐷𝑖) ∈ R𝐵𝑟 ×𝐵𝑐
        dSij[br_idx * Bc + bc_idx] = Pij[br_idx * Bc + bc_idx] * (dPij[br_idx * Bc + d_idx] - Di);

        // Load dQ𝑖 from HBM to SRAM, then on chip, update dQ𝑖 ← dQ𝑖 + dS( 𝑗 )𝑖 K𝑗 ∈ R𝐵𝑟 ×𝑑, and write back to HBM.
        // Bc次元足し合わせ
        dQi[br_idx] += dSij[br_idx * Bc + bc_idx] * Kj[bc_idx * hd + d_idx];
      }
      dQ_list[q_ofst + i * Br + br_idx * hd + d_idx] = dOi[br_idx];
    }

    for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
    {
      float sum = 0;
      for (int br_idx = 0; br_idx < Br; br_idx++)
      {
        // dK𝑗 ← dK𝑗 + dS( 𝑗 )𝑖⊤Q𝑖 ∈ R𝐵𝑐×𝑑
        // Br次元足し合わせ
        sum += dSij[bc_idx * Br + br_idx] * Qi[br_idx];
      }
      dKj[bc_idx] = sum;
    }
  }

  for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
  {
    // Write dK𝑗 , dV𝑗 to HBM.
    dK_list[kv_ofst + bc_idx * hd + d_idx] = dKj[bc_idx];
    dV_list[kv_ofst + bc_idx * hd + d_idx] = dVj[bc_idx];
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

  auto S = torch::empty({B * nh, Br, Bc}).to(dtype).to(Q.device()).data_ptr<float>();
  auto P = S;
  auto PV = torch::empty({B * nh, Br, hd}).to(dtype).to(Q.device()).data_ptr<float>();

  auto prev_O = torch::zeros({B * nh, Br, hd}).to(dtype).to(Q.device()).data_ptr<float>();
  auto prev_l = torch::zeros({B * nh, Br}).to(dtype).to(Q.device()).data_ptr<float>();
  auto prev_m = torch::full({B * nh, Br}, -INFINITY).to(dtype).to(Q.device()).data_ptr<float>();

  auto O = torch::zeros({B * nh, Br, hd}).to(dtype).to(Q.device()).data_ptr<float>();
  auto l = torch::zeros({B * nh, Br}).to(dtype).to(Q.device()).data_ptr<float>();
  auto m = torch::zeros({B * nh, Br}).to(dtype).to(Q.device()).data_ptr<float>();

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

  float alpha = 1;
  float beta = 0;
  size_t mem_size;

  for (int i = 0; i < Tr; i++)
  {
    auto prev_O = torch::zeros({B * nh, Br, hd}).to(dtype).to(Q.device()).data_ptr<float>();
    auto prev_l = torch::zeros({B * nh, Br}).to(dtype).to(Q.device()).data_ptr<float>();
    auto prev_m = torch::full({B * nh, Br}, -INFINITY).to(dtype).to(Q.device()).data_ptr<float>();

    for (int j = 0; j < Tc; j++)
    {
      mem_size = (Br * Bc + Bc * hd) * Q.element_size();
      sgemm_plus<<<gridDim, blockDim, mem_size>>>(
          Br, Bc, hd,
          Q_list.index({Slice(), i, Slice()}).contiguous().data_ptr<float>(),
          K_list.index({Slice(), j, Slice()}).contiguous().data_ptr<float>(),
          mask_list.index({Slice(), j, Slice()}).contiguous().data_ptr<float>(),
          S);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      forward_get_m<<<gridDim, blockDim>>>(
          Br, Bc, Tr, i,
          S,
          prev_m, m);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      auto rowsum_P = torch::zeros({B * nh, Br}).to(dtype).to(Q.device()).data_ptr<float>();
      forward_get_P<<<gridDim, blockDim>>>(
          Br, Bc, Tr, i,
          S,
          m,
          P,
          rowsum_P);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      forward_get_l<<<gridDim, blockDim>>>(
          Br, Bc, Tr, i,
          prev_m, m,
          rowsum_P,
          prev_l, l);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      mem_size = (Br * Bc + Bc * hd) * Q.element_size();
      sgemm<<<gridDim, blockDim, mem_size>>>(
          Br, Bc, hd,
          P,
          V_list.index({Slice(), j, Slice()}).contiguous().data_ptr<float>(),
          PV);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);
      forward_get_O<<<gridDim, blockDim>>>(
          Br, Bc, Tr, i,
          prev_m, m,
          PV,
          prev_O, O);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      gridDim.z = CEIL_DIV(hd, BLOCKSIZE);
      forward_set_Ol<<<gridDim, blockDim>>>(
          Br, hd, Tr, i,
          m,
          O,
          l,
          O_list.data_ptr<float>(),
          L_list.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      m = prev_m;
      l = prev_l;
      O = prev_O;
    }
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

  auto D = torch::sum(dO * O, /* hd= */ 3);
  auto D_list = D.reshape({B * nh, Tr, Br});

  dim3 gridDim(CEIL_DIV(B * nh, BLOCKSIZE), CEIL_DIV(Br, BLOCKSIZE), CEIL_DIV(Bc, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

  size_t mem_size;

  for (int j = 0; j < Tc; j++)
  {
    // Initialize dK𝑗 = (0)𝐵𝑐×𝑑, dV𝑗 = (0)𝐵𝑐×𝑑
    auto dK = torch::empty({B * nh, Br, Bc}).to(Q.dtype()).to(Q.device());
    auto dV = torch::empty({B * nh, Br, Bc}).to(Q.dtype()).to(Q.device());

    for (int i = 0; i < Tr; i++)
    {
      gridDim.y = CEIL_DIV(Br, BLOCKSIZE);
      gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);

      mem_size = (Br * Bc + Bc * hd) * Q.element_size();
      // S( 𝑗 )𝑖 = Q𝑖K𝑇𝑗∈ R𝐵𝑟 ×𝐵𝑐
      sgemm<<<gridDim, blockDim, mem_size>>>(
          Br, Bc, hd,
          Q_list.index({Slice(), i, Slice()}).contiguous().data_ptr<float>(),
          K_list.index({Slice(), j, Slice()}).contiguous().data_ptr<float>(),
          S.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      // P( 𝑗 )𝑖 = exp(S𝑖 𝑗 − 𝐿𝑖) ∈ R𝐵𝑟 ×𝐵𝑐
      auto P = S;
      backward_get_P<<<gridDim, blockDim>>>(
          Br, Bc,
          S.data_ptr<float>(),
          L_list.index({Slice(), i, Slice()}).contiguous().data_ptr<float>(),
          P.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      mem_size = (Br * hd + hd * Bc) * Q.element_size();
      gridDim.z = CEIL_DIV(hd, BLOCKSIZE);
      // dV𝑗 ← dV𝑗 + (P( 𝑗 )𝑖 )⊤dO𝑖 ∈ R𝐵𝑐×𝑑.
      sgemm_add<<<gridDim, blockDim, mem_size>>>(
          Br, hd, Bc,
          P.transpose(-1, -2).contiguous().data_ptr<float>(),
          dO_list.index({Slice(), i, Slice()}).contiguous().data_ptr<float>(),
          dV.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      mem_size = (Br * hd + hd * Bc) * Q.element_size();
      gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);
      auto dP = P;
      // dP( 𝑗 )𝑖 = dO𝑖V⊤𝑗 ∈ R𝐵𝑟 ×𝐵𝑐
      sgemm<<<gridDim, blockDim, mem_size>>>(
          Br, hd, Bc,
          dO_list.index({Slice(), i, Slice()}).contiguous().data_ptr<float>(),
          V_list.index({Slice(), j, Slice()}).transpose(-1, -2).contiguous().data_ptr<float>(),
          dP.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      auto dS = dP;
      // dS( 𝑗 )𝑖 = P( 𝑗 )𝑖 ◦ (dP( 𝑗 )𝑖 − 𝐷𝑖) ∈ R𝐵𝑟 ×𝐵𝑐
      backward_get_dS<<<gridDim, blockDim>>>(
          Br, Bc,
          P.data_ptr<float>(),
          dP.data_ptr<float>(),
          D.index({Slice(), i, Slice()}).contiguous().data_ptr<float>(),
          dS.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      mem_size = (Br * Bc + Bc * hd) * Q.element_size();
      // Load dQ𝑖 from HBM to SRAM, then on chip,
      // update dQ𝑖 ← dQ𝑖 + dS( 𝑗 )𝑖 K𝑗 ∈ R𝐵𝑟 ×𝑑, and write back to HBM.
      sgemm_add<<<gridDim, blockDim, mem_size>>>(
          Br, Bc, hd,
          dS.data_ptr<float>(),
          K_list.index({Slice(), j, Slice()}).contiguous().data_ptr<float>(),
          dQ_list.index({Slice(), i, Slice()}).contiguous().data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());

      mem_size = (Bc * hd + hd * Br) * Q.element_size();
      gridDim.y = CEIL_DIV(Bc, BLOCKSIZE);
      gridDim.z = CEIL_DIV(hd, BLOCKSIZE);
      // dK𝑗 ← dK𝑗 + dS( 𝑗 )𝑖⊤Q𝑖 ∈ R𝐵𝑐×𝑑
      sgemm_add<<<gridDim, blockDim, mem_size>>>(
          Bc, hd, Br,
          dS.transpose(-1, -2).contiguous().data_ptr<float>(),
          Q_list.index({Slice(), i, Slice()}).contiguous().data_ptr<float>(),
          dK.data_ptr<float>());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
    }

    gridDim.y = CEIL_DIV(Br, BLOCKSIZE);
    gridDim.z = CEIL_DIV(Bc, BLOCKSIZE);
    backward_set_dKV<<<gridDim, blockDim>>>(
        Bc, Br,
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        dK_list.data_ptr<float>(),
        dV_list.data_ptr<float>());
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