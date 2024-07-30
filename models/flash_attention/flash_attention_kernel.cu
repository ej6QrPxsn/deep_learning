#include <torch/extension.h>
#include <vector>
#include "../../config.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <iostream>
#include <limits>

// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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

__global__ void flash_attention_cuda_forward_kernel(
    const float *Q_list,
    const float *K_list,
    const float *V_list,
    float *O_list,
    float *l_list,
    const int Tr, const int Tc, const int Br, const int Bc, const int d)
{
  const auto d_idx = threadIdx.x;

  const auto q_ofst = blockIdx.x * gridDim.y * gridDim.z * d +
                      blockIdx.y * gridDim.z * d;

  const auto kv_ofst = blockIdx.x * Tc * Bc * d;

  const auto lm_ofst = blockIdx.x * gridDim.y * gridDim.z +
                       blockIdx.y * gridDim.z +
                       blockIdx.z;

  extern __shared__ float mbm_block[];
  float *Qi = mbm_block;
  float *Kj = Qi + Br;
  float *Vj = Kj + Bc + d;
  float *Sij = Vj + Bc;
  float *Oij = Sij + Bc;
  float *l = Oij + Br;
  float *m = l + Br;
  float *prev_m = l + Br;
  float *temp = prev_m + Br;

  cg::thread_block g = cg::this_thread_block();

  auto Pij = Sij;

  for (int br_idx = 0; br_idx < Br; br_idx++)
  {
    Qi[br_idx] = Q_list[q_ofst + br_idx * d + d_idx];
    Oij[br_idx] = 0;
    l[br_idx] = 0;
    prev_m[br_idx] = -INFINITY;
  }

  // d次元の完了を待つ
  g.sync();

  for (int j = 0; j < Tc; j++)
  {
    // Load Kj to SRAM
    for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
    {
      Kj[bc_idx * d + d_idx] = K_list[kv_ofst + j * Bc * d + bc_idx * d + d_idx];
      Vj[bc_idx * d + d_idx] = V_list[kv_ofst + j * Bc * d + bc_idx * d + d_idx];
    }

    // d次元の完了を待つ
    g.sync();

    for (int br_idx = 0; br_idx < Br; br_idx++)
    {
      float rowmax = 0;

      for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
      {
        // S( 𝑗 )𝑖 = Q𝑖K𝑇𝑗∈ R𝐵𝑟 ×𝐵𝑐
        // d次元足し合わせ
        Sij[bc_idx] = reduce_sum(g, temp, Qi[br_idx] * Kj[d_idx * Br + bc_idx]);
        rowmax = max(Sij[bc_idx], rowmax);
      }
      // 𝑚( 𝑗 )𝑖 = max(𝑚( 𝑗−1)𝑖 , rowmax(S( 𝑗 )𝑖 )) ∈ R𝐵𝑟
      m[br_idx] = max(prev_m[br_idx], rowmax);

      float rowsum = 0;
      float PijVj = 0;
      for (int bc_idx = 0; bc_idx < Bc; bc_idx++)
      {
        // ˜P ( 𝑗 )𝑖 = exp(S( 𝑗 )𝑖 − 𝑚( 𝑗 )𝑖 ) ∈ R𝐵𝑟 ×𝐵𝑐
        Pij[bc_idx] = expf(Sij[bc_idx] - m[br_idx]);
        rowsum += Pij[bc_idx];
        // ˜P ( 𝑗 )𝑖 V𝑗
        PijVj += Pij[bc_idx] * Vj[bc_idx];
      }

      // ℓ( 𝑗 )𝑖 = 𝑒𝑚𝑗−1𝑖 −𝑚( 𝑗 )𝑖 ℓ( 𝑗−1)𝑖 + rowsum(˜P ( 𝑗 )𝑖 ) ∈ R𝐵𝑟
      l[br_idx] = expf(prev_m[br_idx] - m[br_idx]) * l[br_idx] + rowsum;

      // O( 𝑗 )𝑖 = diag(𝑒𝑚( 𝑗−1)𝑖 −𝑚( 𝑗 )𝑖 )−1O( 𝑗−1)𝑖 + ˜P ( 𝑗 )𝑖 V𝑗
      Oij[br_idx] = (1 / expf(prev_m[br_idx] - m[br_idx])) * Oij[br_idx] + PijVj;

      prev_m[br_idx] = m[br_idx];
    }
  }


  for (int br_idx = 0; br_idx < Br; br_idx++)
  {
    // O𝑖 = diag(ℓ(𝑇𝑐 )𝑖 )−1O(𝑇𝑐 )
    O_list[q_ofst + br_idx * d + d_idx] = (1 / l[br_idx]) * Oij[br_idx];

    // 𝐿𝑖 = 𝑚(𝑇𝑐 )𝑖 + log(ℓ(𝑇𝑐 )𝑖 )
    l_list[lm_ofst + br_idx] = m[br_idx] + logf(l[br_idx]);
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
    const int Tr, const int Tc, const int Br, const int Bc, const int d)

{
  const auto d_idx = threadIdx.x;

  const auto q_ofst = blockIdx.x * Tr * Br * d;

  const auto kv_ofst = blockIdx.x * gridDim.y * Bc * d +
                       blockIdx.y * Bc * d;

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
    Kj[bc_idx * d + d_idx] = K_list[kv_ofst + bc_idx * d + d_idx];

    // Load VjT to SRAM
    Vj[bc_idx * d + d_idx] = V_list[kv_ofst + bc_idx * d + d_idx];
  }

  // d次元の完了を待つ
  g.sync();

  auto Pij = Sij;
  auto dSij = dPij;

  for (int i = 0; i < Tr; i++)
  {
    for (int br_idx = 0; br_idx < Br; br_idx++)
    {
      // Load Qi to SRAM
      Qi[br_idx] = Q_list[q_ofst + i * Br + br_idx * d + d_idx];

      // Load dOi to SRAM
      dOi[br_idx] = dO_list[q_ofst + i * Br + br_idx * d + d_idx];

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
        dQi[br_idx] += dSij[br_idx * Bc + bc_idx] * Kj[bc_idx * d + d_idx];
      }
      dQ_list[q_ofst + i * Br + br_idx * d + d_idx] = dOi[br_idx];
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
    dK_list[kv_ofst + bc_idx * d + d_idx] = dKj[bc_idx];
    dV_list[kv_ofst + bc_idx * d + d_idx] = dVj[bc_idx];
  }
}

std::vector<torch::Tensor> flash_attention_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
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
  const int d = Q.size(3);

  const int Bc = Config::Bc;
  const int Br = Config::Br;

  // 切り上げ除算
  const int Tr = (N + Br - 1) / Br;
  const int Tc = (N + Bc - 1) / Bc;

  auto Q_list = Q.reshape({B, nh, Tr, Br, d}).contiguous();
  auto K_list = K.reshape({B, nh, Tc, Bc, d}).contiguous();
  auto V_list = V.reshape({B, nh, Tc, Bc, d}).contiguous();
  auto O_list = torch::zeros({B, nh, Tr, Br, d}).to(Q.device());
  auto l_list = torch::zeros({B, nh, Tr, Br}).to(Q.device());

  int Q_size = Br;
  int K_size = Bc * d;
  int V_size = Bc;
  int S_size = Bc;
  int O_size = Br;
  int l_size = Br;
  int m_size = Br;
  int prev_m_size = Br;
  int temp_size = d;

  int l1_size = Q_size + K_size + V_size + S_size + O_size + l_size + m_size + prev_m_size + temp_size;
  const int mem_size = l1_size * Q.element_size();

  dim3 grid(B * nh, Tr);
  dim3 block(d);

  flash_attention_cuda_forward_kernel<<<grid, block, mem_size>>>(
      Q.data_ptr<float>(),
      K.data_ptr<float>(),
      V.data_ptr<float>(),
      O_list.data_ptr<float>(),
      l_list.data_ptr<float>(),
      Tr, Tc, Br, Bc, d);

  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  return {
      O_list.reshape({B, nh, N, d}),
      l_list.reshape({B, nh, N})};
}

std::vector<torch::Tensor> flash_attention_cuda_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L)
{
  auto B = Q.size(0);
  auto nh = Q.size(1);
  auto N = Q.size(2);
  auto d = Q.size(3);

  auto Bc = Config::Bc;
  auto Br = Config::Br;

  // 切り上げ除算
  auto Tr = (N + Br - 1) / Br;
  auto Tc = (N + Bc - 1) / Bc;

  auto dO_list = dO.reshape({B, nh, Tr, Br, d});
  auto Q_list = Q.reshape({B, nh, Tr, Br, d}).contiguous();
  auto K_list = K.reshape({B, nh, Tc, Bc, d}).contiguous();
  auto V_list = V.reshape({B, nh, Tc, Bc, d}).contiguous();
  auto L_list = torch::zeros({B, nh, Tr, Br}).to(Q.device());
  auto dQ_list = torch::zeros_like(Q_list);
  auto dK_list = torch::zeros_like(K_list);
  auto dV_list = torch::zeros_like(V_list);

  auto D = torch::sum(dO * O, /* d= */ 3);
  auto D_list = D.reshape({B, nh, Tr, Br});

  int Q_size = d;
  int K_size = Bc * d;
  int V_size = Bc * d;
  int S_size = Bc;
  int O_size = d;
  int dQ_size = d;
  int dK_size = Bc * d;
  int dV_size = Bc * d;
  int r_size = Br;

  int l1_size = Q_size + K_size + V_size + S_size + O_size +
                dQ_size + dK_size + dV_size + r_size;
  const int mem_size = l1_size * Q.element_size();

  dim3 grid(B * nh, Tc);
  dim3 block(d);

  flash_attention_cuda_backward_kernel<<<grid, block, mem_size>>>(
      dO_list.data_ptr<float>(),
      Q_list.data_ptr<float>(),
      K_list.data_ptr<float>(),
      V_list.data_ptr<float>(),
      L_list.data_ptr<float>(),
      D_list.data_ptr<float>(),
      dQ_list.data_ptr<float>(),
      dK_list.data_ptr<float>(),
      dV_list.data_ptr<float>(),
      Tr, Tc, Br, Bc, d);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }

  return {
      dQ_list.reshape({B, nh, N, d}),
      dK_list.reshape({B, nh, N, d}),
      dV_list.reshape({B, nh, N, d})};
}