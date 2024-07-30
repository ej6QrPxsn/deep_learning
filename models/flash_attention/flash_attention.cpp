#include <torch/extension.h>
#include <vector>
#include "../../config.h"

// CUDA forward declarations
std::vector<at::Tensor> flash_attention_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V);
    
std::vector<torch::Tensor> flash_attention_cuda_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L);

    
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);

  return flash_attention_cuda_forward(Q, K, V);
}
    
std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L) {
  CHECK_INPUT(dO);
  CHECK_INPUT(Q);
  CHECK_INPUT(K);
  CHECK_INPUT(V);
  CHECK_INPUT(O);
  CHECK_INPUT(L);

  return flash_attention_cuda_backward(dO, Q, K, V, O, L);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def ("forward", &flash_attention_forward, "flash_attention forward (CUDA)");
  m.def ("backward", &flash_attention_backward, "flash_attention backward (CUDA)");
}
