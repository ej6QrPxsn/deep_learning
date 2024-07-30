#include <torch/torch.h>
#include <vector>
#include "toml.hpp"


// CUDA forward declarations
std::vector<torch::Tensor> flash_attention_cuda_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    const int Br, const int Bc);

std::vector<torch::Tensor> flash_attention_cuda_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L,
    const int Br, const int Bc);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

struct FlashAttention : public torch::autograd::Function<FlashAttention>
{
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx, 
      torch::Tensor Q,
      torch::Tensor K,
      torch::Tensor V)
  {

    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);

    auto data = toml::parse("config.toml");
    static const int Br = toml::find<int>(data, "model", "Br");
    static const int Bc = toml::find<int>(data, "model", "Bc");

    auto out = flash_attention_cuda_forward(Q, K, V, Br, Bc);

    // Save data for backward in context
    ctx->saved_data["Q"] = Q;
    ctx->saved_data["K"] = K;
    ctx->saved_data["V"] = V;
    ctx->saved_data["O"] = out[0];
    ctx->saved_data["L"] = out[1];

    return {out[0]};
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) {
    torch::Tensor dO = grad_output[0];
    auto Q = ctx->saved_data["Q"].toTensor();
    auto K = ctx->saved_data["K"].toTensor();
    auto V = ctx->saved_data["V"].toTensor();
    auto O = ctx->saved_data["O"].toTensor();
    auto L = ctx->saved_data["L"].toTensor();

    CHECK_INPUT(dO);
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(O);
    CHECK_INPUT(L);

    auto data = toml::parse("config.toml");
    static const int Br = toml::find<int>(data, "model", "Br");
    static const int Bc = toml::find<int>(data, "model", "Bc");

    return flash_attention_cuda_backward(dO, Q, K, V, O, L, Br, Bc);
  }
};
