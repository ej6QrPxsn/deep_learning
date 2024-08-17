#include <torch/torch.h>
#include "flash_attention.h"

// #include "flash_attention.h"
// #include <pybind11/pybind11.h>

using namespace torch::indexing;

struct RMSNormImpl : torch::nn::Module
{
  RMSNormImpl(int size, torch::Device device, int dim = -1, float eps = 1e-5)
      : eps(eps),
        dim(dim),
        scale(register_parameter("scale", torch::ones(size).to(device)))
  {
  }

  torch::Tensor forward(torch::Tensor x)
  {
    auto norm_x = torch::mean(x * x, dim, true);
    auto x_normed = x * torch::rsqrt(norm_x + eps);
    return scale * x_normed;
  }

  float eps;
  int dim;
  torch::Tensor scale;
};
TORCH_MODULE(RMSNorm);

struct SwiGLUImpl : torch::nn::Module
{
  SwiGLUImpl() {}

  torch::Tensor forward(torch::Tensor x)
  {
    auto ret = x.chunk(2, -1);
    return ret[0] * torch::nn::functional::silu(ret[1]);
  }
};
TORCH_MODULE(SwiGLU);

struct RopeImpl : torch::nn::Module
{
  RopeImpl(torch::Device device) : device(device)
  {
    auto data = toml::parse("config.toml");
    d_model = toml::find<int>(data, "model", "d_model");
  }

  torch::Tensor _rot(torch::Tensor x, int batch, int len)
  {
    auto second = x.index({Slice(), Slice(), Slice(0, None, 2)});
    auto first = x.index({Slice(), Slice(), Slice(1, None, 2)}) * -1;
    auto new_x = torch::stack({first, second}, /* dim = */ 2);
    return new_x.reshape({batch, len, -1});
  }

  torch::Tensor forward(torch::Tensor x)
  {
    auto batch = x.size(0);
    auto len = x.size(1);

    auto half_d = d_model / 2;

    auto m = torch::arange(1, len + 1).to(device);
    auto i = torch::arange(1, half_d + 1).to(device);
    auto theta = torch::pow(10000, -2 * i / d_model);

    auto cos = torch::cos(torch::einsum("i,j->ij", {m, theta}));
    auto sin = torch::sin(torch::einsum("i,j->ij", {m, theta}));

    auto C = torch::stack({cos, cos}, /* dim = */ 1).reshape({1, len, -1});
    auto S = torch::stack({sin, sin}, /* dim =  */ 1).reshape({1, len, -1});

    return x * C + _rot(x, batch, len) * S;
  }

  int d_model;
  torch::Device device;
};
TORCH_MODULE(Rope);

struct SelfAttentionImpl : torch::nn::Module
{
  SelfAttentionImpl(torch::Device device)
      : device(device)
  {
    auto data = toml::parse("config.toml");
    auto d_model = toml::find<int>(data, "model", "d_model");
    num_head = toml::find<int>(data, "model", "num_head");
    head_dim = d_model / num_head;
    W_Q = register_module("W_Q", torch::nn::Linear(d_model, d_model));
    W_K = register_module("W_K", torch::nn::Linear(d_model, d_model));
    W_V = register_module("W_V", torch::nn::Linear(d_model, d_model));
    W_O = register_module("W_O", torch::nn::Linear(d_model, d_model));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask)
  {
    auto batch = x.size(0);
    auto len = x.size(1);

    auto q = W_Q(x).view({batch, len, num_head, head_dim}).transpose(2, 1).contiguous();
    auto k = W_K(x).view({batch, len, num_head, head_dim}).transpose(2, 1).contiguous();
    auto v = W_V(x).view({batch, len, num_head, head_dim}).transpose(2, 1).contiguous();

    auto ret = FlashAttention::apply(q, k, v, mask);

    return W_O(ret[0].reshape({batch, len, -1}));
  }

  int num_head;
  int head_dim;
  torch::Device device;
  torch::nn::Linear W_Q = nullptr;
  torch::nn::Linear W_K = nullptr;
  torch::nn::Linear W_V = nullptr;
  torch::nn::Linear W_O = nullptr;
};
TORCH_MODULE(SelfAttention);

struct FeedForwardNetworksImpl : torch::nn::Module
{
  FeedForwardNetworksImpl(torch::Device device)
      : swi_glu(register_module("swi_glu", SwiGLU()))
  {
    auto data = toml::parse("config.toml");
    auto d_model = toml::find<int>(data, "model", "d_model");
    auto d_ff_scalar = toml::find<int>(data, "model", "d_ff_scalar");
    auto d_ff = d_model * d_ff_scalar;
    linear1 = register_module("linear1", torch::nn::Linear(d_model, d_ff));
    linear2 = register_module("linear2", torch::nn::Linear(d_ff / 2, d_model));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    auto out = swi_glu(linear1(x));
    return linear2(torch::where(out > 0, out, 0));
  }

  torch::nn::Linear linear1 = nullptr;
  torch::nn::Linear linear2 = nullptr;
  SwiGLU swi_glu = nullptr;
};
TORCH_MODULE(FeedForwardNetworks);

struct DecoderLayerImpl : torch::nn::Module
{
  DecoderLayerImpl(torch::Device device)
      : rope(register_module("rope", Rope(device)))
  {
    auto data = toml::parse("config.toml");
    auto d_model = toml::find<int>(data, "model", "d_model");
    auto dropout = toml::find<float>(data, "model", "dropout");

    rms_norm1 = register_module("rms_norm1", RMSNorm(d_model, device));
    attention = register_module("attention", SelfAttention(device));
    dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
    rms_norm2 = register_module("rms_norm2", RMSNorm(d_model, device));
    ffn = register_module("ffn", FeedForwardNetworks(device));
    dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
  }

  std::array<torch::Tensor, 2> forward(std::array<torch::Tensor, 2> input)
  {
    auto[x, mask]  = input;

    auto rope_x = rope(x);

    auto norm_x1 = rms_norm1(rope_x);
    auto attn_out = attention(norm_x1, mask);
    auto out1 = dropout1(attn_out + rope_x);

    auto norm_x2 = rms_norm2(out1);
    auto ffn_out = ffn(norm_x2);
    auto out2 = dropout2(ffn_out + out1);

    return {out2, mask};
  }

  Rope rope = nullptr;
  RMSNorm rms_norm1 = nullptr;
  SelfAttention attention = nullptr;
  torch::nn::Dropout dropout1 = nullptr;
  RMSNorm rms_norm2 = nullptr;
  FeedForwardNetworks ffn = nullptr;
  torch::nn::Dropout dropout2 = nullptr;
};
TORCH_MODULE(DecoderLayer);

struct EmbeddingLayerImpl : torch::nn::Module
{
  EmbeddingLayerImpl(int vocab_size, torch::Device device)
      : device(device)
  {
    auto data = toml::parse("config.toml");
    auto d_model = toml::find<int64_t>(data, "model", "d_model");
    auto dropout_val = toml::find<float>(data, "model", "dropout");

    embedding = register_module("embedding", torch::nn::Embedding(
                                                 static_cast<int64_t>(vocab_size),
                                                 d_model));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_val));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    auto embedd = embedding(x);
    return dropout(embedd);
  }

  torch::Device device;
  torch::nn::Embedding embedding = nullptr;
  torch::nn::Dropout dropout = nullptr;
};
TORCH_MODULE(EmbeddingLayer);

struct DecoderImpl : torch::nn::Module
{
  DecoderImpl(int vocab_size, torch::Device device)
      : embedding(register_module("embedding", EmbeddingLayer(vocab_size, device))),
        layers(register_module("layers", torch::nn::Sequential()))
  {
    auto data = toml::parse("config.toml");
    auto d_model = toml::find<int>(data, "model", "d_model");
    auto num_layer = toml::find<int>(data, "model", "num_layer");

    rms_norm = register_module("rms_norm", RMSNorm(d_model, device));

    for (int i = 0; i < num_layer; i++)
    {
      layers->push_back(DecoderLayer(device));
    }
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask)
  {
    auto embedd = embedding(x);
    std::array<torch::Tensor, 2> in{embedd, mask};
    auto out1 = layers->forward<std::array<torch::Tensor, 2>>(in);
    auto out2 = rms_norm(out1[0]);
    return out2;
  }
  EmbeddingLayer embedding = nullptr;
  torch::nn::Sequential layers = nullptr;
  RMSNorm rms_norm = nullptr;
};
TORCH_MODULE(Decoder);

struct LLaMAImpl : torch::nn::Module
{
  LLaMAImpl(int vocab_size, torch::Device device)
      : device(device),
        decoder(register_module("decoder", Decoder(vocab_size, device)))
  {
    auto data = toml::parse("config.toml");
    auto d_model = toml::find<int>(data, "model", "d_model");
    Bc = toml::find<int>(data, "model", "Bc");
    num_head = toml::find<int>(data, "model", "num_head");
    pad_id = toml::find<int>(data, "train", "pad_id");
    linear = register_module("linear", torch::nn::Linear(d_model, vocab_size));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    auto mask = create_mask(x);
    auto dec_out = decoder(x, mask);
    return linear(dec_out);
  }

  torch::Tensor create_mask(torch::Tensor x)
  {
    auto padding_mask = get_padding_mask(x);
    return padding_mask;
  }

  torch::Tensor get_padding_mask(torch::Tensor x)
  {
    auto batch = x.size(0);
    auto len = x.size(1);
    auto padding_mask = torch::empty({batch, num_head, len}).to(device);

    auto Tc = len / Bc;

    for (int i = 0; i < batch; i++)
    {
      for (int j = 0; j < len; j++)
      {
        // 要素がすべてpaddingなら-inf、そうでないなら0
        padding_mask.index_put_({i, Slice(), j},
                                torch::where(torch::all(x.index({i, j}) == pad_id), -INFINITY, 0));
      }
    }

    return padding_mask.reshape({batch, num_head, Tc, Bc});
  }

  torch::Device device;
  Decoder decoder = nullptr;
  int Bc;
  int num_head;
  int pad_id;
  torch::nn::Linear linear = nullptr;
};
TORCH_MODULE(LLaMA);

PYBIND11_MODULE(llama, m)
{
  torch::python::bind_module<LLaMAImpl>(m, "LLaMA")
      .def(py::init<int, torch::Device>())
      .def("forward", &LLaMAImpl::forward);
}