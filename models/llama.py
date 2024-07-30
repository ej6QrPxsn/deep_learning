import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.flash_attention import FlashAttention


class LLaMA(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(LLaMA, self).__init__()

    self.device = device
    self.decoder = Decoder(vocab_size, Config.d_model)

    self.linear = nn.Linear(Config.d_model, vocab_size)

  def forward(self, x):
    dec_out = self.decoder(x)
    return self.linear(dec_out)


class Embedding(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(Embedding, self).__init__()

    self.device = device
    self.embedding = nn.Embedding(vocab_size, Config.d_model).to(device)
    self.dropout = nn.Dropout(Config.dropout)

  def forward(self, x):
    embedd = self.embedding(x)

    return self.dropout(embedd)


class Decoder(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(Decoder, self).__init__()

    self.embedding = Embedding(vocab_size, Config.d_model)
    self.layers = nn.Sequential(*[DecoderLayer(device) for _ in range(Config.num_layer)])
    self.rms_norm = RMSNorm(Config.d_model, device).to(device)

  def forward(self, x):
    embedd = self.embedding(x)
    out1 = self.layers(embedd)
    out2 = self.rms_norm(out1)
    return out2


class DecoderLayer(nn.Module):
  def __init__(self, device) -> None:
    super(DecoderLayer, self).__init__()
    self.rms_norm1 = RMSNorm(Config.d_model, device).to(device)
    self.attention = SelfAttention(device)
    self.dropout1 = nn.Dropout(Config.dropout)

    self.rms_norm2 = RMSNorm(Config.d_model, device).to(device)
    self.ffn = FeedForwardNetworks(device)
    self.dropout2 = nn.Dropout(Config.dropout)

    self.rope = Rope(device)

  def forward(self, input):
    x = input

    rope_x = self.rope(x)

    norm_x1 = self.rms_norm1(rope_x)
    attn_out = self.attention(norm_x1)
    out1 = self.dropout1(attn_out + rope_x)

    norm_x2 = self.rms_norm2(out1)
    ffn_out = self.ffn(norm_x2)
    out2 = self.dropout2(ffn_out + out1)

    return out2


class SelfAttention(nn.Module):
  def __init__(self, device) -> None:
    super(SelfAttention, self).__init__()

    self.device = device

    self.W_Q = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_K = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_V = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_O = nn.Linear(Config.d_model, Config.d_model).to(device)

    self.flash_attention = FlashAttention.apply

  def forward(self, x):
    batch, len, dim = x.shape
    q = self._split_head(self.W_Q(x))
    k = self._split_head(self.W_K(x))
    v = self._split_head(self.W_V(x))

    attention = torch.empty_like(q)

    for i in range(q.shape[0]):
      for j in range(q.shape[1]):
        attention[i, j] = self.flash_attention(q[i, j], k[i, j], v[i, j])

    return self.W_O(attention.reshape(batch, len, -1))

  def _split_head(self, x):
    batch, len = x.shape[0], x.shape[1]
    return x.view(batch, len, Config.num_head, Config.head_dim).transpose(2, 1)


class FeedForwardNetworks(nn.Module):
  def __init__(self, device) -> None:
    super(FeedForwardNetworks, self).__init__()
    self.linear1 = nn.Linear(Config.d_model, Config.d_ff).to(device)
    self.linear2 = nn.Linear(Config.d_ff // 2, Config.d_model).to(device)
    self.swi_glu = SwiGLU()

  def forward(self, x):
    out = self.swi_glu(self.linear1(x))
    return self.linear2(torch.where(out > 0, out, 0))


class Rope(nn.Module):
  def __init__(self, device) -> None:
    super(Rope, self).__init__()

    self.device = device

  def _rot(self, x):
    batch, len = x.shape[0], x.shape[1]

    second = x[:, :, 0::2]
    first = x[:, :, 1::2] * -1
    new_x = torch.stack((first, second), dim=2)
    return new_x.reshape(batch, len, -1)

  def forward(self, x):
    len = x.shape[1]

    half_d = Config.d_model // 2

    m = torch.arange(1, len + 1).to(self.device)
    i = torch.arange(1, half_d + 1).to(self.device)
    theta = (10000 ** (-2 * i / Config.d_model))

    cos = torch.cos(torch.einsum('i,j->ij', m, theta))
    sin = torch.sin(torch.einsum('i,j->ij', m, theta))

    C = torch.stack((cos, cos), dim=1).reshape(1, len, -1)
    S = torch.stack((sin, sin), dim=1).reshape(1, len, -1)

    return x * C + self._rot(x) * S


class RMSNorm(nn.Module):
  """Root Mean Square Layer Normalization.

  Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
  https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
  """

  def __init__(self, size: int, device, dim: int = -1, eps: float = 1e-5) -> None:
    super().__init__()
    self.scale = nn.Parameter(torch.ones(size))
    self.eps = eps
    self.dim = dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # NOTE: the original RMSNorm paper implementation is not equivalent
    # norm_x = x.norm(2, dim=self.dim, keepdim=True)
    # rms_x = norm_x * d_x ** (-1. / 2)
    # x_normed = x / (rms_x + self.eps)
    norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
    x_normed = x * torch.rsqrt(norm_x + self.eps)
    return self.scale * x_normed


class SwiGLU(nn.Module):
  def forward(self, x):
    a, b = x.chunk(2, dim=-1)
    return a * F.silu(b)
