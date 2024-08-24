import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.flash_attention import FlashAttentionFunction


class LLaMA(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(LLaMA, self).__init__()

    self.device = device
    self.decoder = Decoder(vocab_size, Config.d_model)

    self.linear = nn.Linear(Config.d_model, vocab_size)

  def forward(self, x):
    mask = self._create_masks(x)
    dec_out = self.decoder(x, mask)
    return self.linear(dec_out)

  def _create_masks(self, x):
    dec_padding_mask = self._get_padding_mask(x)

    return dec_padding_mask

  def _get_padding_mask(self, x):
    batch, len = x.shape[0], x.shape[1]
    padding_mask = torch.empty(batch, Config.num_head, len).to(self.device)
    Tc = (len + Config.Bc - 1) // Config.Bc

    for i in range(batch):
      for j in range(len):
        # 要素がすべてpaddingなら0、そうでないなら1
        padding_mask[i, :, j] = torch.where(torch.all(x[i, j] == Config.pad_id), 0, 1)
    return padding_mask.reshape(batch, Config.num_head, Tc, Config.Bc)


class Embedding(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(Embedding, self).__init__()

    self.device = device
    self.embedding = nn.Embedding(vocab_size, Config.d_model)
    self.dropout = nn.Dropout(Config.dropout)

  def forward(self, x):
    embedd = self.embedding(x)
    return self.dropout(embedd)


class Decoder(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(Decoder, self).__init__()

    self.embedding = Embedding(vocab_size, Config.d_model)
    self.layers = nn.Sequential(*[DecoderLayer(device) for _ in range(Config.num_layer)])

  def forward(self, x, mask):
    embedd = self.embedding(x)
    out = self.layers((embedd, mask))
    return out[0]


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


class DecoderLayer(nn.Module):
  def __init__(self, device) -> None:
    super(DecoderLayer, self).__init__()
    self.rope = Rope(device)
    self.attention_block = SubLayerBlock(SelfAttention(device))
    self.ffn_block = SubLayerBlock(FeedForwardNetworks())

  def forward(self, input):
    x, mask = input
    x_with_pos = self.rope(x)
    out1 = self.attention_block(x_with_pos, mask)
    out2 = self.ffn_block(out1, None)
    return (out2, mask)


class SubLayerBlock(nn.Module):
  def __init__(self, sublayer) -> None:
    super(SubLayerBlock, self).__init__()

    self.rms_norm = RMSNorm()
    self.sublayer = sublayer
    self.dropout = nn.Dropout(Config.dropout)

  def forward(self, x, mask):
    norm_out = self.rms_norm(x)
    sublayer_out = self.sublayer(norm_out, mask)
    dropped_out = self.dropout(sublayer_out)

    return x + dropped_out


class SelfAttention(nn.Module):
  def __init__(self, device) -> None:
    super(SelfAttention, self).__init__()

    self.device = device

    self.W_Q = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_K = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_V = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_O = nn.Linear(Config.d_model, Config.d_model).to(device)

    self.flash_attention = FlashAttentionFunction.apply

  def forward(self, x, mask):
    q = self._split_head(self.W_Q(x))
    k = self._split_head(self.W_K(x))
    v = self._split_head(self.W_V(x))

    attention = self.flash_attention(q, k, v, mask)

    return self.W_O(self._concat_head(attention))

  def _split_head(self, x):
    batch, len = x.shape[0], x.shape[1]
    return x.view(batch, len, Config.num_head, Config.head_dim).transpose(2, 1).reshape(-1, len, Config.head_dim)

  def _concat_head(self, x):
    _, len, hd = x.shape
    return x.reshape(-1, Config.num_head, len, hd).transpose(2, 1).reshape(-1, len, Config.num_head * hd)


class FeedForwardNetworks(nn.Module):
  def __init__(self) -> None:
    super(FeedForwardNetworks, self).__init__()
    self.linear1 = nn.Linear(Config.d_model, Config.d_ff)
    self.swi_glu = SwiGLU()
    self.linear2 = nn.Linear(Config.d_ff, Config.d_model)

  def forward(self, x, mask):
    out = self.swi_glu(self.linear1(x))
    return self.linear2(torch.where(out > 0, out, 0))


class RMSNorm(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, x):
    x_norm = 1 / x.pow(2).mean(dim=-1).sqrt()
    return x * x_norm.unsqueeze(-1)


class SwiGLU(nn.Module):
  def __init__(self) -> None:
    super(SwiGLU, self).__init__()
    self.linear = nn.Linear(Config.d_ff, Config.d_ff * 2)

  def forward(self, x):
    a, b = self.linear(x).chunk(2, dim=-1)
    return a * F.silu(b)
