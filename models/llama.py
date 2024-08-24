import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class LLaMA(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(LLaMA, self).__init__()

    self.device = device
    self.decoder = Decoder(vocab_size, Config.d_model)

    self.linear = nn.Linear(Config.d_model, vocab_size)

  def forward(self, x):
    dec_padding_mask, lookahead_mask = self._create_masks(x)
    dec_out = self.decoder(x, dec_padding_mask, lookahead_mask)
    return self.linear(dec_out)

  def _create_masks(self, x):
    dec_padding_mask = self._get_padding_mask(x)
    lookahead_mask = self._get_lookahead_mask(x)

    return dec_padding_mask, lookahead_mask

  def _get_lookahead_mask(self, x):
    batch, len = x.shape[0], x.shape[1]
    mask = torch.ones(batch, Config.num_head, len, len).tril().to(self.device)
    return torch.where(mask == 1, 0, torch.tensor(-float('inf')).to(self.device))

  def _get_padding_mask(self, x):
    batch, len = x.shape[0], x.shape[1]
    padding_mask = torch.empty(batch, Config.num_head, 1, len, dtype=bool).to(self.device)
    for i in range(batch):
      for j in range(len):
        # 要素がすべてpaddingならtrue、そうでないならfalse
        padding_mask[i, :, :, j] = torch.all(x[i, j] == Config.pad_id)
    return torch.where(padding_mask == 1, torch.tensor(-float('inf')).to(self.device), padding_mask)


class Embedding(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(Embedding, self).__init__()

    self.device = device
    self.embedding = nn.Embedding(vocab_size, Config.d_model).to(device)
    self.dropout = nn.Dropout(Config.dropout)

  def _positional_encoding(self, len):
    pos = torch.arange(len).to(self.device)
    i = torch.arange(Config.d_model).to(self.device)

    sin = torch.sin(torch.einsum('i,j->ij', pos, 10000 ** (-2 * i[0::2] / Config.d_model)))
    cos = torch.cos(torch.einsum('i,j->ij', pos, 10000 ** (-2 * i[1::2] / Config.d_model)))
    return torch.stack((sin, cos), dim=1).reshape(len, -1).unsqueeze(0)

  def forward(self, x):
    batch = x.shape[0]
    seq = x.shape[1]

    embedd = self.embedding(x)
    pos_enc = self._positional_encoding(seq).repeat(batch, 1, 1)

    return self.dropout(embedd + pos_enc)


class Decoder(nn.Module):
  def __init__(self, vocab_size, device) -> None:
    super(Decoder, self).__init__()

    self.embedding = Embedding(vocab_size, Config.d_model)
    self.layers = nn.Sequential(*[DecoderLayer(device) for _ in range(Config.num_layer)])

  def forward(self, x, pad_mask, lookahead_mask):
    embedd = self.embedding(x)
    out = self.layers((embedd, pad_mask, lookahead_mask))
    return out[0]


class DecoderLayer(nn.Module):
  def __init__(self, device) -> None:
    super(DecoderLayer, self).__init__()
    self.masked_block = SubLayerBlock(SelfAttention(device), device)
    self.ffn_block = SubLayerBlock(FeedForwardNetworks(device), device)

  def forward(self, input):
    x, pad_mask, lookahead_mask = input
    out1 = self.masked_block(x, x, lookahead_mask)
    out2 = self.ffn_block(out1, None, None)
    return (out2, pad_mask, lookahead_mask)


class SubLayerBlock(nn.Module):
  def __init__(self, sublayer, device) -> None:
    super(SubLayerBlock, self).__init__()

    self.sublayer = sublayer
    self.dropout = nn.Dropout(Config.dropout)
    self.layer_norm = nn.LayerNorm(Config.d_model).to(device)

  def forward(self, query, memory, mask):
    sublayer_out = self.sublayer(query, memory, mask)
    dropped_out = self.dropout(sublayer_out)
    norm_out = self.layer_norm(query + dropped_out)

    return norm_out


class SelfAttention(nn.Module):
  def __init__(self, device) -> None:
    super(SelfAttention, self).__init__()

    self.device = device

    self.W_Q = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_K = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_V = nn.Linear(Config.d_model, Config.d_model).to(device)
    self.W_O = nn.Linear(Config.d_model, Config.d_model).to(device)

  def forward(self, query, memory, mask):
    q = self._split_head(self.W_Q(query))
    k = self._split_head(self.W_K(memory))
    v = self._split_head(self.W_V(memory))

    attention = self._scaled_dot_product_attention(q, k, v, mask)

    concat_attention = self._concat_head(attention)

    return self.W_O(concat_attention)

  def _split_head(self, x):
    batch, len = x.shape[0], x.shape[1]
    return x.view(batch, len, Config.num_head, Config.head_dim).transpose(2, 1)

  def _scaled_dot_product_attention(self, q, k, v, mask):

    # batch * num_head * q_len * head_dim, batch * num_head * kv_len * head_dim
    # batch * num_head * q_len * kv_len
    qk = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(k.shape[-1]))

    # batch * num_head * q_len * kv_len, batch * num_head * kv_len * kv_len
    # batch * num_head * q_len * kv_len
    return F.softmax(qk + mask, dim=-1) @ v

  def _concat_head(self, x):
    batch, len = x.shape[0], x.shape[2]
    return x.transpose(2, 1).reshape(batch, len, -1)


class FeedForwardNetworks(nn.Module):
  def __init__(self, device) -> None:
    super(FeedForwardNetworks, self).__init__()
    self.linear1 = nn.Linear(Config.d_model, Config.d_ff).to(device)
    self.linear2 = nn.Linear(Config.d_ff, Config.d_model).to(device)

  def forward(self, query, memory, mask):
    out = F.relu(self.linear1(query))
    return self.linear2(torch.where(out > 0, out, 0))
