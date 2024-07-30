import torch

import flash_attention_cpp


class FlashAttention(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q, K, V):
    out = flash_attention_cpp.forward(Q, K, V)
    ctx.save_for_backward(*out[1:])
    return out[0]

  @staticmethod
  def backward(ctx, dO):
    Q, K, V, O, L = ctx.saved_tensors
    out = flash_attention_cpp.backward(dO, Q.to(torch.float), K.to(torch.float), V.to(torch.float), O, L)
    dQ, dK, dV = out
    return dQ, dK, dV
