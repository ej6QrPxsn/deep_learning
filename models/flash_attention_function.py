import torch

import flash_attention


class FlashAttentionFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q, K, V):
    O, L = flash_attention.forward(
      Q.contiguous(),
      K.contiguous(),
      V.contiguous())
    ctx.save_for_backward(Q, K, V, O, L)
    return O

  @staticmethod
  def backward(ctx, dO):
    Q, K, V, O, L = ctx.saved_tensors
    out = flash_attention.backward(
      dO,
      Q.contiguous(),
      K.contiguous(),
      V.contiguous(),
      O.contiguous(),
      L.contiguous())
    dQ, dK, dV = out
    return dQ, dK, dV
