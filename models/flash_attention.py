import torch
from line_profiler import profile

from deep_learning.config import Config


def diag(x):
  return x.unsqueeze(-1).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[2]).tril().triu()


def dropout(x, rng, p):
  Zij = torch.where(torch.rand(x.shape, generator=rng, device=x.device) > p, 1 / (1 - p), 0).to(x.dtype)
  return x * Zij, Zij


class FlashAttentionFunction(torch.autograd.Function):
  @staticmethod
  # @profile
  def forward(ctx, Q, K, V, mask):
    B, N, hd = Q.size()

    Br = Config.Br
    Bc = Config.Bc
    # 切り上げ除算
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    if N < Config.Br and N % Config.Br != 0:
      Tr = 1
      Br = N

    if N < Config.Bc and N % Config.Bc != 0:
      Tc = 1
      Bc = N

    Qi = Q.reshape(B, Tr, Br, hd)
    KT_list = K.reshape(B, Tc, 1, Bc, hd).transpose(-1, -2)
    V_list = V.reshape(B, Tc, 1, Bc, hd)
    mask_list = mask.reshape(B, Tc, 1, 1, Bc)
    lookahead_mask = torch.full((B, Br, Bc), torch.tensor(-float('inf')), device=Q.device).triu(diagonal=1)

    O = torch.empty(B, Tr, Br, hd, dtype=Q.dtype, device=Q.device)
    L = torch.empty(B, Tr, Br, dtype=Q.dtype, device=Q.device)

    Oij_1 = torch.zeros(B, Tr, Br, hd, dtype=Q.dtype, device=Q.device)
    lij_1 = torch.zeros(B, Tr, Br, dtype=Q.dtype, device=Q.device)
    mij_1 = torch.full((B, Tr, Br), torch.tensor(-float('inf')), dtype=Q.dtype, device=Q.device)

    minus_inf = torch.tensor(-float('inf'), dtype=Q.dtype, device=Q.device)

    rng = torch.Generator(Q.device)
    R = rng.get_state()

    softmax_scaling = Config.softmax_scaling
    P_drop = Config.dropout_probabilty

    for j in range(Tc):
      KjT = KT_list[:, j]
      Vj = V_list[:, j]

      Sij = softmax_scaling * Qi @ KjT
      Sij_masked = Sij + mask_list[:, j]

      # ブロックが正方形の場合、行ブロックと同じ列ブロックのみ、マスクが必要になる
      Sij_masked[:, j] += lookahead_mask
      # 列より小さい行ブロックはすべて-inf
      Sij_masked[:, :j] += minus_inf

      mij = torch.maximum(mij_1, torch.max(Sij_masked, dim=-1)[0])
      Pij = torch.exp(Sij_masked - mij.unsqueeze(-1))

      lij = torch.exp(mij_1 - mij) * lij_1 + torch.sum(Pij, dim=-1)

      Pij_dropped, _ = dropout(Pij, rng, P_drop)
      mij_exp = torch.exp(mij_1 - mij)
      Oij = torch.where(mij_exp == 0, 0, 1 / mij_exp).unsqueeze(-1) * Oij_1 + Pij_dropped @ Vj

      Oij_1[:] = Oij
      lij_1[:] = lij
      mij_1[:] = mij

    O[:] = torch.where(lij == 0, 0, 1 / lij).unsqueeze(-1) * Oij
    L[:] = mij + torch.log(lij)

    ctx.save_for_backward(Q, K, V, O.reshape(B, -1, hd), L.reshape(B, -1), mask, R)
    return O.reshape(B, -1, hd)[:, :N]

  @ staticmethod
  def backward(ctx, dO):
    Q, K, V, O, L, mask, R = ctx.saved_tensors
    B, N, hd = Q.size()

    Br = Config.Br
    Bc = Config.Bc
    # 切り上げ除算
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    if N < Config.Br and N % Config.Br != 0:
      Tr = 1
      Br = N

    if N < Config.Bc and N % Config.Bc != 0:
      Tc = 1
      Bc = N

    D = torch.sum(dO * O, dim=2)
    D_list = D.reshape(B, Tr, 1, Br).to(Q.dtype)

    Q_list = Q.reshape(B, Tr, 1, Br, hd)
    Kj = K.reshape(B, Tc, Bc, hd)
    Vj = V.reshape(B, Tc, Bc, hd)

    dO_list = dO.reshape(B, Tr, 1, Br, hd).to(Q.dtype)
    L_list = L.reshape(B, Tr, 1, Br).to(Q.dtype)

    dQ_list = torch.zeros_like(Q_list)
    dKj = torch.zeros_like(Kj)
    dVj = torch.zeros_like(Vj)

    softmax_scaling = Config.softmax_scaling
    P_drop = Config.dropout_probabilty

    mask_list = mask.reshape(B, Tc, 1, Bc).to(Q.dtype)
    lookahead_mask = torch.full((B, Br, Bc), torch.tensor(-float('inf')), device=Q.device).triu(diagonal=1)
    minus_inf = torch.tensor(-float('inf'), dtype=Q.dtype, device=Q.device)

    rng = torch.Generator(Q.device)
    rng.set_state(R)

    for i in range(Tr):
      Qi = Q_list[:, i]
      dOi = dO_list[:, i]
      dQi = dQ_list[:, i]
      Li = L_list[:, i]
      Di = D_list[:, i]

      Sij = softmax_scaling * Qi @ Kj.transpose(-1, -2)
      Sij_masked = Sij + mask_list

      # ブロックが正方形の場合、行ブロックと同じ列ブロックのみ、マスクが必要になる
      Sij_masked[:, i] += lookahead_mask
      # 行より大きい列ブロックはすべて-inf
      Sij_masked[:, i:] += minus_inf

      Pij = torch.exp(Sij_masked - Li.unsqueeze(-1))

      Pij_dropped, Zij = dropout(Pij, rng, P_drop)

      dVj = dVj + Pij_dropped.transpose(-1, -2) @ dOi
      dPij_dropped = dOi @ Vj.transpose(-1, -2)
      dPij = dPij_dropped * Zij

      dSij = Pij * (dPij - Di.unsqueeze(-1))

      dQi = dQi + softmax_scaling * dSij @ Kj

      dQ_list[:, i] = dQi.sum(1).unsqueeze(1)

      dKj = dKj + softmax_scaling * dSij.transpose(-1, -2) @ Qi

    return (dQ_list.reshape(B, -1, hd)[:, :N],
            dKj.reshape(B, -1, hd)[:, :N],
            dVj.reshape(B, -1, hd)[:, :N],
            None
            )
