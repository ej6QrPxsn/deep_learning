import torch
from config import Config


def diag(x):
  return x.unsqueeze(-1).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[2]).tril().triu()


def dropout(x, rng, p):
  Zij = torch.where(torch.rand(x.shape, generator=rng, device=x.device) > p, 1 / (1 - p), 0).to(x.dtype)
  return x * Zij, Zij


class FlashAttentionFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q, K, V, mask):
    B, N, hd = Q.size()

    Bc = Config.Bc
    Br = Config.Br

    # 切り上げ除算
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    Qi = Q.reshape(B, Tr, Br, hd)
    KT_list = K.reshape(B, Tc, 1, Bc, hd).transpose(-1, -2)
    V_list = V.reshape(B, Tc, 1, Bc, hd)
    mask_list = mask.reshape(B, Tc, 1, 1, Bc)
    lookahead_mask = torch.ones(B, Br, Bc).tril().to(Q.device)
    lookahead_mask = torch.where(lookahead_mask == 0, -float('inf'), 0)

    O = torch.empty(B, Tr, Br, hd, dtype=Q.dtype).to(Q.device)
    L = torch.empty(B, Tr, Br, dtype=Q.dtype).to(Q.device)

    Oi_1 = torch.zeros(B, Tr, Br, hd, dtype=Q.dtype).to(Q.device)
    li_1 = torch.zeros(B, Tr, Br, dtype=Q.dtype).to(Q.device)
    mij_1 = torch.full((B, Tr, Br), torch.tensor(-float('inf')), dtype=Q.dtype).to(Q.device)

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
      Sij_masked[:, :j] += -float('inf')

      mij = torch.maximum(mij_1, torch.max(Sij_masked, dim=-1)[0])
      Pij = torch.exp(Sij_masked - mij.unsqueeze(-1))

      lij = torch.exp(mij_1 - mij) * li_1 + torch.sum(Pij, dim=-1)

      Pij_dropped, _ = dropout(Pij, rng, P_drop)
      Oij = torch.linalg.pinv(diag(torch.exp(mij_1 - mij))) @ Oi_1 + Pij_dropped @ Vj

      Oi_1 = Oij
      li_1 = lij
      mij_1 = mij

    O[:] = torch.linalg.pinv(diag(lij)) @ Oij
    L[:] = mij + torch.log(lij)

    ctx.save_for_backward(Q, K, V, O.reshape(B, N, hd), L.reshape(B, N), mask, R)
    return O.reshape(B, N, hd)

  @ staticmethod
  def backward(ctx, dO):
    Q, K, V, O, L, mask, R = ctx.saved_tensors
    B, N, hd = Q.size()

    Bc = Config.Bc
    Br = Config.Br

    # 切り上げ除算
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    D = torch.sum(dO * O, dim=2)
    D_list = D.reshape(B, Tr, 1, Br)

    Q_list = Q.reshape(B, Tr, 1, Br, hd)
    Kj = K.reshape(B, Tc, Bc, hd)
    Vj = V.reshape(B, Tc, Bc, hd)

    dO_list = dO.reshape(B, Tr, 1, Br, hd)
    L_list = L.reshape(B, Tr, 1, Br)

    dQ_list = torch.zeros_like(Q_list)
    dKj = torch.zeros_like(Kj)
    dVj = torch.zeros_like(Vj)

    softmax_scaling = Config.softmax_scaling
    P_drop = Config.dropout_probabilty

    mask_list = mask.reshape(B, Tc, 1, Bc).to(Q.dtype)
    lookahead_mask = torch.ones(B, Br, Bc).tril().to(Q.dtype).to(Q.device)

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
      Sij_masked[:, i:] += -float('inf')

      Pij = torch.exp(Sij_masked - Li.unsqueeze(-1))

      Pij_dropped, Zij = dropout(Pij, rng, P_drop)

      dVj = dVj + Pij_dropped.transpose(-1, -2) @ dOi
      dPij_dropped = dOi @ Vj.transpose(-1, -2)
      dPij = dPij_dropped * Zij

      dSij = Pij * (dPij - Di.unsqueeze(-1))

      dQi = dQi + softmax_scaling * dSij @ Kj

      dQ_list[:, i] = dQi.sum(1).unsqueeze(1)

      dKj = dKj + softmax_scaling * dSij.transpose(-1, -2) @ Qi

    return dQ_list.reshape(B, N, hd), dKj.reshape(B, N, hd), dVj.reshape(B, N, hd), None
