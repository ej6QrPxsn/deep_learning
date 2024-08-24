import torch
from config import Config


def diag(x):
  return x.unsqueeze(-1).expand(x.shape[0], x.shape[1], x.shape[2], x.shape[2]).tril().triu()


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
    KT_list = K.reshape(B, 1, Tc, Bc, hd).transpose(-1, -2)
    V_list = V.reshape(B, 1, Tc, Bc, hd)
    mask_list = mask.reshape(B, 1, Tc, Bc, 1)
    lookahead_mask = torch.ones(B, Tr, Br, Bc).tril().to(Q.device)

    O = torch.empty(B, Tr, Br, hd, dtype=Q.dtype).to(Q.device)
    L = torch.empty(B, Tr, Br, dtype=Q.dtype).to(Q.device)

    Oi_1 = torch.zeros(B, Tr, Br, hd, dtype=Q.dtype).to(Q.device)
    li_1 = torch.zeros(B, Tr, Br, dtype=Q.dtype).to(Q.device)
    mij_1 = torch.full((B, Tr, Br), torch.tensor(-float('inf')), dtype=Q.dtype).to(Q.device)

    for j in range(Tc):
      KjT = KT_list[:, :, j]
      Vj = V_list[:, :, j]

      Sij = Qi @ KjT

      mij = torch.maximum(mij_1, torch.max(Sij, dim=-1)[0])
      Pij = (torch.exp(Sij - mij.unsqueeze(-1)) * mask_list[:, :, j]) * lookahead_mask
      lij = torch.exp(mij_1 - mij) * li_1 + torch.sum(Pij, dim=-1)
      Oij = torch.linalg.pinv(diag(torch.exp(mij_1 - mij))) @ Oi_1 + Pij @ Vj

      Oi_1 = Oij
      li_1 = lij
      mij_1 = mij

    O[:] = torch.linalg.pinv(diag(lij)) @ Oij
    L[:] = mij + torch.log(lij)

    ctx.save_for_backward(Q, K, V, O.reshape(B, -1, hd), L.reshape(B, N))
    return O.reshape(B, -1, hd)

  @ staticmethod
  def backward(ctx, dO):
    Q, K, V, O, L = ctx.saved_tensors
    B, N, hd = Q.size()

    Bc = Config.Bc
    Br = Config.Br

    # 切り上げ除算
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    D = torch.sum(dO * O, dim=2)
    D_list = D.reshape(B, 1, Tr, Br)

    Q_list = Q.reshape(B, 1, Tr, Br, hd)
    Kj = K.reshape(B, Tc, Bc, hd)
    Vj = V.reshape(B, Tc, Bc, hd)

    dO_list = dO.reshape(B, 1, Tr, Br, hd)
    L_list = L.reshape(B, 1, Tr, Br)

    dQ_list = torch.zeros_like(Q_list)
    dKj = torch.zeros_like(Kj)
    dVj = torch.zeros_like(Vj)

    for i in range(Tr):
      Qi = Q_list[:, :, i]
      dOi = dO_list[:, :, i]
      dQi = dQ_list[:, :, i]
      Li = L_list[:, :, i]
      Di = D_list[:, :, i]

      Sij = Qi @ Kj.transpose(-1, -2)
      Pij = torch.exp(Sij - Li.unsqueeze(-1))
      dVj = dVj + Pij.transpose(-1, -2) @ dOi
      dPij = dOi @ Vj.transpose(-1, -2)
      dSij = Pij * (dPij - Di.unsqueeze(-1))
      dQi = dQi + dSij @ Kj

      dQ_list[:, :, i] = dQi.sum(1).unsqueeze(1)

      dKj = dKj + dSij.transpose(-1, -2) @ Qi

    return dQ_list.reshape(B, N, hd), dKj.reshape(B, N, hd), dVj.reshape(B, N, hd), None
