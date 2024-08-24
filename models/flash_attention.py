import torch

from config import Config


class FlashAttention(torch.autograd.Function):
  def _get_lookahead_mask(self, batch, x, y):
    mask = torch.ones(batch, Config.num_head, x, y).tril().to(self.device)
    return torch.where(mask == 1, 0, torch.tensor(-float('inf')).to(self.device))

  @staticmethod
  def forward(ctx, Q, K, V, mask):
    B, N, hd = Q.size()

    Bc = Config.Bc
    Br = Config.Br

    # 切り上げ除算
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    Q_list = Q.reshape(B, Tr, Br, hd)
    K_list = K.reshape(B, Tc, Bc, hd)
    V_list = V.reshape(B, Tc, Bc, hd)
    mask_list = mask.reshape(B, Tc, Bc, 1)
    lookahead_mask = torch.ones(B, Br, Bc).tril().to(Q.device)

    O = torch.empty(B, Tr, Br, hd, dtype=Q.dtype).to(Q.device)
    L = torch.empty(B, Tr, Br, dtype=Q.dtype).to(Q.device)

    for i in range(Tr):
      Qi = Q_list[:, i]

      Oi_1 = torch.zeros(B, Br, hd, dtype=Q.dtype).to(Q.device)
      li_1 = torch.zeros(B, Br, dtype=Q.dtype).to(Q.device)
      mij_1 = torch.full((B, Br), torch.tensor(-float('inf')), dtype=Q.dtype).to(Q.device)

      for j in range(Tc):
        Kj = K_list[:, j]
        Vj = V_list[:, j]

        Sij = Qi @ Kj.transpose(-1, -2)

        mij = torch.maximum(mij_1, torch.max(Sij, dim=-1)[0])
        Pij = torch.exp(Sij - mij.unsqueeze(-1))  # * mask_list[:, j]) * lookahead_mask
        lij = torch.exp(mij_1 - mij) * li_1 + torch.sum(Pij, dim=-1)
        Oij = (1 / torch.exp(mij_1 - mij)).unsqueeze(-1) * Oi_1 + Pij @ Vj

        Oi_1 = Oij
        li_1 = lij
        mij_1 = mij

      Oi = (1 / lij.unsqueeze(-1)) * Oij
      Li = mij + torch.log(lij)

      O[:, i] = Oi
      L[:, i] = Li

    ctx.save_for_backward(Q, K, V, O.reshape(B, -1, hd), L.reshape(B, -1))
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

    Q_list = Q.reshape(B, Tr, Br, hd)
    K_list = K.reshape(B, Tc, Bc, hd)
    V_list = V.reshape(B, Tc, Bc, hd)

    dO_list = dO.reshape(B, Tr, Br, hd)
    L_list = L.reshape(B, Tr, Br)

    dQ_list = torch.zeros_like(Q_list)
    dK_list = torch.zeros_like(K_list)
    dV_list = torch.zeros_like(V_list)

    D = torch.sum(dO * O, dim=2)
    D_list = D.reshape(B, Tr, Br)

    for j in range(Tc):
      Kj = K_list[:, j]
      Vj = V_list[:, j]

      dKj = dK_list[:, j]
      dVj = dV_list[:, j]

      for i in range(Tr):
        Qi = Q_list[:, i]
        dOi = dO_list[:, i]
        dQi = dQ_list[:, i]
        Li = L_list[:, i]
        Di = D_list[:, i]

        Sij = Qi @ Kj.transpose(-1, -2)
        Pij = torch.exp(Sij - Li.unsqueeze(-1))
        dVj = dVj + Pij.transpose(-1, -2) @ dOi
        dPij = dOi @ Vj.transpose(-1, -2)
        dSij = Pij * (dPij - Di.unsqueeze(-1))
        dQi = dQi + dSij @ Kj

        dQ_list[:, i] = dQi

        dKj = dKj + dSij.transpose(-1, -2) @ Qi

      dK_list[:, j] = dKj
      dV_list[:, j] = dVj
    return dQ_list.reshape(B, N, hd), dK_list.reshape(B, N, hd), dV_list.reshape(B, N, hd), None
