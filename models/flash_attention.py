import torch

from config import FlashAttentionConfig


class FlashAttention(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q, K, V):
    N, d = Q.size()

    Bc = FlashAttentionConfig.Bc
    Br = FlashAttentionConfig.Br

    # 切り上げ除算
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    Q_list = Q.split(Br, dim=0)
    K_list = K.split(Bc, dim=0)
    V_list = V.split(Bc, dim=0)

    O = torch.empty(Tr, Br, d, dtype=torch.bfloat16).to(Q.device)
    L = torch.empty(Tr, Br, dtype=torch.bfloat16).to(Q.device)

    for i in range(Tr):
      Qi = Q_list[i]

      Oi_1 = torch.zeros(Br, d).to(Q.device)
      li_1 = torch.zeros(Br).to(Q.device)
      mij_1 = torch.empty(Br).to(Q.device)
      mij_1[:] = torch.tensor(-float('inf')).to(Q.device)

      for j in range(Tc):
        Kj = K_list[j]
        Vj = V_list[j]

        Sij = Qi @ Kj.transpose(1, 0)

        if j > 0:
          mij = torch.maximum(mij_1, torch.max(Sij, dim=1)[0])
          Pij = torch.exp(Sij - mij.unsqueeze(-1))
          lij = torch.exp(mij_1 - mij) * li_1 + torch.sum(Pij, dim=1)
          Oij = torch.linalg.inv(torch.diag(torch.exp(mij_1 - mij), 0)) @ Oi_1 + Pij @ Vj
        else:
          mij = torch.max(Sij, dim=1)[0]
          lij = torch.sum(torch.exp(Sij - mij.unsqueeze(-1)), dim=1)
          Oij = torch.exp(Sij - mij.unsqueeze(-1)) @ Vj

        Oi_1 = Oij
        li_1 = lij
        mij_1 = mij

      Oi = torch.linalg.inv(torch.diag(lij, 0)) @ Oij
      Li = mij + torch.log(lij)

      O[i] = Oi
      L[i] = Li

    ctx.save_for_backward(Q, K, V, O.reshape(-1, d), L.reshape(-1))
    return O.reshape(-1, d)

  @staticmethod
  def backward(ctx, dO):
    Q, K, V, O, L = ctx.saved_tensors
    N, d = Q.size()

    Bc = FlashAttentionConfig.Bc
    Br = FlashAttentionConfig.Br

    # 切り上げ除算
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    Q_list = torch.stack(Q.split(Br, dim=0), dim=0)
    K_list = torch.stack(K.split(Bc, dim=0), dim=0)
    V_list = torch.stack(V.split(Bc, dim=0), dim=0)

    dO_list = dO.split(Br, dim=0)
    L_list = L.split(Br, dim=0)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dQ_list = torch.stack(dQ.split(Br, dim=0), dim=0)
    dK_list = torch.stack(dK.split(Bc, dim=0), dim=0)
    dV_list = torch.stack(dV.split(Bc, dim=0), dim=0)

    D = torch.sum(dO * O, dim=1)
    D_list = D.split(Br, dim=0)

    for j in range(Tc):
      Kj = K_list[j]
      Vj = V_list[j]

      dKj = dK_list[j]
      dVj = dV_list[j]
      for i in range(Tr):
        Qi = Q_list[i]
        dOi = dO_list[i]
        dQi = dQ_list[i]
        Li = L_list[i]
        Di = D_list[i]

        Sij = Qi @ Kj.transpose(1, 0)
        Pij = torch.exp(Sij - Li.unsqueeze(-1))
        dVj = dVj + Pij.transpose(1, 0) @ dOi
        dPij = dOi @ Vj.transpose(1, 0)
        dSij = Pij * (dPij - Di.unsqueeze(-1))
        dQi = dQi + dSij @ Kj

        dQ_list[i] = dQi

        dKj = dKj + dSij.transpose(1, 0) @ Qi

      dK_list[j] = dKj
      dV_list[j] = dVj
    return dQ.reshape(N, d), dK.reshape(N, d), dV.reshape(N, d)
