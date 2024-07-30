#include <torch/extension.h>
#include <vector>
#include "../../config.h"


std::vector<at::Tensor> flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V) {

    auto N = Q.size(0);
    auto d = Q.size(1);

    auto Bc = Config::Bc;
    auto Br = Config::Br;

    // 切り上げ除算
    auto Tr = (N + Br - 1) / Br;
    auto Tc = (N + Bc - 1) / Bc;

    auto Q_list = Q.split(Br, /* dim= */0);
    auto K_list = K.split(Bc, /* dim= */0);
    auto V_list = V.split(Bc, /* dim= */0);

    auto O = torch::empty({Tr, Br, d}).to(Q.device());
    auto L = torch::empty({Tr, Br}).to(Q.device());

    float minus_inf = numeric_limits<float>::infinity() * -1;

    for (int i = 0; i < Tr; i++) {
      auto Qi = Q_list[i];

      auto Oi_1 = torch::zeros({Br, d}).to(Q.device());
      auto li_1 = torch::zeros({Br}).to(Q.device());
      auto mij_1 = torch::empty({Br}).to(Q.device());
      mij_1.index({torch::indexing::Slice()}) = torch::tensor(minus_inf).to(Q.device());

      torch::Tensor mij;
      torch::Tensor lij;
      torch::Tensor Oij;

      for (int j = 0; j < Tc; j++) {
        auto Kj = K_list[j];
        auto Vj = V_list[j];

        auto Sij = Qi.matmul(Kj.transpose(1, 0));

        if (j > 0) {
          mij = mij_1.maximum(std::get<0>(Sij.max(/* dim= */1)));
          auto Pij = at::exp(Sij - mij.unsqueeze(-1));
          lij = at::exp(mij_1 - mij) * li_1 + Pij.sum(/* dim= */1);
          Oij = torch::linalg::inv(at::diag(at::exp(mij_1 - mij), 0)).matmul(Oi_1) + Pij.matmul(Vj);
        } else {
          mij = std::get<0>(Sij.max(/* dim= */1));
          lij = at::exp(Sij - mij.unsqueeze(-1)).sum(/* dim= */1);
          Oij = at::exp(Sij - mij.unsqueeze(-1)).matmul(Vj);
        }

        Oi_1 = Oij;
        li_1 = lij;
        mij_1 = mij;
      }

      auto Oi = torch::linalg::inv(at::diag(lij, 0)).matmul(Oij);
      auto Li = mij + lij.log();

      O.index_put_({i}, Oi);
      L.index_put_({i}, Li);
    }

    return {O.reshape({-1, d}), Q, K, V, O.reshape({-1, d}), L.reshape({-1})};
}

std::vector<torch::Tensor> flash_attention_backward(
    torch::Tensor dO,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L
    ) {

    auto N = Q.size(0);
    auto d = Q.size(1);

    auto Bc = Config::Bc;
    auto Br = Config::Br;

    // 切り上げ除算
    auto Tr = (N + Br - 1) / Br;
    auto Tc = (N + Bc - 1) / Bc;

    auto Q_list = Q.split(Br, /* dim= */0);
    auto K_list = K.split(Bc, /* dim= */0);
    auto V_list = V.split(Bc, /* dim= */0);

    auto dO_list = dO.split(Br, /* dim= */0);
    auto L_list = L.split(Br, /* dim= */0);

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    auto dQ_list = torch::stack(dQ.split(Br, /* dim= */0), /* dim= */0);
    auto dK_list = torch::stack(dK.split(Bc, /* dim= */0), /* dim= */0);
    auto dV_list = torch::stack(dV.split(Bc, /* dim= */0), /* dim= */0);

    auto D = torch::sum(dO * O, /* dim= */1);
    auto D_list = D.split(Br, /* dim= */0);

    for (int j = 0; j < Tc; j++) {
      auto Kj = K_list[j];
      auto Vj = V_list[j];

      auto dKj = dK_list.index({j});
      auto dVj = dV_list.index({j});

      for (int i = 0; i < Tr; i++) {
        auto Qi = Q_list[i];
        auto dOi = dO_list[i];
        auto dQi = dQ_list.index({i});
        auto Li = L_list[i];
        auto Di = D_list[i];

        auto Sij = Qi.matmul(Kj.transpose(1, 0));
        auto Pij = at::exp(Sij - Li.unsqueeze(-1));
        dVj = dVj + Pij.transpose(1, 0).matmul(dOi);
        auto dPij = dOi.matmul(Vj.transpose(1, 0));
        auto dSij = Pij * (dPij - Di.unsqueeze(-1));
        dQi = dQi + dSij.matmul(Kj);

        dQ_list[i] = dQi;

        dKj = dKj + dSij.transpose(1, 0).matmul(Qi);
      }

      dK_list.index_put_({j}, dKj);
      dV_list.index_put_({j}, dVj);
    }
    return {dQ.reshape({N, d}), dK.reshape({N, d}), dV.reshape({N, d})};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def ("forward", &flash_attention_forward, "flash_attention forward");
  m.def ("backward", &flash_attention_backward, "flash_attention backward");
}
