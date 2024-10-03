
import math
from typing import NamedTuple


class Config(NamedTuple):
  use_amp = True

  pad_id = 0

  d_model = 128
  d_ff = math.ceil(2 / 3 * 4 * d_model)
  num_layer = 16

  num_head = 16
  head_dim = d_model // num_head

  batch_size = 128
  accum_iter = 1

  adam_lr = 3e-5
  adam_eps = 1e-4
  scheduler_lr_min = 3e-7
  scheduler_lr_init = 3e-7
  adam_betas = (0.9, 0.95)
  adam_weight_decay = 0.1
  warmup_steps = 2000
  clip_value = 1.0 / accum_iter

  dropout = 0.1
  label_smoothing = 0.1
  # nan回避のため、16ビットで表せる最小の数6.1e-5より大きい値を設定する
  rms_norm_eps = 1e-4

  pre_train_model_path = "pre_train_model.pt"
  model_path = "model.pt"

  Br = 50
  Bc = 50
  softmax_scaling = 1 / math.sqrt(d_model)
  dropout_probabilty = 0.1
