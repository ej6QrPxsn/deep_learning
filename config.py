
from typing import NamedTuple


class Config(NamedTuple):
  use_amp = True

  pad_id = 3

  d_model = 512
  d_ff = 2 // 3 * 4 * d_model
  num_layer = 8

  num_head = 8
  head_dim = d_model // num_head

  batch_size = 64
  accum_iter = 1

  adam_lr = 1e-9
  adam_betas = (0.9, 0.98)
  adam_weight_decay = 0.01
  warmup_steps = 4000

  dropout = 0.1
  label_smoothing = 0.1

  model_path = "model.pt"
