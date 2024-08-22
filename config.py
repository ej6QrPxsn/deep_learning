
from typing import NamedTuple


class Config(NamedTuple):
  use_amp = True

  pad_id = 3

  d_model = 512
  d_ff = 2048
  num_layer = 16

  num_head = 16
  head_dim = d_model // num_head

  batch_size = 32
  accum_iter = 2

  adam_lr = 3e-4
  adam_betas = (0.9, 0.95)
  adam_weight_decay = 0.1
  warmup_steps = 2000
  lr_min = 3e-5

  dropout = 0.1
  label_smoothing = 0.1
  grad_clip = 1.0

  model_path = "model.pt"

  Bc = 32
  Br = 32
