import time

import torch

from config import Config
from models.flash_attention import FlashAttention
from models.llama import SelfAttention

device = "cuda" if torch.cuda.is_available() else "cpu"
self_attention = SelfAttention(device)

x = torch.randn(Config.batch_size, 128, Config.d_model).to(device)
y = torch.randn(Config.batch_size, 128, Config.d_model).to(device)

forward = 0
backward = 0
for _ in range(10):
  start = time.time()
  with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=Config.use_amp):
    out = self_attention(x)
    forward += time.time() - start

    loss = out.sum()

  start = time.time()
  loss.backward()
  backward += time.time() - start

print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))
