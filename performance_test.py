
import time
import torch

from config import Config
from models.llama import SelfAttention

device = "cuda" if torch.cuda.is_available() else "cpu"
self_attention = SelfAttention(device)

x = torch.randn(Config.batch_size, 128, Config.d_model).to(device)
y = torch.randn(Config.batch_size, 128, Config.d_model).to(device)

forward = 0
backward = 0
self_attention = SelfAttention(device)
# for _ in range(10):
start = time.time_ns()
out = self_attention(x)
forward += time.time_ns() - start

loss = out.sum()

start = time.time_ns()
loss.backward()
backward += time.time_ns() - start

print('Forward: {:.3f} ms | Backward {:.3f} ms'.format(forward / 1e+6, backward / 1e+6))
