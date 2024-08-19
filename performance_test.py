
import time
import torch
from llama import LLaMA
import toml

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vocab_size = 20000

config = toml.load(open('config.toml'))
x = torch.randint(low=0, high=vocab_size, size=(config["train"]["batch_size"], 128)).to(device)
y = torch.randint(low=0, high=vocab_size, size=(config["train"]["batch_size"], 128)).to(device)

forward = 0
backward = 0

model = LLaMA(vocab_size, device)
model.to(device)

# for _ in range(10):
start = time.time_ns()

# with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=Config.use_amp):
out = model(x)
forward += time.time_ns() - start

loss = out.sum()

start = time.time_ns()
loss.backward()
backward += time.time_ns() - start

print('Forward: {:.3f} ms | Backward {:.3f} ms'.format(forward / 1e+6, backward / 1e+6))
