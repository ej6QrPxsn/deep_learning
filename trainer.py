
import datetime
import io
import torch

from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import CosineLRScheduler

import os
import sys

from models.llama import LLaMA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, FlashAttentionConfig
import torch.nn.utils.rnn as rnn
from datasets import load_dataset
import sentencepiece as spm
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def build_vocab(data, vocab_size):
  spm_model = io.BytesIO()
  spm.SentencePieceTrainer.train(
    sentence_iterator=iter(data),
    model_writer=spm_model,
    vocab_size=vocab_size,
    pad_id=Config.pad_id,
    user_defined_symbols=['<SEP>']
  )

  return spm.SentencePieceProcessor(model_proto=spm_model.getvalue())


def pad_batch_fn(batch):
  inputs = []
  labels = []
  max_len = 0
  for i, item in enumerate(batch):
    if len(item[0]) > max_len:
      max_len = len(item[0])
    inputs.append(item[0])
    labels.append(item[1])

  rest_len = max_len % FlashAttentionConfig.Br
  if rest_len != 0:
    max_len += FlashAttentionConfig.Br - rest_len

  return (
    rnn.pad_packed_sequence(
      rnn.pack_sequence(inputs, enforce_sorted=False),
      batch_first=True, padding_value=Config.pad_id, total_length=max_len)[0],
    rnn.pad_packed_sequence(
      rnn.pack_sequence(labels, enforce_sorted=False),
      batch_first=True, padding_value=Config.pad_id, total_length=max_len)[0]
  )


class CustomDataSet(torch.utils.data.Dataset):
  def __init__(self, vocab, src_data, dst_data):
    super(CustomDataSet, self).__init__()

    self.vocab = vocab

    self.src_lines = src_data
    self.dst_lines = dst_data

    self.sep_id = self.vocab.piece_to_id('<SEP>')

  def __len__(self):
    return len(self.src_lines)

  def __getitem__(self, index):
    inputs = torch.Tensor(
      [self.vocab.bos_id()]
      + self.vocab.encode(self.src_lines[index])
      + [self.sep_id]
      + self.vocab.encode(self.dst_lines[index])
    )

    outputs = torch.Tensor(
      self.vocab.encode(self.src_lines[index])
      + [self.sep_id]
      + self.vocab.encode(self.dst_lines[index])
      + [self.vocab.eos_id()]
    )

    return (inputs, outputs)


class ValidateDataSet(CustomDataSet):
  def __init__(self, vocab, src_data, dst_data):
    super(ValidateDataSet, self).__init__(vocab, src_data, dst_data)

  def __len__(self):
    return super(ValidateDataSet, self).__len__()

  def __getitem__(self, index):
    inputs = torch.Tensor(
      [self.vocab.bos_id()]
      + self.vocab.encode(self.src_lines[index])
    )

    outputs = torch.Tensor(
      self.vocab.encode(self.dst_lines[index])
      + [self.vocab.eos_id()]
    )

    return (inputs, outputs)

  def get_tokens(self, indicies):
    return self.vocab.decode(indicies)


class TransformerScheduler:
  def __init__(self, optimizer):
    self.optimizer = optimizer
    self.step_num = 0

  def step(self):
    self.step_num += 1
    lr = Config.d_model ** -0.5 * min(self.step_num ** -0.5, self.step_num * Config.warmup_steps ** -1.5)

    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr


class Trainer:
  def __init__(self) -> None:
    datasets = load_dataset("davidstap/ted_talks", "ja_en")

    src_dataset = datasets["train"]["en"]
    dst_dataset = datasets["train"]["ja"]

    self.vocab_size = 20000
    self.vocab = build_vocab(src_dataset + dst_dataset, self.vocab_size)

    train_dataset = CustomDataSet(
      self.vocab,
      src_dataset,
      dst_dataset,
    )

    self.train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=pad_batch_fn)

    self.validation_dataset = ValidateDataSet(
      self.vocab,
      datasets["validation"]["en"],
      datasets["validation"]["ja"]
    )

    self.validation_data_loader = torch.utils.data.DataLoader(
      self.validation_dataset, batch_size=1, shuffle=True,
      num_workers=2, pin_memory=True, collate_fn=pad_batch_fn)

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.model = LLaMA(self.vocab_size, self.device).to(self.device)

    self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing, reduction="none")
    self.scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)

    self.writer = SummaryWriter(log_dir="./logs")

    if os.path.exists(Config.model_path):
      self.model.load_state_dict(torch.load(Config.model_path))

  def _compute_loss(self, label, output):
    logits = self.criterion(
      output.view(-1, self.vocab_size),
      label.reshape(-1).to(torch.long).to(self.device))

    # パディングマスク
    mask = torch.where(label == Config.pad_id, 0, 1).to(self.device)
    logits = logits.reshape(*label.shape)
    logits *= mask

    # 長さ合計
    logits = torch.sum(logits, dim=-1) / torch.sum(mask, dim=-1)
    # バッチ平均
    return torch.mean(logits)

  def _compute_accuracy(self, label, output):
    indecies = torch.argmax(output, dim=-1)
    accuracies = torch.where(label.to(self.device) == indecies, 1, 0)

    # パディングマスク
    mask = torch.where(label == Config.pad_id, 0, 1).to(self.device)
    accuracies *= mask

    # 長さ合計
    accuracies = torch.sum(accuracies, dim=-1) / torch.sum(mask, dim=-1)
    # バッチ平均
    return torch.mean(accuracies)

  def train(self):
    TOTAL_EPOCH = 100

    optimizer = torch.optim.AdamW(self.model.parameters(),
                                  lr=Config.adam_lr, betas=Config.adam_betas)
    scheduler = CosineLRScheduler(optimizer, t_initial=TOTAL_EPOCH, lr_min=Config.lr_min,
                                  warmup_t=1, warmup_lr_init=Config.adam_lr, warmup_prefix=True)
    scaler = torch.GradScaler()

    self.model.train()

    total_steps = 0
    loss_item = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=80),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False
    ) as progress:

      task1 = progress.add_task("Epoch", total=TOTAL_EPOCH)

      for epoch in range(TOTAL_EPOCH):
        task2 = progress.add_task("Train", total=len(self.train_data_loader))
        for steps, (inputs, labels) in enumerate(self.train_data_loader):
          with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=Config.use_amp):
            output = self.model(
              inputs.to(torch.int).to(self.device))

            loss = self._compute_loss(labels, output)
            accuracy = self._compute_accuracy(labels, output)

          scaler.scale(loss).backward()

          if (steps + 1) % Config.accum_iter == 0 or (steps + 1) == len(self.train_data_loader):
            scaler.step(optimizer)
            scaler.update()

          loss_item = loss.item()

          total_steps += 1
          self.writer.add_scalar("train/loss", loss_item, total_steps)
          self.writer.add_scalar("train/accuracy", accuracy.item(), total_steps)

          del loss
          del output
          del inputs, labels
          torch.cuda.empty_cache()

          if total_steps % 100 == 0:
            self.test(total_steps)
            torch.save(self.model.state_dict(), Config.model_path)
            self.model.train()

          progress.update(task2, advance=1)
          progress.refresh()

        scheduler.step(epoch + 1)
        progress.update(task1, advance=1)

  def test(self, steps):
    self.model.eval()
    for inputs, labels in self.validation_data_loader:
      id = self.validation_dataset.sep_id

      decode_output = torch.tensor(id).reshape(1, 1)
      decode_output = torch.cat((inputs, decode_output), dim=-1).to(torch.int).to(self.device)

      for i in range(labels.shape[1]):
        with torch.no_grad():
          output = self.model(
            decode_output
          )

        id = torch.argmax(output[:, -1], dim=-1).reshape(1, 1)
        if id == self.vocab.eos_id():
          break

        decode_output = torch.cat((decode_output, id), dim=-1)

      input_tokens = self.validation_dataset.get_tokens(inputs.to(torch.int).tolist()[0])

      output2 = torch.softmax(output, dim=2)
      indicies = torch.argmax(output2, dim=-1)[0].to(torch.int).tolist()
      label_tokens = self.validation_dataset.get_tokens(indicies)

      dt_now = datetime.datetime.now()
      d = dt_now.strftime('%Y/%m/%d %H:%M:%S')
      with open("validation.txt", "+a") as f:
        f.write(f"{d} ----------------\n")
        f.write(f"{input_tokens}\n")
        f.write(f"{label_tokens}\n")

      del output
      del inputs, labels
      torch.cuda.empty_cache()

      return


if __name__ == '__main__':
  torch.autograd.set_detect_anomaly(True)
  trainer = Trainer()
  trainer.train()
  # trainer.test_output()
