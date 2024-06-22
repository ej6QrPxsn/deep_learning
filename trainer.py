
import datetime
import io
import torch

from torch.utils.tensorboard import SummaryWriter

import os
import sys

from models.transformer import Transformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
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
  )

  return spm.SentencePieceProcessor(model_proto=spm_model.getvalue())


def pad_batch_fn(batch):
  inputs = []
  labels = []
  for i, item in enumerate(batch):
    inputs.append(item[0])
    labels.append(item[1])

  return (
    rnn.pad_sequence(inputs, batch_first=True, padding_value=Config.pad_id),
    rnn.pad_sequence(labels, batch_first=True, padding_value=Config.pad_id),
  )


class CustomDataSet(torch.utils.data.Dataset):
  def __init__(self, src_vocab, dst_vocab, src_data, dst_data):
    super(CustomDataSet, self).__init__()

    self.src_vocab = src_vocab
    self.dst_vocab = dst_vocab

    self.src_lines = src_data
    self.dst_lines = dst_data

  def __len__(self):
    return len(self.src_lines)

  def __getitem__(self, index):
    inputs = torch.Tensor(
      [self.dst_vocab.bos_id()]
      + self.src_vocab.encode(self.src_lines[index])
      + [self.dst_vocab.eos_id()]
    )

    outputs = torch.Tensor(
      [self.dst_vocab.bos_id()]
      + self.dst_vocab.encode(self.dst_lines[index])
      + [self.dst_vocab.eos_id()]
    )

    return (inputs, outputs)

  def get_src_tokens(self, indicies):
    return self.src_vocab.decode(indicies)

  def get_dst_tokens(self, indicies):
    return self.dst_vocab.decode(indicies)


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

    self.src_vocab_size = 10000
    self.dst_vocab_size = 10000
    src_vocab = build_vocab(src_dataset, self.src_vocab_size)
    self.dst_vocab = build_vocab(dst_dataset, self.dst_vocab_size)

    train_dataset = CustomDataSet(
      src_vocab,
      self.dst_vocab,
      src_dataset,
      dst_dataset
    )

    self.train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=pad_batch_fn)

    validation_dataset = CustomDataSet(
      src_vocab,
      self.dst_vocab,
      datasets["validation"]["en"],
      datasets["validation"]["ja"]
    )

    self.validation_data_loader = torch.utils.data.DataLoader(
      validation_dataset, batch_size=1, shuffle=True,
      num_workers=2, pin_memory=True, collate_fn=pad_batch_fn)

    self.test_dataset = CustomDataSet(
      src_vocab,
      self.dst_vocab,
      datasets["test"]["en"],
      datasets["test"]["ja"]
    )

    self.test_data_loader = torch.utils.data.DataLoader(
      self.test_dataset, batch_size=1, shuffle=True, collate_fn=pad_batch_fn)

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.model = Transformer(self.src_vocab_size, self.dst_vocab_size, self.device).to(self.device)

    self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing, reduction="none")
    self.scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)

    self.writer = SummaryWriter(log_dir="./logs")

    if os.path.exists(Config.model_path):
      self.model.load_state_dict(torch.load(Config.model_path))

  def _compute_loss(self, label, output):
    logits = self.criterion(
      output.view(-1, self.dst_vocab_size),
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
    optimizer = torch.optim.Adam(self.model.parameters(),
                                 lr=Config.adam_lr, betas=Config.adam_betas)
    scheduler = TransformerScheduler(optimizer)
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

      TOTAL_EPOCH = 30
      task1 = progress.add_task("Epoch", total=TOTAL_EPOCH)

      for epoch in range(TOTAL_EPOCH):
        task2 = progress.add_task("Train", total=len(self.train_data_loader))
        for steps, (inputs, labels) in enumerate(self.train_data_loader):
          decoder_input = labels[:, :-1]
          output_label = labels[:, 1:]
          with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=Config.use_amp):
            output = self.model(
              inputs.to(torch.int).to(self.device),
              decoder_input.to(torch.int).to(self.device))

            loss = self._compute_loss(output_label, output)
            accuracy = self._compute_accuracy(output_label, output)

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

          scheduler.step()
        progress.update(task1, advance=1)

  def test(self, steps):
    self.model.eval()
    for inputs, labels in self.validation_data_loader:
      input = inputs.to(torch.int).to(self.device)

      id = self.dst_vocab.bos_id()

      decode_output = torch.tensor(id).reshape(1, 1).to(self.device)

      for i in range(labels.shape[1]):
        with torch.no_grad():
          output = self.model(
            input,
            decode_output
          )

        id = torch.argmax(output[:, -1], dim=-1).reshape(1, 1)
        if id == self.dst_vocab.eos_id():
          break

        decode_output = torch.cat((decode_output, id), dim=-1)

      input_tokens = self.test_dataset.get_src_tokens(inputs.to(torch.int).tolist()[0])

      output2 = torch.softmax(output, dim=2)
      indicies = torch.argmax(output2, dim=-1)[0].to(torch.int).tolist()
      label_tokens = self.test_dataset.get_dst_tokens(indicies)

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
