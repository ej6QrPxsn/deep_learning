
import datetime
import io
import torch

from torch.utils.tensorboard import SummaryWriter

import os
import sys

from models.llama import LLaMA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
import torch.nn.utils.rnn as rnn
from datasets import load_dataset
import sentencepiece as spm
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from timm.scheduler import CosineLRScheduler

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def build_vocab(data, vocab_size, model_name=None):
  spm_model = io.BytesIO()
  spm.SentencePieceTrainer.train(
    sentence_iterator=iter(data),
    model_writer=spm_model,
    vocab_size=vocab_size,
    pad_id=Config.pad_id,
    train_extremely_large_corpus=True,
    user_defined_symbols=['<sep>']
  )

  if model_name:
    with open(model_name, 'wb') as f:
      f.write(spm_model.getvalue())

  return spm.SentencePieceProcessor(model_proto=spm_model.getvalue())


def pad_batch_fn(batch):
  list_len = [len(i) for i in batch]
  max_len = max(list_len)

  actual_max_len = max_len - 1
  if actual_max_len % Config.Br != 0:
    rest_len = actual_max_len % Config.Br
    if rest_len != 0:
      actual_max_len += Config.Br - rest_len

  return rnn.pad_packed_sequence(
      rnn.pack_sequence(batch, enforce_sorted=False),
      batch_first=True, padding_value=Config.pad_id, total_length=actual_max_len + 1)[0]


def pad_batch(inputs):
  max_len = len(inputs[0])

  rest_len = max_len % Config.Br
  if rest_len != 0:
    max_len += Config.Br - rest_len

  return rnn.pad_packed_sequence(
      rnn.pack_sequence(inputs, enforce_sorted=False),
      batch_first=True, padding_value=Config.pad_id, total_length=max_len)[0]


class PreTrainDataSet(torch.utils.data.Dataset):
  def __init__(self, vocab, data):
    super(PreTrainDataSet, self).__init__()

    self.vocab = vocab
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    src = self.vocab.encode(self.data[index][:1000])
    inputs = torch.Tensor(
      src
    )
    return inputs


class TrainDataSet(torch.utils.data.Dataset):
  def __init__(self, vocab, src_data, dst_data):
    super(TrainDataSet, self).__init__()

    self.vocab = vocab

    self.src_lines = src_data
    self.dst_lines = dst_data

    self.sep_id = self.vocab.piece_to_id('<sep>')

  def __len__(self):
    return len(self.src_lines)

  def __getitem__(self, index):
    src = self.vocab.encode(self.src_lines[index])
    dst = self.vocab.encode(self.dst_lines[index])
    inputs = torch.Tensor(
      [self.vocab.bos_id()]
      + src
      + [self.sep_id]
      + dst
      + [self.vocab.eos_id()]
    )
    return inputs


class ValidateDataSet(torch.utils.data.Dataset):
  def __init__(self, vocab, src_data, dst_data):
    super(ValidateDataSet, self).__init__()

    self.vocab = vocab

    self.src_lines = src_data
    self.dst_lines = dst_data

    self.sep_id = self.vocab.piece_to_id('<sep>')

  def __len__(self):
    return len(self.src_lines)

  def __getitem__(self, index):
    src = self.vocab.encode(self.src_lines[index])
    dst = self.vocab.encode(self.dst_lines[index])
    inputs = torch.Tensor(
      [self.vocab.bos_id()]
      + src
    )

    labels = torch.Tensor(
      dst
      + [self.vocab.eos_id()]
    )
    return inputs, labels

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
    # pre_train_ja_datasets = load_dataset("wikimedia/wikipedia", "20231101.ja")["train"]["text"]
    # pre_train_en_datasets = load_dataset("wikimedia/wikipedia", "20231101.en")["train"]["text"]
    # self.vocab_size = 20000

    # ja_len = len(pre_train_ja_datasets)

    # vocab_name = "vocab"
    # if os.path.exists(vocab_name):
    #   vocab = spm.SentencePieceProcessor(model_file=vocab_name)
    # else:
    #   vocab = build_vocab(
    #     data=pre_train_ja_datasets + pre_train_en_datasets[:ja_len],
    #     vocab_size=self.vocab_size,
    #     model_name=vocab_name)

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.vocab_size = 16000

    self.model = LLaMA(self.vocab_size, self.device).to(self.device)
    self.writer = SummaryWriter(log_dir="./logs")

    # self.pre_train(vocab, pre_train_ja_datasets)
    # self.pre_train(vocab, pre_train_en_datasets[:ja_len])
    self.train()

  def _compute_loss(self, label, output):
    logits = self.criterion(
      output.reshape(-1, self.vocab_size),
      label.reshape(-1).to(torch.long).to(self.device))

    # パディングマスク
    mask = torch.where(label == Config.pad_id, 0, 1).to(self.device)
    logits = logits.reshape(*label.shape)

    mask_logits = logits * mask
    # 長さ合計
    sum_logits = torch.sum(mask_logits, dim=-1) / torch.sum(mask, dim=-1)
    # バッチ平均
    return torch.mean(sum_logits)

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

  def pre_train(self, vocab, datasets):
    pre_train_dataset = PreTrainDataSet(
      vocab,
      datasets
    )

    data_loader = torch.utils.data.DataLoader(
      pre_train_dataset, batch_size=Config.batch_size, shuffle=True,
      num_workers=2, pin_memory=True, collate_fn=pad_batch_fn)

    optimizer = torch.optim.SGD(self.model.parameters(), lr=3e-3)

    self.model.train()

    TOTAL_EPOCH = 10
    batch_steps = 0

    scheduler = CosineLRScheduler(optimizer, t_initial=len(data_loader) / Config.accum_iter,
                                  lr_min=3e-5, cycle_limit=TOTAL_EPOCH,
                                  warmup_t=Config.warmup_steps, warmup_lr_init=3e-5)

    self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing, reduction="none")

    # for p in self.model.parameters():
    #   p.register_hook(lambda grad: grad.clamp(min=-Config.clip_value, max=Config.clip_value))

    if os.path.exists(Config.pre_train_model_path):
      self.model.load_state_dict(torch.load(Config.pre_train_model_path))
      print("load pre train")

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
        task2 = progress.add_task("Train", total=len(data_loader))

        for steps, batches in enumerate(data_loader):

          inputs = batches[:, :-1]
          labels = batches[:, 1:]

          with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=Config.use_amp):
            output = self.model(
              inputs.to(torch.int).to(self.device))

            loss = self._compute_loss(labels, output)
            loss_item = loss.item()
            loss /= Config.accum_iter

          loss.backward()
          # scaler.scale(loss).backward()

          if (steps + 1) % Config.accum_iter == 0 or (steps + 1) == len(data_loader):

            # scaler.unscale_(optimizer)

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.clip_value)

            # scaler.step(optimizer)
            # scaler.update()
            # if scaler._scale < min_scale:
            #   scaler._scale = torch.tensor(min_scale).to(scaler._scale)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.clip_value)

            optimizer.step()
            optimizer.zero_grad()

            batch_steps += 1

            self.writer.add_scalar("pre_train/lr", scheduler._get_lr(batch_steps)[-1], batch_steps)
            scheduler.step(batch_steps)

          self.writer.add_scalar("pre_train/loss", loss_item, steps + 1)

          if (steps + 1) % 100 == 0:
            torch.save(self.model.state_dict(), Config.pre_train_model_path)
            torch.cuda.empty_cache()

          del loss
          del output
          del inputs, labels

          progress.update(task2, advance=1)
          progress.refresh()

        progress.update(task1, advance=1)

  def train(self):
    datasets = load_dataset("davidstap/ted_talks", "ja_en")

    src_dataset = datasets["train"]["en"]
    dst_dataset = datasets["train"]["ja"]

    vocab = build_vocab(src_dataset + dst_dataset, self.vocab_size)

    train_dataset = TrainDataSet(
      vocab,
      src_dataset,
      dst_dataset
    )

    self.train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=pad_batch_fn)

    self.validation_dataset = ValidateDataSet(
      vocab,
      datasets["validation"]["en"],
      datasets["validation"]["ja"]
    )

    self.validation_data_loader = iter(torch.utils.data.DataLoader(
      self.validation_dataset, batch_size=1, shuffle=True,
      num_workers=2, pin_memory=True))

    self.sep_id = self.validation_dataset.sep_id
    self.eos_id = vocab.eos_id()

    self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing, reduction="none")
    self.scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)

    if os.path.exists(Config.model_path):
      self.model.load_state_dict(torch.load(Config.model_path))

    optimizer = torch.optim.Adam(self.model.parameters(), eps=Config.adam_eps,
                                 lr=Config.adam_lr, betas=Config.adam_betas)

    self.model.train()

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

      TOTAL_EPOCH = 200
      task1 = progress.add_task("Epoch", total=TOTAL_EPOCH)
      # for p in self.model.parameters():
      #   p.register_hook(lambda grad: torch.clamp(grad, -Config.clip_value, Config.clip_value))

      scheduler = CosineLRScheduler(optimizer, t_initial=len(self.train_data_loader) / Config.accum_iter,
                                    lr_min=Config.scheduler_lr_min, cycle_limit=TOTAL_EPOCH,
                                    warmup_t=Config.warmup_steps, warmup_lr_init=Config.scheduler_lr_init)

      batch_steps = 0

      for epoch in range(TOTAL_EPOCH):
        task2 = progress.add_task("Train", total=len(self.train_data_loader))

        for steps, batches in enumerate(self.train_data_loader):

          inputs = batches[:, :-1]
          labels = batches[:, 1:]
          with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=Config.use_amp):
            output = self.model(
              inputs.to(torch.int).to(self.device))

            loss = self._compute_loss(labels, output)
            accuracy = self._compute_accuracy(labels, output)

          (loss / Config.accum_iter).backward()
          # scaler.scale(loss).backward()

          torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.clip_value)

          if (steps + 1) % Config.accum_iter == 0 or (steps + 1) == len(self.train_data_loader):
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()

            batch_steps += 1

            self.writer.add_scalar("train/lr", scheduler._get_lr(batch_steps)[-1], batch_steps)
            scheduler.step(batch_steps)

          if (steps + 1) % 100 == 0:
            self.test(batch_steps)
            torch.save(self.model.state_dict(), Config.model_path)
            self.model.train()
            torch.cuda.empty_cache()

            loss_item = loss.item()

            self.writer.add_scalar("train/loss", loss_item, batch_steps)
            self.writer.add_scalar("train/accuracy", accuracy.item(), batch_steps)

          del loss
          del output
          del inputs, labels

          progress.update(task2, advance=1)
          progress.refresh()

        progress.update(task1, advance=1)

  def test(self, steps):
    self.model.eval()

    for inputs, labels in self.validation_data_loader:

      # input_tokens = self.validation_dataset.get_tokens(inputs.to(torch.int).tolist()[0])
      # label_tokens = self.validation_dataset.get_tokens(labels.to(torch.int).tolist()[0])
      # print("valid ", input_tokens, label_tokens)

      decode_input = torch.tensor(self.sep_id).reshape(1, 1)
      decode_input = torch.cat((inputs, decode_input), dim=-1).to(torch.int).to(self.device)

      for i in range(labels.shape[1]):
        with torch.no_grad():
          output = self.model(pad_batch(decode_input))

        id = torch.argmax(output[:, decode_input.shape[1] - 1], dim=-1).reshape(1, 1)
        if id == self.eos_id:
          break

        decode_input = torch.cat((decode_input, id), dim=-1)

      input_tokens = self.validation_dataset.get_tokens(inputs.to(torch.int).tolist()[0])

      output2 = torch.softmax(output[:, inputs.shape[1] + 1:decode_input.shape[1]], dim=2)
      indicies = torch.argmax(output2, dim=-1)[0].to(torch.int).tolist()
      label_tokens = self.validation_dataset.get_tokens(indicies)

      dt_now = datetime.datetime.now()
      d = dt_now.strftime('%Y/%m/%d %H:%M:%S')
      with open("validation.txt", "+a") as f:
        f.write(f"{d} ----------------\n")
        f.write(f"{input_tokens}\n")
        f.write(f"{label_tokens}\n")

      break


if __name__ == '__main__':
  # torch.autograd.set_detect_anomaly(True)
  trainer = Trainer()
  # trainer.train()
  # trainer.test_output()
