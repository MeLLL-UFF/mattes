import os
import sys
import time
import gc

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def memReport():
  for obj in gc.get_objects():
    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
      print(type(obj), obj.size())

def reorder(x, index):
  """original x is reordered in terms of index to get x,
  this function is to recover original index

  Args:
    x: list
    index: numpy array, index[i] == j means the ith element
           in x was located at j originally
  """

  assert(len(x) == len(index))
  new_x = [0 for _ in range(len(x))]

  for i, j in enumerate(index):
    new_x[j] = x[i]

  return new_x

def get_criterion(hparams):
  loss_reduce = False
  crit = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, size_average=False, reduce=loss_reduce)
  if hparams.cuda:
    crit = crit.cuda()
  return crit

def get_performance(crit, trans_logits, noise_logits, labels, hparams, x_len, sum_loss=True):
  # average over length
  x_len_t = torch.tensor(x_len, dtype=torch.float, requires_grad=False, device=hparams.device)
  x_len_t = x_len_t - 1
  batch_size = len(x_len)
  mask = (labels == hparams.pad_id)
  if hparams.bt:
    trans_logits = trans_logits.view(-1, hparams.src_vocab_size)
    trans_loss = crit(trans_logits, labels)
    trans_loss = trans_loss.view(batch_size, -1).sum(-1)
    _, trans_preds = torch.max(trans_logits, dim=1)
    trans_acc = torch.eq(trans_preds, labels).int().masked_fill_(mask, 0).sum().item()
  else:
    trans_loss = torch.zeros((batch_size), requires_grad=False, device=hparams.device)
    trans_acc = 0    

  if hparams.noise_flag:
    noise_logits = noise_logits.view(-1, hparams.src_vocab_size)
    noise_loss = crit(noise_logits, labels)
    noise_loss = noise_loss.view(batch_size, -1).sum(-1)
    _, preds = torch.max(noise_logits, dim=1)
    acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum().item()
  else:
    noise_loss = torch.zeros((batch_size), requires_grad=False, device=hparams.device)
    acc = 0

  if hparams.avg_len:
    noise_loss = noise_loss / x_len_t
    trans_loss = trans_loss / x_len_t

  trans_loss = trans_loss.sum()
  noise_loss = noise_loss.sum()
  loss = trans_loss + hparams.noise_weight * noise_loss
  #loss = noise_loss.sum()
  return loss, trans_loss, noise_loss, acc, trans_acc

def count_params(params):
  num_params = sum(p.data.nelement() for p in params)
  return num_params

def save_checkpoint(extra, model, optimizer, hparams, path):
  print("Saving model to '{0}'".format(path))
  torch.save(extra, os.path.join(path, "extra.pt"))
  torch.save(model, os.path.join(path, "model.pt"))
  torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
  torch.save(hparams, os.path.join(path, "hparams.pt"))
  torch.save(model.state_dict(), os.path.join(path, "model.dict"))

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr

def init_param(p, init_type="uniform", init_range=None):
  if init_type == "xavier_normal":
    init.xavier_normal(p)
  elif init_type == "xavier_uniform":
    init.xavier_uniform(p)
  elif init_type == "kaiming_normal":
    init.kaiming_normal(p)
  elif init_type == "kaiming_uniform":
    init.kaiming_uniform(p)
  elif init_type == "uniform":
    #assert init_range is not None and init_range > 0
    init.uniform_(p, -init_range, init_range)
  else:
    raise ValueError("Unknown init_type '{0}'".format(init_type))


def get_attn_subsequent_mask(seq, pad_id=0):
  """ Get an attention mask to avoid using the subsequent info."""

  assert seq.dim() == 2
  batch_size, max_len = seq.size()
  sub_mask = torch.triu(
    torch.ones(max_len, max_len), diagonal=1).unsqueeze(0).repeat(
      batch_size, 1, 1).type(torch.ByteTensor)
  if seq.is_cuda:
    sub_mask = sub_mask.cuda()
  return sub_mask

def grad_clip(params, grad_bound=None):
  """Clipping gradients at L-2 norm grad_bound. Returns the L-2 norm."""

  params = list(filter(lambda p: p.grad is not None, params))
  total_norm = 0
  for p in params:
    if p.grad is None:
      continue
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm ** 2
  total_norm = total_norm ** 0.5

  if grad_bound is not None:
    clip_coef = grad_bound / (total_norm + 1e-6)
    if clip_coef < 1:
      for p in params:
        p.grad.data.mul_(clip_coef)
  return total_norm

def tensor2text(data, tensor):
    tensor = tensor.cpu().numpy()
    text = []
    for sample in tensor:
        sample = data.tokenizer.decode(sample)
        sample = sample.split('[SEP]')[0]
        text.append(sample)

    return text

def calc_ppl(log_probs, tokens_mask):
    return (log_probs.sum() / tokens_mask.sum()).exp()

def idx2onehot(x, num_classes):
    y = x.unsqueeze(-1)
    x_onehot = torch.zeros_like(y.expand(x.size() + torch.Size((num_classes, ))))
    x_onehot.scatter_(-1, y, 1)
    return x_onehot.float()

def word_shuffle(x, l, shuffle_len):
    if not shuffle_len:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    pad_mask = (pos_idx >= l.unsqueeze(1)).float()

    scores = pos_idx.float() + ((1 - pad_mask) * noise + pad_mask) * shuffle_len
    x2 = x.clone()
    x2 = x2.gather(1, scores.argsort(1))

    return x2



def word_dropout_raw(x, l, unk_drop_prob, rand_drop_prob, vocab):
    if not unk_drop_prob and not rand_drop_prob:
        return x

    assert unk_drop_prob + rand_drop_prob <= 1

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < l.unsqueeze(1)

    x2 = x.clone()
    
    # drop to <unk> token
    if unk_drop_prob:
        unk_idx = vocab.stoi['<unk>']
        unk_drop_mask = (noise < unk_drop_prob) & token_mask
        x2.masked_fill_(unk_drop_mask, unk_idx)

    # drop to random_mask
    if rand_drop_prob:
        rand_drop_mask = (noise > 1 - rand_drop_prob) & token_mask
        rand_tokens = torch.randint_like(x, len(vocab))
        rand_tokens.masked_fill_(1 - rand_drop_mask, 0)
        x2.masked_fill_(rand_drop_mask, 0)
        x2 = x2 + rand_tokens
    
    return x2

def unk_dropout_(x, l, drop_prob, unk_idx):
    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < l.unsqueeze(1)
    unk_drop_mask = (noise < drop_prob) & token_mask
    x.masked_fill_(unk_drop_mask, unk_idx)

def rand_dropout_(x, l, drop_prob, vocab_size):
    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < l.unsqueeze(1)
    rand_drop_mask = (noise < drop_prob) & token_mask
    rand_tokens = torch.randint_like(x, vocab_size)
    rand_tokens.masked_fill_(1 - rand_drop_mask, 0)
    x.masked_fill_(rand_drop_mask, 0)
    x += rand_tokens

def word_dropout_new(x, l, unk_drop_fac, rand_drop_fac, drop_prob, vocab):
    if not unk_drop_fac and not rand_drop_fac:
        return x

    assert unk_drop_fac + rand_drop_fac <= 1

    batch_size = x.size(0)
    unk_idx = vocab.stoi['<unk>']
    unk_drop_idx = int(batch_size * unk_drop_fac)
    rand_drop_idx = int(batch_size * rand_drop_fac)

    shuffle_idx = torch.argsort(torch.rand(batch_size))
    orignal_idx = torch.argsort(shuffle_idx)

    x2 = x.clone()
    x2 = x2[shuffle_idx]
    
    if unk_drop_idx:
        unk_dropout_(x2[:unk_drop_idx], l[:unk_drop_idx], drop_prob, unk_idx)

    if rand_drop_idx:
        rand_dropout_(x2[-rand_drop_idx:], l[-rand_drop_idx:], drop_prob, len(vocab))

    x2 = x2[orignal_idx]

    return x2

def word_dropout(x, l, drop_prob, unk_idx):
    if not drop_prob:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < l.unsqueeze(1)

    drop_mask = (noise < drop_prob) & token_mask
    x2 = x.clone()
    x2.masked_fill_(drop_mask, unk_idx)
    
    return x2

def word_drop(x, l, drop_prob):
    if not drop_prob:
        return x

    noise = torch.rand(x.size(), dtype=torch.float).to(x.device)
    pos_idx = torch.arange(x.size(1)).unsqueeze(0).expand_as(x).to(x.device)
    token_mask = pos_idx < (l.unsqueeze(1) - 1)

    drop_mask = (noise < drop_prob) & token_mask
    x2 = x.clone()
    pos_idx.masked_fill_(drop_mask, x.size(1) - 1)
    pos_idx = torch.sort(pos_idx, 1)[0]
    x2 = x2.gather(1, pos_idx)
    
    return x2

def add_noise(words, lengths, shuffle_len, drop_prob, unk_idx):
    words = word_shuffle(words, lengths, shuffle_len)
    words = word_dropout(words, lengths, drop_prob, unk_idx)
    return words 

def kd_loss(log_prob, teacher_outputs, temperature, mask):
    """ our own temp scaling """
    # NOTE: the temperature scaling is kind of non-standard, as we observe
    #       better empirical performance this way
    T = temperature
    topk_prob, topk_idx = teacher_outputs
    topk_prob = topk_prob.view(-1,topk_prob.size(2))
    topk_idx = topk_idx.view(-1,topk_idx.size(2))
    topk_prob = F.softmax(topk_prob/T, dim=-1)
    loss = -(log_prob.gather(dim=-1, index=topk_idx) * topk_prob)[mask.bool()].sum()
    return loss
