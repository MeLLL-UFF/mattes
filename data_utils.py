import io
import random
import numpy as np
import os
from transformers import AlbertTokenizer

import torch
import shelve

class DataUtil(object):

  def __init__(self, hparams, k=8):
    self.hparams = hparams
    self.tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
    self.hparams.src_vocab_size = len(self.tokenizer)
    self.hparams.pad_id = self.tokenizer.pad_token_id
    self.hparams.unk_id = self.tokenizer.unk_token_id
    self.hparams.bos_id = self.tokenizer.cls_token_id
    self.hparams.eos_id = self.tokenizer.sep_token_id
    self.hparams.mask_id = self.tokenizer.mask_token_id
    self.hparams.pred_probs = torch.FloatTensor([hparams.word_mask, hparams.word_keep, hparams.word_rand])
    self.topk_db0 = shelve.open(f'{hparams.bert_dump0}/topk', 'r')
    self.topk_db1 = shelve.open(f'{hparams.bert_dump1}/topk', 'r')
    self.k = k

    self.trg_i2w, self.trg_w2i = self._build_vocab(self.hparams.trg_vocab)
    self.hparams.trg_vocab_size = len(self.trg_i2w)
    #self.hparams.trg_pad_id = self.trg_w2i["<pad>"]
    self.hparams.trg_pad_id = self.hparams.pad_id
    print("src_vocab_size={}".format(self.hparams.src_vocab_size))
    print("trg_vocab_size={}".format(self.hparams.trg_vocab_size))

    if not self.hparams.decode:

      self.train_size = 0
      self.n_train_batches = 0

      self.train_x0, self.train_y0, _ , self.index0 = self._build_parallel(self.hparams.train_src_file0, self.hparams.train_trg_file)
      self.train_size = len(self.train_x0)
      self.train_x1, self.train_y1, _ , self.index1 = self._build_parallel(self.hparams.train_src_file1, self.hparams.train_trg_file)
      assert self.train_size == len(self.train_x1)

      self.dev_x0, self.dev_y0, _ , _ = self._build_parallel(self.hparams.dev_src_file0, self.hparams.dev_trg_file, is_train=False)
      self.dev_x1, self.dev_y1, _ , _ = self._build_parallel(self.hparams.dev_src_file1, self.hparams.dev_trg_file, is_train=False)
      self.dev_size = len(self.dev_x0)
      assert self.dev_size == len(self.dev_x1)
      self.dev_index = 0
      self.reset_train()
    else:
      #test_src_file = os.path.join(self.hparams.data_path, self.hparams.test_src_file)
      #test_trg_file = os.path.join(self.hparams.data_path, self.hparams.test_trg_file)
      test_src_file = self.hparams.test_src_file
      test_trg_file = self.hparams.test_trg_file
      self.test_x, self.test_y, _ , _ = self._build_parallel(test_src_file, test_trg_file, is_train=False)
      self.test_size = len(self.test_x)
      self.test_index = 0

  def load_pretrained(self, pretrained_emb_file):
    f = open(pretrained_emb_file, 'r', encoding='utf-8')
    header = f.readline().split(' ')
    count = int(header[0])
    dim = int(header[1])
    #matrix = np.zeros((len(w2i), dim), dtype=np.float32)
    matrix = np.zeros((count, dim), dtype=np.float32)
    #i2w = ['<pad>', '<unk>', '<s>', '<\s>']
    i2w = []
    #w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
    w2i = {}

    for i in range(count):
      word, vec = f.readline().split(' ', 1)
      w2i[word] = len(w2i)
      i2w.append(word)
      matrix[i] = np.fromstring(vec, sep=' ', dtype=np.float32)
      #if not word in w2i:
      #  print("{} no in vocab".format(word))
      #  continue
      #matrix[w2i[word]] = np.fromstring(vec, sep=' ', dtype=np.float32)
    return torch.FloatTensor(matrix), i2w, w2i

  def reset_train(self):
    if not self.n_train_batches:
      self.n_train_batches = (self.train_size + self.hparams.batch_size - 1) // self.hparams.batch_size
    self.train_queue = np.random.permutation(self.n_train_batches)
    self.train_index = 0

  def next_train(self):
    start_index = (self.train_queue[self.train_index] * self.hparams.batch_size)
    end_index = min(start_index + self.hparams.batch_size, self.train_size)

    x_train0 = self.train_x0[start_index:end_index]
    x_train1 = self.train_x1[start_index:end_index]

    self.train_index += 1
    batch_size = len(x_train0)
    # pad
    x_train0, _, _, _, _ = self._pad(x_train0, self.hparams.pad_id)
    x_train1, _, _, _, _ = self._pad(x_train1, self.hparams.pad_id)

    if self.train_index >= self.n_train_batches:
      self.reset_train()
      eop = True
    else:
      eop = False

    assert x_train0.size(0) == x_train1.size(0)

    batch, len_ = x_train0.size()
    len1 = x_train1.size(1)
    topk_logit0 = torch.zeros(batch, len_, self.k)
    topk_index0 = torch.zeros(batch, len_, self.k , dtype = torch.long)
    topk_logit1 = torch.zeros(batch, len1, self.k)
    topk_index1 = torch.zeros(batch, len1, self.k , dtype = torch.long)
    for i, j in zip(range(start_index, end_index, 1), range(end_index - start_index)):
      topk_logits0, topk_inds0 = load_topk(self.topk_db0[str(int(self.index0[i]))])
      topk_logits1, topk_inds1 = load_topk(self.topk_db1[str(int(self.index1[i]))])
      topk_logits0 = topk_logits0[:, :self.k].float()
      #print(topk_logits0.size())
      topk_inds0 = topk_inds0[:, :self.k]
      topk_logits1 = topk_logits1[:, :self.k].float()
      topk_inds1 = topk_inds1[:, :self.k]
      
      topk_logit0.data[j, :topk_logits0.size(0), :] = topk_logits0.data
      topk_index0.data[j, :topk_inds0.size(0), :] = topk_inds0.data
      topk_logit1.data[j, :topk_logits1.size(0), :] = topk_logits1.data
      topk_index1.data[j, :topk_inds1.size(0), :] = topk_inds1.data

    if torch.cuda.is_available():
      topk_logit0 = topk_logit0.cuda()
      topk_logit1 = topk_logit1.cuda()
      topk_index0 = topk_index0.cuda()
      topk_index1 = topk_index1.cuda()

    return (x_train0, x_train1, topk_logit0, topk_logit1, topk_index0, topk_index1), batch_size, eop

  def sample_y(self):
    # first how many attrs?
    attn_num = random.randint(1, (self.hparams.trg_vocab_size-1)//2)
    # then select attrs
    y = np.random.binomial(1, 0.5, attn_num)
    y = y + np.arange(attn_num) * 2
    return y.tolist()

  def next_dev(self, dev_batch_size=1, sort=True):
    start_index = self.dev_index
    end_index = min(start_index + dev_batch_size, self.dev_size)
    batch_size = end_index - start_index

    x_dev0 = self.dev_x0[start_index:end_index]
    x_dev1 = self.dev_x1[start_index:end_index]
    y_dev0 = self.dev_y0[start_index:end_index]
    y_dev1 = self.dev_y1[start_index:end_index]
    if sort:
      x_dev0, y_dev0, _ = self.sort_by_xlen(x_dev0, y_dev0)
      x_dev1, y_dev1, _ = self.sort_by_xlen(x_dev1, y_dev1)

    x_dev0, _, _, _, _ = self._pad(x_dev0, self.hparams.pad_id)
    x_dev1, _, _, _, _ = self._pad(x_dev1, self.hparams.pad_id)

    if end_index >= self.dev_size:
      eop = True
      self.dev_index = 0
    else:
      eop = False
      self.dev_index += batch_size

    return (x_dev0, x_dev1),  batch_size, eop

  def reset_test(self, test_src_file, test_trg_file):
    self.test_x, self.test_y, src_len, _ = self._build_parallel(test_src_file, test_trg_file, is_train=False)
    self.test_size = len(self.test_x)
    self.test_index = 0

  def next_test(self, test_batch_size=10):
    start_index = self.test_index
    end_index = min(start_index + test_batch_size, self.test_size)
    batch_size = end_index - start_index

    x_test = self.test_x[start_index:end_index]
    y_test = self.test_y[start_index:end_index]

    x_test, y_test, index = self.sort_by_xlen(x_test, y_test)

    x_test, x_mask, x_count, x_len, x_pos_emb_idxs = self._pad(x_test, self.hparams.pad_id)
    y_test, y_mask, y_count, y_len, y_pos_emb_idxs = self._pad(y_test, self.hparams.trg_pad_id)

    y_neg = 1 - y_test

    if end_index >= self.test_size:
      eop = True
      self.test_index = 0
    else:
      eop = False
      self.test_index += batch_size

    return x_test, x_mask, x_count, x_len, x_pos_emb_idxs, y_test, y_mask, y_count, y_len, y_pos_emb_idxs, y_neg, batch_size, eop, index

  def sort_by_xlen(self, x, y, x_char_kv=None, y_char_kv=None, file_index=None, descend=True):
    x = np.array(x)
    y = np.array(y)
    x_len = [len(i) for i in x]
    index = np.argsort(x_len)
    if descend:
      index = index[::-1]
    x, y = x[index].tolist(), y[index].tolist()
    return x, y, index

  def _pad(self, sentences, pad_id, char_kv=None, char_dim=None, char_sents=None):
    batch_size = len(sentences)
    lengths = [len(s) for s in sentences]
    count = sum(lengths)
    max_len = max(lengths)
    padded_sentences = [s + ([pad_id]*(max_len - len(s))) for s in sentences]

    mask = [[0]*len(s) + [1]*(max_len - len(s)) for s in sentences]
    padded_sentences = torch.LongTensor(padded_sentences)
    mask = torch.ByteTensor(mask)
    pos_emb_indices = [[i+1 for i in range(len(s))] + ([0]*(max_len - len(s))) for s in sentences]
    pos_emb_indices = torch.FloatTensor(pos_emb_indices)
    if torch.cuda.is_available():
      padded_sentences = padded_sentences.cuda()
      pos_emb_indices = pos_emb_indices.cuda()
      mask = mask.cuda()
    return padded_sentences, mask, count, lengths, pos_emb_indices

  def _build_parallel(self, src_file_name, trg_file_name, is_train=True):
    print("loading parallel sentences from {} {}".format(src_file_name, trg_file_name))
    with open(src_file_name, 'r', encoding='utf-8') as f:
      src_lines = f.read().split('\n')
    with open(trg_file_name, 'r', encoding='utf-8') as f:
      trg_lines = f.read().split('\n')
    src_data = []
    trg_data = []
    line_count = 0
    skip_line_count = 0
    src_unk_count = 0
    trg_unk_count = 0

    src_lens = []
    src_unk_id = self.hparams.unk_id
    for src_line, trg_line in zip(src_lines, trg_lines):
      src_tokens = self.tokenizer.tokenize(src_line)
      trg_tokens = trg_line.split()
      if not src_tokens or not trg_tokens:
        skip_line_count += 1
        continue
      #if is_train and not self.hparams.decode and self.hparams.max_len and (len(src_tokens) > self.hparams.max_len or len(trg_tokens) > self.hparams.max_len):
      #  skip_line_count += 1
      #  continue

      src_lens.append(len(src_tokens))
      src_indices, trg_indices = [], []
      src_indices += self.tokenizer.convert_tokens_to_ids(src_tokens)
      if len(src_indices) >= self.hparams.max_length:
        src_indices = src_indices[:(self.hparams.max_length - 1)]

      trg_w2i = self.trg_w2i
      for trg_tok in trg_tokens:
        if trg_tok not in trg_w2i:
          print("trg attribute cannot have oov!")
          exit(0)
        else:
          trg_indices.append(trg_w2i[trg_tok])
        # calculate char ngram emb for trg_tok

      src_indices.append(self.hparams.eos_id)
      src_data.append(src_indices)
      trg_data.append(trg_indices)
      line_count += 1
      if line_count % 10000 == 0:
        print("processed {} lines".format(line_count))
    index = None
    if is_train:
      src_data, trg_data, index = self.sort_by_xlen(src_data, trg_data, descend=False)
    print("src_unk={}, trg_unk={}".format(src_unk_count, trg_unk_count))
    assert len(src_data) == len(trg_data)
    print("lines={}, skipped_lines={}".format(len(src_data), skip_line_count))
    return src_data, trg_data, src_lens, index

  def _build_vocab(self, vocab_file, max_vocab_size=None):
    i2w = []
    w2i = {}
    i = 0
    with open(vocab_file, 'r', encoding='utf-8') as f:
      for line in f:
        w = line.strip()
        #if i == 0 and w != "<pad>":
        #  i2w = ['<pad>', '<unk>', '<s>', '<\s>']
        #  w2i = {'<pad>': 0, '<unk>':1, '<s>':2, '<\s>':3}
        #  i = 4
        w2i[w] = i
        i2w.append(w)
        i += 1
        if max_vocab_size and i >= max_vocab_size:
          break

    #if "<pad>" not in w2i:
    #    w2i["<pad>"] = i
    #    i2w.append("<pad>")
    #assert i2w[self.hparams.pad_id] == '<pad>'
    #assert i2w[self.hparams.unk_id] == '<unk>'
    #assert i2w[self.hparams.bos_id] == '<s>'
    #assert i2w[self.hparams.eos_id] == '<\s>'
    #assert w2i['<pad>'] == self.hparams.pad_id
    #assert w2i['<unk>'] == self.hparams.unk_id
    #assert w2i['<s>'] == self.hparams.bos_id
    #assert w2i['<\s>'] == self.hparams.eos_id
    return i2w, w2i

def load_topk(dump):
    with io.BytesIO(dump) as reader:
        topk = torch.load(reader)
    return topk


