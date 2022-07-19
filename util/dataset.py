# -*- coding: utf-8 -*-

import random
import time
import numpy as np

import torch
import torch.utils.data


class MattesDataset(torch.utils.data.Dataset):
    """ Seq2Seq Dataset """

    def __init__(self, src_inst, tgt_inst, batch_size):

        self.src_inst = src_inst
        self.tgt_inst = tgt_inst
        self.batch_size = batch_size
        self.n_train_batches = (len(self.src_inst) + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self.n_train_batches

    def __getitem__(self, idx):
        id0=idx * self.batch_size
        id1=min(len(self.src_inst),(idx+1) * self.batch_size)
        return self.src_inst[id0:id1], self.tgt_inst[id0:id1]


class MattesIterator(object):
    """ Data iterator for fine-tuning BART """

    def __init__(self, tokenizer, opt):

        self.tokenizer = tokenizer
        self.opt = opt

        self.train_src, self.train_tgt = self.read_insts('train', opt.shuffle, opt)
        self.valid_src, self.valid_tgt = self.read_insts('valid', False, opt)
        print('[Info] {} instances from train set'.format(len(self.train_src)))
        print('[Info] {} instances from valid set'.format(len(self.valid_src)))

        self.loader = self.gen_loader(self.train_src, self.train_tgt, 
                                      self.valid_src, self.valid_tgt)

    def read_insts(self, mode, shuffle, opt):
        """
        Read instances from input file
        Args:
            mode (str): 'train' or 'valid'.
            shuffle (bool): whether randomly shuffle training data.
            opt: it contains the information of transfer direction.
        Returns:
            src_seq: list of the lists of token ids for each source sentence.
            tgt_seq: list of the lists of token ids for each tgrget sentence.
        """

        src_dir = 'data/{}/{}.{}'.format(opt.dataset, mode, opt.style)
        tgt_dir = 'data/{}/{}.{}'.format(opt.dataset, mode, bool(opt.style-1).real)

        src_seq, tgt_seq = [], []
        with open(src_dir, 'r') as f1, open(tgt_dir, 'r') as f2:
            start = time.time()
            '''
            f1 = f1.readlines()
            f2 = f2.readlines()
            if shuffle:
                random.seed(opt.seed)
                random.shuffle(f1)
                random.shuffle(f2)
            for i in range(len(f1)):
            '''
            for (i, (line1, line2)) in enumerate(zip(f1, f2)):
                #print(i)
                #print(line1,"\n", line2)
                s = self.tokenizer.encode(line1.strip()[:150])
                t = self.tokenizer.encode(line2.strip()[:150])
                s=s[1:]
                t=t[1:]
                if len(s) > self.opt.max_length:
                    s = s[:(self.opt.max_length)]
                    s[self.opt.max_length-1] = self.tokenizer.sep_token_id
                if len(t) > self.opt.max_length:
                    t = t[:(self.opt.max_length)]
                    t[self.opt.max_length-1] = self.tokenizer.sep_token_id
                src_seq.append(s)
                tgt_seq.append(t)
                #if i==320000:
                #    break
            end = time.time()
            print("Execution time in seconds: ",(end-start))
            if shuffle:
                src_seq, index = self.sort_by_xlen(src_seq)
                tgt_seq = np.array(tgt_seq)[index].tolist()

            return src_seq, tgt_seq


    def gen_loader(self, train_src, train_tgt, valid_src, valid_tgt):
        """Generate pytorch DataLoader."""

        train_loader = torch.utils.data.DataLoader(
            MattesDataset(
                src_inst=train_src,
                tgt_inst=train_tgt,
                batch_size=self.opt.batch_size),
            num_workers=2,
            batch_size=1,
            collate_fn=self.paired_collate_fn,
            shuffle=True)

        valid_loader = torch.utils.data.DataLoader(
            MattesDataset(
                src_inst=valid_src,
                tgt_inst=valid_tgt,
                batch_size=self.opt.batch_size),
            num_workers=2,
            batch_size=1,
            collate_fn=self.paired_collate_fn)

        return train_loader, valid_loader


    def collate_fn(self, insts):
        """Pad the instance to the max seq length in batch"""

        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(inst) for inst in insts)

        batch_seq = [inst + [pad_id]*(max_len - len(inst))
                     for inst in insts]
        batch_seq = torch.LongTensor(batch_seq)

        return batch_seq


    def paired_collate_fn(self, insts):
        src_inst, tgt_inst = list(zip(*insts))
        src_inst = src_inst[0]
        tgt_inst = tgt_inst[0]
        src_inst = self.collate_fn(src_inst)
        tgt_inst = self.collate_fn(tgt_inst)

        return src_inst, tgt_inst

    def sort_by_xlen(self, x, descend=False):
        x = np.array(x)
        x_len = [len(i) for i in x]
        index = np.argsort(x_len)
        if descend:
          index = index[::-1]
        x= x[index].tolist()
        return x, index


