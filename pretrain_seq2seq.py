# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
import numpy as np

#from bleurt import score
#import tensorflow as tf

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BartTokenizer
from transformers import GPT2LMHeadModel

from models import StyleTransformer
from util.helper import optimize
from util.dataset import MattesIterator
#from classifier.textcnn import TextCNN
from util.optim import ScheduledOptim
from main import Config
from data_utils_yelp import DataUtil

device = 'cuda' if cuda.is_available() else 'cpu'


def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths

def evaluate(model, valid_loader, tokenizer, step, train_log_file):
    """
    Evaluation function for fine-tuning BART

    Args:
        model: the BART model.
        valid_loader: pytorch valid DataLoader.
        tokenizer: BART tokenizer
        step: the current training step.

    Returns:
        the average cross-entropy loss
    """
    
    with torch.no_grad():
        model.eval()
        report_tokens = report_loss = report_sents = 0
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            inp_lengths = get_lengths(src, tokenizer.eos_token_id)
            styles = torch.ones_like(src[:, 0])*2
            batch_size = src.size(0)
            report_sents += batch_size
            report_tokens += inp_lengths.sum().item()
            loss_fn = nn.NLLLoss(reduction='none')
            mask = tgt.ne(tokenizer.pad_token_id).float()
            para_log_probs = model(src, tgt, inp_lengths, styles)
            loss = loss_fn(para_log_probs.transpose(1, 2), tgt) * mask
            loss = loss.sum().item()
            report_loss += loss
        model.train()
    with open(train_log_file, 'a') as fl:
        print('[Info] valid {:05d} | loss {:.4f} | '
              'ppl {:.4f}'.format(step, report_loss / report_sents, np.exp(report_loss / report_tokens)
        ), file=fl)
        #print('[Info] valid {:05d} | ppl {:.4f}'.format(step, np.exp(report_loss / report_tokens)))

    return report_loss / report_sents


def main():
    config = Config()
    data = DataUtil(config)
    parser = argparse.ArgumentParser('Supervised training with sentence-pair')
    parser.add_argument('-seed', default=42, type=int, help='the random seed')
    parser.add_argument('-lr', default=1e-5, type=float, help='the learning rate')
    parser.add_argument('-order', default=0, type=str, help='the order of training')
    parser.add_argument('-style', default=0, type=int, help='transfer inf. to for.')
    parser.add_argument('-model', default='bart', type=str, help='the name of model')
    parser.add_argument('-dataset', default='ye', type=str, help='the name of dataset')
    parser.add_argument('-task', default='ye', type=str, help='a specific target task')
    parser.add_argument('-shuffle', default=False, type=bool, help='sort train data by xlen')
    parser.add_argument('-steps', default=300001, type=int, help='force stop at x steps')
    parser.add_argument('-batch_size', default=256, type=int, help='the size in a batch')
    parser.add_argument('-patience', default=3, type=int, help='early stopping fine-tune')
    parser.add_argument('-eval_step', default=10000, type=int, help='evaluate every x step')
    parser.add_argument('-log_step', default=100, type=int, help='print logs every x step')
    parser.add_argument('-max_length', default=64, type=int, help='max lenght of sequences')
    parser.add_argument('-train_log_file', default='checkpoints', type=str, help='path to log file')

    opt = parser.parse_args()
    if opt.task=='fr':
        opt.steps=10001
    print('[Info]', opt)
    torch.manual_seed(opt.seed)

    #base = BartModel.from_pretrained("facebook/bart-base")
    #model = BartForMaskedLM.from_pretrained('facebook/bart-base',
    #                                         config=base.config)
    if config.load_ckpt:
        state_dict = torch.load(config.f_ckpt)
        model = StyleTransformer(config, data)
        model.load_state_dict(state_dict)
    else:
        model = StyleTransformer(config, data)

    model.to(device).train()

    tokenizer = data.tokenizer
    eos_token_id = tokenizer.eos_token_id

    # load data for training
    data_iter = MattesIterator(tokenizer, opt)
    train_loader, valid_loader = data_iter.loader

    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr, config.d_model, len(train_loader))

    tab = 0
    avg_loss = 1e9
    loss_list = []
    start = time.time()
    train_iter = iter(iter(train_loader))
    train_log_file = opt.train_log_file + '/train_log.txt'
    print("Start Training")
        
    for step in range(1, opt.steps):

        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        #print(batch[0][0],batch[1][0])
        src_seq, tgt_seq = map(lambda x: x.to(device), batch)
        inp_lengths = get_lengths(src_seq, eos_token_id)
        styles = torch.ones_like(src_seq[:, 0])*2
        batch_size = src_seq.size(0)
        loss_fn = nn.NLLLoss(reduction='none')

        mask = tgt_seq.ne(tokenizer.pad_token_id).float()
        para_log_probs = model(src_seq, tgt_seq, inp_lengths, styles)
        loss = loss_fn(para_log_probs.transpose(1, 2), tgt_seq) * mask
        loss = loss.sum() / batch_size
        loss_list.append(loss.item())
        optimize(optimizer, loss)

        if step % opt.log_step == 0:
            lr = optimizer._optimizer.param_groups[0]['lr']
            with open(train_log_file, 'a') as fl:
                print('[Info] steps {:05d} | loss {:.4f} | '
                      'lr {:.6f} | second {:.2f}'.format(
                    step, np.mean(loss_list), lr, time.time() - start
                ), file=fl)
            loss_cen_list = []
            start = time.time()

        if step % opt.eval_step == 0:
            eval_loss = evaluate(model, valid_loader, tokenizer, step, train_log_file)
            if avg_loss >= eval_loss:
                model_dir ='checkpoints/{}_{}_{}_{}.chkpt'.format(
                            opt.model, 'fur', opt.task, opt.style)
                torch.save(model.state_dict(), model_dir)
                print('[Info] The checkpoint file has been updated.')
                avg_loss = eval_loss
                tab = 0
            else:
                tab += 1
            if tab == opt.patience:
                exit()

if __name__ == "__main__":
    main()

