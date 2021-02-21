"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

precompute hidden states of CMLM teacher to speedup KD training
"""
import argparse
import io
import os
import shelve

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#from pytorch_pretrained_bert import BertTokenizer
from toolz.sandbox import unzip
from transformers import AlbertTokenizer, AlbertForMaskedLM

#from cmlm.model import BertForSeq2seq
#from cmlm.data import convert_token_to_bert, CLS, SEP, MASK


def tensor_dumps(tensor):
    with io.BytesIO() as writer:
        np.save(writer, tensor.cpu().numpy().astype(np.float16),
                allow_pickle=False)
        dump = writer.getvalue()
    return dump


def gather_hiddens(hiddens, masks):
    outputs = []
    for hid, mask in zip(hiddens.split(1, dim=1), masks.split(1, dim=1)):
        if mask.sum().item() == 0:
            continue
        mask = mask.unsqueeze(-1).expand_as(hid)
        mask = mask.to(dtype=torch.bool)
        outputs.append(hid.masked_select(mask))
    output = torch.stack(outputs, dim=0)
    return output


class BertSampleDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, style, num_samples=7):
        self.db = shelve.open(corpus_path, 'r')
        self.ids = []
        self.style = style
        if style == 'classic':
            for i, ex in self.db.items():
                if len(ex['src']) <= 10000:
                    self.ids.append(i)
        else:
            for i, ex in self.db.items():
                if len(ex['tgt']) <= 10000:
                    self.ids.append(i)
        self.toker = tokenizer
        self.num_samples = num_samples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.db[id_]
        if self.style == 'classic':
            features = convert_example(example['src'], self.toker, self.num_samples)
        else:
            features = convert_example(example['tgt'], self.toker, self.num_samples)

        return (id_, ) + features


def convert_example(tgt, toker, num_samples):
    #src = [convert_token_to_bert(tok) for tok in src]
    tgt = tgt + ['[SEP]']

    # build the random masks
    tgt_len = len(tgt)
    if tgt_len <= num_samples:
        masks = torch.eye(tgt_len).byte()
        num_samples = tgt_len
    else:
        mask_inds = [list(range(i, tgt_len, num_samples))
                     for i in range(num_samples)]
        masks = torch.zeros(num_samples, tgt_len).byte()
        for i, indices in enumerate(mask_inds):
            for j in indices:
                masks.data[i, j] = 1
    assert (masks.sum(dim=0) != torch.ones(tgt_len).long()).sum().item() == 0
    assert masks.sum().item() == tgt_len
    masks = torch.cat([torch.zeros(num_samples, 1).byte(), masks],
                      dim=1)
    masks = masks.to(dtype=torch.bool)

    # make BERT inputs
    input_ids = toker.convert_tokens_to_ids(['[CLS]'] + tgt)
    mask_id = toker.convert_tokens_to_ids(['[MASK]'])[0]
    input_ids = torch.tensor([input_ids for _ in range(num_samples)])
    input_ids.data.masked_fill_(masks, mask_id)
    token_ids = torch.tensor([[0] * (len(tgt) + 1)
                              for _ in range(num_samples)])
    return input_ids, token_ids, masks


def batch_features(features):
    ids, all_input_ids, all_token_ids, all_masks = map(list, unzip(features))
    #print(all_input_ids, 'teste')
    batch_size = sum(input_ids.size(0) for input_ids in all_input_ids)
    max_len = max(input_ids.size(1) for input_ids in all_input_ids)
    input_ids = torch.zeros(batch_size, max_len).long()
    token_ids = torch.zeros(batch_size, max_len).long()
    attn_mask = torch.zeros(batch_size, max_len).long()
    i = 0
    for inp, tok in zip(all_input_ids, all_token_ids):
        block, len_ = inp.size()
        input_ids.data[i: i+block, :len_] = inp.data
        token_ids.data[i: i+block, :len_] = tok.data
        attn_mask.data[i: i+block, :len_].fill_(1)
        i += block
    return ids, input_ids, token_ids, attn_mask, all_masks


def process_batch(batch, bert, toker, num_samples=7):
    input_ids, token_ids, attn_mask, all_masks = batch
    input_ids = input_ids.cuda()
    token_ids = token_ids.cuda()
    attn_mask = attn_mask.cuda()
    hiddens = bert.albert(input_ids, token_type_ids=token_ids, attention_mask=attn_mask)
    hiddens = hiddens[0]
    hiddens = bert.predictions.dense(hiddens)
    hiddens = bert.predictions.activation(hiddens)
    hiddens = bert.predictions.LayerNorm(hiddens)

    i = 0
    outputs = []
    for masks in all_masks:
        block, len_ = masks.size()
        hids = hiddens[i:i+block, :len_, :]
        masks = masks.cuda()
        outputs.append(gather_hiddens(hids, masks))
        i += block
    return outputs


def build_db_batched(corpus, out_db, bert, toker, style, batch_size=8):
    dataset = BertSampleDataset(corpus, toker, style)
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=4, collate_fn=batch_features)
    with tqdm(desc='computing ALBERT features', total=len(dataset)) as pbar:
        for ids, *batch in loader:
            outputs = process_batch(batch, bert, toker)
            for id_, output in zip(ids, outputs):
                out_db[id_] = tensor_dumps(output)
            pbar.update(len(ids))


def main(opts):
    # load BERT
    state_dict = torch.load(opts.ckpt)
    vsize = state_dict['predictions.decoder.weight'].size(0)
    albert = AlbertForMaskedLM.from_pretrained('albert-large-v2').eval().half().cuda()
    #bert.update_output_layer_by_size(vsize)
    albert.load_state_dict(state_dict)
    toker = AlbertTokenizer.from_pretrained('albert-large-v2')

    # save the final projection layer
    linear = torch.nn.Linear(albert.config.embedding_size, albert.config.vocab_size)
    linear.weight.data = state_dict['predictions.decoder.weight']
    linear.bias.data = state_dict['predictions.bias']
    os.makedirs(opts.output)
    torch.save(linear, f'{opts.output}/linear.pt')

    # create DB
    with shelve.open(f'{opts.output}/db') as out_db, \
            torch.no_grad():
        build_db_batched(opts.db, out_db, albert, toker, opts.style)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', required=True, choices=['classic', 'modern'])
    parser.add_argument('--ckpt', required=True, help='ALBERT checkpoint')
    parser.add_argument('--db', required=True, help='dataset to compute')
    parser.add_argument('--output', required=True, help='path to dump output')
    args = parser.parse_args()

    main(args)