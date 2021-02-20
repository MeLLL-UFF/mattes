"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

use BERT tokenizer to process seq2seq data
"""
import argparse
import glob
import gzip
import multiprocessing as mp
import os
from os.path import basename, exists, join

from transformers import AlbertTokenizer, BertTokenizer
from tqdm import tqdm
from cytoolz import curry, partition_all


IN_WORD = '@@'  # This prefix is used for reconstructing the original
                # tokenization after generation. (BERT tokenizer does not
                # preserve white spaces)
                # it seems not conflicting for the corpus we test on

# special chars in moses tokenizer
MOSES_SPECIALS = {'&amp;': '&', '&#124;': '|', '&lt;': '<', '&gt;': '>',
                  '&apos;': "'", '&quot;': '"', '&#91;': '[', '&#93;': ']'}
HYPHEN = '@-@'

UNK = '<unk>'

BUF = 65536
CHUNK = 4096


@curry
def tokenize(albert_toker, line):
    
    return albert_toker.tokenize(line)


def write(writer, tokens):
    writer.write(' '.join(tokens) + '\n')


def process(reader, writer, tokenizer):
    with mp.Pool() as pool, tqdm(desc='tokenizing') as pbar:
        for lines in partition_all(BUF, reader):
            for tokens in pool.imap(tokenize(tokenizer), lines,
                                    chunksize=CHUNK):
                write(writer, tokens)
            pbar.update(len(lines))


def main(opts):
    #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    tokenizer = AlbertTokenizer.from_pretrained('albert-large-v2')
    for prefix in opts.prefixes:
        input_files = glob.glob(f'{prefix}*')
        if not exists(opts.output_dir):
            os.makedirs(opts.output_dir)
        for input_file in input_files:
            if input_file.endswith('.gz'):
                out_name = basename(input_file)[:-3]
                reader = gzip.open(input_file, 'rt')
            else:
                out_name = basename(input_file)
                reader = open(input_file, 'r')
            output_file = join(opts.output_dir, f'{out_name}.albert')
            with open(output_file, 'w') as writer:
                process(reader, writer, tokenizer)
            reader.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefixes', action='store', required=True, nargs='+',
                        help='line by line text file for data '
                             '(will apply to all prefix)')
    parser.add_argument('--output_dir', action='store', required=True,
                        help='path to output')
    args = parser.parse_args()
    main(args)
