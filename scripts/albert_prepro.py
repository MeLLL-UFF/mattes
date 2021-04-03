"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess text files for C-MLM finetuning
"""
import argparse
import shelve

from tqdm import tqdm


def make_db(src_reader, db):
    print()
    for i, src in tqdm(enumerate(src_reader)):
        src_toks = src.strip().split()
        if src_toks:
            dump = {'src': src_toks}
        db[str(i)] = dump


def main(args):
    # process the dataset
    with open(args.src) as src_reader, \
         shelve.open(args.output, 'n') as db:
        make_db(src_reader, db)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', action='store', required=True,
                        help='line by line text file for source data ')
    #parser.add_argument('--tgt', action='store', required=True,
    #                    help='line by line text file for target data ')
    parser.add_argument('--output', action='store', required=True,
                        help='path to output')
    args = parser.parse_args()
    main(args)