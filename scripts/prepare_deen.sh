#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
set -x
set -e
export PYTHONUNBUFFERED=1

TMP=$(pwd)/data/tmp_yelp
DUMP=$(pwd)/data/dump_yelp
RAW=$(pwd)/data/yelp

mkdir -p $TMP


# ALBERT tokenization
python $(pwd)/scripts/albert_tokenize.py \
    --prefixes $RAW/cleaned_train_0.txt $RAW/cleaned_train_1.txt $RAW/cleaned_dev_0.txt $RAW/cleaned_dev_1.txt $RAW/cleaned_test_0.txt $RAW/cleaned_test_1.txt \
    --output_dir $TMP

# prepare albert teacher training dataset
mkdir -p $DUMP
python $(pwd)/scripts/albert_prepro.py --src $TMP/cleaned_train_0.txt.albert \
                                   --output $DUMP/NEGA.db

python $(pwd)/scripts/albert_prepro.py --src $TMP/cleaned_train_1.txt.albert \
                                   --output $DUMP/POSI.db

# OpenNMT preprocessing
#VSIZE=30000
#FREQ=0
#SHARD_SIZE=200000
#python $(pwd)/opennmt/preprocess.py \
#    -train_src $TMP/cleaned_train_0.txt.albert \
#    -train_tgt $TMP/cleaned_train_1.txt.albert\
#    -valid_src $TMP/cleaned_dev_0.txt.albert \
#    -valid_tgt $TMP/cleaned_dev_1.txt.albert \
#    -save_data $DUMP/DEEN \
#    -src_seq_length 64 \
#    -tgt_seq_length 64 \
#    -src_vocab_size $VSIZE \
#    -tgt_vocab_size $VSIZE \
#    -vocab_size_multiple 8 \
#    -src_words_min_frequency $FREQ \
#    -tgt_words_min_frequency $FREQ \
#    -share_vocab \
#    -shard_size $SHARD_SIZE
