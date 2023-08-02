#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"

python cnn_classify.py \
  --dataset gyafc \
  --output_dir "pretrained_classifer/gyafc2/" \
  --clean_mem_every 5 \
  --reset_output_dir \
  --train_src_file data/gyafc/cleaned_train.txt \
  --train_trg_file data/gyafc/train.attr \
  --dev_src_file data/gyafc/cleaned_dev.txt \
  --dev_trg_file data/gyafc/dev.attr \
  --dev_trg_ref data/gyafc/cleaned_dev.txt \
  --trg_vocab  data/gyafc/attr.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=100 \
  --eval_every=5000 \
  --out_c_list="1,2,3,4" \
  --k_list="3,3,3,3" \
  --batch_size 32 \
  --valid_batch_size=64 \
  --patience 5 \
  --lr_dec 0.8 \
  --dropout 0.3 \
  --cuda \
  --classifer cnn \

