#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0
export PYTHONPATH="$(pwd)"
export TRAIN_FILE=/home/ubuntu/mattes/data/gyafc/cleaned_train_1.txt
export TEST_FILE=/home/ubuntu/mattes/data/gyafc/cleaned_dev_1.txt
export CUDA_VISIBLE_DEVICES=0

python src/run_finetune_albert.py \
    --output_dir=masked_lm/formal-gyafc \
    --model_type=albert \
    --model_name_or_path=albert-large-v2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --block_size 64 \
    --num_train_epochs 100 \
    --line_by_line \
    --adam_epsilon 1e-6 \
    --learning_rate 1e-5 \
    --save_steps 10000 \
    --per_gpu_train_batch_size 64 \
    --warmup_steps 8000