#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python ./scripts/run_experiment.py \
--model_type roberta_mc \
--model_name_or_path roberta-large \
--task_name winogrande \
--do_eval \
--do_lower_case \
--data_dir ./data \
--max_seq_length 128 \
--per_gpu_eval_batch_size 16 \
--per_gpu_train_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--output_dir ./output/roberta-42 \
--do_train \
--logging_steps 0 \
--save_steps 4750 \
--seed 42 \
--data_cache_dir ./output/roberta-cache-42 \
--warmup_pct 0.1 \
--evaluate_during_training \
--overwrite_output_dir