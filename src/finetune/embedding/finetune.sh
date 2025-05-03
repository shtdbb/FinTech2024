#!/bin/bash
# -*- coding: utf-8 -*-

export CUDA_VISIBLE_DEVICES=0
current_datetime=$(date +%Y_%m_%d_%H_%M_%S)

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir ../models/checkpoint_$current_datetime \
    --model_name_or_path ../models/bge-large-zh-v1.5 \
    --train_data "./data/train/query_chunk_neg.jsonl" \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 128 \
    --passage_max_len 512 \
    --train_group_size 6 \
    --negatives_cross_device \
    --logging_steps 3 \
    --query_instruction_for_retrieval "为这个句子生成表示以用于检索相关文章："
