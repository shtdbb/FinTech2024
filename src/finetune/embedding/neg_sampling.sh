#!/bin/bash
# -*- coding: utf-8 -*-

export CUDA_VISIBLE_DEVICES=0

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path "../models/bge-large-zh-v1.5" \
    --input_file "./data/train/query_chunk.jsonl" \
    --output_file "./data/train/query_chunk_neg.jsonl" \
    --range_for_sampling 2-20 \
    --negative_number 5 \
    --use_gpu_for_searching
