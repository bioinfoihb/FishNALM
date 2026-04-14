#!/bin/bash
torchrun --nproc_per_node=4 model_pretrain.py \
  --model_name_or_path /nvme0/wb_ssd0/fish_model/pretrain_fish8/checkpoint-120000 \
  --train_data /nvme0/wb_ssd0/FishNALM/examples_pretrain/test_sequences.txt \
  --is_mlm True \
  --from_scratch True \
  --bf16 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --logging_steps 100 \
  --save_steps 1000 \
  --output_dir /nvme0/wb_ssd0/FishNALM/pretrain_fish_try \
  --model_max_length 512 \
  --max_steps 200 \
